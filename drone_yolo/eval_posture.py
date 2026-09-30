"""
eval_posture.py — 자세 분류 판정 (상태 인지 A2 · 2026-09-30)

  장소 분리 평가셋에서만 본다 (make_posture_crops.py 가 만든 정답 박스 크롭):
    aihub_test     AI-Hub 산악5 (= test_kr 장소 · 수직 90° · 한국)
    okutama_obl    Okutama 비스듬 25편 (학습 미사용 · 일본)
    okutama_nadir  Okutama 수직 15편

  지표
    · 자세별 재현율 · 혼동 행렬 · 매크로 F1 (불균형 — 정확도 하나로 보지 않는다)
    · 누움 vs 나머지 AUROC (누움 확률) · 서기 오인 10 % 에서 누움 재현율
    · 기준선 = 박스 모양만 (비스듬: 세로/가로 · 수직: 길쭉함) — 분류기가 이걸 넘어야 의미가 있다
    · 95 % 구간 = 영상(Okutama) · 고도(AI-Hub) 블록 부트스트랩 1,000회

실행: python eval_posture.py weights/s1_aihub.pt [weights/…] [--limit 50]
출력: metrics/posture_<모델>.json + 콘솔 표
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
EVAL = BASE / "data" / "pose_eval"
SETS = ("aihub_test", "okutama_obl", "okutama_nadir")
CLASSES = ("lying", "sitting", "standing")


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return float("nan")
    r = np.argsort(np.argsort(np.r_[pos, neg], kind="mergesort"), kind="mergesort") + 1
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def recall_at_fpr(score, is_lying, is_standing, fpr=0.10):
    """서기를 누움으로 잘못 부르는 비율이 fpr 이하가 되는 임계에서 누움 재현율."""
    th = np.quantile(score[is_standing], 1 - fpr) if is_standing.any() else np.inf
    return float((score[is_lying] > th).mean()) if is_lying.any() else float("nan")


def metrics(y, prob, geo, block, rng=None):
    """y: 정답 인덱스 · prob: (n,3) · geo: 박스 모양 점수 (클수록 누움) · block: 부트스트랩 블록."""
    pred = prob.argmax(1)
    cm = np.zeros((3, 3), int)
    for t, p in zip(y, pred):
        cm[t, p] += 1
    rec = cm.diagonal() / np.maximum(cm.sum(1), 1)
    prec = cm.diagonal() / np.maximum(cm.sum(0), 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    ly, st = y == 0, y == 2
    out = {"n": {c: int((y == i).sum()) for i, c in enumerate(CLASSES)},
           "재현율": {c: round(float(rec[i]), 3) for i, c in enumerate(CLASSES)},
           "정밀도": {c: round(float(prec[i]), 3) for i, c in enumerate(CLASSES)},
           "매크로F1": round(float(f1.mean()), 3),
           "혼동(행=정답·열=예측 lying·sitting·standing)": cm.tolist(),
           "누움AUROC_분류기": round(auc(prob[ly, 0], prob[~ly, 0]), 3),
           "누움AUROC_박스모양": round(auc(geo[ly], geo[~ly]), 3),
           "누움재현율@서기오인10%_분류기": round(recall_at_fpr(prob[:, 0], ly, st), 3),
           "누움재현율@서기오인10%_박스모양": round(recall_at_fpr(geo, ly, st), 3)}
    if rng is not None:                                   # 블록 부트스트랩 — 누움 AUROC · 매크로 F1
        ub = np.unique(block)
        idx = {b: np.where(block == b)[0] for b in ub}
        A, G, F = [], [], []
        for _ in range(1000):
            s = np.concatenate([idx[b] for b in rng.choice(ub, len(ub))])
            yy, pp, gg = y[s], prob[s], geo[s]
            l = yy == 0
            if l.any() and (~l).any():
                A.append(auc(pp[l, 0], pp[~l, 0])); G.append(auc(gg[l], gg[~l]))
            pr = pp.argmax(1)
            f = []
            for i in range(3):
                tp = ((pr == i) & (yy == i)).sum(); rr = tp / max((yy == i).sum(), 1); qq = tp / max((pr == i).sum(), 1)
                f.append(2 * rr * qq / max(rr + qq, 1e-9))
            F.append(np.mean(f))
        ci = lambda v: [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)] if v else None
        out["95%구간"] = {"누움AUROC_분류기": ci(A), "누움AUROC_박스모양": ci(G), "매크로F1": ci(F),
                         "블록": f"{len(ub)}개"}
    return out


def load(set_name, limit=None):
    rows = list(csv.DictReader(open(EVAL / set_name / "crops.csv")))
    if limit:
        rng = np.random.default_rng(0)
        rows = [rows[i] for i in sorted(rng.choice(len(rows), min(limit, len(rows)), replace=False))]
    y = np.array([CLASSES.index(r["pose"]) for r in rows])
    w = np.array([float(r["w1080"]) for r in rows]); h = np.array([float(r["h1080"]) for r in rows])
    nadir = rows[0]["view"] == "nadir"
    # 박스 모양 기준선 — 비스듬: 납작할수록 누움 (−h/w) · 수직: 길쭉할수록 누움 (긴/짧은)
    geo = np.maximum(w, h) / np.minimum(w, h) if nadir else -h / w
    block = np.array([r["place"] if r["source"] == "okutama" else str(r["alt"]) for r in rows])
    return [str(EVAL / set_name / r["file"]) for r in rows], y, geo, block


def predict(model, files, imgsz=128, bs=512):
    probs = []
    for i in range(0, len(files), bs):
        for r in model(files[i:i + bs], imgsz=imgsz, verbose=False):
            names = r.names
            p = r.probs.data.float().cpu().numpy()
            probs.append([p[[k for k, v in names.items() if v == c][0]] for c in CLASSES])
    return np.array(probs)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    limit = next((int(a.split("=")[1]) for a in sys.argv[1:] if a.startswith("--limit=")), None)
    from ultralytics import YOLO
    for wp in args:
        model, name = YOLO(wp), Path(wp).stem
        res = {}
        for s in SETS:
            if not (EVAL / s / "crops.csv").exists():
                continue
            files, y, geo, block = load(s, limit)
            prob = predict(model, files)
            res[s] = metrics(y, prob, geo, block, None if limit else np.random.default_rng(0))
            r = res[s]
            print(f"\n== {name} · {s}  n={r['n']}")
            print(f"   재현율 {r['재현율']} · 매크로 F1 {r['매크로F1']}")
            print(f"   누움 AUROC  분류기 {r['누움AUROC_분류기']} · 박스 모양 {r['누움AUROC_박스모양']}"
                  f"  | 서기 오인 10 % 에서 누움 재현율  분류기 {r['누움재현율@서기오인10%_분류기']} · 박스 모양 {r['누움재현율@서기오인10%_박스모양']}")
            if "95%구간" in r:
                print(f"   95 % {r['95%구간']}")
        if not limit:
            (BASE / "metrics" / f"posture_{name}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
