"""
posture_fusion.py — 시점별로 나눠 쓰는 자세 판정 (상태 인지 A2 · 2026-10-01)

왜
  크롭 분류기는 수직 → 수직 (한국 → 일본) 은 버티지만 비스듬 Okutama 에서 박스 모양(누움 AUROC 0.97)을 못 넘는다
  (s1 0.72~0.91 · s2 0.87 · 앉음 붕괴). 마운트각을 아는 우리 구조에서는 **비스듬 = 박스 모양 · 수직 = 외형 분류기** 로 나눠 쓴다.

판정기
  비스듬  3자세 다항 로지스틱 — 입력 = 박스 모양 [log(세로/가로) · log(긴 변 px)] (+ 선택: 외형 확률)
          맞추는 데이터 = **SARD + NOMAD 비스듬 크롭** (사람 라벨 · 활동 유도 라벨) — Okutama 는 평가만
          조합(박스만 · +SigLIP 2 제로샷 · +DINOv2 선형 · +YOLO cls)은 **학습 데이터 5겹 교차검증 매크로 F1** 로 미리 고른다
  수직    DINOv2 특징 + 로지스틱 (AI-Hub 5곳 학습 · 산악8 로 C 고름) — 09-30 측정 최고 (Okutama 수직 누움 AUROC 0.93)
  경로    크롭의 시점(view) = 운용에선 짐벌 마운트각 → 비스듬(< 75°) · 수직

평가: eval_posture.metrics 와 같은 지표 · 같은 블록 부트스트랩 (aihub_test · okutama_obl · okutama_nadir)
실행 (state 환경 · GPU): /home/se/venvs/state/bin/python posture_fusion.py
출력: metrics/posture_fusion.json · 콘솔 표
"""
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from eval_posture import CLASSES, EVAL, SETS, load, metrics
from posture_foundation import Dino, Siglip, train_files

BASE = Path(__file__).resolve().parent
CLS = BASE / "data" / "pose_cls"
YOLO_CLS = BASE / "weights" / "s1_aihub_sard.pt"


def rows_of(root):
    rows = list(csv.DictReader(open(root / "crops.csv")))
    files = [str(root / r["file"]) for r in rows]
    y = np.array([CLASSES.index(r["pose"]) for r in rows])
    wh = np.array([[float(r["w1080"]), float(r["h1080"])] for r in rows])
    return files, y, wh


def box_feat(wh):
    w, h = np.maximum(wh[:, 0], 1), np.maximum(wh[:, 1], 1)
    return np.c_[np.log(h / w), np.log(np.maximum(w, h))]


def lg(p):
    return np.log(np.clip(p, 1e-4, 1))


def main():
    # ── 외형 확률 세 가지를 모든 크롭에 한 번씩 ─────────────────────────────
    sets = {s: load(s) for s in SETS if (EVAL / s / "crops.csv").exists()}
    evwh = {s: rows_of(EVAL / s)[2] for s in sets}
    ob_parts = [rows_of(CLS / p / "train") for p in ("sard", "nomad")]
    ob_files = sum((p[0] for p in ob_parts), []); ob_y = np.concatenate([p[1] for p in ob_parts])
    ob_wh = np.concatenate([p[2] for p in ob_parts]); ob_src = np.concatenate([[i] * len(p[0]) for i, p in enumerate(ob_parts)])
    allf = ob_files + sum((sets[s][0] for s in sets), [])
    print(f"비스듬 학습 크롭 {len(ob_files):,} (SARD {len(ob_parts[0][0]):,} · NOMAD {len(ob_parts[1][0]):,}) · 평가 크롭 {len(allf) - len(ob_files):,}", flush=True)

    sig = Siglip(); T = sig.text()
    z = 100 * (sig.image(allf) @ T.T); z = np.exp(z - z.max(1, keepdims=True))
    P_sig = dict(zip(allf, z / z.sum(1, keepdims=True)))
    del sig
    dino = Dino()
    Xtr_f, ytr = train_files("train"); Xva_f, yva = train_files("val")
    Ftr, Fva = dino.image(Xtr_f), dino.image(Xva_f)
    best = max(((float((c.predict(Fva) == yva).mean()), C, c) for C in (0.1, 0.3, 1.0, 3.0)
                for c in [LogisticRegression(C=C, max_iter=2000, class_weight="balanced").fit(Ftr, ytr)]), key=lambda t: t[0])
    print(f"DINOv2 선형 C={best[1]} · 산악8 정확도 {best[0]:.3f}", flush=True)
    P_dino = dict(zip(allf, best[2].predict_proba(dino.image(allf))))
    del dino
    from ultralytics import YOLO
    ym = YOLO(str(YOLO_CLS)); P_cls = {}
    for i in range(0, len(allf), 512):
        for f, r in zip(allf[i:i + 512], ym(allf[i:i + 512], imgsz=128, verbose=False)):
            p = r.probs.data.float().cpu().numpy(); n = r.names
            P_cls[f] = np.array([p[[k for k, v in n.items() if v == c][0]] for c in CLASSES])

    def feats(files, wh, kind):
        X = box_feat(wh)
        if kind == "box+siglip":
            X = np.c_[X, lg(np.stack([P_sig[f] for f in files]))]
        elif kind == "box+dino":
            X = np.c_[X, lg(np.stack([P_dino[f] for f in files]))]
        elif kind == "box+cls":
            X = np.c_[X, lg(np.stack([P_cls[f] for f in files]))]
        return X

    # ── 비스듬 판정기 조합 고르기 — 학습 데이터 교차검증 (Okutama 는 보지 않는다) ──
    kinds = ("box", "box+siglip", "box+dino", "box+cls")
    cv = {}
    strat = ob_y * 2 + ob_src                                 # 자세 × 출처 층화
    for k in kinds:
        X = feats(ob_files, ob_wh, k); f1 = []
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, strat):
            clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X[tr], ob_y[tr])
            pr = clf.predict(X[te]); yy = ob_y[te]
            f = []
            for i in range(3):
                tp = ((pr == i) & (yy == i)).sum(); r_ = tp / max((yy == i).sum(), 1); q_ = tp / max((pr == i).sum(), 1)
                f.append(2 * r_ * q_ / max(r_ + q_, 1e-9))
            f1.append(np.mean(f))
        cv[k] = round(float(np.mean(f1)), 3)
    chosen = max(cv, key=cv.get)
    print(f"비스듬 조합 교차검증 매크로 F1 {cv} → 고름: {chosen}", flush=True)

    # ── 평가 ─────────────────────────────────────────────────────────────────
    res = {"비스듬_교차검증F1": cv, "비스듬_고른조합": chosen, "평가": {}}
    fitted = {k: LogisticRegression(max_iter=2000, class_weight="balanced").fit(feats(ob_files, ob_wh, k), ob_y) for k in kinds}
    for s, (files, y, geo, block) in sets.items():
        view = "nadir" if s != "okutama_obl" else "oblique"
        cand = {f"비스듬판정_{k}": fitted[k].predict_proba(feats(files, evwh[s], k)) for k in kinds}
        cand["DINOv2선형"] = np.stack([P_dino[f] for f in files])
        cand["SigLIP2제로샷"] = np.stack([P_sig[f] for f in files])
        cand["YOLOcls_s1_aihub_sard"] = np.stack([P_cls[f] for f in files])
        cand["경로판정(비스듬→고른조합 · 수직→DINOv2)"] = cand[f"비스듬판정_{chosen}"] if view == "oblique" else cand["DINOv2선형"]
        res["평가"][s] = {}
        print(f"\n== {s} ({view})")
        for name, prob in cand.items():
            m = metrics(y, prob, geo, block, np.random.default_rng(0))
            res["평가"][s][name] = m
            print(f"   {name:42s} 누움 AUROC {m['누움AUROC_분류기']:.3f} {m['95%구간']['누움AUROC_분류기']} · 매크로 F1 {m['매크로F1']:.3f}"
                  f" {m['95%구간']['매크로F1']} · 재현율 {m['재현율']}", flush=True)
        print(f"   (박스 모양 기준선 누움 AUROC {m['누움AUROC_박스모양']})")
    (BASE / "metrics" / "posture_fusion.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
