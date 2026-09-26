"""
analyze_models.py — 두 모델이 **어디서 · 왜** 다른가 (2026-09-26)

  예측(conf ≥ 0.01)을 한 번 뽑아 runs_person/<모델>/preds_<평가셋>.npz 에 캐시하고, 운용 NMS 로 다음을 낸다
    AP50 (판정과 같은 규칙) · AP50[운용 범위] (정답 긴 변 < 24 px 는 '무시' — 맞혀도 틀려도 안 셈)
    크기 구간별 재현율@0.15 · 놓친 원인(diag_misses 와 같은 정의)
    찾은 사람의 IoU 분포 · 예측/정답 박스 너비·높이 비 (박스 그리는 습관)
    (test_obl) 자세 · 가림별 재현율
실행: python analyze_models.py v6_obl_r2 v6_nwd v6_obl v6_p2m
"""
import json, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import diag_misses as D

BASE = Path(__file__).resolve().parent
TW, TH = 1280, 720
SETS = {"test_obl": ("single", 0.7), "test_v2": ("tile", 0.6)}
SMALL = 24.0


def predict_cached(name, tag):
    f = BASE / "runs_person" / name / f"preds_{tag}.npz"
    if f.exists():
        d = np.load(f, allow_pickle=True)
        return list(d["recs"])
    import cv2
    from ultralytics import YOLO
    model = YOLO(str(BASE / "runs_person" / name / "weights" / "best.pt"))
    mode = SETS[tag][0]; recs = []
    for p in sorted((BASE / "data" / tag / "images").glob("*.jpg")):
        img = cv2.imread(str(p)); h, w = img.shape[:2]
        if mode == "tile":
            offs = [(x, y) for y in (0, h - TH) for x in (0, w - TW)]; tiles = [img[y:y + TH, x:x + TW] for x, y in offs]
        else:
            offs, tiles = [(0, 0)], [img]
        res = model.predict(tiles, imgsz=1280, conf=0.01, iou=0.9, max_det=1000, batch=len(tiles), quantize="fp16", verbose=False)
        B, C = [], []
        for (ox, oy), r in zip(offs, res):
            if len(r.boxes):
                B.append(r.boxes.xyxy.cpu().numpy() + np.array([ox, oy, ox, oy], np.float32)); C.append(r.boxes.conf.cpu().numpy())
        B = np.concatenate(B) if B else np.zeros((0, 4), np.float32); C = np.concatenate(C) if C else np.zeros(0, np.float32)
        recs.append((p.stem, D.gt_boxes(p, w, h), B, C))
    np.savez_compressed(f, recs=np.array(recs, dtype=object))
    return recs


def analyze(name, tag, omet):
    recs = predict_cached(name, tag); op = SETS[tag][1]
    S, T, n = [], [], 0; Si, Ti, ni = [], [], 0
    tp_iou, rw, rh = [], [], []
    cause = Counter(); strata = defaultdict(lambda: [0, 0])
    for stem, G, B, C in recs:
        k = D.nms(B, C, op); B2, C2 = B[k], C[k]
        tp, cs, gconf = D.greedy(G, B2, C2); S += list(cs); T += list(tp); n += len(G)
        L = np.maximum(G[:, 2] - G[:, 0], G[:, 3] - G[:, 1]) if len(G) else np.zeros(0)
        # 운용 범위 AP: 작은 정답은 무시 (그 정답과 IoU≥0.5 인 남는 예측도 버림)
        big = L >= SMALL
        tpb, csb, _ = D.greedy(G[big], B2, C2)
        o = np.argsort(-C2); Bs = B2[o]
        if (~big).any() and len(Bs):
            Mi = D.iou(G[~big], Bs); ign = (Mi.max(0) >= 0.5) & ~tpb
        else:
            ign = np.zeros(len(Bs), bool)
        Si += list(csb[~ign]); Ti += list(tpb[~ign]); ni += int(big.sum())
        # 찾은 사람의 IoU · 박스 비 (conf ≥ 0.15)
        hi = C2 >= 0.15
        if len(G) and hi.any():
            M = D.iou(G, B2[hi])
            for i in range(len(G)):
                j = M[i].argmax()
                if gconf[i] >= 0.15 and M[i, j] >= 0.5:
                    g, b = G[i], B2[hi][j]; tp_iou.append(M[i, j])
                    rw.append((b[2] - b[0]) / (g[2] - g[0])); rh.append((b[3] - b[1]) / (g[3] - g[1]))
        # 놓친 원인 · 층
        Bh = B2[hi]; IA = D.ioa_gt(G, Bh) if len(Bh) and len(G) else np.zeros((len(G), len(Bh)))
        merged = set(np.where((IA >= 0.5).sum(0) >= 2)[0]) if len(Bh) else set()
        Mall = D.iou(G, B2)
        meta = omet.get(stem) if tag == "test_obl" else None
        for i in range(len(G)):
            if gconf[i] >= 0.15: c = "찾음"
            elif gconf[i] >= 0: c = "확신도 낮음"
            elif len(Bh) and any(IA[i, j] >= 0.5 for j in merged): c = "묶임"
            elif len(B2) and Mall[i].max() >= 0.3: c = "위치 어긋남"
            else: c = "아예 못 봄"
            cause[c] += 1; hit = c == "찾음"
            sz = "<16" if L[i] < 16 else "16-24" if L[i] < 24 else "24-36" if L[i] < 36 else "36-50" if L[i] < 50 else "50+"
            strata[("크기", sz)][0] += hit; strata[("크기", sz)][1] += 1
            strata[("원인×크기", f"{c}|{'<24' if L[i] < 24 else '≥24'}")][1] += 1
            if meta and len(meta) == len(G):
                occ, act = meta[i]
                strata[("가림", "예" if occ else "아니오")][0] += hit; strata[("가림", "예" if occ else "아니오")][1] += 1
                if act in ("Lying", "Sitting", "Standing", "Walking"):
                    strata[("자세", act)][0] += hit; strata[("자세", act)][1] += 1
    tot = sum(cause.values())
    q = lambda a: [round(float(x), 3) for x in np.percentile(a, [25, 50, 75])] if len(a) else None
    return {"AP50": round(D.ap101(np.array(S), np.array(T, bool), n), 4),
            "AP50_운용범위(≥24px)": round(D.ap101(np.array(Si), np.array(Ti, bool), ni), 4),
            "찾은_IoU_25/50/75": q(tp_iou), "너비비_25/50/75": q(rw), "높이비_25/50/75": q(rh),
            "놓친_원인_%": {k: round(v / tot * 100, 1) for k, v in cause.most_common()},
            "층": {f"{a}:{b}": (round(h / t, 3) if a != "원인×크기" else t, t) for (a, b), (h, t) in sorted(strata.items())}}


def main():
    names = sys.argv[1:]
    omet = D.okutama_meta(); out = {}
    for tag in SETS:
        for nm in names:
            out[f"{nm}|{tag}"] = r = analyze(nm, tag, omet)
            print(f"\n== {nm} · {tag}\n" + json.dumps(r, ensure_ascii=False))
    (BASE / "metrics" / "analyze_models.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
