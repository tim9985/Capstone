"""
fair_metrics.py — 판넬 표용 지표 (11-07) → metrics/fair_metrics.json
  runs_fair 검출 (fair_eval.py) · 모델 = 우리 v12 (운용 4조각) + 기성 13개 (방식마다 따로 → 셋마다 AP50 최고 방식)
  지표  AP50 · AP50-95 (IoU 0.50~0.95 · 10단계) · 최적 F1 문턱에서 P · R · F1 · 문턱 0.25 (YOLO 기본) 의 P · R · 도로 헛탐지 (AU-AIR 사람 없는 장 · 건/장)
  우리 운용 문턱 0.20 값도 따로
"""
import json
import numpy as np
from fair_eval_report import BASE, RUN, U, B
from eval_external_det import ap101, iou_mat

IOUS = np.round(np.arange(0.5, 0.96, 0.05), 2)

def match(s, tag):
    g = np.load(RUN / s / "gt.npz"); d = np.load(RUN / s / f"{tag}.npz"); n = len(g["names"])
    S, T, NG, NEG = [], {t: [] for t in IOUS}, 0, []
    for i in range(n):
        gt = g["gt"][g["gt_img"] == i]; m = d["img"] == i; box, cf = d["box"][m], d["conf"][m]
        o = np.argsort(-cf); box, cf = box[o], cf[o]; NG += len(gt); S.append(cf)
        if not len(gt):
            NEG.append(cf)
        M = iou_mat(gt, box) if len(gt) else None
        for t in IOUS:
            tp = np.zeros(len(cf), bool)
            if len(gt):
                used = np.zeros(len(gt), bool)
                for j in range(len(cf)):
                    c = np.where((M[:, j] >= t) & ~used)[0]
                    if len(c):
                        used[c[np.argmax(M[c, j])]] = True; tp[j] = True
            T[t].append(tp)
    S = np.concatenate(S); T = {t: np.concatenate(v) for t, v in T.items()}
    return S, T, NG, NEG

def prf(S, T, NG, thr):
    sel = S >= thr; tp = int(T[sel].sum()); p = tp / max(sel.sum(), 1); r = tp / max(NG, 1)
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)

out = {}
road_cache = {}
for s in list(U) + list(B):
    for p in sorted((RUN / s).glob("*@*.npz")):
        tag = p.stem; S, T, NG, _ = match(s, tag); T5 = T[0.5]
        ap50 = ap101(S, T5, NG); ap5095 = float(np.mean([ap101(S, T[t], NG) for t in IOUS]))
        ths = np.round(np.arange(0.05, 0.96, 0.01), 2); f1s = [prf(S, T5, NG, t)[2] for t in ths]; tb = float(ths[int(np.argmax(f1s))])
        pb, rb, fb = prf(S, T5, NG, tb); p25, r25, f25 = prf(S, T5, NG, 0.25); p20, r20, f20 = prf(S, T5, NG, 0.20)
        if tag not in road_cache:
            rp = RUN / "test_auair" / f"{tag}.npz"
            if rp.exists():
                _, _, _, NEG = match("test_auair", tag); road_cache[tag] = NEG
        neg = road_cache.get(tag)
        fp = (lambda t: float(sum((c >= t).sum() for c in neg) / len(neg))) if neg else None
        out.setdefault(s, {})[tag] = {"AP50": round(ap50, 4), "AP50_95": round(ap5095, 4), "best_thr": tb, "P_best": round(pb, 4), "R_best": round(rb, 4), "F1_best": round(fb, 4),
                                      "P25": round(p25, 4), "R25": round(r25, 4), "P20": round(p20, 4), "R20": round(r20, 4), "n_gt": NG,
                                      "road_fp_best": None if fp is None else round(fp(tb), 3), "road_fp25": None if fp is None else round(fp(0.25), 3), "road_fp20": None if fp is None else round(fp(0.20), 3)}
        print(s, tag, out[s][tag]["AP50"], out[s][tag]["AP50_95"], out[s][tag]["F1_best"], flush=True)
    json.dump(out, open(BASE / "metrics" / "fair_metrics.json", "w"), ensure_ascii=False, indent=1)
