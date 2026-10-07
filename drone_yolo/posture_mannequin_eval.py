"""
posture_mannequin_eval.py — Archangel 마네킹 (무릎 → 앉음) 을 3자세 박스 판정기에 더하면 나아지나 (10-07 · K2 다음)
  기준 V3 = Archangel 실제 (무릎 → 앉음) + UE level01 (자세마다 3,000) · 박스 2특징 · 로지스틱 (K2 최고)
  M1  V3 + 마네킹 자세마다 3,000
  M2  V3 + 마네킹 전부
  M3  Archangel 실제 + 마네킹 (UE 없이)
  평가 Okutama 비스듬 정답 박스 · 실전 파이프라인 판정 영상 B · 짝 = V3 대비 · 영상 블록 부트스트랩 1,000회
  채택 (미리 정함): 앉음 F1 또는 매크로 F1 짝 구간 > 0 그리고 누움 AUROC · 다른 F1 이 유의하게 나쁘지 않음
"""
import csv, json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from posture_fusion import box_feat
from posture_v2 import auc
from state_trackfix import videos

BASE = Path(__file__).resolve().parent; rng = np.random.default_rng(0)
MAP = {"lying": 0, "kneeling": 1, "sitting": 1, "standing": 2}


def table(path, keep=lambda r: True):
    R = [r for r in csv.DictReader(open(path)) if r["pose"] in MAP and keep(r)]
    return box_feat(np.array([[float(r["w1080"]), float(r["h1080"])] for r in R])), np.array([MAP[r["pose"]] for r in R])


def capk(X, y, n):
    i = np.concatenate([rng.permutation(np.where(y == k)[0])[:n] for k in range(3)]); return X[i], y[i]


Xr, yr = table(BASE / "data/pose_eval/archangel_real/crops.csv")
Xu, yu = capk(*table(BASE / "data/pose_cls/ue_level01/crops.csv", lambda r: r["cut"] == "0"), 3000)
Xm, ym = table(BASE / "data/pose_eval/archangel_mannequin/crops.csv")
Xmc, ymc = capk(Xm, ym, 3000)
V = {"V3 실제+UE": (np.r_[Xr, Xu], np.r_[yr, yu]), "M1 +마네킹 3,000": (np.r_[Xr, Xu, Xmc], np.r_[yr, yu, ymc]),
     "M2 +마네킹 전부": (np.r_[Xr, Xu, Xm], np.r_[yr, yu, ym]), "M3 실제+마네킹": (np.r_[Xr, Xm], np.r_[yr, ym])}
O = list(csv.DictReader(open(BASE / "data/pose_eval/okutama_obl/crops.csv")))
Xo = box_feat(np.array([[float(r["w1080"]) * 1.5, float(r["h1080"]) * 1.5] for r in O])); yo = np.array([MAP[r["pose"]] for r in O]); po = np.array([r["place"] for r in O])
S = [s for s in json.load(open(BASE / "runs_state/samples_a5.json")) if s.get("h_wh") and s["pose"] in ("Lying", "Sitting", "Standing", "Walking", "Running")]
_, _, B = videos(); S = [s for s in S if s["vid"] in B]
Xp = box_feat(np.array([s["h_wh"][-1] for s in S])); yp = np.array([{"Lying": 0, "Sitting": 1}.get(s["pose"], 2) for s in S]); pp = np.array([s["vid"] for s in S])


def m(y, P):
    pr = P.argmax(1); f = []
    for k in range(3):
        tp = ((pr == k) & (y == k)).sum(); r_ = tp / max((y == k).sum(), 1); q = tp / max((pr == k).sum(), 1); f.append(2 * r_ * q / max(r_ + q, 1e-9))
    l = y == 0
    return {"앉음F1": f[1], "누움F1": f[0], "서기F1": f[2], "매크로F1": float(np.mean(f)),
            "누움AUROC": auc(P[l, 0], P[~l, 0]) if l.any() and (~l).any() else float("nan")}


def boot(y, Pa, Pb, place):
    up = np.unique(place); idx = {p: np.where(place == p)[0] for p in up}; d = {k: [] for k in ("앉음F1", "매크로F1", "누움AUROC", "서기F1")}
    for _ in range(1000):
        s = np.concatenate([idx[p] for p in rng.choice(up, len(up))]); a, b = m(y[s], Pa[s]), m(y[s], Pb[s])
        for k in d:
            if not (np.isnan(a[k]) or np.isnan(b[k])): d[k].append(b[k] - a[k])
    return {k: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)] for k, v in d.items()}


P = {k: (lambda mdl: (mdl.predict_proba(Xo), mdl.predict_proba(Xp)))(make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced")).fit(X, y))
     for k, (X, y) in V.items()}
res = {"학습 수": {"실제": len(yr), "UE": len(yu), "마네킹": {k: int((ym == i).sum()) for i, k in enumerate(("누움", "무릎", "서기"))}}}
for k, (a, b) in P.items():
    res[k] = {"정답박스": {x: round(float(v), 4) for x, v in m(yo, a).items()}, "실전B": {x: round(float(v), 4) for x, v in m(yp, b).items()}}
    if k != "V3 실제+UE":
        res[k]["짝 정답박스"] = boot(yo, P["V3 실제+UE"][0], a, po); res[k]["짝 실전B"] = boot(yp, P["V3 실제+UE"][1], b, pp)
(BASE / "metrics/posture_mannequin_eval.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
print(res["학습 수"])
for k, v in res.items():
    if k == "학습 수": continue
    a, b = v["정답박스"], v["실전B"]
    print(f"{k:16s} | 정답 앉음F1 {a['앉음F1']:.3f} 매크로 {a['매크로F1']:.3f} 누움AUROC {a['누움AUROC']:.3f} 서기F1 {a['서기F1']:.3f} | 실전B 앉음F1 {b['앉음F1']:.3f} 매크로 {b['매크로F1']:.3f} 누움AUROC {b['누움AUROC']:.3f}")
    for z in ("짝 정답박스", "짝 실전B"):
        if z in v: print(f"   {z}: {v[z]}")
