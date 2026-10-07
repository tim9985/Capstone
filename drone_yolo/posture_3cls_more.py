"""
posture_3cls_more.py — 3자세 (누움 · 앉음 · 서기) 박스 판정기 더 개선하기 (10-07 · K 다음)
  기준 K-only = Archangel 실제 사람 (무릎 → 앉음) · 박스 2특징 · 로지스틱
  V2  + Archangel 기어감 → 누움
  V3  + UE level01 합성 (앉음 · 웅크림 → 앉음 · 누움 · 걷기 → 서기 · 잘린 박스 제외 · 자세마다 3,000) — 박스 기하만 쓰니 외형 습관 위험 작음
  V4  K-only + 2차 다항 특징 (비선형 경계)
  V5  K-only + 그라디언트 부스팅
  V6  V3 + 2차 다항
  평가 Okutama 비스듬 정답 박스 (720p ×1.5) · 실전 파이프라인 판정 영상 B (runs_state/samples_a5.json) · 영상 블록 부트스트랩 1,000회 · 짝 = K-only 대비
"""
import csv, json, sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from posture_fusion import box_feat
from posture_v2 import auc
from state_trackfix import videos

BASE = Path(__file__).resolve().parent; rng = np.random.default_rng(0)
A = list(csv.DictReader(open(BASE / "data/pose_eval/archangel_real/crops.csv")))
amap = {"lying": 0, "kneeling": 1, "standing": 2}
def arc(crawl):
    rows = [r for r in A if r["pose"] in amap or (crawl and r["pose"] == "crawling")]
    y = np.array([0 if r["pose"] == "crawling" else amap[r["pose"]] for r in rows])
    return box_feat(np.array([[float(r["w1080"]), float(r["h1080"])] for r in rows])), y
U = [r for r in csv.DictReader(open(BASE / "data/pose_cls/ue_level01/crops.csv")) if r["cut"] == "0" and r["pose"] in ("lying", "sitting", "standing")]
yu = np.array([{"lying": 0, "sitting": 1, "standing": 2}[r["pose"]] for r in U]); Xu = box_feat(np.array([[float(r["w1080"]), float(r["h1080"])] for r in U]))
uc = np.concatenate([rng.permutation(np.where(yu == k)[0])[:3000] for k in range(3)])
Xk, yk = arc(False); Xc, yc = arc(True)
LR = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))
POLY = lambda: make_pipeline(StandardScaler(), PolynomialFeatures(2), LogisticRegression(max_iter=5000, class_weight="balanced"))
V = {"K-only": (LR, Xk, yk), "V2 +기어감→누움": (LR, Xc, yc), "V3 +UE": (LR, np.r_[Xk, Xu[uc]], np.r_[yk, yu[uc]]),
     "V4 2차": (POLY, Xk, yk), "V5 부스팅": ("gb", Xk, yk), "V6 +UE·2차": (POLY, np.r_[Xk, Xu[uc]], np.r_[yk, yu[uc]])}
O = list(csv.DictReader(open(BASE / "data/pose_eval/okutama_obl/crops.csv")))
Xo = box_feat(np.array([[float(r["w1080"]) * 1.5, float(r["h1080"]) * 1.5] for r in O])); yo = np.array([{"lying": 0, "sitting": 1, "standing": 2}[r["pose"]] for r in O])
po = np.array([r["place"] for r in O])
S = [s for s in json.load(open(BASE / "runs_state/samples_a5.json")) if s.get("h_wh") and s["pose"] in ("Lying", "Sitting", "Standing", "Walking", "Running")]
_, _, B = videos(); S = [s for s in S if s["vid"] in B]
Xp = box_feat(np.array([s["h_wh"][-1] for s in S])); yp = np.array([{"Lying": 0, "Sitting": 1}.get(s["pose"], 2) for s in S]); pp = np.array([s["vid"] for s in S])

def m(y, P):
    pr = P.argmax(1); f = []
    for k in range(3):
        tp = ((pr == k) & (y == k)).sum(); r_ = tp / max((y == k).sum(), 1); q = tp / max((pr == k).sum(), 1); f.append(2 * r_ * q / max(r_ + q, 1e-9))
    l = y == 0
    return {"앉음F1": f[1], "매크로F1": float(np.mean(f)), "누움AUROC": auc(P[l, 0], P[~l, 0]) if l.any() and (~l).any() else float("nan"),
            "서기재현율": float(((pr == 2) & (y == 2)).sum() / max((y == 2).sum(), 1))}
P = {}
for k, (mk, X, y) in V.items():
    if mk == "gb":
        mdl = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05).fit(X, y, sample_weight=compute_sample_weight("balanced", y))
    else:
        mdl = mk().fit(X, y)
    P[k] = (mdl.predict_proba(Xo), mdl.predict_proba(Xp))
res = {}
for k, (a, b) in P.items():
    res[k] = {"정답박스": {x: round(float(v), 4) for x, v in m(yo, a).items()}, "실전B": {x: round(float(v), 4) for x, v in m(yp, b).items()}}
def boot(y, Pa, Pb, place, n=1000):
    up = np.unique(place); idx = {p: np.where(place == p)[0] for p in up}; d = {"앉음F1": [], "매크로F1": [], "누움AUROC": []}
    for _ in range(n):
        s = np.concatenate([idx[p] for p in rng.choice(up, len(up))]); a, b = m(y[s], Pa[s]), m(y[s], Pb[s])
        for kk in d:
            if not (np.isnan(a[kk]) or np.isnan(b[kk])): d[kk].append(b[kk] - a[kk])
    return {kk: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)] for kk, v in d.items()}
for k in list(P)[1:]:
    res[k]["짝 정답박스 (K-only 대비)"] = boot(yo, P["K-only"][0], P[k][0], po)
    res[k]["짝 실전B (K-only 대비)"] = boot(yp, P["K-only"][1], P[k][1], pp)
(BASE / "metrics/posture_3cls_more.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
print(f"{'판정기':14s} | 정답: 앉음F1 매크로F1 누움AUROC 서기재현 | 실전B: 앉음F1 매크로F1 누움AUROC")
for k, v in res.items():
    a, b = v["정답박스"], v["실전B"]
    print(f"{k:14s} | {a['앉음F1']:.3f} {a['매크로F1']:.3f} {a['누움AUROC']:.3f} {a['서기재현율']:.2f} | {b['앉음F1']:.3f} {b['매크로F1']:.3f} {b['누움AUROC']:.3f}")
    for z in ("짝 정답박스 (K-only 대비)", "짝 실전B (K-only 대비)"):
        if z in v: print(f"   {z}: {v[z]}")
