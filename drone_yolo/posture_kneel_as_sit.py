"""
posture_kneel_as_sit.py — Archangel 무릎 꿇음을 "앉음 (낮은 자세)" 으로 묶어 3자세 (누움 · 앉음 · 서기) 박스 판정기를 맞추면
Okutama 비스듬 (진짜 앉음 1,564) 에서 앉음이 나아지나 (10-07 · 사용자 제안)

  판정기 (박스 2특징 · 로지스틱 · class_weight balanced)
    B0      SARD + NOMAD (지금)
    K-all   SARD + NOMAD + Archangel 전부 (무릎 → 앉음)
    K-cap   SARD + NOMAD + Archangel 자세마다 1,500 (Archangel 이 압도하지 않게)
    K-only  Archangel 만 (무릎 → 앉음)
  평가  Okutama 비스듬 정답 박스 크롭 10,508 (누움 268 · 앉음 1,564 · 서기 8,676 · 720p → ×1.5)
  지표  앉음 F1 · 앉음 재현율 · 누움 AUROC · 매크로 F1 · 짝 = 영상 블록 부트스트랩 1,000회 (B0 대비)
  채택 (미리 정함): 앉음 F1 짝 구간 > 0 그리고 누움 AUROC 가 유의하게 나쁘지 않음
"""
import csv, json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from posture_fusion import CLS, box_feat, rows_of
from posture_v2 import auc

BASE = Path(__file__).resolve().parent
rng = np.random.default_rng(0)
parts = [rows_of(CLS / p / "train") for p in ("sard", "nomad")]
Xs = box_feat(np.concatenate([p[2] for p in parts])); ys = np.concatenate([p[1] for p in parts])
A = [r for r in csv.DictReader(open(BASE / "data/pose_eval/archangel_real/crops.csv")) if r["pose"] in ("lying", "kneeling", "standing")]
Ya = np.array([{"lying": 0, "kneeling": 1, "standing": 2}[r["pose"]] for r in A])
Xa = box_feat(np.array([[float(r["w1080"]), float(r["h1080"])] for r in A]))
cap = np.concatenate([rng.permutation(np.where(Ya == k)[0])[:1500] for k in range(3)])
sets = {"B0": (Xs, ys), "K-all": (np.r_[Xs, Xa], np.r_[ys, Ya]), "K-cap": (np.r_[Xs, Xa[cap]], np.r_[ys, Ya[cap]]), "K-only": (Xa, Ya)}
O = list(csv.DictReader(open(BASE / "data/pose_eval/okutama_obl/crops.csv")))
Xo = box_feat(np.array([[float(r["w1080"]) * 1.5, float(r["h1080"]) * 1.5] for r in O]))
yo = np.array([{"lying": 0, "sitting": 1, "standing": 2}[r["pose"]] for r in O]); place = np.array([r["place"] for r in O])
up = np.unique(place); pidx = {p: np.where(place == p)[0] for p in up}

def m(y, P):
    pr = P.argmax(1); f = []
    for k in range(3):
        tp = ((pr == k) & (y == k)).sum(); r_ = tp / max((y == k).sum(), 1); q = tp / max((pr == k).sum(), 1); f.append(2 * r_ * q / max(r_ + q, 1e-9))
    l = y == 0
    return {"누움AUROC": auc(P[l, 0], P[~l, 0]), "앉음F1": f[1], "앉음재현율": float(((pr == 1) & (y == 1)).sum() / max((y == 1).sum(), 1)),
            "매크로F1": float(np.mean(f)), "서기재현율": float(((pr == 2) & (y == 2)).sum() / max((y == 2).sum(), 1))}

P = {}
for k, (X, y) in sets.items():
    P[k] = LogisticRegression(max_iter=3000, class_weight="balanced").fit(X, y).predict_proba(Xo)
res = {k: {a: round(float(b), 4) for a, b in m(yo, p).items()} for k, p in P.items()}
for k in ("K-all", "K-cap", "K-only"):
    d = {"앉음F1": [], "누움AUROC": []}
    for _ in range(1000):
        s = np.concatenate([pidx[p] for p in rng.choice(up, len(up))])
        a, b = m(yo[s], P["B0"][s]), m(yo[s], P[k][s])
        for kk in d: d[kk].append(b[kk] - a[kk])
    res[f"짝 B0 → {k}"] = {kk: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)] for kk, v in d.items()}
(BASE / "metrics/posture_kneel_as_sit.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
print(json.dumps(res, ensure_ascii=False, indent=1))
