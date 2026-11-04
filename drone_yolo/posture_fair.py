"""
posture_fair.py — 자세 (배포 posture.json score · 박스 2특징 로지스틱) 공정 확인 (11-04) → metrics/posture_fair.json
  Okutama 비스듬 정답 크롭 (×1.5 · 720p px 보정) · Archangel 실사람 정답 크롭 — 누움 vs 나머지 ROC · 3자세 혼동
  ⚠ 기어감 (crawling) 은 뺌 (누움으로 묶으면 정답이 바뀜 · 10-31 fair_viz_data 는 묶었음) · 무릎 → 앉음
  ⚠ 정답 박스 = 자세 단계만 · 탐지 박스까지 이은 값은 state_worker_check_v12_c20 (점수 누움 AUROC 0.922 · 영상 블록 95 %)
"""
import csv, json, sys
from pathlib import Path
import numpy as np
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / "deploy" / "vision_worker"))
from app.vision_core.state import PostureLR
P = PostureLR(json.load(open(BASE / "runs_person" / "_pack_v12" / "posture.json"))["score"])
MAP = {"lying": 0, "sitting": 1, "kneeling": 1, "standing": 2}
rng = np.random.default_rng(0)
def auc(s, pos):
    from scipy.stats import rankdata                     # 동점 평균 순위
    r = rankdata(s); n1 = pos.sum(); n0 = (~pos).sum()
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
out = {}
for s, k in (("okutama_obl", 1.5), ("archangel_real", 1.0)):
    rows = [r for r in csv.DictReader(open(BASE / "data" / "pose_eval" / s / "crops.csv")) if r["pose"] in MAP]
    wh = np.array([[float(r["w1080"]) * k, float(r["h1080"]) * k] for r in rows]); pr = P.proba(wh)
    y = np.array([MAP[r["pose"]] for r in rows]); pos = y == 0; sc = pr[:, 0]
    # 장소 (영상 · 회차) 블록 부트스트랩
    grp = np.array([r["place"] for r in rows]); ug = np.unique(grp); idx = {g: np.where(grp == g)[0] for g in ug}; bs = []
    for _ in range(1000):
        ii = np.concatenate([idx[g] for g in rng.choice(ug, len(ug))])
        if pos[ii].any() and (~pos[ii]).any(): bs.append(auc(sc[ii], pos[ii]))
    o = np.argsort(-sc); tpr = np.r_[0, np.cumsum(pos[o]) / pos.sum()]; fpr = np.r_[0, np.cumsum(~pos[o]) / (~pos).sum()]
    sel = np.unique(np.r_[np.linspace(0, len(tpr) - 1, 300).astype(int), len(tpr) - 1])
    cm = np.zeros((3, 3), int)
    for t, p in zip(y, pr.argmax(1)): cm[t, p] += 1
    out[s] = {"n": int(len(y)), "n_lying": int(pos.sum()), "places": int(len(ug)), "AUROC": round(auc(sc, pos), 4),
              "CI": [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)],
              "roc": [[round(float(fpr[i]), 4), round(float(tpr[i]), 4)] for i in sel], "confusion": cm.tolist(), "classes": ["누움", "앉음·무릎", "서기"]}
    print(s, out[s]["n"], out[s]["n_lying"], out[s]["AUROC"], out[s]["CI"], cm.tolist())
w = json.load(open(BASE / "metrics" / "state_worker_check_v12_c20.json"))
out["pipeline_okutama"] = {"AUROC": w["점수 누움 AUROC"], "CI": w["95% (영상 블록)"], "videos": w["영상 (B)"], "lying_samples": w["누움 정답 1초 표본"],
                           "lying_caught": w["누움 잡힌 비율"], "grade_lying": w["정답 자세 × 등급"]["Lying"]}
json.dump(out, open(BASE / "metrics" / "posture_fair.json", "w"), ensure_ascii=False)
