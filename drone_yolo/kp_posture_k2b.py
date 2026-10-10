"""
kp_posture_k2b.py — K2b: K2 로지스틱을 Archangel 전부로 맞추고 우리 UE level01 에서 시험 (10-10 · 판정 기준은 _학습 큐 「K2b · E1」)
  A 박스 2특징 · B 박스 + FlyPose-H 관절 (kp_posture_k2 와 같은 특징) · UE 는 학습에 안 씀
  구역: 세로≥가로 (주) · 전체 · 시선 방향 누움 (|rel_az| ≤ 20° 또는 ≥ 160°) vs 서기 · 사진 (장면 번호) 블록 부트스트랩 1,000회
실행: /home/se/miniconda3/envs/drone/bin/python kp_posture_k2b.py → metrics/kp_posture_k2b.json
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from kp_posture_k2 import ENGINE, HPose, extract, load_rows
from posture_v2 import auc

BASE = Path(__file__).resolve().parent
UE = BASE / "data" / "pose_cls" / "ue_level01"
UCACHE = BASE / "runs_state" / "k2b_ue_keypoints.npz"


def feats(w, h, xy, sc):
    L = np.maximum(w, h); box = np.c_[np.log(h / w), np.log(L)]
    return box, np.c_[box, xy.reshape(len(w), -1), sc]


def main():
    ar = load_rows(); AXY, ASC = extract(ar)
    ya = np.array([r["pose"] == "lying" for r in ar])
    Aa, Ba = feats(np.array([r["w"] for r in ar]), np.array([r["h"] for r in ar]), AXY, ASC)
    ue = []
    for r in csv.DictReader(open(UE / "crops.csv", encoding="utf-8")):
        w, h = float(r["w1080"]), float(r["h1080"])
        if r["pose"] in ("standing", "lying") and max(w, h) >= 45:
            ue.append(r | {"w": w, "h": h, "L": max(w, h)})
    if UCACHE.exists() and list(np.load(UCACHE, allow_pickle=True)["files"]) == [r["file"] for r in ue]:
        z = np.load(UCACHE, allow_pickle=True); UXY, USC = z["xy"], z["sc"]
    else:
        hp = HPose(ENGINE); UXY, USC = [], []
        for r in ue:
            img = cv2.imread(str(UE / "crops" / r["file"])); side = img.shape[0] / 1.5; c = img.shape[0] / 2
            bw, bh = side * r["w"] / r["L"], side * r["h"] / r["L"]
            xy, sc = hp(img, (c - bw / 2, c - bh / 2, c + bw / 2, c + bh / 2))
            UXY.append(((xy - c) / side).astype(np.float32)); USC.append(sc)
        UXY, USC = np.array(UXY), np.array(USC)
        np.savez(UCACHE, files=np.array([r["file"] for r in ue]), xy=UXY, sc=USC)
    yu = np.array([r["pose"] == "lying" for r in ue])
    Au, Bu = feats(np.array([r["w"] for r in ue]), np.array([r["h"] for r in ue]), UXY, USC)
    P = {}
    for name, Xa, Xu in (("A_box", Aa, Au), ("B_box+kp", Ba, Bu)):
        m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000)).fit(Xa, ya); P[name] = m.predict_proba(Xu)[:, 1]
    amb = np.array([r["h"] >= r["w"] for r in ue])
    rel = np.array([abs(((float(r["rel_az"]) + 180) % 360) - 180) for r in ue])
    look = (rel <= 20) | (rel >= 160)
    zones = {"세로≥가로 (주)": (amb, amb), "전체": (np.ones(len(ue), bool),) * 2, "시선 방향 누움 vs 서기": (look, np.ones(len(ue), bool))}
    shot = np.array([r["file"].split("/")[-1].split("_Person")[0] for r in ue])
    res = {"ue crops": len(ue), "누움 · 서기": [int(yu.sum()), int((~yu).sum())], "zones": {}}
    for zn, (zl, zs) in zones.items():
        L, S = zl & yu, zs & ~yu
        res["zones"][zn] = {"n 누움 / 서기": [int(L.sum()), int(S.sum())]} | {k: round(float(auc(p[L], p[S])), 4) for k, p in P.items()} | {
            "누움을 서기로 (0.5) A → B": [round(float((P["A_box"][L] < 0.5).mean()), 3), round(float((P["B_box+kp"][L] < 0.5).mean()), 3)]}
    blocks = sorted(set(shot)); idx = defaultdict(list)
    for i, v in enumerate(shot):
        idx[v].append(i)
    rng = np.random.default_rng(0); d = defaultdict(list)
    for _ in range(1000):
        s = np.concatenate([idx[v] for v in rng.choice(blocks, len(blocks))])
        for zn, (zl, zs) in zones.items():
            L, S = (zl & yu)[s], (zs & ~yu)[s]
            if L.any() and S.any():
                d[zn].append(auc(P["B_box+kp"][s][L], P["B_box+kp"][s][S]) - auc(P["A_box"][s][L], P["A_box"][s][S]))
    res["AUROC (B − A) 95%"] = {k: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)] for k, v in d.items()}
    res["K2b 판정"] = bool(res["AUROC (B − A) 95%"]["세로≥가로 (주)"][0] > 0)
    (BASE / "metrics" / "kp_posture_k2b.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
