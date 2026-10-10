"""
kp_posture_k2_extra.py — K2 보조 확인 (10-10 · K2 판정 뒤 · 판정을 바꾸지 않음)

  ① 회차 단위 교차검증 — Archangel 은 같은 사람이 여러 영상 (회차 · 고도) 에 나온다 → 영상이 아니라 회차 (AA_BP_<n>) 로 묶어 나눠도 남는가
  ② 다른 관절점 모델 — yolo11m-pose (COCO · K1 방식 크롭) 로 같은 시험 → 이점이 FlyPose-H 만의 것인가
     yolo 가 사람을 못 찾으면 관절 0 + 찾음 표시 0 (특징에 「찾음」 하나 더)
실행: /home/se/miniconda3/envs/drone/bin/python kp_posture_k2_extra.py   → metrics/kp_posture_k2_extra.json
"""
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from ultralytics import YOLO

from kp_flypose import yolo_kp
from kp_posture_k2 import CACHE, SRC, extract, load_rows
from posture_v2 import auc

BASE = Path(__file__).resolve().parent
YCACHE = BASE / "runs_state" / "k2_keypoints_yolo11m.npz"


def yolo_feats(rows):
    if YCACHE.exists():
        z = np.load(YCACHE, allow_pickle=True)
        if list(z["files"]) == [r["file"] for r in rows]:
            return z["xy"], z["sc"], z["found"]
    m = YOLO(str(BASE.parent / "data" / "raw" / "pose_weights" / "yolo11m-pose.pt"))
    XY, SC, FD = [], [], []
    for i, r in enumerate(rows):
        img = cv2.imread(str(SRC / r["file"])); side = img.shape[0] / 1.5; c = img.shape[0] / 2
        bw, bh = side * r["w"] / r["L"], side * r["h"] / r["L"]
        p = yolo_kp(m, img, (c - bw / 2, c - bh / 2, c + bw / 2, c + bh / 2))
        if p is None:
            XY.append(np.zeros((17, 2), np.float32)); SC.append(np.zeros(17, np.float32)); FD.append(0)
        else:
            XY.append(((p[0] - c) / side).astype(np.float32)); SC.append(p[1].astype(np.float32)); FD.append(1)
        if i % 3000 == 0:
            print(f"  yolo 관절 {i}/{len(rows)}", flush=True)
    XY, SC, FD = np.array(XY), np.array(SC), np.array(FD)
    np.savez(YCACHE, files=np.array([r["file"] for r in rows]), xy=XY, sc=SC, found=FD)
    return XY, SC, FD


def oof(X, y, groups):
    p = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, groups):
        p[te] = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000)).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return p


def main():
    rows = load_rows(); HXY, HSC = extract(rows); YXY, YSC, YFD = yolo_feats(rows)
    y = np.array([r["pose"] == "lying" for r in rows]); amb = np.array([r["h"] >= r["w"] for r in rows])
    box = np.c_[np.log(np.array([r["h"] / r["w"] for r in rows])), np.log(np.array([r["L"] for r in rows]))]
    n = len(rows)
    feats = {"A_box": box, "B_H": np.c_[box, HXY.reshape(n, -1), HSC], "B_yolo11m": np.c_[box, YXY.reshape(n, -1), YSC, YFD]}
    groupings = {"영상": np.array([r["place"] for r in rows]), "회차": np.array(["_".join(r["place"].split("_")[:3]) for r in rows])}
    res = {"yolo11m 사람 찾음": round(float(YFD.mean()), 3), "yolo11m 찾음 (누움 · 서기)": [round(float(YFD[y].mean()), 3), round(float(YFD[~y].mean()), 3)],
           "회차 수": len(set(groupings["회차"])), "결과": {}}
    for gname, g in groupings.items():
        P = {k: oof(X, y, g) for k, X in feats.items()}
        blocks = sorted(set(g)); idx = defaultdict(list)
        for i, v in enumerate(g):
            idx[v].append(i)
        rng = np.random.default_rng(0); d = defaultdict(list)
        for _ in range(1000):
            s = np.concatenate([idx[v] for v in rng.choice(blocks, len(blocks))])
            for zn, z0 in (("세로≥가로", amb), ("전체", np.ones(n, bool))):
                z = z0[s]; ys = y[s]
                if (z & ys).any() and (z & ~ys).any():
                    a = auc(P["A_box"][s][z & ys], P["A_box"][s][z & ~ys])
                    for k in ("B_H", "B_yolo11m"):
                        d[(k, zn)].append(auc(P[k][s][z & ys], P[k][s][z & ~ys]) - a)
        out = {}
        for zn, z in (("세로≥가로", amb), ("전체", np.ones(n, bool))):
            out[zn] = {k: round(float(auc(P[k][z & y], P[k][z & ~y])), 4) for k in feats} | {
                f"{k} − A 95%": [round(float(np.percentile(d[(k, zn)], 2.5)), 4), round(float(np.percentile(d[(k, zn)], 97.5)), 4)] for k in ("B_H", "B_yolo11m")} | {
                "누움을 서기로 (0.5)": {k: round(float((P[k][z & y] < 0.5).mean()), 3) for k in feats}}
        res["결과"][f"{gname} 단위"] = out
    (BASE / "metrics" / "kp_posture_k2_extra.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
