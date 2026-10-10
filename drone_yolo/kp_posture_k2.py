"""
kp_posture_k2.py — K2: FlyPose-H 관절점이 「박스가 서기처럼 보이는 누움」 을 박스 판정기보다 잘 가르는가 (10-10 · 판정 기준은 _학습 큐 「10-10 K2」)

  데이터  Archangel-Real 크롭 (data/pose_eval/archangel_real · 128 px 정사각 · 사람 긴 변 × 1.5) · 서기 vs 누움 · 긴 변 (1080) ≥ 45 px
  관절점  FlyPose-H TensorRT (FP32 · 14 ms) — kp_flypose 의 공식 전처리 그대로 · 사람 박스 = 크롭 가운데 (긴 변 128/1.5 · 가로세로 w1080 : h1080)
          관절은 한 번만 뽑아 runs_state/k2_keypoints.npz 에 둔다 (git 밖 · Archangel 파생)
  비교    A 박스 2특징 [log(세로/가로) · log(긴 변)] · B = A + 관절 17점 (박스 중심 · 긴 변으로 정규화한 좌표 34 + 확신 17)
          같은 로지스틱 (표준화 · C=1) · 영상 묶음 5겹 교차검증 바깥 예측
  지표    누움 vs 서기 AUROC — 주: 세로 ≥ 가로 구역 · 보조: 전체 · 크기별 · 영상 블록 짝 부트스트랩 1,000회 (B − A)
실행: /home/se/miniconda3/envs/drone/bin/python kp_posture_k2.py   → metrics/kp_posture_k2.json
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import tensorrt as trt
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from kp_flypose import fp_decode, fp_preprocess
from posture_v2 import auc

BASE = Path(__file__).resolve().parent
SRC = BASE.parent / "data" / "pose_eval" / "archangel_real"
ENGINE = BASE.parent / "data" / "raw" / "flypose" / "checkpoints" / "pose" / "flypose_h" / "flypose_h_fp32_3090.engine"
CACHE = BASE / "runs_state" / "k2_keypoints.npz"
MIN_L = 45
SIZES = [(45, 61), (61, 81), (81, 1e9)]


class HPose:
    def __init__(self, path):
        log = trt.Logger(trt.Logger.WARNING)
        self.e = trt.Runtime(log).deserialize_cuda_engine(open(path, "rb").read()); self.ctx = self.e.create_execution_context()
        self.x = torch.empty(1, 3, 256, 192, device="cuda"); self.y = torch.empty(1, 17, 64, 48, device="cuda"); self.st = torch.cuda.Stream()
        self.ctx.set_tensor_address("input", self.x.data_ptr()); self.ctx.set_tensor_address("output", self.y.data_ptr())

    def __call__(self, frame, box):
        t, inv = fp_preprocess(frame, box)
        self.x.copy_(torch.from_numpy(t)); self.ctx.execute_async_v3(self.st.cuda_stream); self.st.synchronize()
        return fp_decode(self.y.cpu().numpy()[0], inv)


def load_rows():
    rows = []
    for r in csv.DictReader(open(SRC / "crops.csv", encoding="utf-8")):
        w, h = float(r["w1080"]), float(r["h1080"])
        if r["pose"] in ("standing", "lying") and max(w, h) >= MIN_L:
            rows.append(r | {"w": w, "h": h, "L": max(w, h)})
    return rows


def extract(rows):
    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=True)
        if list(z["files"]) == [r["file"] for r in rows]:
            return z["xy"], z["sc"]
    hp = HPose(ENGINE); XY, SC = [], []
    for i, r in enumerate(rows):
        img = cv2.imread(str(SRC / r["file"])); side = img.shape[0] / 1.5          # 사람 긴 변 (크롭 px)
        bw, bh = side * r["w"] / r["L"], side * r["h"] / r["L"]; c = img.shape[0] / 2
        box = (c - bw / 2, c - bh / 2, c + bw / 2, c + bh / 2)
        xy, sc = hp(img, box)
        XY.append(((xy - c) / side).astype(np.float32)); SC.append(sc)        # 박스 중심 · 긴 변으로 정규화
        if i % 3000 == 0:
            print(f"  관절 {i}/{len(rows)}", flush=True)
    XY, SC = np.array(XY), np.array(SC)
    CACHE.parent.mkdir(exist_ok=True); np.savez(CACHE, files=np.array([r["file"] for r in rows]), xy=XY, sc=SC)
    return XY, SC


def main():
    rows = load_rows(); print("크롭", len(rows), {p: sum(r["pose"] == p for r in rows) for p in ("standing", "lying")})
    XY, SC = extract(rows)
    y = np.array([r["pose"] == "lying" for r in rows]); g = np.array([r["place"] for r in rows])
    box = np.c_[np.log(np.array([r["h"] / r["w"] for r in rows])), np.log(np.array([r["L"] for r in rows]))]
    feats = {"A_box": box, "B_box+kp": np.c_[box, XY.reshape(len(rows), -1), SC]}
    oof = {}
    for name, X in feats.items():
        p = np.zeros(len(rows))
        for tr, te in GroupKFold(5).split(X, y, g):
            m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000)).fit(X[tr], y[tr])
            p[te] = m.predict_proba(X[te])[:, 1]
        oof[name] = p
    amb = np.array([r["h"] >= r["w"] for r in rows]); L = np.array([r["L"] for r in rows])
    zones = {"세로≥가로 (주)": amb, "전체": np.ones(len(rows), bool)} | {f"긴 변 {a}~{b if b < 1e9 else '∞'}": (L >= a) & (L < b) for a, b in SIZES}
    res = {"crops": len(rows), "videos": len(set(g)), "zones": {}}
    for zn, z in zones.items():
        res["zones"][zn] = {"n 누움 / 서기": [int((z & y).sum()), int((z & ~y).sum())]} | {
            name: round(float(auc(p[z & y], p[z & ~y])), 4) for name, p in oof.items()}
        lr = (oof["A_box"][z & y] < 0.5).mean(), (oof["B_box+kp"][z & y] < 0.5).mean()
        res["zones"][zn]["누움을 서기로 (0.5) A → B"] = [round(float(lr[0]), 3), round(float(lr[1]), 3)]
    vids = sorted(set(g)); idx = defaultdict(list)
    for i, v in enumerate(g):
        idx[v].append(i)
    rng = np.random.default_rng(0); d = defaultdict(list)
    for _ in range(1000):
        s = np.concatenate([idx[v] for v in rng.choice(vids, len(vids))])
        for zn in ("세로≥가로 (주)", "전체"):
            z = zones[zn][s]; ys = y[s]
            if (z & ys).any() and (z & ~ys).any():
                a = auc(oof["A_box"][s][z & ys], oof["A_box"][s][z & ~ys]); b = auc(oof["B_box+kp"][s][z & ys], oof["B_box+kp"][s][z & ~ys])
                d[zn].append(b - a)
    res["AUROC (B − A) 95%"] = {k: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)] for k, v in d.items()}
    ci = res["AUROC (B − A) 95%"]
    res["K2 판정"] = bool(ci["세로≥가로 (주)"][0] > 0 and ci["전체"][0] > -0.01)
    (BASE / "metrics" / "kp_posture_k2.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
