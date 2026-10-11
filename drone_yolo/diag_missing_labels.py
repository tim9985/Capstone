"""
diag_missing_labels.py — 진단: v7 학습 크롭에 라벨이 빠진 사람이 얼마나 있나 (10-11 · v7 넘기 검토)

  가설  학습 크롭에 라벨 없는 (작은) 사람이 있으면 모델은 「사람처럼 보이는 작은 덩어리 = 배경」 으로 배운다
        → 작은 사람 확신을 낮춘다 (Karen 사람의 60 % 를 확신 0.01 에서도 못 봄)
  방법  학습에 쓰이지 않은 독립 모델 (FlyPose 검출기 · RT-DETRv2 · 공식 문턱 0.7) 로 v7 학습 크롭을 훑어
        정답 박스와 IoU < 0.3 인 확신 높은 검출 = 「라벨 빠짐 후보」 · 출처별 · 크기별 · 표본 그림 (눈으로 확인)
  ⚠ 후보 중 일부는 FlyPose 의 오탐 (물체) — 그림으로 비율을 본다
실행: /home/se/miniconda3/envs/drone/bin/python diag_missing_labels.py [--per-source 600]
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from eval_external_det import ENGINE, Det
from eval_test_v2 import iou_mat

BASE = Path(__file__).resolve().parent
OUT = Path("/tmp/claude-1001/-home-se-JupyterLAB/027ea9bd-abdf-476f-ad0d-25f3e2ea7ffc/scratchpad/missing")


def src(p):
    q = p.split("/data/")[1].split("/"); return q[0] + ("/" + q[1] if len(q) > 2 else "")


def gt(p, w, h):
    t = Path(p.replace("/images/", "/labels/")).with_suffix(".txt")
    if not t.exists():
        return np.zeros((0, 4), np.float32)
    a = np.array([[float(v) for v in l.split()[1:5]] for l in open(t) if l.strip()], np.float32).reshape(-1, 4)
    return np.c_[(a[:, 0] - a[:, 2] / 2) * w, (a[:, 1] - a[:, 3] / 2) * h, (a[:, 0] + a[:, 2] / 2) * w, (a[:, 1] + a[:, 3] / 2) * h]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-source", type=int, default=600)
    ap.add_argument("--thr", type=float, default=0.7)
    a = ap.parse_args()
    L = [l.strip() for l in open(BASE / "configs" / "lists" / "train_v7.txt") if l.strip()]
    by = defaultdict(list)
    for p in L:
        by[src(p)].append(p)
    det = Det(ENGINE); rnd = random.Random(0); res = {}; samples = []
    OUT.mkdir(parents=True, exist_ok=True)
    for s, ps in sorted(by.items()):
        n_img = n_gt = n_miss = 0; miss_px = []; img_with = 0
        for p in rnd.sample(ps, min(a.per_source, len(ps))):
            img = cv2.imread(p); h, w = img.shape[:2]; g = gt(p, w, h)
            b, sc = det([img])[0]; keep = sc >= a.thr; b, sc = b[keep], sc[keep]
            n_img += 1; n_gt += len(g)
            if len(b):
                m = iou_mat(b, g).max(1) if len(g) else np.zeros(len(b))
                miss = b[m < 0.3]; n_miss += len(miss); img_with += int(len(miss) > 0)
                miss_px += [float(max(x2 - x1, y2 - y1)) for x1, y1, x2, y2 in miss]
                for (x1, y1, x2, y2), c in zip(miss, sc[m < 0.3]):
                    samples.append((s, p, (x1, y1, x2, y2), float(c)))
        res[s] = {"크롭": n_img, "정답 사람": n_gt, "라벨 빠짐 후보": n_miss, "후보 / 정답": round(n_miss / max(n_gt, 1), 3),
                  "후보 있는 크롭 비율": round(img_with / max(n_img, 1), 3), "후보 키 px 중앙": round(float(np.median(miss_px)), 1) if miss_px else None}
        print(s, res[s], flush=True)
    # 출처마다 후보 크롭 표본 12개 (눈으로 확인)
    for s in res:
        cand = [x for x in samples if x[0] == s]
        tiles = []
        for _, p, (x1, y1, x2, y2), c in rnd.sample(cand, min(12, len(cand))):
            img = cv2.imread(p); cx, cy = (x1 + x2) / 2, (y1 + y2) / 2; r = max(48, 2.5 * max(x2 - x1, y2 - y1))
            X1, Y1, X2, Y2 = int(max(0, cx - r)), int(max(0, cy - r)), int(min(img.shape[1], cx + r)), int(min(img.shape[0], cy + r))
            crop = img[Y1:Y2, X1:X2].copy(); cv2.rectangle(crop, (int(x1 - X1), int(y1 - Y1)), (int(x2 - X1), int(y2 - Y1)), (0, 0, 255), 1)
            t = cv2.resize(crop, (200, 200)); cv2.putText(t, f"{c:.2f}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1); tiles.append(t)
        if tiles:
            tiles += [np.zeros_like(tiles[0])] * ((-len(tiles)) % 6)
            cv2.imwrite(str(OUT / f"{s.replace('/', '_')}.jpg"), np.vstack([np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)]))
    (BASE / "metrics" / "diag_missing_labels.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
