"""
verify_detector.py — 컨테이너에서도 우리 수치가 나오는지 확인 (인수 검증)

  평가셋 (images/ + labels/ · YOLO 형식 · 1920×1080) 에 app.vision.detector 를 돌려
  AP50 (101점 · 확신도 ≥0.01 · IoU 0.5 · eval_test_v2.py 와 같은 식) · 재현율@0.15 · 한 장 ms 를 낸다
  --expect 를 주면 |AP50 − 기대값| ≤ --tol 인지 판정 (exit 0 통과 · 1 실패)

  기대값 (우리 환경 · soup_v7r2 · test_v2 1,407장): AP50 0.5927 (.pt FP16)
실행: python -m app.tools.verify_detector --data /data/eval/test_v2 --weights /data/models/soup_v7r2/best.pt [--engine …] [--limit 300] [--expect 0.5927]
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from app.vision.detector import PersonDetector


def load_gt(img_path, w, h):
    t = Path(str(img_path).replace("/images/", "/labels/")).with_suffix(".txt")
    if not t.exists():
        return np.zeros((0, 4), np.float32)
    rows = [l.split() for l in t.read_text().splitlines() if l.strip()]
    if not rows:
        return np.zeros((0, 4), np.float32)
    a = np.array([[float(v) for v in r[1:5]] for r in rows], np.float32)
    return np.c_[(a[:, 0] - a[:, 2] / 2) * w, (a[:, 1] - a[:, 3] / 2) * h, (a[:, 0] + a[:, 2] / 2) * w, (a[:, 1] + a[:, 3] / 2) * h]


def iou_mat(g, p):
    if not len(g) or not len(p):
        return np.zeros((len(g), len(p)), np.float32)
    x1 = np.maximum(g[:, None, 0], p[None, :, 0]); y1 = np.maximum(g[:, None, 1], p[None, :, 1])
    x2 = np.minimum(g[:, None, 2], p[None, :, 2]); y2 = np.minimum(g[:, None, 3], p[None, :, 3])
    it = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    ar = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])
    return it / (ar(g)[:, None] + ar(p)[None, :] - it + 1e-9)


def ap101(s, t, n_gt):
    if not len(s) or n_gt == 0:
        return 0.0
    o = np.argsort(-s); t = t[o]
    tp = np.cumsum(t); fp = np.cumsum(~t)
    rec = tp / n_gt; prec = tp / np.maximum(tp + fp, 1)
    return float(np.mean([prec[rec >= r].max() if (rec >= r).any() else 0 for r in np.linspace(0, 1, 101)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--engine", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--expect", type=float, default=None)
    ap.add_argument("--tol", type=float, default=0.003)
    a = ap.parse_args()
    det = PersonDetector(a.weights, engine=a.engine)
    imgs = sorted((Path(a.data) / "images").glob("*.jpg"))
    if a.limit:
        imgs = imgs[:a.limit]
    scores, tps, n_gt, hit15, ms = [], [], 0, 0, []
    for i, p in enumerate(imgs):
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]
        gt = load_gt(p, w, h); n_gt += len(gt)
        box, cf, t = det.detect(img, conf=0.01)
        if i >= 10:
            ms.append(t)
        o = np.argsort(-cf); box, cf = box[o], cf[o]
        M = iou_mat(gt, box); used = np.zeros(len(gt), bool)
        for j in range(len(box)):
            k = -1
            if len(gt):
                cand = np.where((M[:, j] >= 0.5) & ~used)[0]
                if len(cand):
                    k = cand[np.argmax(M[cand, j])]
            scores.append(float(cf[j])); tps.append(k >= 0)
            if k >= 0:
                used[k] = True
                hit15 += int(cf[j] >= 0.15)
    s, t = np.array(scores), np.array(tps, bool)
    out = {"모델": det.name, "백엔드": det.backend, "장수": len(imgs), "정답": int(n_gt),
           "AP50": round(ap101(s, t, n_gt), 4), "재현율@0.15": round(hit15 / max(n_gt, 1), 4),
           "한 장 ms (확신도 0.01 · 중앙)": round(float(np.median(ms)), 1) if ms else None}
    ok = True
    if a.expect is not None:
        ok = abs(out["AP50"] - a.expect) <= a.tol
        out["판정"] = f"{'통과' if ok else '실패'} (기대 {a.expect} ± {a.tol})"
    print(json.dumps(out, ensure_ascii=False))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
