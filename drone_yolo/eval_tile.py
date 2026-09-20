"""
eval_tile.py — 타일 추론 평가 (2026-09-19)

왜
  학습과 추론의 **입력 형식이 다르다.**
    학습: 1920 프레임에서 **1280×720 창을 1:1 배율로** 오려낸 것
    추론: 1920×1080 전체를 한 번에 (imgsz 로 축소·확대됨)
  1920×1080 을 1280×720 타일로 잘라 넣으면 **학습과 같은 형식 · 같은 배율**이 된다.
  (문헌의 SAHI — Slicing Aided Hyper Inference. 우리는 학습 자체가 타일이라 더 잘 맞는다)

타일 배치 (겹침 50 %)
  x = 0, 640   y = 0, 360   → 2×2 = 4장. 1920×1080 을 빈틈없이 덮는다
  타일 하나가 정확히 1280×720 이므로 **리사이즈가 일어나지 않는다**

합치기
  타일 좌표 → 프레임 좌표로 옮긴 뒤 전체에 NMS (IoU 0.6).
  겹침 구간에서 같은 사람이 두 번 잡히는 것을 없앤다.

채점 (eval_fullframe.py 와 같은 규칙)
  정답은 거울 복제 칸까지 매칭 대상에 넣고, 점수는 orig & !seam & near_ref 박스로만 낸다.
  오탐 = 어떤 정답과도 매칭되지 않은 예측.

실행: python eval_tile.py --weights runs_person/fov_11m_1280_all/weights/best.pt
출력: metrics/tile_fullframe_<실행>.csv
"""
import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")
BASE_DIR = Path(__file__).resolve().parent
W, H = 1920, 1080
TW, TH = 1280, 720
TILES = [(x, y) for y in (0, H - TH) for x in (0, W - TW)]      # 4장 · 겹침 50 %


def load_gt(p):
    t = Path(str(p).replace("/images/", "/labels/")).with_suffix(".txt")
    if not t.exists():
        return np.zeros((0, 4))
    a = np.array([list(map(float, l.split()[1:])) for l in t.read_text().splitlines() if l.strip()]).reshape(-1, 4)
    return np.c_[(a[:, 0] - a[:, 2] / 2) * W, (a[:, 1] - a[:, 3] / 2) * H,
                 (a[:, 0] + a[:, 2] / 2) * W, (a[:, 1] + a[:, 3] / 2) * H]


def nms(box, conf, thr=0.6):
    if not len(box):
        return np.zeros(0, dtype=int)
    idx = np.argsort(-conf); keep = []
    area = (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])
    while len(idx):
        i = idx[0]; keep.append(i)
        if len(idx) == 1:
            break
        r = idx[1:]
        x1 = np.maximum(box[i, 0], box[r, 0]); y1 = np.maximum(box[i, 1], box[r, 1])
        x2 = np.minimum(box[i, 2], box[r, 2]); y2 = np.minimum(box[i, 3], box[r, 3])
        inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        idx = r[inter / (area[i] + area[r] - inter + 1e-9) < thr]
    return np.array(keep, dtype=int)


def matched(gt, pr, thr=0.5):
    """정답마다 매칭 여부 + 매칭된 예측 수"""
    hit = np.zeros(len(gt), dtype=bool)
    if not len(gt) or not len(pr):
        return hit, 0
    x1 = np.maximum(gt[:, None, 0], pr[None, :, 0]); y1 = np.maximum(gt[:, None, 1], pr[None, :, 1])
    x2 = np.minimum(gt[:, None, 2], pr[None, :, 2]); y2 = np.minimum(gt[:, None, 3], pr[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])
    iou = inter / (area(gt)[:, None] + area(pr)[None, :] - inter + 1e-9)
    used = set()
    for gi, pj in zip(*np.unravel_index(np.argsort(-iou, axis=None), iou.shape)):
        if iou[gi, pj] < thr:
            break
        if hit[gi] or pj in used:
            continue
        hit[gi] = True; used.add(pj)
    return hit, len(used)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="runs_person/fov_11m_1280_all/weights/best.pt")
    ap.add_argument("--testsets", default=str(BASE_DIR / "data" / "det_fullframe"))
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--nms-iou", type=float, default=0.6)
    args = ap.parse_args()

    import cv2
    from ultralytics import YOLO

    root = Path(args.testsets).resolve()
    sets = [d.name for d in sorted(root.iterdir()) if d.is_dir() and (d / "images").exists()]
    name = Path(args.weights).parent.parent.name
    model = YOLO(args.weights)
    rows = []

    for s in sets:
        d = root / s
        imgs = sorted((d / "images").glob("*.jpg"))
        rec, fps, times = [], [], []
        for p in imgs:
            img = cv2.imread(str(p))
            if img is None:
                continue
            t0 = time.perf_counter()
            tiles = [img[y:y + TH, x:x + TW] for x, y in TILES]
            res = model.predict(tiles, imgsz=1280, conf=args.conf, verbose=False)
            box, cf = [], []
            for (x, y), r in zip(TILES, res):
                b = r.boxes.xyxy.cpu().numpy()
                if len(b):
                    box.append(b + np.array([x, y, x, y])); cf.append(r.boxes.conf.cpu().numpy())
            if box:
                box = np.concatenate(box); cf = np.concatenate(cf)
                k = nms(box, cf, args.nms_iou); box = box[k]
            else:
                box = np.zeros((0, 4))
            times.append((time.perf_counter() - t0) * 1000)

            gt = load_gt(p)
            mp = d / "meta" / f"{p.stem}.json"
            if mp.exists():
                meta = json.loads(mp.read_text())
                meta = meta["boxes"] if isinstance(meta, dict) else meta
                keep = np.array([m["orig"] and not m["seam"] and m["near_ref"] for m in meta], dtype=bool)
                keep = keep[:len(gt)] if len(keep) >= len(gt) else np.pad(keep, (0, len(gt) - len(keep)))
            else:
                keep = np.zeros(len(gt), dtype=bool)
            hit, n_match = matched(gt, box)
            if keep.any():
                rec.append(float(hit[keep].mean()))
            fps.append(max(0, len(box) - n_match))
        rows.append({"model": name, "set": s, "mode": f"tile 4×({TW}×{TH})", "images": len(imgs),
                     f"recall@{args.conf:g}": round(float(np.mean(rec)), 4) if rec else "",
                     "fp_per_frame": round(float(np.mean(fps)), 3),
                     "ms_per_frame": round(float(np.median(times)), 1)})
        print(f"  {s}: {len(imgs)}장 · 재현율 {rows[-1][f'recall@{args.conf:g}']} · "
              f"오탐 {rows[-1]['fp_per_frame']}/장 · {rows[-1]['ms_per_frame']} ms", flush=True)

    out = BASE_DIR / "metrics" / f"tile_fullframe_{name}.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    print(f"\n→ {out}")
    print("비교 대상: metrics/eval_fullframe_%s.csv (단일 패스 1280/1600/1920/2560/3200)" % name)


if __name__ == "__main__":
    main()
