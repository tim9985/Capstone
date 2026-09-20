"""
eval_tile_edge.py — 타일 가장자리에서 성능이 떨어지는지 (2026-09-20)

왜
  학습 크롭은 사람이 거의 항상 창 **가운데**에 온다 (크롭 위치 지터 ±30 %).
  저장된 크롭 5,770개를 세어 보니 가장자리 100 px 안에 있는 사람이 **1.8 %** 뿐이다.
  그런데 타일 추론에서는 사람이 타일 **어디에나** 온다 → 학습/추론 불일치일 수 있다.

  ⚠ 단, 학습 중에는 `translate 0.15` 증강이 이미 이미지를 ±15 % 흔든다.
    모델이 실제로 보는 분포는 저장된 크롭보다 넓다. 이 실험은 **그래도 남는 차이**를 잰다.

무엇을 재나
  타일마다 그 안에 **온전히 들어온** 채점 대상 정답을 골라,
  **가장자리까지 거리**(박스 경계에서 타일 경계까지, 가로·세로 중 작은 값) 구간별로
  그 **타일 자신의 예측**이 맞혔는지 본다 (NMS 합치기 전 — 다른 타일이 구해 주는 효과를 뺀다).

실행: python eval_tile_edge.py --weights runs_person/fov_11m_1280_all/weights/best.pt
출력: metrics/tile_edge_<실행>.csv
"""
import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")
BASE_DIR = Path(__file__).resolve().parent
W, H = 1920, 1080
TW, TH = 1280, 720
TILES = [(x, y) for y in (0, H - TH) for x in (0, W - TW)]
BANDS = [(0, 20), (20, 50), (50, 100), (100, 200), (200, 10000)]


def load_gt(p):
    t = Path(str(p).replace("/images/", "/labels/")).with_suffix(".txt")
    if not t.exists():
        return np.zeros((0, 4))
    a = np.array([list(map(float, l.split()[1:])) for l in t.read_text().splitlines() if l.strip()]).reshape(-1, 4)
    return np.c_[(a[:, 0] - a[:, 2] / 2) * W, (a[:, 1] - a[:, 3] / 2) * H,
                 (a[:, 0] + a[:, 2] / 2) * W, (a[:, 1] + a[:, 3] / 2) * H]


def matched(gt, pr, thr=0.5):
    hit = np.zeros(len(gt), dtype=bool)
    if not len(gt) or not len(pr):
        return hit
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
    return hit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="runs_person/fov_11m_1280_all/weights/best.pt")
    ap.add_argument("--testsets", default=str(BASE_DIR / "data" / "det_fullframe"))
    ap.add_argument("--conf", type=float, default=0.15)
    args = ap.parse_args()

    import cv2
    from ultralytics import YOLO

    root = Path(args.testsets).resolve()
    name = Path(args.weights).parent.parent.name if Path(args.weights).parent.name == "weights" else Path(args.weights).stem
    engine = str(args.weights).endswith(".engine")
    model = YOLO(args.weights)
    sz = (((TH + 31) // 32) * 32, TW) if engine else 1280

    # (세트, 구간) → [맞힘 여부]
    acc = {}
    for s in ("s38", "l128"):
        d = root / s
        imgs = sorted((d / "images").glob("*.jpg"))
        for p in imgs:
            img = cv2.imread(str(p))
            if img is None:
                continue
            gt = load_gt(p)
            mp = d / "meta" / f"{p.stem}.json"
            if not mp.exists() or not len(gt):
                continue
            meta = json.loads(mp.read_text())
            meta = meta["boxes"] if isinstance(meta, dict) else meta
            keep = np.array([m["orig"] and not m["seam"] and m["near_ref"] for m in meta], dtype=bool)
            keep = keep[:len(gt)] if len(keep) >= len(gt) else np.pad(keep, (0, len(gt) - len(keep)))
            if not keep.any():
                continue

            tiles = [img[y:y + TH, x:x + TW] for x, y in TILES]
            res = model.predict(tiles, imgsz=sz, conf=args.conf, batch=len(tiles), verbose=False)

            for (ox, oy), r in zip(TILES, res):
                # 이 타일에 **온전히** 들어온 채점 대상만
                inside = keep & (gt[:, 0] >= ox) & (gt[:, 1] >= oy) & (gt[:, 2] <= ox + TW) & (gt[:, 3] <= oy + TH)
                if not inside.any():
                    continue
                g = gt[inside] - np.array([ox, oy, ox, oy])          # 타일 좌표계로
                pr = r.boxes.xyxy.cpu().numpy()
                hit = matched(g, pr)
                # 가장자리까지 거리 = 박스 경계와 타일 경계 사이 최소값
                dist = np.minimum.reduce([g[:, 0], g[:, 1], TW - g[:, 2], TH - g[:, 3]])
                for dd, hh in zip(dist, hit):
                    for lo, hi in BANDS:
                        if lo <= dd < hi:
                            acc.setdefault((s, (lo, hi)), []).append(bool(hh))
                            break
        print(f"  {s} 처리 완료", flush=True)

    rows = []
    print(f"\n=== {name} · 타일 가장자리까지 거리별 재현율@{args.conf:g} (타일 자신의 예측만 · NMS 전)")
    print(f"{'세트':>6} {'거리 구간':>14} {'정답 수':>8} {'재현율':>8}")
    for s in ("s38", "l128"):
        for lo, hi in BANDS:
            v = acc.get((s, (lo, hi)), [])
            if not v:
                continue
            r = float(np.mean(v))
            band = f"{lo}~{hi if hi < 9999 else '∞'} px"
            print(f"{s:>6} {band:>14} {len(v):>8,} {r:>8.4f}")
            rows.append({"model": name, "set": s, "edge_lo": lo, "edge_hi": hi,
                         "n": len(v), f"recall@{args.conf:g}": round(r, 4)})

    out = BASE_DIR / "metrics" / f"tile_edge_{name}.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
