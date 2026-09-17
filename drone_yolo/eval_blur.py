"""
eval_blur.py — 기체 흔들림(모션 블러) 내성 측정 (2026-09-17)

왜
  "흔들림은 jitter 로 커버한다" 는 오해를 정리하려고 만들었다.
  우리 `--target-dist jitter` 는 **사람 크기 지터**다 (몇 px 로 찍히게 할지). 흔들림과 무관하다.
  흔들림이 영상에 남기는 것은 두 가지고, 지금 학습이 덮는 범위가 다르다:
    · 겨냥이 틀어짐(회전 · 이동) → degrees 180 · translate 0.15 · flipud/fliplr 로 덮인다
    · **모션 블러** → 덮는 증강이 하나도 없다 (albumentations 미설치라 ultralytics 기본 Blur 도 꺼져 있다)
  그래서 블러만 따로 합성해 넣고 재현율이 언제 무너지는지 잰다.

블러 길이 ↔ 각속도
  L[px] = ω[rad/s] × f[px] × 노출[s]      (회전 흔들림 · 1920 입력 기준 f)
  L[px] = v[m/s] / 고도[m] × f[px] × 노출[s]  (수평 이동)
  → 잰 L 한계를 노출 시간으로 나누면 "몇 °/s 까지 버티나" 가 나온다

실행: python eval_blur.py --weights runs_person/fov_11m_1280_all/weights/best.pt
출력: metrics/eval_blur_<실행>.csv + 화면 표
"""
import argparse
import csv
import json
import os
from pathlib import Path

import cv2
import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")
BASE_DIR = Path(__file__).resolve().parent
# 가림 지형 시험셋의 칸 — (이름, 폴더, 1920 기준 f, 고도, 사람 px)
CELLS = (
    ("75° 25 m 서 있음", "s20_41", 1251, 25, 41), ("93° 25 m 서 있음", "s20_30", 911, 25, 30),
    ("75° 30 m 서 있음", "s20_34", 1251, 30, 34), ("93° 30 m 서 있음", "s20_25", 911, 30, 25),
    ("75° 30 m 누움", "l20_71", 1251, 30, 71), ("93° 30 m 누움", "l20_52", 911, 30, 52),
)
BLURS = (0, 2, 4, 6, 9, 13)      # 모션 블러 길이(px)
SEED = 42


def motion_kernel(length, angle_deg):
    k = np.zeros((length, length), np.float32)
    k[length // 2, :] = 1.0
    M = cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), angle_deg, 1.0)
    k = cv2.warpAffine(k, M, (length, length))
    s = k.sum()
    return k / s if s > 0 else k


def load_gt(img):
    t = Path(str(img).replace("/images/", "/labels/")).with_suffix(".txt")
    a = np.array([list(map(float, l.split()[1:])) for l in t.read_text().splitlines() if l.strip()]).reshape(-1, 4)
    return np.c_[(a[:, 0] - a[:, 2] / 2) * 1280, (a[:, 1] - a[:, 3] / 2) * 720,
                 (a[:, 0] + a[:, 2] / 2) * 1280, (a[:, 1] + a[:, 3] / 2) * 720]


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
    ap.add_argument("--weights", nargs="+", required=True)
    ap.add_argument("--testsets", default=str(BASE_DIR / "data" / "det_fov_test_budget"))
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--limit", type=int, default=700, help="칸마다 쓸 원본 수")
    args = ap.parse_args()

    from ultralytics import YOLO
    root = Path(args.testsets).resolve()

    for w in args.weights:
        wp = Path(w)
        name = wp.parent.parent.name if wp.parent.name == "weights" else wp.stem
        model = YOLO(str(wp))
        rows = []
        for label, dname, f_px, alt, px in CELLS:
            d = root / dname
            imgs = sorted((d / "images").glob("*.jpg"))[: args.limit]
            metas, gts = {}, {}
            for p in imgs:
                m = json.loads((d / "meta" / f"{p.stem}.json").read_text())
                keep = np.array([q["orig"] and not q["seam"] and q["near_ref"] for q in m], dtype=bool)
                if keep.any():
                    metas[p] = keep; gts[p] = load_gt(p)
            for L in BLURS:
                rnd = np.random.default_rng(SEED)
                recalls = []
                batch, keys = [], []
                for p in metas:
                    img = cv2.imread(str(p))
                    if L >= 2:
                        img = cv2.filter2D(img, -1, motion_kernel(L, float(rnd.uniform(0, 180))))
                    batch.append(img); keys.append(p)
                    if len(batch) == 32:
                        for k, res in zip(keys, model.predict(batch, imgsz=args.imgsz, conf=args.conf, verbose=False)):
                            recalls.append(matched(gts[k], res.boxes.xyxy.cpu().numpy())[metas[k]].mean())
                        batch, keys = [], []
                if batch:
                    for k, res in zip(keys, model.predict(batch, imgsz=args.imgsz, conf=args.conf, verbose=False)):
                        recalls.append(matched(gts[k], res.boxes.xyxy.cpu().numpy())[metas[k]].mean())
                r = float(np.mean(recalls))
                rows.append({"cell": label, "dir": dname, "f_px": f_px, "alt": alt, "px": px,
                             "blur_px": L, "images": len(recalls), "recall": r})
                print(f"  {name} {label} 블러 {L:>2}px → 재현율 {r:.3f} ({len(recalls)}장)", flush=True)

        out = BASE_DIR / "metrics" / f"eval_blur_{name}.csv"
        with open(out, "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0]) + ["recall_ratio", "blur_over_px"])
            wr.writeheader()
            base = {r["cell"]: next(x["recall"] for x in rows if x["cell"] == r["cell"] and x["blur_px"] == 0) for r in rows}
            for r in rows:
                wr.writerow({**r, "recall_ratio": round(r["recall"] / base[r["cell"]], 4) if base[r["cell"]] else "",
                             "blur_over_px": round(r["blur_px"] / r["px"], 3)})

        print(f"\n=== {name} · 재현율@{args.conf:g} (모션 블러 길이별) ===")
        print("  " + "칸".ljust(18) + "".join(f"{L}px".rjust(9) for L in BLURS))
        for label, dname, f_px, alt, px in CELLS:
            rs = [next(x["recall"] for x in rows if x["cell"] == label and x["blur_px"] == L) for L in BLURS]
            print("  " + label.ljust(18) + "".join(f"{v:9.3f}" for v in rs))
        print(f"→ {out}\n")


if __name__ == "__main__":
    main()
