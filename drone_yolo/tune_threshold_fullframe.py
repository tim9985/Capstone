"""
tune_threshold_fullframe.py — 추론 입력 크기별 F2 임계값 탐색 (2026-09-19)

왜
  [[NFR-V05 신뢰도 기준]] 은 "F2 로 임계값을 고른다"고 정해 놓고 기본값 0.15 를 그냥 쓰고 있었다.
  [[GSD 정규화 - 추론 입력 크기 스윕]] 에서 **입력 크기가 바뀌면 신뢰도 분포가 이동한다**는 것이
  확인됐다 (재현율과 오탐이 같은 방향으로 움직였다) → 크기마다 임계값을 따로 정해야 한다.

방법
  임계값마다 추론을 다시 돌리면 느리다. **conf=0.01 로 입력 크기마다 한 번만 추론**해
  모든 후보를 신뢰도와 함께 모아 두고, 임계값을 바꿔 가며 오프라인 집계한다.

채점 (eval_fullframe.py 와 같은 규칙)
  · 정답은 거울 복제 칸까지 전부 매칭 대상에 넣는다
  · 점수는 **원본 칸 · 이음새 아님 · 기준 크기**(orig & !seam & near_ref) 박스로만 낸다
  · TP = 채점 대상 정답이 매칭됨 · FN = 매칭 안 됨
  · FP = 어떤 정답(복제 포함)과도 매칭되지 않은 예측 → 거울 채움이 오탐을 부풀리지 않는다
  · 음성 세트(neg)는 정답이 없으므로 모든 예측이 FP

  F_beta = (1+b^2)·P·R / (b^2·P + R),  b=2 → 재현율에 4배 가중

실행: python tune_threshold_fullframe.py --weights runs_person/fov_11m_1280_all/weights/best.pt
출력: metrics/threshold_fullframe_<실행>.csv + 화면 요약
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
THRESHOLDS = np.round(np.arange(0.05, 0.61, 0.025), 3)


def load_gt(p):
    t = Path(str(p).replace("/images/", "/labels/")).with_suffix(".txt")
    if not t.exists():
        return np.zeros((0, 4))
    a = np.array([list(map(float, l.split()[1:])) for l in t.read_text().splitlines() if l.strip()]).reshape(-1, 4)
    return np.c_[(a[:, 0] - a[:, 2] / 2) * W, (a[:, 1] - a[:, 3] / 2) * H,
                 (a[:, 0] + a[:, 2] / 2) * W, (a[:, 1] + a[:, 3] / 2) * H]


def iou_matrix(gt, pr):
    if not len(gt) or not len(pr):
        return np.zeros((len(gt), len(pr)))
    x1 = np.maximum(gt[:, None, 0], pr[None, :, 0]); y1 = np.maximum(gt[:, None, 1], pr[None, :, 1])
    x2 = np.minimum(gt[:, None, 2], pr[None, :, 2]); y2 = np.minimum(gt[:, None, 3], pr[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])
    return inter / (area(gt)[:, None] + area(pr)[None, :] - inter + 1e-9)


def count(iou, keep, conf, thr, iou_thr=0.5):
    """임계값 thr 에서 (TP, FN, FP) — 신뢰도 높은 예측부터 1:1 탐욕 매칭"""
    sel = conf >= thr
    n_g = iou.shape[0]
    if not sel.any():
        return 0, int(keep.sum()), 0
    idx = np.where(sel)[0][np.argsort(-conf[sel])]
    hit = np.zeros(n_g, dtype=bool)
    used = 0
    for pj in idx:
        cand = np.where((~hit) & (iou[:, pj] >= iou_thr))[0]
        if len(cand):
            hit[cand[np.argmax(iou[cand, pj])]] = True
            used += 1
    tp = int((hit & keep).sum())
    return tp, int(keep.sum()) - tp, len(idx) - used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="runs_person/fov_11m_1280_all/weights/best.pt")
    ap.add_argument("--testsets", default=str(BASE_DIR / "data" / "det_fullframe"))
    ap.add_argument("--imgsz", type=int, nargs="+", default=[1280, 1600, 1920])
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--min-conf", type=float, default=0.01)
    args = ap.parse_args()

    from ultralytics import YOLO

    root = Path(args.testsets).resolve()
    sets = [d.name for d in sorted(root.iterdir()) if d.is_dir() and (d / "images").exists()]
    name = Path(args.weights).parent.parent.name
    rows = []

    for sz in args.imgsz:
        model = YOLO(args.weights)      # 정밀도·크기마다 새 인스턴스 (predictor 재사용 시 인자가 무시된다)
        cache = []                      # (iou 행렬, 채점 마스크, 신뢰도, 세트)
        for s in sets:
            d = root / s
            imgs = sorted((d / "images").glob("*.jpg"))
            for i in range(0, len(imgs), 8):          # 32 → 8 (09-19 OOM · L2 와 공존 대비)
                chunk = imgs[i:i + 8]
                for p, r in zip(chunk, model.predict([str(x) for x in chunk], imgsz=sz, conf=args.min_conf,
                                                     batch=args.batch, verbose=False)):
                    gt = load_gt(p)
                    mp = d / "meta" / f"{p.stem}.json"
                    if mp.exists():
                        meta = json.loads(mp.read_text())
                        meta = meta["boxes"] if isinstance(meta, dict) else meta   # det_fullframe 는 dict, det_fov_test 는 list
                        keep = np.array([m["orig"] and not m["seam"] and m["near_ref"] for m in meta], dtype=bool)
                    else:
                        keep = np.zeros(len(gt), dtype=bool)
                    keep = keep[:len(gt)] if len(keep) >= len(gt) else np.pad(keep, (0, len(gt) - len(keep)))
                    pr = r.boxes.xyxy.cpu().numpy()
                    cf = r.boxes.conf.cpu().numpy()
                    cache.append((iou_matrix(gt, pr), keep, cf, s))
            print(f"  입력 {sz} · {s}: {len(imgs)}장 추론 완료", flush=True)

        for thr in THRESHOLDS:
            tp = fn = fp = 0
            per = {}
            for iou, keep, cf, s in cache:
                a, b, c = count(iou, keep, cf, thr)
                tp += a; fn += b; fp += c
                q = per.setdefault(s, [0, 0, 0])
                q[0] += a; q[1] += b; q[2] += c
            P = tp / (tp + fp) if tp + fp else 0.0
            R = tp / (tp + fn) if tp + fn else 0.0
            F2 = 5 * P * R / (4 * P + R) if (P + R) else 0.0
            F1 = 2 * P * R / (P + R) if (P + R) else 0.0
            rows.append({"model": name, "imgsz": sz, "conf": thr, "TP": tp, "FN": fn, "FP": fp,
                         "precision": round(P, 4), "recall": round(R, 4),
                         "F1": round(F1, 4), "F2": round(F2, 4),
                         "fp_per_frame": round(fp / len(cache), 3),
                         **{f"recall_{s}": round(v[0] / (v[0] + v[1]), 4) if v[0] + v[1] else "" for s, v in per.items()}})

    out = BASE_DIR / "metrics" / f"threshold_fullframe_{name}.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader(); wr.writerows(rows)

    print(f"\n=== {name} · 입력 크기별 F2 최적 임계값 (전체 프레임 {len(cache)}장)")
    print(f"{'입력':>6} {'최적 conf':>9} {'F2':>7} {'정밀도':>7} {'재현율':>7} {'오탐/장':>8}  |  {'conf 0.15 에서':>14}")
    for sz in args.imgsz:
        sub = [r for r in rows if r["imgsz"] == sz]
        best = max(sub, key=lambda r: r["F2"])
        cur = min(sub, key=lambda r: abs(r["conf"] - 0.15))
        print(f"{sz:>6} {best['conf']:>9.3f} {best['F2']:>7.4f} {best['precision']:>7.4f} "
              f"{best['recall']:>7.4f} {best['fp_per_frame']:>8.2f}  |  F2 {cur['F2']:.4f} · 재현율 {cur['recall']:.4f}")
    print(f"→ {out}")


if __name__ == "__main__":
    main()
