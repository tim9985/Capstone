"""
eval_tilt_groups.py — 마운트각 태그가 실제로 성능을 가르는지 검증 (2026-09-22)

왜
  survey_tilt2.py 가 만든 "비스듬/수직" 태그는 **추정**이다 (박스 크기 vs y 상관).
  지형 경사·배우 이동이 섞이므로, 그 태그로 학습 데이터를 자르기 전에
  **태그가 실제로 성능 차이를 설명하는지** 먼저 확인해야 한다.

무엇을
  Okutama 는 같은 장소·같은 배우를 여러 시퀀스로 찍었다.
  상관 r 이 뚜렷한 두 부류에 **같은 모델**을 돌려 재현율 차이를 본다.
    비스듬  r ≥ 0.60
    수직    r ≤ 0.15
  차이가 크면 태그가 의미 있다. 작으면 그 태그는 잡음이다.

실행: python eval_tilt_groups.py --weights <pt>
"""
import argparse, csv, glob
from collections import defaultdict
from pathlib import Path
import numpy as np

BASE = Path(__file__).resolve().parent
W, H = 1280, 720          # Extracted-Frames-1280x720
SRC_W, SRC_H = 3840, 2160


def load_labels(seq):
    """시퀀스 라벨 → {frame: [xyxy…]} (1280x720 좌표)"""
    per = defaultdict(list)
    for lp in glob.glob(str(BASE / f"data/raw/okutama/**/Labels/SingleActionLabels/3840x2160/{seq}.txt"), recursive=True):
        for ln in open(lp, encoding="utf-8", errors="replace"):
            t = ln.split()
            if len(t) < 6:
                continue
            try:
                x1, y1, x2, y2, fr = (int(t[i]) for i in (1, 2, 3, 4, 5))
            except ValueError:
                continue
            s = W / SRC_W
            per[fr].append([x1 * s, y1 * s, x2 * s, y2 * s])
    return per


def matched(gt, pr, thr=0.5):
    hit = np.zeros(len(gt), bool)
    if not len(gt) or not len(pr):
        return hit
    x1 = np.maximum(gt[:, None, 0], pr[None, :, 0]); y1 = np.maximum(gt[:, None, 1], pr[None, :, 1])
    x2 = np.minimum(gt[:, None, 2], pr[None, :, 2]); y2 = np.minimum(gt[:, None, 3], pr[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    ar = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])
    iou = inter / (ar(gt)[:, None] + ar(pr)[None, :] - inter + 1e-9)
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
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--per-seq", type=int, default=120, help="시퀀스당 프레임 수")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(BASE / "metrics/survey_tilt2.csv", encoding="utf-8")))
    ob = [r["group"].split()[1] for r in rows
          if r["group"].startswith("Okutama") and float(r["corr_cy_px"]) >= 0.60]
    nd = [r["group"].split()[1] for r in rows
          if r["group"].startswith("Okutama") and float(r["corr_cy_px"]) <= 0.15]
    print(f"비스듬 시퀀스 {len(ob)}: {ob}")
    print(f"수직   시퀀스 {len(nd)}: {nd}\n", flush=True)

    import cv2
    from ultralytics import YOLO
    model = YOLO(args.weights)
    name = Path(args.weights).parent.parent.name

    out = []
    for tag, seqs in (("비스듬", ob), ("수직", nd)):
        hits, npx = [], []
        for seq in seqs:
            lab = load_labels(seq)
            if not lab:
                continue
            dirs = glob.glob(str(BASE / f"data/raw/okutama/**/Extracted-Frames-1280x720/{seq}"), recursive=True)
            if not dirs:
                continue
            frames = sorted(lab.keys())
            step = max(1, len(frames) // args.per_seq)
            for fr in frames[::step][:args.per_seq]:
                p = Path(dirs[0]) / f"{fr}.jpg"
                if not p.exists():
                    continue
                img = cv2.imread(str(p))
                if img is None:
                    continue
                gt = np.array(lab[fr], dtype=np.float32)
                gt = gt[(gt[:, 2] - gt[:, 0] > 3) & (gt[:, 3] - gt[:, 1] > 3)]
                if not len(gt):
                    continue
                r = model.predict(img, imgsz=1280, conf=args.conf, quantize="fp16", verbose=False)[0]
                pr = r.boxes.xyxy.cpu().numpy()
                h = matched(gt, pr)
                hits += list(h)
                npx += list(np.maximum(gt[:, 2] - gt[:, 0], gt[:, 3] - gt[:, 1]))
        a, px = np.array(hits), np.array(npx)
        print(f"{tag}: 정답 {len(a):,} · 재현율@{args.conf:g} {a.mean():.4f} · 사람 px 중앙 {np.median(px):.1f}", flush=True)
        out.append({"model": name, "tilt": tag, "n": len(a),
                    f"recall@{args.conf:g}": round(float(a.mean()), 4),
                    "px_median": round(float(np.median(px)), 1)})
    if len(out) == 2:
        d = out[0][f"recall@{args.conf:g}"] - out[1][f"recall@{args.conf:g}"]
        print(f"\n차이(비스듬−수직): {d*100:+.1f} %p")
        print("→ |차이| ≥ 5 %p 면 태그가 의미 있다 · 미만이면 잡음")
    f = BASE / "metrics" / f"tilt_groups_{name}.csv"
    with open(f, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    print(f"→ {f}")


if __name__ == "__main__":
    main()
