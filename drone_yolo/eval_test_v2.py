"""
eval_test_v2.py — NFR-V03 측정 (2026-09-23)

왜 전용 도구인가
  NFR-V03 은 "지정한 평가셋(학습과 **장소 분리**)에서 사람 **AP50 ≥0.80**" 이다.
  · eval_fullframe.py 는 s38/l128/neg 하위 폴더 구조를 전제해 test_v2(평면)에 못 쓴다
  · 기존 도구들은 대부분 **재현율@0.15** 만 낸다. NFR-V03 은 **AP50**(임계값 무관)이다

무엇을
  배포 파이프라인 그대로 — 1920×1080 전체 프레임을 1280×720 타일 4장(겹침 50 %)으로 나눠
  추론하고 NMS(IoU 0.6)로 합친 뒤, conf 전 구간 PR 곡선으로 AP50 을 낸다.

실행: python eval_test_v2.py --weights runs_person/v3_place/weights/best.pt
"""
import argparse, csv, json
from pathlib import Path
import numpy as np

BASE = Path(__file__).resolve().parent
W, H, TW, TH = 1920, 1080, 1280, 720
TILES = [(x, y) for y in (0, H - TH) for x in (0, W - TW)]


def load_gt(p):
    t = Path(str(p).replace("/images/", "/labels/")).with_suffix(".txt")
    if not t.exists():
        return np.zeros((0, 4), np.float32)
    rows = [l.split() for l in t.read_text().splitlines() if l.strip()]
    if not rows:
        return np.zeros((0, 4), np.float32)
    a = np.array([[float(v) for v in r[1:5]] for r in rows], np.float32)
    return np.c_[(a[:, 0] - a[:, 2] / 2) * W, (a[:, 1] - a[:, 3] / 2) * H,
                 (a[:, 0] + a[:, 2] / 2) * W, (a[:, 1] + a[:, 3] / 2) * H]


def iou_mat(g, p):
    if not len(g) or not len(p):
        return np.zeros((len(g), len(p)), np.float32)
    x1 = np.maximum(g[:, None, 0], p[None, :, 0]); y1 = np.maximum(g[:, None, 1], p[None, :, 1])
    x2 = np.minimum(g[:, None, 2], p[None, :, 2]); y2 = np.minimum(g[:, None, 3], p[None, :, 3])
    it = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    ar = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])
    return it / (ar(g)[:, None] + ar(p)[None, :] - it + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="runs_person/v3_place/weights/best.pt")
    ap.add_argument("--data", default=str(BASE / "data" / "test_v2"))
    ap.add_argument("--conf-min", type=float, default=0.01, help="PR 곡선용 하한")
    ap.add_argument("--op-conf", type=float, default=0.15, help="운용 임계값")
    ap.add_argument("--batch", type=int, default=4)
    args = ap.parse_args()

    import cv2
    from ultralytics import YOLO
    model = YOLO(args.weights)
    name = Path(args.weights).parent.parent.name

    imgs = sorted((Path(args.data) / "images").glob("*.jpg"))
    print(f"{name} · {len(imgs):,}장 · 타일 4×(1280×720) · conf ≥{args.conf_min}", flush=True)

    scores, tps, n_gt, fp_neg, n_neg = [], [], 0, 0, 0
    for i, p in enumerate(imgs):
        if i % 200 == 0:
            print(f"  {i:,}/{len(imgs):,}", flush=True)
        img = cv2.imread(str(p))
        if img is None:
            continue
        gt = load_gt(p); n_gt += len(gt)
        tiles = [img[y:y + TH, x:x + TW] for x, y in TILES]
        res = model.predict(tiles, imgsz=1280, conf=args.conf_min, batch=len(tiles),
                            quantize="fp16", verbose=False)
        box, cf = [], []
        for (ox, oy), r in zip(TILES, res):
            b = r.boxes
            if not len(b):
                continue
            box.append(b.xyxy.cpu().numpy() + np.array([ox, oy, ox, oy], np.float32))
            cf.append(b.conf.cpu().numpy())
        if box:
            box = np.concatenate(box); cf = np.concatenate(cf)
            keep = cv2.dnn.NMSBoxes(
                [[float(x1), float(y1), float(x2 - x1), float(y2 - y1)] for x1, y1, x2, y2 in box],
                cf.tolist(), args.conf_min, 0.6)
            keep = np.array(keep).ravel().astype(int) if len(keep) else np.array([], int)
            box, cf = box[keep], cf[keep]
        else:
            box, cf = np.zeros((0, 4), np.float32), np.zeros(0, np.float32)

        if not len(gt):                      # 음성 프레임
            n_neg += 1
            fp_neg += int((cf >= args.op_conf).sum())
        order = np.argsort(-cf)
        box, cf = box[order], cf[order]
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

    s = np.array(scores); t = np.array(tps, bool)
    o = np.argsort(-s); t = t[o]; s = s[o]
    tp = np.cumsum(t); fp = np.cumsum(~t)
    rec = tp / max(n_gt, 1); prec = tp / np.maximum(tp + fp, 1)
    # 101점 보간 AP
    ap50 = float(np.mean([prec[rec >= r].max() if (rec >= r).any() else 0
                          for r in np.linspace(0, 1, 101)]))
    m = s >= args.op_conf
    r_op = float(tp[m][-1] / max(n_gt, 1)) if m.any() else 0.0
    p_op = float(prec[m][-1]) if m.any() else 0.0

    print(f"\n=== NFR-V03 — 장소 분리 평가셋 (Carnation · Karen)")
    print(f"  정답 {n_gt:,} · 예측 {len(s):,} · 음성 프레임 {n_neg}")
    print(f"  **AP50            {ap50:.4f}**   (목표 ≥0.80 → {'✅ 통과' if ap50>=0.80 else '❌ 미달'})")
    print(f"  재현율@{args.op_conf:g}      {r_op:.4f}")
    print(f"  정밀도@{args.op_conf:g}      {p_op:.4f}")
    print(f"  음성 프레임 오탐   {fp_neg/max(n_neg,1):.3f} 건/프레임")

    out = BASE / "metrics" / f"test_v2_{name}.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model", "n_gt", "AP50", f"recall@{args.op_conf:g}",
                                          f"precision@{args.op_conf:g}", "fp_per_neg_frame"])
        w.writeheader()
        w.writerow({"model": name, "n_gt": n_gt, "AP50": round(ap50, 4),
                    f"recall@{args.op_conf:g}": round(r_op, 4),
                    f"precision@{args.op_conf:g}": round(p_op, 4),
                    "fp_per_neg_frame": round(fp_neg / max(n_neg, 1), 3)})
    print(f"→ {out}")


if __name__ == "__main__":
    main()
