"""
eval_visibility.py — NOMAD 가시도 구간별 재현율 · AP50 (2026-09-16)

왜
  NOMAD 검증 정답의 22 % 가 가시도 30 미만(거의 안 보임)이다. 이 구간이 전체 점수를 눌러
  "모델이 나쁜 것"과 "애초에 안 보이는 것"이 섞인다. VisDrone 처럼 **무시 영역**으로 다루는 게 맞는지 본다.

무엇을 재나 (화각 시험셋 v4 의 운용 기준 칸)
  · 가시도 0~30 / 30~70 / 70~100 구간별 재현율@0.15
  · AP50 두 가지: ① 전부 정답으로 ② 가시도 <30 을 무시 (그 정답도 빼고, 거기 맞은 예측도 오탐에서 뺀다)

실행: python eval_visibility.py --weights runs_person/fov_11m_1280_all/weights/best.pt
"""
import argparse, csv, json, os
from pathlib import Path
import numpy as np
os.environ.setdefault("MPLBACKEND", "Agg")
BASE_DIR = Path(__file__).resolve().parent
W, H = 1280, 720


def boxes_of(p):
    t = Path(str(p).replace("/images/", "/labels/")).with_suffix(".txt")
    a = np.array([list(map(float, l.split()[1:])) for l in t.read_text().splitlines() if l.strip()]).reshape(-1, 4)
    return np.c_[(a[:, 0] - a[:, 2] / 2) * W, (a[:, 1] - a[:, 3] / 2) * H, (a[:, 0] + a[:, 2] / 2) * W, (a[:, 1] + a[:, 3] / 2) * H]


def iou_mat(a, b):
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    ar = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])
    return inter / (ar(a)[:, None] + ar(b)[None, :] - inter + 1e-9)


def ap50(records, n_gt):
    """records: [(conf, is_tp, is_ignored)] — ignored 예측은 빼고 계산"""
    r = sorted([x for x in records if not x[2]], key=lambda x: -x[0])
    if not r or n_gt == 0:
        return float("nan")
    tp = np.cumsum([x[1] for x in r]); fp = np.cumsum([not x[1] for x in r])
    rec, prec = tp / n_gt, tp / np.maximum(tp + fp, 1e-9)
    mrec, mpre = np.r_[0, rec, rec[-1]], np.r_[1, prec, 0]
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", nargs="+", default=["runs_person/fov_11m_1280_all/weights/best.pt"])
    ap.add_argument("--testsets", default=str(BASE_DIR / "data" / "det_fov_test"))
    ap.add_argument("--cells", nargs="+", default=["s20_38", "s20_24", "l20_128"])
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.15)
    args = ap.parse_args()
    from ultralytics import YOLO
    root, rows = Path(args.testsets), []
    for w in args.weights:
        wp = Path(w); name = wp.parent.parent.name if wp.parent.name == "weights" else wp.stem
        model = YOLO(str(wp))
        for cell in args.cells:
            d = root / cell
            imgs = [p for p in sorted((d / "images").glob("*.jpg")) if p.name.startswith("nomad_")]
            bins = {"0~30": [0, 0], "30~70": [0, 0], "70~100": [0, 0]}
            recs_all, recs_ign, n_gt_all, n_gt_ign = [], [], 0, 0
            for i in range(0, len(imgs), 64):
                chunk = imgs[i:i + 64]
                for p, res in zip(chunk, model.predict([str(x) for x in chunk], imgsz=args.imgsz, conf=0.001, verbose=False, batch=8)):
                    meta = json.loads((d / "meta" / f"{p.stem}.json").read_text())
                    gt, pr = boxes_of(p), res.boxes.xyxy.cpu().numpy()
                    cf = res.boxes.conf.cpu().numpy()
                    keep = np.array([m["orig"] and not m["seam"] and m["near_ref"] for m in meta], dtype=bool)
                    vis = np.array([m["vis"] for m in meta])
                    low = keep & (vis >= 0) & (vis < 30)
                    M = iou_mat(gt, pr)
                    # 신뢰도 높은 예측부터 IoU 0.5 이상인 정답에 1:1 매칭.
                    # 채점 대상이 아닌 정답(거울 복제 칸 · 이음새 · 기준 크기 밖)에 맞은 예측은
                    # 정답으로도 오탐으로도 세지 않는다 (무시) — 안 그러면 복제가 많은 작은 칸의 AP 가 부풀려진다
                    taken, gt_conf = set(), np.full(len(gt), -1.0)
                    rec_a, rec_i = [], []
                    for j in np.argsort(-cf):
                        gi = -1
                        for g in (np.argsort(-M[:, j]) if len(gt) else []):
                            if M[g, j] < 0.5:
                                break
                            if int(g) not in taken:
                                gi = int(g); break
                        if gi >= 0:
                            taken.add(gi); gt_conf[gi] = cf[j]
                            scored = bool(keep[gi])
                            rec_a.append((float(cf[j]), scored, not scored))
                            rec_i.append((float(cf[j]), scored and not bool(low[gi]), (not scored) or bool(low[gi])))
                        else:
                            rec_a.append((float(cf[j]), False, False)); rec_i.append((float(cf[j]), False, False))
                    recs_all += rec_a; recs_ign += rec_i
                    hit = gt_conf >= args.conf
                    for k, (lo, hi) in (("0~30", (0, 30)), ("30~70", (30, 70)), ("70~100", (70, 101))):
                        sel = keep & (vis >= lo) & (vis < hi)
                        bins[k][0] += int(hit[sel].sum()); bins[k][1] += int(sel.sum())
                    n_gt_all += int(keep.sum()); n_gt_ign += int((keep & ~low).sum())
            a_all, a_ign = ap50(recs_all, n_gt_all), ap50(recs_ign, n_gt_ign)
            print(f"\n[{name} · {cell} · NOMAD {len(imgs)}장]")
            for k, (h, n) in bins.items():
                print(f"  가시도 {k:>7}: 정답 {n:>5} · 재현율@{args.conf:g} {h/n if n else float('nan'):.3f}")
            print(f"  AP50 전부 {a_all:.3f} · 가시도<30 무시 {a_ign:.3f}  (정답 {n_gt_all} → {n_gt_ign})")
            rows.append([name, cell, len(imgs), *[f"{bins[k][0]/bins[k][1]:.4f}" if bins[k][1] else "" for k in bins],
                         *[bins[k][1] for k in bins], f"{a_all:.4f}", f"{a_ign:.4f}"])
    out = BASE_DIR / "metrics" / "eval_visibility.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f); wr.writerow(["model", "cell", "images", "recall_vis0_30", "recall_vis30_70", "recall_vis70_100",
                                         "n_vis0_30", "n_vis30_70", "n_vis70_100", "AP50_all", "AP50_ignore_low_vis"]); wr.writerows(rows)
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
