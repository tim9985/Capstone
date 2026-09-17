"""
eval_budget.py — 예산안 1안 vs 2안 렌즈 화각 비교 채점 (2026-09-17)

무엇을 재나 (make_budget_testset.py 가 만든 칸마다)
  · 재현율@0.15 — NFR-V05 기본 임계값. 예측(conf ≥0.15)과 정답(거울 복제 칸 포함 전부)을 IoU ≥0.5 로 1:1 매칭한 뒤
    **원본 칸 · 이음새에 안 붙은** 박스만으로 이미지마다 재현율을 낸다.
  · **실제 자세 라벨**(standing · sitting · lying)별로 따로 낸다 — AI-Hub `person_pose` 다. 크기 가정이 아니다.
  · 프레임당 오탐 — 정답과 매칭되지 않은 예측 수 (거울 복제와 매칭된 것은 오탐으로 세지 않는다)
  · 평균은 **모든 칸에서 채점된 원본의 교집합**으로 낸다 → 칸마다 같은 프레임으로 비교

  ⚠ 모델은 NOMAD · WiSARD · SARD 로 배웠고 이 시험셋은 한국 산악 겨울이다 → **절대값은 도메인 차이로 낮게** 나온다.
    쓰는 방법은 **칸 사이 상대 비교**다. (덤으로 이 숫자가 AI-Hub 학습 전 도메인 차이의 기준선이 된다)

실행: python eval_budget.py --weights runs_person/fov_11m_1280_all/weights/best.pt
출력: metrics/eval_budget_<실행>.csv + 화면에 카메라 × 고도 표
"""
import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")
BASE_DIR = Path(__file__).resolve().parent
POSES = ("standing", "sitting", "lying")


def load_gt(img):
    t = Path(str(img).replace("/images/", "/labels/")).with_suffix(".txt")
    a = np.array([list(map(float, l.split()[1:])) for l in t.read_text().splitlines() if l.strip()]).reshape(-1, 4)
    return np.c_[(a[:, 0] - a[:, 2] / 2) * 1280, (a[:, 1] - a[:, 3] / 2) * 720,
                 (a[:, 0] + a[:, 2] / 2) * 1280, (a[:, 1] + a[:, 3] / 2) * 720]


def match(gt, pr, thr=0.5):
    """IoU 높은 쌍부터 1:1 → (정답마다 맞았는지, 매칭된 예측 수)"""
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


def mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", nargs="+", required=True)
    ap.add_argument("--testsets", default=str(BASE_DIR / "data" / "det_budget"))
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    from ultralytics import YOLO

    root = Path(args.testsets).resolve()
    man = json.load(open(root / "manifest.json"))
    cams = [c[0] for c in man["cameras"]]
    alts = man["alts"]

    for w in args.weights:
        wp = Path(w)
        name = wp.parent.parent.name if wp.parent.name == "weights" else wp.stem
        model = YOLO(str(wp))
        per_cell = {}
        for t in man["table"]:
            d = root / t["dir"]
            imgs = sorted((d / "images").glob("*.jpg"))
            stems = {}
            for i in range(0, len(imgs), 64):
                chunk = imgs[i:i + 64]
                for p, res in zip(chunk, model.predict([str(x) for x in chunk], imgsz=args.imgsz,
                                                       conf=args.conf, batch=args.batch, verbose=False)):
                    meta = json.loads((d / "meta" / f"{p.stem}.json").read_text())
                    keep = np.array([m["orig"] and not m["seam"] for m in meta], dtype=bool)
                    if not keep.any():
                        continue
                    pr = res.boxes.xyxy.cpu().numpy()
                    hit, n_matched = match(load_gt(p), pr)
                    poses = [m["pose"] for m in meta]
                    by_pose = {q: float(hit[[k for k in range(len(meta)) if keep[k] and poses[k] == q]].mean())
                               for q in POSES if any(keep[k] and poses[k] == q for k in range(len(meta)))}
                    stems[p.stem] = {"recall": float(hit[keep].mean()), "pose": by_pose,
                                     "fp": max(0, len(pr) - n_matched)}
            per_cell[t["dir"]] = stems
            print(f"  {name} {t['dir']}: 채점 {len(stems)}장 · 재현율@{args.conf:g} "
                  f"{mean([v['recall'] for v in stems.values()]):.3f} · 오탐 {mean([v['fp'] for v in stems.values()]):.2f}/장", flush=True)

        # 칸마다 원본 프레임이 고도별로 다르다 → 교집합은 **같은 고도 안 카메라끼리**만 낸다
        common = {a: set.intersection(*[set(per_cell[t["dir"]]) for t in man["table"] if t["alt"] == a])
                  for a in alts}
        rows = {}
        for t in man["table"]:
            cm = common[t["alt"]]
            vals = [per_cell[t["dir"]][s] for s in cm]
            rows[t["dir"]] = {"common": len(cm), "recall": mean([v["recall"] for v in vals]),
                              "fp": mean([v["fp"] for v in vals]),
                              **{f"r_{q}": mean([v["pose"].get(q) for v in vals]) for q in POSES}}

        out = BASE_DIR / "metrics" / f"eval_budget_{name}.csv"
        with open(out, "w", newline="", encoding="utf-8") as f:
            wr = csv.writer(f)
            wr.writerow(["model", "camera", "label", "f_px", "altitude_m", "scale", "px_standing_calc",
                         "px_standing_real", "px_lying_real", "orig_frac", "common_images",
                         f"recall@{args.conf:g}", "recall_standing", "recall_sitting", "recall_lying", "fp_per_frame"])
            for t in man["table"]:
                m = rows[t["dir"]]
                wr.writerow([name, t["cam"], t["label"], t["f_px"], t["alt"], t["scale"], t["px_standing_calc"],
                             t["px_by_pose"].get("standing", ""), t["px_by_pose"].get("lying", ""), t["orig_frac"],
                             m["common"], f"{m['recall']:.4f}", f"{m['r_standing']:.4f}", f"{m['r_sitting']:.4f}",
                             f"{m['r_lying']:.4f}", f"{m['fp']:.3f}"])

        px_of = {(t["cam"], t["alt"]): t["px_by_pose"] for t in man["table"]}
        for title, key in (("전체", "recall"), ("서 있는 사람 (실제 자세 라벨)", "r_standing"),
                           ("누운 사람 (실제 자세 라벨)", "r_lying")):
            print(f"\n=== {name} · {title} · 재현율@{args.conf:g} (고도별 공통 원본 "
                  + " · ".join(f"{a}m {len(common[a])}" for a in alts) + ") ===")
            print("        " + "  ".join(f"{a} m".rjust(13) for a in alts))
            for cam in cams:
                cells = []
                for a in alts:
                    t = next((x for x in man["table"] if x["cam"] == cam and x["alt"] == a), None)
                    if t is None:
                        cells.append("—".rjust(13)); continue
                    pose_px = px_of[(cam, a)].get("standing" if key == "r_standing" else
                                                  "lying" if key == "r_lying" else "standing", 0)
                    cells.append(f"{pose_px:>3.0f}px {rows[t['dir']][key]:.3f}".rjust(13))
                print(f"  {cam:>5}  " + "  ".join(cells))
        print(f"\n오탐/장: " + " · ".join(
            f"{cam} {mean([rows[t['dir']]['fp'] for t in man['table'] if t['cam'] == cam]):.2f}" for cam in cams))
        print(f"→ {out}\n")


if __name__ == "__main__":
    main()
