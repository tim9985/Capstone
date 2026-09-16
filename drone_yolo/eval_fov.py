"""
eval_fov.py — 화각 · 고도별 시험셋(make_fov_testsets.py v4)으로 모델을 재서 화각 × 고도 표를 만든다

지표 (칸마다)
  · 재현율@0.15 (주 지표) — NFR-V05 기본 임계값. 예측(conf ≥0.15)과 정답(복제 칸 포함 전부)을 IoU ≥0.5 로 1:1 매칭한 뒤,
    채점 박스(meta orig & !seam & near_ref = 원본 칸 · 이음새에 안 붙음 · 그 칸의 목표 크기)만으로 이미지마다 재현율.
    평균은 **세트(자세 × 10 m / 20~40 m) 안 모든 칸에서 채점된 원본의 교집합**으로 낸다 → 칸마다 같은 원본으로 비교.
    거울 채움 · 이음새 제외는 작은 칸에서만 생겨, 칸별 평균만 쓰면 작은 칸일수록 표본이 달라진다 (로컬 세션 검토).
  · 보조: 칸에서 채점된 원본 전체 평균 · NOMAD · WiSARD · NOMAD 가시도 ≥30 (모두 교집합 기준) · AP50(복제 칸 포함 박스 단위)
  ⚠ 시험 원본은 best 선택용 val 과 같은 배우 · 비행이다 → 절대값은 낙관적, 모델 간 · 칸 간 상대 비교용
  ⚠ 10 m 행은 원본 세트가 달라 20 m 이상 행과 직접 비교하지 않는다

실행:
  python eval_fov.py --weights runs_person/fov_11s_1280_all/weights/best.pt runs_person/m1_11m_1280/weights/best.pt
출력: metrics/eval_fov_<실행>.csv (화각·고도·자세 한 줄씩) + 화면에 화각 × 고도 표
"""
import argparse
import csv
import json
import os
import tempfile
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")
BASE_DIR = Path(__file__).resolve().parent
MANIFEST_VERSION = 4
TRAIN_PX = (16, 160)        # 학습 크기 범위 (scale 0.3 증강 전) — 밖이면 외삽 표시


def load_gt(img):
    t = Path(str(img).replace("/images/", "/labels/")).with_suffix(".txt")
    a = np.array([list(map(float, l.split()[1:])) for l in t.read_text().splitlines() if l.strip()]).reshape(-1, 4)
    return np.c_[(a[:, 0] - a[:, 2] / 2) * 1280, (a[:, 1] - a[:, 3] / 2) * 720,
                 (a[:, 0] + a[:, 2] / 2) * 1280, (a[:, 1] + a[:, 3] / 2) * 720]


def matched(gt, pr, thr=0.5):
    """IoU 높은 쌍부터 1:1 매칭 → 정답마다 맞았는지 (bool 배열)"""
    hit = np.zeros(len(gt), dtype=bool)
    if not len(gt) or not len(pr):
        return hit
    x1 = np.maximum(gt[:, None, 0], pr[None, :, 0]); y1 = np.maximum(gt[:, None, 1], pr[None, :, 1])
    x2 = np.minimum(gt[:, None, 2], pr[None, :, 2]); y2 = np.minimum(gt[:, None, 3], pr[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])
    iou = inter / (area(gt)[:, None] + area(pr)[None, :] - inter + 1e-9)
    used_p = set()
    for gi, pj in zip(*np.unravel_index(np.argsort(-iou, axis=None), iou.shape)):
        if iou[gi, pj] < thr:
            break
        if hit[gi] or pj in used_p:
            continue
        hit[gi] = True; used_p.add(pj)
    return hit


def mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", nargs="+", required=True)
    ap.add_argument("--testsets", default=str(BASE_DIR / "data" / "det_fov_test"))
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    from ultralytics import YOLO

    root = Path(args.testsets).resolve()
    man = json.load(open(root / "manifest.json"))
    if man.get("version") != MANIFEST_VERSION:
        raise SystemExit(f"manifest version {MANIFEST_VERSION} 시험셋이 필요하다 (지금 {man.get('version')}) — make_fov_testsets.py 로 다시 만든다")
    cells = sorted({(t["set"], t["px"], t["dir"]) for t in man["table"]})

    for w in args.weights:
        wp = Path(w)
        name = wp.parent.parent.name if wp.parent.name == "weights" else wp.stem
        model = YOLO(str(wp))
        per_cell = {}     # dir → {"ap50", "stems": {stem: (kind, 재현율, 가시도≥30 재현율 또는 None)}}
        with tempfile.TemporaryDirectory() as tmp:
            for set_name, px, dname in cells:
                d = root / dname
                imgs = sorted((d / "images").glob("*.jpg"))
                yml = Path(tmp) / f"{dname}.yaml"      # 폴더 이름이 바뀌어도 되게 절대경로 yaml 을 그때 만든다
                yml.write_text(f"path: {d}\ntrain: images\nval: images\nnc: 1\nnames: ['person']\n")
                r = model.val(data=str(yml), imgsz=args.imgsz, batch=args.batch, plots=False,
                              verbose=False, project=tmp, name=dname, exist_ok=True)
                stems = {}
                for i in range(0, len(imgs), 64):
                    chunk = imgs[i:i + 64]
                    for p, res in zip(chunk, model.predict([str(x) for x in chunk], imgsz=args.imgsz, conf=args.conf,
                                                           batch=args.batch, verbose=False)):
                        meta = json.loads((d / "meta" / f"{p.stem}.json").read_text())
                        keep = np.array([m["orig"] and not m["seam"] and m["near_ref"] for m in meta], dtype=bool)
                        if not keep.any():
                            continue
                        hit = matched(load_gt(p), res.boxes.xyxy.cpu().numpy())
                        kind = p.name.split("_")[0]
                        v30 = keep & np.array([m["vis"] >= 30 for m in meta], dtype=bool)
                        stems[p.stem] = (kind, float(hit[keep].mean()),
                                         float(hit[v30].mean()) if kind == "nomad" and v30.any() else None)
                per_cell[dname] = {"ap50": r.box.map50, "stems": stems}
                print(f"  {name} {dname}: 채점 원본 {len(stems)} · 재현율@{args.conf:g}(칸 전체) "
                      f"{mean([v[1] for v in stems.values()]):.3f} · AP50 {r.box.map50:.3f}", flush=True)

        common = {}
        for set_name in {c[0] for c in cells}:
            ks = [set(per_cell[c[2]]["stems"]) for c in cells if c[0] == set_name]
            common[set_name] = set.intersection(*ks) if ks else set()

        rows = {}
        for t in man["table"]:
            pc, cm = per_cell[t["dir"]], common[t["set"]]
            vals = [pc["stems"][s] for s in cm]
            rows[t["dir"]] = {
                "common": len(cm), "recall": mean([v[1] for v in vals]),
                "nomad": mean([v[1] for v in vals if v[0] == "nomad"]), "wisard": mean([v[1] for v in vals if v[0] == "wisard"]),
                "nomad_vis30": mean([v[2] for v in vals if v[0] == "nomad"]),
                "scored": len(pc["stems"]), "recall_scored": mean([v[1] for v in pc["stems"].values()]), "ap50": pc["ap50"]}

        out = BASE_DIR / "metrics" / f"eval_fov_{name}.csv"
        with open(out, "w", newline="", encoding="utf-8") as f:
            wr = csv.writer(f)
            wr.writerow(["model", "fov_deg", "altitude_m", "pose", "person_px", "source_set", "extrapolated",
                         "common_images", f"recall@{args.conf:g}", "recall_nomad", "recall_nomad_vis30", "recall_wisard",
                         "scored_images", "recall_all_scored", "AP50_with_copies"])
            for t in man["table"]:
                m = rows[t["dir"]]
                wr.writerow([name, t["fov"], t["alt"], t["pose"], t["px"], t["set"], int(not TRAIN_PX[0] <= t["px"] <= TRAIN_PX[1]),
                             m["common"], f"{m['recall']:.4f}", f"{m['nomad']:.4f}", f"{m['nomad_vis30']:.4f}", f"{m['wisard']:.4f}",
                             m["scored"], f"{m['recall_scored']:.4f}", f"{m['ap50']:.4f}"])

        alts = man["alts"]
        print(f"\n=== {name} · 재현율@{args.conf:g} (세트 공통 원본 평균 · 화면 중앙 · 하향 · 입력 1920 기준 크기) ===")
        print("    * = 학습 크기 범위(16~160 px) 밖 외삽 · 10 m 열은 원본 세트가 달라 20 m 이상과 직접 비교 주의")
        for pose, label in (("standing", "서 있는 사람 크기 기준 0.5 m (자세 라벨 아님)"), ("lying", "누운 사람 크기 기준 1.7 m")):
            print(f"\n[{label}]  " + "  ".join(f"{a} m".rjust(14) for a in alts))
            for fov in man["fovs"]:
                cells_txt = []
                for a in alts:
                    t = next(x for x in man["table"] if x["fov"] == fov and x["alt"] == a and x["pose"] == pose)
                    star = "*" if not TRAIN_PX[0] <= t["px"] <= TRAIN_PX[1] else " "
                    cells_txt.append(f"{t['px']:>3}px{star}{rows[t['dir']]['recall']:.3f}".rjust(14))
                print(f"  {fov:>2}°  " + "  ".join(cells_txt))
        print("\n공통 원본 수: " + " · ".join(f"{k} {len(v)}" for k, v in sorted(common.items())))
        print(f"→ {out}\n")


if __name__ == "__main__":
    main()
