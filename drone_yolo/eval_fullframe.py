"""
eval_fullframe.py — 1920×1080 전체 프레임 평가 (2026-09-16)

무엇을 보나
  1) 크롭(1280×720)으로 학습한 모델이 **전체 프레임에서 오탐이 늘지 않는지** — 프레임당 오탐 (NFR-V06 ≤1)
  2) **추론 입력 크기**(1280 / 1600 / 1920)에 따른 재현율 · 오탐 · 한 장 처리 시간 (NFR-V01 ≤30 ms)
  3) **선명도(라플라시안 분산) 구간별 재현율** — 흔들린 프레임을 어디서 버릴지 기준 잡기

시험셋: make_fullframe_testset.py (s38 서 있는 사람 38 px · l128 누운 사람 128 px · neg 사람 없는 프레임)
채점: 정답(복제 칸 포함 전부)과 예측을 IoU ≥0.5 로 1:1 매칭 → 원본 칸 · 이음새 아님 · 기준 크기(orig & !seam & near_ref) 만 재현율
      오탐 = 어떤 정답과도 안 맞은 예측 (복제 칸 정답까지 고려하므로 거울 채움이 오탐을 부풀리지 않는다)

실행: python eval_fullframe.py --weights runs_person/fov_11m_1280_all/weights/best.pt
"""
import argparse, csv, json, os, time
from pathlib import Path
import numpy as np
os.environ.setdefault("MPLBACKEND", "Agg")
from eval_fov import matched
BASE_DIR = Path(__file__).resolve().parent
W, H = 1920, 1080


def load_gt(p):
    t = Path(str(p).replace("/images/", "/labels/")).with_suffix(".txt")
    a = np.array([list(map(float, l.split()[1:])) for l in t.read_text().splitlines() if l.strip()]).reshape(-1, 4)
    return np.c_[(a[:, 0] - a[:, 2] / 2) * W, (a[:, 1] - a[:, 3] / 2) * H, (a[:, 0] + a[:, 2] / 2) * W, (a[:, 1] + a[:, 3] / 2) * H]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", nargs="+", default=["runs_person/fov_11m_1280_all/weights/best.pt"])
    ap.add_argument("--testsets", default=str(BASE_DIR / "data" / "det_fullframe"))
    ap.add_argument("--imgsz", type=int, nargs="+", default=[1280, 1600, 1920])
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--batch", type=int, default=4)
    args = ap.parse_args()
    from ultralytics import YOLO
    root = Path(args.testsets)
    man = json.load(open(root / "manifest.json"))
    names = list(man["sets"]) + ["neg"]
    rows = []
    for w in args.weights:
        wp = Path(w)
        model_name = wp.parent.parent.name if wp.parent.name == "weights" else wp.stem
        model = YOLO(str(wp))
        per = {}
        for imgsz in args.imgsz:
            for name in names:
                d = root / name
                imgs = sorted((d / "images").glob("*.jpg"))
                recs, fps, sharps, ms = [], [], [], []
                # 1920 입력에서는 리스트 전체가 한 배치로 들어가 OOM 이 난다 → --batch 장씩 끊어서 보낸다
                for i in range(0, len(imgs), args.batch):
                    chunk = imgs[i:i + args.batch]
                    for p, res in zip(chunk, model.predict([str(x) for x in chunk], imgsz=imgsz, conf=args.conf,
                                                           batch=args.batch, verbose=False)):
                        meta = json.loads((d / "meta" / f"{p.stem}.json").read_text())
                        pr = res.boxes.xyxy.cpu().numpy()
                        gt = load_gt(p) if meta["boxes"] else np.zeros((0, 4))
                        hit_g = matched(gt, pr)
                        hit_p = matched(pr, gt) if len(pr) else np.zeros(0, dtype=bool)   # 예측 쪽 매칭 → 나머지가 오탐
                        fps.append(int((~hit_p).sum())); sharps.append(meta["sharpness"])
                        ms.append(sum(res.speed.values()))
                        keep = np.array([m["orig"] and not m["seam"] and m["near_ref"] for m in meta["boxes"]], dtype=bool)
                        if keep.any():
                            recs.append(float(hit_g[keep].mean()))
                per[(imgsz, name)] = {"images": len(imgs), "recall": float(np.mean(recs)) if recs else float("nan"),
                                      "fp": float(np.mean(fps)), "ms": float(np.median(ms)),
                                      "sharp": sharps, "recs": recs}
                m = per[(imgsz, name)]
                print(f"  {model_name} 입력 {imgsz} · {name}: 장수 {m['images']} · 재현율 {m['recall']:.3f} · 오탐/장 {m['fp']:.2f} · {m['ms']:.1f} ms", flush=True)
                rows.append([model_name, name, imgsz, m["images"], f"{m['recall']:.4f}", f"{m['fp']:.3f}", f"{m['ms']:.1f}"])

        print(f"\n=== {model_name} · 전체 프레임 1920×1080 (운용 기준 54° · 25 m) ===")
        print(f"{'입력':>6} | {'서 있는 크기(0.5m) 재현율':>18} | {'누운 크기(1.7m) 재현율':>16} | {'오탐/장 (사람 없음)':>18} | {'오탐/장 (양성)':>14} | {'한 장 ms':>8}")
        for imgsz in args.imgsz:
            s, l, n = per[(imgsz, "s38")], per[(imgsz, "l128")], per[(imgsz, "neg")]
            print(f"{imgsz:>6} | {s['recall']:>18.3f} | {l['recall']:>16.3f} | {n['fp']:>18.2f} | {(s['fp']+l['fp'])/2:>14.2f} | {n['ms']:>8.1f}")

        big = max(args.imgsz)
        print(f"\n=== 선명도(라플라시안 분산) 구간별 재현율 · 입력 {big} ===")
        for name, label in (("s38", "서 있는 크기 38 px"), ("l128", "누운 크기 128 px")):
            m = per[(big, name)]
            sh = np.array(m["sharp"][:len(m["recs"])]); rc = np.array(m["recs"])
            qs = np.percentile(sh, [25, 50, 75])
            print(f"  [{label}] 4분위 경계 {qs.round(1)}")
            for lo, hi, tag in ((-1, qs[0], "하위 25% (가장 흐림)"), (qs[0], qs[1], "25~50%"), (qs[1], qs[2], "50~75%"), (qs[2], 1e18, "상위 25% (가장 선명)")):
                sel = (sh > lo) & (sh <= hi)
                if sel.any():
                    print(f"    {tag:<22} n={int(sel.sum()):>5} 재현율 {rc[sel].mean():.3f}")
            rows.append([model_name, name + "_sharp_q", big, len(rc), "", "", ""])
        out = BASE_DIR / "metrics" / f"eval_fullframe_{model_name}.csv"
        with open(out, "w", newline="", encoding="utf-8") as f:
            wr = csv.writer(f); wr.writerow(["model", "set", "imgsz", "images", f"recall@{args.conf:g}", "fp_per_frame", "ms_per_frame"]); wr.writerows(rows)
        print(f"→ {out}")


if __name__ == "__main__":
    main()
