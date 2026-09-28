"""
f2_tune.py — NFR-V05: 최종 모델의 확신도 임계값을 F2 로 고른다 (2026-09-28)

  F2 = 5·P·R / (4·P + R) — 재현율을 정밀도보다 4배 무겁게 (놓치는 게 더 나쁘다)
  고르는 곳은 **val** (configs/lists/val_v6b.txt · 1280×720 한 장 추론) — 평가셋(test_*)으로 고르지 않는다
  평가셋의 재현율·정밀도는 eval_test_v2.py 결과(metrics/test_*_<모델>.csv) 를 쓴다
실행: python f2_tune.py --weights runs_person/soup_v7r2/weights/best.pt
"""
import argparse, json
from pathlib import Path
import numpy as np
from ultralytics import YOLO
import diag_misses as D

BASE = Path(__file__).resolve().parent
THRS = [0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.25, 0.3, 0.4]


def f2_table(S, T, n):
    rows = []
    for t in THRS:
        k = S >= t; tp = int(T[k].sum()); fp = int(k.sum() - tp)
        P = tp / max(tp + fp, 1); R = tp / max(n, 1)
        rows.append({"conf": t, "P": round(P, 3), "R": round(R, 3), "F2": round(5 * P * R / max(4 * P + R, 1e-9), 3)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    args = ap.parse_args()
    name = Path(args.weights).parent.parent.name
    model = YOLO(args.weights)
    paths = [l.strip() for l in open(BASE / "configs" / "lists" / "val_v6b.txt") if l.strip()]
    S, T, n = [], [], 0
    for i in range(0, len(paths), 16):
        batch = paths[i:i + 16]
        for p, r in zip(batch, model.predict(batch, imgsz=1280, conf=0.01, iou=0.7, half=True, verbose=False)):
            h, w = r.orig_shape
            lbl = Path(p.replace("/images/", "/labels/")).with_suffix(".txt")
            G = []
            if lbl.exists():
                for ln in lbl.read_text().split("\n"):
                    t = ln.split()
                    if len(t) >= 5:
                        cx, cy, bw, bh = map(float, t[1:5])
                        G.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
            G = np.array(G, np.float32).reshape(-1, 4)
            tp, cs, _ = D.greedy(G, r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy())
            S += list(cs); T += list(tp); n += len(G)
    val = f2_table(np.array(S), np.array(T, bool), n)
    best = max(val, key=lambda r: r["F2"])
    out = {"모델": name, "val": "val_v6b (1,172장)", "val_F2표": val, "F2_최대": best}
    print(json.dumps({k: out[k] for k in ("val_F2표", "F2_최대")}, ensure_ascii=False))
    (BASE / "metrics" / f"nfr_v05_{name}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
