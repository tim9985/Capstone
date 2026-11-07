"""ens_probe.py — 학습 없이 합치기 시험 (11-07) · runs_fair 검출 재사용 · v12 + 개방형 (GDINO · YOLO-World) 박스 합침 → NMS 0.6 · AP50 · 도로 헛탐지"""
import numpy as np, cv2
from fair_eval_report import RUN, load, ap_of, recall_at, fp_neg
import fair_eval_report as fr
S = ["test_sar", "test_obl", "test_cy", "test_v2", "test_kr", "test_visdrone", "test_nomad", "test_ladd_h", "test_auair"]
OURS = "v12_f91_swa@tile1280"

def fuse(s, tags, w):
    ds = [np.load(RUN / s / f"{t}.npz") for t in tags]; out = {"img": [], "box": [], "conf": []}
    n = len(np.load(RUN / s / "gt.npz")["names"])
    for i in range(n):
        B = []; C = []
        for d, wt in zip(ds, w):
            m = d["img"] == i; B.append(d["box"][m]); C.append(d["conf"][m] * wt)
        b, c = np.concatenate(B), np.concatenate(C)
        if len(b):
            k = cv2.dnn.NMSBoxes([[float(a), float(q), float(e - a), float(f - q)] for a, q, e, f in b], c.tolist(), 0.01, 0.6)
            k = np.array(k).ravel().astype(int) if len(k) else np.array([], int); b, c = b[k], c[k]
        out["img"].append(np.full(len(c), i, np.int32)); out["box"].append(b.reshape(-1, 4)); out["conf"].append(c)
    tag = "ens_" + "+".join(t.split("@")[0] for t in tags) + f"_w{w[1]}"
    np.savez_compressed(RUN / s / f"{tag}.npz", img=np.concatenate(out["img"]), box=np.concatenate(out["box"]).astype(np.float32), conf=np.concatenate(out["conf"]).astype(np.float32))
    return tag

for other in ("sota_gdino@tile", "sota_yoloworld@tile1280"):
    for w in (0.5, 1.0):
        row = []
        for s in S:
            t = fuse(s, [OURS, other], (1.0, w)); per, names = load(s, t)
            if s == "test_auair":
                row.append(f"도로FP@0.20={fp_neg(per, 0.20):.2f}")
            else:
                row.append(f"{s[5:9]} {ap_of(per, range(len(names))):.3f}")
        print(other, "w", w, " · ".join(row), flush=True)
