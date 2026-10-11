"""
diag_scale_infer.py — 진단: 그림을 키우거나 줄여 추론하면 어느 크기 · 장소가 나아지나 (10-11 · v7 넘기 검토 · 운용용 아님)

  가설  학습 크롭의 사람이 운용보다 크다 (WiSARD 94 px 규칙 · 54° 렌즈 가정) → 작은 사람을 못 잡는다
        → 그림을 s 배로 키워 (작은 사람을 크게) 넣었을 때 작은 사람 · Karen 이 크게 오르면 「크기 분포 문제」
  방법  s ∈ {0.75, 1.0, 1.5} 로 그림 크기를 바꾸고 1280×720 타일 (겹침 50 %) 로 덮어 추론 · NMS 0.6 · 박스를 원래 좌표로
        채점 = analyze_dets 와 같음 (AP50 · 장소별 · 크기별 재현율@0.15)
실행: /home/se/miniconda3/envs/drone/bin/python diag_scale_infer.py [--sets test_v2,test_obl]
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from analyze_dets import ap_at, match, nms
from eval_test_v2 import load_gt

BASE = Path(__file__).resolve().parent
SETS = {"test_v2": BASE / "data" / "test_v2", "test_obl": BASE / "data" / "test_obl", "test_kr": BASE / "data" / "test_kr"}
TW, TH = 1280, 720


def tiles(w, h):
    xs = sorted({min(x, max(w - TW, 0)) for x in range(0, max(w - TW, 0) + 1, TW // 2)} | {max(w - TW, 0)})
    ys = sorted({min(y, max(h - TH, 0)) for y in range(0, max(h - TH, 0) + 1, TH // 2)} | {max(h - TH, 0)})
    return [(x, y) for y in ys for x in xs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="test_v2,test_obl")
    ap.add_argument("--scales", default="0.75,1.0,1.5")
    a = ap.parse_args()
    m = YOLO(str(BASE / "runs_person" / "soup_v7r2" / "weights" / "best.pt"))
    res = {}
    for tag in a.sets.split(","):
        imgs = sorted((SETS[tag] / "images").glob("*.jpg")); R = {}
        for s in map(float, a.scales.split(",")):
            dets, gts, names = {}, [], []
            for i, p in enumerate(imgs):
                img = cv2.imread(str(p)); h, w = img.shape[:2]; gts.append(load_gt(p, w, h)); names.append(p.stem)
                im = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_LINEAR if s > 1 else cv2.INTER_AREA) if s != 1 else img
                H, W = im.shape[:2]
                if W < TW or H < TH:
                    im = cv2.copyMakeBorder(im, 0, max(TH - H, 0), 0, max(TW - W, 0), cv2.BORDER_CONSTANT, value=(114, 114, 114)); H, W = im.shape[:2]
                offs = tiles(W, H); bl, sl = [], []
                for k in range(0, len(offs), 8):
                    part = offs[k:k + 8]
                    for (x, y), r in zip(part, m.predict([im[y:y + TH, x:x + TW] for x, y in part], imgsz=1280, conf=0.01, iou=0.7, quantize="fp16", verbose=False)):
                        if len(r.boxes):
                            bl.append(r.boxes.xyxy.cpu().numpy() + np.array([x, y, x, y], np.float32)); sl.append(r.boxes.conf.cpu().numpy())
                b = np.concatenate(bl) / s if bl else np.zeros((0, 4), np.float32); c = np.concatenate(sl) if sl else np.zeros(0, np.float32)
                k = nms(b, c, 0.6); dets[i] = (b[k], c[k])
            r = {"AP50": ap_at(dets, gts, 0.5), "타일 수 (한 장)": len(offs)}
            groups = sorted({n.rsplit("_", 1)[0] for n in names})
            if tag == "test_v2":
                r["장소별 AP50"] = {g.split("_")[1] + g[-4:]: ap_at(dets, gts, 0.5, sel=lambda i, g=g: names[i].rsplit("_", 1)[0] == g) for g in groups}
            bins = [(0, 24), (24, 36), (36, 50), (50, 80), (80, 1e9)]; hit = defaultdict(lambda: [0, 0])
            for i, gt in enumerate(gts):
                bb, cc = dets[i]; kk = cc >= 0.15; rows, used = match(gt, bb[kk], cc[kk], 0.5)
                L = np.maximum(gt[:, 2] - gt[:, 0], gt[:, 3] - gt[:, 1]) if len(gt) else np.zeros(0)
                for lo, hi in bins:
                    sel = (L >= lo) & (L < hi); hit[(lo, hi)][0] += int(used[sel].sum()); hit[(lo, hi)][1] += int(sel.sum())
            r["크기별 재현율@0.15"] = {f"{lo}-{hi if hi < 1e9 else '∞'}": round(v[0] / max(v[1], 1), 3) for (lo, hi), v in hit.items()}
            R[f"s={s}"] = r; print(tag, s, json.dumps(r, ensure_ascii=False), flush=True)
        res[tag] = R
    (BASE / "metrics" / "diag_scale_infer.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
