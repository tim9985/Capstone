"""
eval_external_yolo.py — E1b: 공개 VisDrone 검출기 (HF dronefreak · mAP50 1~3위) 를 우리 평가셋으로 (10-10 · 판정 기준은 _학습 큐 「E1b」)

  모델  data/raw/ext_det/visdrone-{yolov9e, yolo11x, yolo26x}/best.pt (AGPL-3.0 · VisDrone2019 · imgsz 640 학습)
  사람  pedestrian + people → 사람 하나 (클래스 무시 NMS 0.6)
  조건  tile = 우리 운용 방식 (1920×1080 → 1280×720 타일 4장 · imgsz 1280 · FP16 · NMS 0.6 · test_obl 은 한 장) · full = 한 장 통째 imgsz 1280
  채점  eval_external_det.run 그대로 (E1 과 같은 정답 · IoU 0.5 · 101점 AP50 · 블록) → runs_person/ext_<모델>_<조건>/eval_<tag>.npz
실행: /home/se/miniconda3/envs/drone/bin/python eval_external_yolo.py [--models visdrone-yolov9e,…] [--modes tile,full]
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from eval_external_det import SETS, run

BASE = Path(__file__).resolve().parent
EXT = BASE.parent / "data" / "raw" / "ext_det"


class YoloPerson:
    def __init__(self, path):
        self.m = YOLO(str(path)); names = {v.lower(): k for k, v in self.m.names.items()}
        self.cls = [names[c] for c in ("pedestrian", "people") if c in names]
        assert self.cls, f"사람 클래스 없음: {self.m.names}"

    def __call__(self, imgs, rgb=False):
        res = self.m.predict(imgs, imgsz=1280, conf=0.01, classes=self.cls, half=True, verbose=False, batch=len(imgs))
        out = []
        for r in res:
            b = r.boxes.xyxy.cpu().numpy(); c = r.boxes.conf.cpu().numpy()
            if len(b):
                keep = cv2.dnn.NMSBoxes([[float(x1), float(y1), float(x2 - x1), float(y2 - y1)] for x1, y1, x2, y2 in b], c.tolist(), 0.01, 0.6)
                keep = np.array(keep).ravel().astype(int) if len(keep) else np.array([], int); b, c = b[keep], c[keep]
            out.append((b, c))
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="visdrone-yolov9e,visdrone-yolo11x,visdrone-yolo26x")
    ap.add_argument("--modes", default="tile,full")
    a = ap.parse_args()
    rows = []
    for mname in a.models.split(","):
        det = YoloPerson(EXT / mname / "best.pt"); print(mname, "사람 클래스", det.cls, {i: det.m.names[i] for i in det.cls}, flush=True)
        for tag in SETS:
            for mode in a.modes.split(","):
                r = run(det, SETS[tag], "tile" if mode == "tile" else "official", tag, prefix=f"ext_{mname.replace('-', '_')}")
                r["model"] = mname; r["mode"] = mode; rows.append(r); print(r, flush=True)
    with open(BASE / "metrics" / "eval_external_yolo.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["model", "set", "mode", "n_gt", "AP50", "ci_lo", "ci_hi", "recall@0.3", "recall@0.5", "fp_neg@0.3", "fp_neg@0.5", "ms"])
        for r in rows:
            w.writerow([r["model"], r["set"], r["mode"], r["n_gt"], r["AP50"], *r["ci"], r["recall/precision@conf"][0.3][0], r["recall/precision@conf"][0.5][0],
                        r["fp_per_neg_frame@conf"][0.3], r["fp_per_neg_frame@conf"][0.5], r["ms_median"]])


if __name__ == "__main__":
    main()
