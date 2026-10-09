"""
fo_dd_fhd.py — 디지털관 사진을 운용 입력 (1920×1080 한 장) 으로 바꿔 worker 와 똑같이 탐지 (10-09) → FiftyOne `digital_department_test` 에 필드 추가

  왜  fo_dd_test 의 op_* 는 긴 변 1920 으로 줄인 4:3 (1920×1440 · 타일 6장) — 운용은 16:9 한 장 · 타일 4장
      full_* 은 원본 그대로 (사람 2.1배) 라 유리 → 운용과 같은 조건으로 다시 본다
  변환 원본 4:3 → 가로 전체 · 아래쪽 16:9 띠 (하늘을 버림 · 높이 = 가로 × 9/16) → 1920×1080 으로 줄임
       = worker PersonDetector 의 1920×1080 경로 그대로 (타일 4장 1280×720 · conf 0.15 · NMS 0.6 · FP16 · v7r2 는 TensorRT)
  ⚠ 휴대폰 1배 화각 (iPhone 13 Pro ~69° · S24 ~74°) 이 운용 카메라 (80.2°) 보다 좁다 → 같은 거리면 사람이 1.1~1.2배 크게 찍힌다 (아직 약간 유리)
  필드  fhd_v7r2 · fhd_v9x2 (원본 좌표로 되돌려 겹쳐 봄 · 박스마다 fhd_h = 1920 기준 사람 키 px) · fhd_area (잘라 쓴 구역)
실행: /home/se/miniconda3/envs/drone/bin/python fo_dd_fhd.py
"""
import json
import sys
from pathlib import Path

import cv2
import fiftyone as fo

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / "deploy" / "vision_worker"))
from app.vision_core.detector import PersonDetector   # noqa: E402

MODELS = {"fhd_v7r2": ("soup_v7r2", True), "fhd_v9x2": ("soup_v9x2", False)}


def fhd(img):
    h, w = img.shape[:2]
    ch = round(w * 9 / 16); y0 = h - ch
    return cv2.resize(img[y0:, :], (1920, 1080), interpolation=cv2.INTER_AREA), y0, ch


def main():
    ds = fo.load_dataset("digital_department_test")
    dets = {}
    for field, (name, use_engine) in MODELS.items():
        wd = BASE / "runs_person" / name / "weights"
        eng = wd / "best.engine" if use_engine and (wd / "best.engine").exists() else None
        dets[field] = PersonDetector(wd / "best.pt", engine=eng, conf=0.15)
    summary = {f: {"boxes": 0, "ms": []} for f in MODELS}
    for s in ds:
        img = cv2.imread(s.filepath); h, w = img.shape[:2]
        frame, y0, ch = fhd(img)
        s["fhd_area"] = fo.Detections(detections=[fo.Detection(label="fhd_frame", bounding_box=[0, y0 / h, 1, ch / h])])
        for field, det in dets.items():
            boxes, confs, ms = det.detect(frame)
            out = []
            for (x1, y1, x2, y2), c in zip(boxes, confs):
                X1, X2 = x1 / 1920 * w, x2 / 1920 * w; Y1, Y2 = y0 + y1 / 1080 * ch, y0 + y2 / 1080 * ch
                out.append(fo.Detection(label="person", confidence=float(c), fhd_h=round(float(y2 - y1), 1),
                                        bounding_box=[X1 / w, Y1 / h, (X2 - X1) / w, (Y2 - Y1) / h]))
            s[field] = fo.Detections(detections=out)
            summary[field]["boxes"] += len(out); summary[field]["ms"].append(ms)
        s.save()
    for f, v in summary.items():
        ms = sorted(v["ms"][1:]) or v["ms"]
        print(f, "박스", v["boxes"], "· 탐지 ms 중앙", round(ms[len(ms) // 2], 1))


if __name__ == "__main__":
    main()
