"""
detector.py — 사람 1클래스 탐지 (운용 설정 · 2026-10-05)

  1920×1080 → 1280×720 타일 4장 (겹침 50 %) → 모델 → 프레임 좌표로 합치기 → NMS 0.6
  `drone_yolo/eval_test_v2.py` 의 tile 모드와 같은 식이다 — NFR-V03 수치(AP50)가 이 경로에서 나왔다
  그 밖의 크기는 한 장 그대로 (imgsz 1280 · eval_test_v2 single 모드)

  백엔드: TensorRT 엔진 (FP16 · [736,1280] · batch 4) 이 있으면 엔진 · 없으면 .pt FP16
  엔진은 GPU · TensorRT 버전마다 따로 만든다 → tools/build_engine.py
"""
import time
from pathlib import Path

import cv2
import numpy as np

TW, TH = 1280, 720
FULL = (1920, 1080)


class PersonDetector:
    def __init__(self, weights, engine=None, conf=0.15, nms_iou=0.6, imgsz=1280):
        from ultralytics import YOLO
        self.conf, self.nms_iou, self.imgsz = conf, nms_iou, imgsz
        self.weights = Path(weights)
        self.engine = Path(engine) if engine else None
        if self.engine and self.engine.exists():
            self.model, self.backend = YOLO(str(self.engine), task="detect"), "tensorrt"
        else:
            self.model, self.backend = YOLO(str(self.weights)), "pytorch_fp16"
        self.name = self.weights.parent.parent.name if self.weights.parent.name == "weights" else self.weights.parent.name

    def _predict(self, imgs, imgsz, conf):
        kw = dict(conf=conf, batch=len(imgs), verbose=False)
        if self.backend == "tensorrt":
            return self.model.predict(imgs, imgsz=[736, 1280], **kw)
        return self.model.predict(imgs, imgsz=imgsz, quantize="fp16", **kw)

    def detect(self, frame, conf=None):
        """frame: BGR · 반환 (boxes N×4 xyxy float32 프레임 px, conf N float32, ms)"""
        conf = self.conf if conf is None else conf
        t0 = time.perf_counter()
        h, w = frame.shape[:2]
        if (w, h) == FULL:
            offs = [(x, y) for y in (0, h - TH) for x in (0, w - TW)]
            imgs = [frame[y:y + TH, x:x + TW] for x, y in offs]
        elif self.backend == "tensorrt":            # 엔진은 타일 4장 고정 → 1920×1080 으로 맞춘다
            return self.detect(cv2.resize(frame, FULL), conf)
        else:
            offs, imgs = [(0, 0)], [frame]
        res = self._predict(imgs, self.imgsz, conf)
        B, C = [], []
        for (ox, oy), r in zip(offs, res):
            if len(r.boxes):
                B.append(r.boxes.xyxy.cpu().numpy() + np.array([ox, oy, ox, oy], np.float32))
                C.append(r.boxes.conf.cpu().numpy())
        if not B:
            return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), (time.perf_counter() - t0) * 1000
        B, C = np.concatenate(B).astype(np.float32), np.concatenate(C).astype(np.float32)
        keep = cv2.dnn.NMSBoxes([[float(a), float(b), float(c - a), float(d - b)] for a, b, c, d in B],
                                C.tolist(), conf, self.nms_iou)
        keep = np.array(keep).ravel().astype(int) if len(keep) else np.array([], int)
        return B[keep], C[keep], (time.perf_counter() - t0) * 1000
