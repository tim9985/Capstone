"""
detector.py — 사람 1클래스 탐지 (운용 설정 · 2026-10-05)

  1920×1080 → 1280×720 타일 4장 (겹침 50 %) → 모델 → 프레임 좌표로 합치기 → NMS 0.6
  `drone_yolo/eval_test_v2.py` 의 tile 모드와 같은 식이다 — NFR-V03 수치(AP50)가 이 경로에서 나왔다
  그 밖의 크기는 한 장 그대로 (imgsz 1280 · eval_test_v2 single 모드)

  백엔드: TensorRT 엔진 (FP16 · [736,1280] · batch 4) 이 있으면 엔진 · 없으면 .pt FP16
  엔진은 GPU · TensorRT 버전마다 따로 만든다 → tools/build_engine.py

  edge_merge (10-11 · **10-12 기본 켜짐** · 끄려면 edge_merge=False): NMS 0.6 만으론 타일 안쪽 경계에서 잘린 박스가 남는다
  (온전한 박스와 IoU < 0.6 → 한 사람이 후보 둘) → 잘린 박스가 다른 타일의 더 큰 박스 안에 60 % 이상이면 지우고 점수는 큰 박스로
  soup_v7r2 그대로 test_v2 AP50 +0.8 · test_kr +2.2 %p (drone_yolo/tile_merge.py 와 같은 규칙)
"""
import time
from pathlib import Path

import cv2
import numpy as np

TW, TH = 1280, 720
FULL = (1920, 1080)
OFFS = [(x, y) for y in (0, FULL[1] - TH) for x in (0, FULL[0] - TW)]


def edge_merge(b, s, t, e=3.0, thr=0.6):
    """1920×1080 타일 박스 (NMS 뒤) 에서 안쪽 경계에 닿은 잘린 박스를 지운다 — b xyxy · s 점수 · t 타일 번호 (OFFS 순서)"""
    if not len(b):
        return b, s
    s = s.copy(); area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]); drop = np.zeros(len(b), bool)
    for i in np.argsort(area):                                   # 작은 (잘린) 박스부터
        x0, y0 = OFFS[t[i]]
        if not ((x0 > 0 and b[i, 0] - x0 <= e) or (x0 + TW < FULL[0] and x0 + TW - b[i, 2] <= e)
                or (y0 > 0 and b[i, 1] - y0 <= e) or (y0 + TH < FULL[1] and y0 + TH - b[i, 3] <= e)):
            continue
        c = np.flatnonzero((t != t[i]) & ~drop & (area > area[i]))
        if not len(c):
            continue
        ix = np.clip(np.minimum(b[i, 2], b[c, 2]) - np.maximum(b[i, 0], b[c, 0]), 0, None)
        iy = np.clip(np.minimum(b[i, 3], b[c, 3]) - np.maximum(b[i, 1], b[c, 1]), 0, None)
        r = ix * iy / max(float(area[i]), 1e-6)
        if r.max() >= thr:
            j = c[np.argmax(r)]; s[j] = max(s[j], s[i]); drop[i] = True
    return b[~drop], s[~drop]


class PersonDetector:
    def __init__(self, weights, engine=None, conf=0.20, nms_iou=0.6, imgsz=1280, edge_merge=True):
        from ultralytics import YOLO
        self.conf, self.nms_iou, self.imgsz, self.edge_merge = conf, nms_iou, imgsz, edge_merge
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
            offs = OFFS
            imgs = [frame[y:y + TH, x:x + TW] for x, y in offs]
        elif self.backend == "tensorrt":            # 엔진은 타일 4장 고정 → 1920×1080 으로 맞춘다
            boxes, scores, ms = self.detect(cv2.resize(frame, FULL), conf)      # 10-14 통합 판 수정 반영 — 원래 크기로 되돌림
            boxes *= np.array([w / FULL[0], h / FULL[1], w / FULL[0], h / FULL[1]], np.float32)
            return boxes, scores, ms
        else:
            offs, imgs = [(0, 0)], [frame]
        res = self._predict(imgs, self.imgsz, conf)
        B, C, T = [], [], []
        for k, ((ox, oy), r) in enumerate(zip(offs, res)):
            if len(r.boxes):
                B.append(r.boxes.xyxy.cpu().numpy() + np.array([ox, oy, ox, oy], np.float32))
                C.append(r.boxes.conf.cpu().numpy()); T.append(np.full(len(r.boxes), k))
        if not B:
            return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), (time.perf_counter() - t0) * 1000
        B, C, T = np.concatenate(B).astype(np.float32), np.concatenate(C).astype(np.float32), np.concatenate(T)
        keep = cv2.dnn.NMSBoxes([[float(a), float(b), float(c - a), float(d - b)] for a, b, c, d in B],
                                C.tolist(), conf, self.nms_iou)
        keep = np.array(keep).ravel().astype(int) if len(keep) else np.array([], int)
        B, C, T = B[keep], C[keep], T[keep]
        if self.edge_merge and len(offs) > 1:
            B, C = edge_merge(B, C, T)
        return B, C, (time.perf_counter() - t0) * 1000
