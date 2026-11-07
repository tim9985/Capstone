"""
fair_eval_sota.py — 일반 탐지 SOTA 를 같은 틀로 (11-07 · 사용자 「기성 모델 중 SOTA 와 비교 · 철저 · 냉정」) → runs_fair/<셋>/<모델>@<방식>.npz
  모델  YOLO11x (COCO) · RT-DETR-x (COCO) · YOLO-World v8x v2 (개방형 · 「person」) — 사람 클래스만
  방식  4조각 imgsz 1280 · 4조각 640 · 통째 1280 (fair_eval 과 같은 전처리 · NMS · 확신 ≥ 0.01)
  Grounding DINO 는 transformers 가 있는 state 환경에서 따로 (gdino_eval.py)
"""
import sys, time
import cv2, numpy as np
from ultralytics import YOLO, RTDETR
from bench_all import BASE, tiles
from fair_eval import OUT, SETS

W = BASE.parent / "data" / "raw" / "ext_det" / "sota"


class Net:
    def __init__(self, kind, path, imgsz, full):
        self.m = RTDETR(str(path)) if kind == "rtdetr" else YOLO(str(path))
        if kind == "world":
            self.m.set_classes(["person"])
        names = {v.lower(): k for k, v in self.m.names.items()}; self.cls = [names["person"]]; self.z = imgsz; self.full = full

    def __call__(self, img):
        h, w = img.shape[:2]; s = 1920 / w
        im = cv2.resize(img, (1920, max(1, round(h * s))), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        if self.full:
            r = self.m.predict(im, imgsz=self.z, conf=0.01, classes=self.cls, half=True, verbose=False)[0]
            return r.boxes.xyxy.cpu().numpy() / s, r.boxes.conf.cpu().numpy()
        T = tiles(1920, im.shape[0]); crops = [im[y:y + 720, x:x + 1280] for x, y in T]
        res = self.m.predict(crops, imgsz=self.z, conf=0.01, classes=self.cls, half=True, verbose=False, batch=len(crops))
        B, C = [], []
        for (x, y), r in zip(T, res):
            B.append(r.boxes.xyxy.cpu().numpy() + np.array([x, y, x, y], np.float32)); C.append(r.boxes.conf.cpu().numpy())
        b, c = np.concatenate(B), np.concatenate(C)
        if len(b):
            keep = cv2.dnn.NMSBoxes([[float(a), float(q), float(e - a), float(d - q)] for a, q, e, d in b], c.tolist(), 0.01, 0.6)
            keep = np.array(keep).ravel().astype(int) if len(keep) else np.array([], int); b, c = b[keep], c[keep]
        return b / s, c


MODELS = {"coco_yolo11m": ("yolo", BASE / "yolo11m.pt"), "sota_yolo11x": ("yolo", W / "yolo11x.pt"), "sota_rtdetrx": ("rtdetr", W / "rtdetr-x.pt"), "sota_yoloworld": ("world", W / "yolov8x-worldv2.pt")}
PROTO = {"tile1280": (1280, False), "tile640": (640, False), "full1280": (1280, True)}


def main():
    nets = {f"{m}@{p}": Net(k, path, z, full) for m, (k, path) in MODELS.items() for p, (z, full) in PROTO.items()}
    for s in SETS:
        d = OUT / s; g = np.load(d / "gt.npz"); names = list(g["names"])
        runs = {k: v for k, v in nets.items() if not (d / f"{k}.npz").exists()}
        if not runs:
            continue
        acc = {k: ([], [], []) for k in runs}; t0 = time.time()
        for i, nm in enumerate(names):
            img = cv2.imread(str(BASE / "data" / s / "images" / nm))
            for k, n in runs.items():
                b, c = n(img); acc[k][0].append(np.full(len(c), i, np.int32)); acc[k][1].append(np.asarray(b, np.float32).reshape(-1, 4)); acc[k][2].append(np.asarray(c, np.float32))
        for k, (ii, bb, cc) in acc.items():
            np.savez_compressed(d / f"{k}.npz", img=np.concatenate(ii), box=np.concatenate(bb), conf=np.concatenate(cc))
        print("done", s, f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
