"""
ue_reid_feats.py — 후보 잇기 외형 특징 (10-15 R · 덤프 = ue_group_dump.py · state 환경)

  덤프의 탐지마다 크롭 → ① 색 히스토그램 (HSV 16×4×4 · 박스 가로 가운데 60 % · 세로 15~70 %) ② DINOv2 (timm vit_base_patch14_dinov2 · 224 · CLS · L2 정규화)
  출력 runs_state/reid_<run>.pkl — 프레임 순서 · 탐지 순서 그대로 [(hist, dino) …] · 탐지 하나당 시간 (ms)
실행  /home/se/venvs/state/bin/python ue_reid_feats.py S1_a16 …
"""
import pickle, sys, time
from pathlib import Path

import cv2
import numpy as np
import torch
import timm

BASE = Path(__file__).resolve().parent


def hist(img, b):
    x1, y1, x2, y2 = b; w, h = x2 - x1, y2 - y1
    c = img[int(y1 + 0.15 * h):int(y1 + 0.70 * h) + 1, int(x1 + 0.2 * w):int(x2 - 0.2 * w) + 1]
    if c.size == 0:
        return np.zeros(256, np.float32)
    hsv = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
    hh = cv2.calcHist([hsv], [0, 1, 2], None, [16, 4, 4], [0, 180, 0, 256, 0, 256]).ravel()
    return (hh / max(hh.sum(), 1)).astype(np.float32)


def main():
    dev = "cuda"
    m = timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=True, num_classes=0, img_size=224).to(dev).eval().half()
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1); std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    for name in sys.argv[1:]:
        d = pickle.loads((BASE / "runs_state" / f"group_{name}.pkl").read_bytes())
        rows = list(__import__("csv").DictReader(open(BASE / "ue_flight" / name / "flight.csv")))
        out, ms = [], []
        for f, r in zip(d["frames"], rows):
            img = cv2.imread(str(BASE / "ue_flight" / name / "frames" / f"{r['frame']}.jpg"))
            t0 = time.perf_counter()
            H = [hist(img, det["box"]) for det in f["dets"]]
            crops = []
            for det in f["dets"]:
                x1, y1, x2, y2 = det["box"]; mw, mh = 0.1 * (x2 - x1), 0.1 * (y2 - y1)
                c = img[max(0, int(y1 - mh)):int(y2 + mh) + 1, max(0, int(x1 - mw)):int(x2 + mw) + 1]
                c = cv2.resize(c, (224, 224)) if c.size else np.zeros((224, 224, 3), np.uint8)
                crops.append(c[..., ::-1])
            if crops:
                x = torch.from_numpy(np.stack(crops)).to(dev).permute(0, 3, 1, 2).half() / 255
                with torch.no_grad():
                    e = torch.nn.functional.normalize(m(((x - mean) / std).half()).float(), dim=1).cpu().numpy()
            else:
                e = np.zeros((0, 768), np.float32)
            if crops:
                ms.append((time.perf_counter() - t0) * 1000 / len(crops))
            out.append([(h, e[i]) for i, h in enumerate(H)])
        (BASE / "runs_state" / f"reid_{name}.pkl").write_bytes(pickle.dumps({"feats": out, "ms_per_det": float(np.median(ms)) if ms else None}))
        print(name, len(out), "장 · 탐지당", round(float(np.median(ms)), 2) if ms else None, "ms")


if __name__ == "__main__":
    main()
