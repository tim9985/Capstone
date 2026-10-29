"""
ue_reid_siglip.py — 후보 잇기 외형 특징 = SigLIP 2 임베딩 (10-30 R2 · worker 1.3 꾸러미 인코더 그대로 · 덤프 = ue_group_dump.py)

  탐지마다 ① 색 히스토그램 (ue_reid_feats.hist · R 과 같음) ② SigLIP 2 임베딩 — part = upper (상의 크롭 · 색 판정과 같은 입력) · person (박스 전체)
  출력 runs_state/reid_siglip_<part>_<run>.pkl — {"feats": [[(hist, emb) …] 프레임마다], "ms_per_det"} (R 의 reid_<run>.pkl 과 같은 모양)
실행  /home/se/miniconda3/envs/drone/bin/python ue_reid_siglip.py upper S1_a16 …
"""
import csv, pickle, sys, time
from pathlib import Path

import cv2
import numpy as np

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / "deploy" / "vision_worker" / "integration_v1_3"))
from app.vision_core.color import SiglipColorProfiler  # noqa: E402


def hist(img, b):
    """ue_reid_feats.hist 와 같음 (그 파일은 timm 을 불러와 drone 환경에서 못 읽음)"""
    x1, y1, x2, y2 = b; w, h = x2 - x1, y2 - y1
    c = img[int(y1 + 0.15 * h):int(y1 + 0.70 * h) + 1, int(x1 + 0.2 * w):int(x2 - 0.2 * w) + 1]
    if c.size == 0:
        return np.zeros(256, np.float32)
    hsv = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
    hh = cv2.calcHist([hsv], [0, 1, 2], None, [16, 4, 4], [0, 180, 0, 256, 0, 256]).ravel()
    return (hh / max(hh.sum(), 1)).astype(np.float32)

PACK = BASE / "runs_person" / "_pack_color_siglip2"


def main():
    part, names = sys.argv[1], sys.argv[2:]
    prof = SiglipColorProfiler(PACK)
    torch = prof._torch
    for name in names:
        d = pickle.loads((BASE / "runs_state" / f"group_{name}.pkl").read_bytes())
        rows = list(csv.DictReader(open(BASE / "ue_flight" / name / "flight.csv")))
        out, ms = [], []
        for f, r in zip(d["frames"], rows):
            img = cv2.imread(str(BASE / "ue_flight" / name / "frames" / f"{r['frame']}.jpg"))
            t0 = time.perf_counter()
            H = [hist(img, det["box"]) for det in f["dets"]]
            crops = []
            for det in f["dets"]:
                c = prof._crop(img, det["box"], None) if part == "upper" else img[int(max(0, det["box"][1])):int(det["box"][3]), int(max(0, det["box"][0])):int(det["box"][2])]
                crops.append(c if c.size and c.shape[0] >= 2 and c.shape[1] >= 2 else np.zeros((8, 8, 3), np.uint8))
            if crops:
                x = torch.tensor(np.stack([prof._prep(c) for c in crops]).transpose(0, 3, 1, 2), device=prof.dev, dtype=prof.dtype)
                with torch.no_grad():
                    e = prof.enc(x).cpu().numpy()
                ms.append((time.perf_counter() - t0) * 1000 / len(crops))
            else:
                e = np.zeros((0, 768), np.float32)
            out.append([(h, e[i]) for i, h in enumerate(H)])
        (BASE / "runs_state" / f"reid_siglip_{part}_{name}.pkl").write_bytes(pickle.dumps({"feats": out, "ms_per_det": float(np.median(ms)) if ms else None}))
        print(part, name, len(out), "장 · 탐지당", round(float(np.median(ms)), 2) if ms else None, "ms", flush=True)


if __name__ == "__main__":
    main()
