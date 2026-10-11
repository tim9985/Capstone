"""
dump_dets.py — 평가셋 검출을 박스째 한 번 뽑아 둔다 (10-11 · v7 넘기 검토용) → runs_person/dumps/<이름>_<tag>.npz

  뽑는 법  우리 모델 (ultralytics .pt) = eval_test_v2 와 같은 추론 — 1920×1080 은 타일 4장 1280×720 · 1280×720 은 한 장 · imgsz 1280 · FP16
           타일마다 conf ≥ 0.01 · 타일 안 NMS 0.7 (느슨) → 타일 합치기 전 박스를 그대로 저장 (합치기 · 앙상블은 오프라인)
           flypose = FlyPose 검출기 TensorRT 공식 전처리 (통째 1280×1280 · BGR) · 상위 300
  저장     img (그림 번호) · box (xyxy 원본 좌표) · score · tile (타일 번호 · 통째는 −1) · names · gt (그림별 정답) · W · H
실행: /home/se/miniconda3/envs/drone/bin/python dump_dets.py --model soup_v7r2 [--sets test_v2,test_obl,test_kr]
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from eval_test_v2 import TH, TILES, TW, load_gt

BASE = Path(__file__).resolve().parent
SETS = {t: BASE / "data" / t for t in ("test_v2", "test_obl", "test_kr", "test_ladd", "test_nii", "test_cy", "test_auair")}
OUT = BASE / "runs_person" / "dumps"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="runs_person/<이름> 또는 flypose")
    ap.add_argument("--sets", default="test_v2,test_obl,test_kr")
    a = ap.parse_args()
    if a.model == "flypose":
        from eval_external_det import ENGINE, Det
        det = Det(ENGINE); yolo = None
    else:
        from ultralytics import YOLO
        yolo = YOLO(str(BASE / "runs_person" / a.model / "weights" / "best.pt"))
    OUT.mkdir(parents=True, exist_ok=True)
    for tag in a.sets.split(","):
        imgs = sorted((SETS[tag] / "images").glob("*.jpg"))
        I, B, S, T, names, gts, wh = [], [], [], [], [], [], []
        for i, p in enumerate(imgs):
            img = cv2.imread(str(p)); h, w = img.shape[:2]
            gts.append(load_gt(p, w, h)); names.append(p.stem); wh.append((w, h))
            if yolo is None:
                b, s = det([img])[0]; tl = np.full(len(b), -1)
            else:
                if (w, h) == (1920, 1080):
                    offs = TILES; tiles = [img[y:y + TH, x:x + TW] for x, y in TILES]
                else:
                    offs, tiles = [(0, 0)], [img]
                res = yolo.predict(tiles, imgsz=1280, conf=0.01, iou=0.7, batch=len(tiles), quantize="fp16", verbose=False)
                bl, sl, tl = [], [], []
                for k, ((ox, oy), r) in enumerate(zip(offs, res)):
                    if len(r.boxes):
                        bl.append(r.boxes.xyxy.cpu().numpy() + np.array([ox, oy, ox, oy], np.float32)); sl.append(r.boxes.conf.cpu().numpy())
                        tl.append(np.full(len(r.boxes), k if len(offs) > 1 else -1))
                b = np.concatenate(bl) if bl else np.zeros((0, 4), np.float32); s = np.concatenate(sl) if sl else np.zeros(0, np.float32)
                tl = np.concatenate(tl) if tl else np.zeros(0, int)
            I.append(np.full(len(b), i)); B.append(b); S.append(s); T.append(tl)
            if i % 300 == 0:
                print(f"  {a.model} {tag} {i}/{len(imgs)}", flush=True)
        np.savez_compressed(OUT / f"{a.model}_{tag}.npz", img=np.concatenate(I), box=np.concatenate(B).astype(np.float32), score=np.concatenate(S).astype(np.float32),
                            tile=np.concatenate(T), names=np.array(names), wh=np.array(wh), gt=np.array(gts, dtype=object), allow_pickle=True)
        print(a.model, tag, "검출", sum(len(x) for x in B), flush=True)


if __name__ == "__main__":
    main()
