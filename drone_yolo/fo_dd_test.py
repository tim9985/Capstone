"""
fo_dd_test.py — 디지털관 높은 층 사진 (휴대폰 · 4032×3024) 간단 탐지 시험 → FiftyOne `digital_department_test` (10-08)

  사진: data/digital_department_test/*.jpg (정답 없음 · 눈으로 확인용 · git 밖)
  탐지 (conf 0.15 · 타일 1280×720 겹침 50 % · NMS 0.6 — worker 와 같은 방식을 크기만 일반화)
    op_v7r2   운용 크기 — 긴 변 1920 으로 줄임 (1920×1440 · 타일 6장) · soup_v7r2 (배포 모델)
    op_v9x2   같은 크기 · soup_v9x2 (상태 인지용)
    full_v7r2 원본 4032×3024 그대로 타일 (참고 상한 · 사람이 2.1배 큼)
  자세: 탐지마다 V3 표시 자세 (posture.json · 1080p 환산 박스) · B0 누움 확률
  ⚠ 휴대폰 1배 화각 (~69°) 은 운용 카메라 (80.2°) 보다 좁아 사람이 약 1.2배 크게 찍힌다 → 운용 크기 결과가 실제보다 약간 유리
실행: /home/se/miniconda3/envs/drone/bin/python fo_dd_test.py
"""
import glob
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / "deploy" / "vision_worker"))
from app.vision_core.state import load_posture   # noqa: E402

SRC = BASE.parent / "data" / "digital_department_test"
POSTURE = "/tmp/claude-1001/-home-se-JupyterLAB/027ea9bd-abdf-476f-ad0d-25f3e2ea7ffc/scratchpad/stage/models/posture.json"
TW, TH = 1280, 720


def tiles(w, h):
    xs = sorted({min(x, w - TW) for x in range(0, max(w - TW, 0) + 1, TW // 2)} | {max(w - TW, 0)})
    ys = sorted({min(y, h - TH) for y in range(0, max(h - TH, 0) + 1, TH // 2)} | {max(h - TH, 0)})
    return [(x, y) for y in ys for x in xs]


def detect(model, img, conf=0.15, iou=0.6):
    h, w = img.shape[:2]
    offs = tiles(w, h)
    B, C = [], []
    for i in range(0, len(offs), 8):
        part = offs[i:i + 8]
        res = model.predict([img[y:y + TH, x:x + TW] for x, y in part], imgsz=1280, conf=conf, half=True, verbose=False)
        for (x, y), r in zip(part, res):
            if len(r.boxes):
                B.append(r.boxes.xyxy.cpu().numpy() + np.array([x, y, x, y], np.float32)); C.append(r.boxes.conf.cpu().numpy())
    if not B:
        return np.zeros((0, 4)), np.zeros(0), len(offs)
    B, C = np.concatenate(B), np.concatenate(C)
    keep = cv2.dnn.NMSBoxes([[float(a), float(b), float(c - a), float(d - b)] for a, b, c, d in B], C.tolist(), conf, iou)
    keep = np.array(keep).ravel().astype(int) if len(keep) else np.array([], int)
    return B[keep], C[keep], len(offs)


def main():
    import fiftyone as fo
    from ultralytics import YOLO
    score_m, disp_m = load_posture(POSTURE)
    M = {"v7r2": YOLO(str(BASE / "runs_person/soup_v7r2/weights/best.pt")), "v9x2": YOLO(str(BASE / "runs_person/soup_v9x2/weights/best.pt"))}
    name = "digital_department_test"
    if fo.dataset_exists(name):
        fo.delete_dataset(name)
    ds = fo.Dataset(name); ds.persistent = True
    summary = []
    for f in sorted(glob.glob(str(SRC / "*.jpg"))):
        img = cv2.imread(f)
        H0, W0 = img.shape[:2]
        truncated = os.path.getsize(f) < 3_000_000                  # 05 는 전송이 덜 됐다 (아래 55 % 회색)
        s = fo.Sample(filepath=f, tags=["truncated"] if truncated else [])
        row = {"file": os.path.basename(f), "truncated": truncated}
        for key, mname, scale in (("op_v7r2", "v7r2", 1920 / W0), ("op_v9x2", "v9x2", 1920 / W0), ("full_v7r2", "v7r2", 1.0)):
            im = cv2.resize(img, (round(W0 * scale), round(H0 * scale))) if scale != 1.0 else img
            b, c, nt = detect(M[mname], im)
            b = b / scale                                           # 원본 좌표로
            dets = []
            if len(c):
                wh = np.c_[b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]] * (1920 / W0)   # 운용 크기 (폭 1920) px — 자세 판정기 입력 단위
                D, P = disp_m.proba(wh), score_m.proba(wh)
                for (x1, y1, x2, y2), cf, d, p, (ww, hh) in zip(b, c, D, P, wh):
                    lab = ("lying", "sitting", "standing")[int(d.argmax())]
                    dets.append(fo.Detection(label="person", confidence=float(cf),
                                             bounding_box=[x1 / W0, y1 / H0, (x2 - x1) / W0, (y2 - y1) / H0],
                                             posture=lab, p_lying=round(float(d[0]), 3), p_sitting=round(float(d[1]), 3),
                                             score_lying_b0=round(float(p[0]), 3), px_1920=[round(float(ww), 1), round(float(hh), 1)]))
            s[key] = fo.Detections(detections=dets)
            row[key] = {"n": len(dets), "n_conf50": int(sum(d.confidence >= 0.5 for d in dets)), "tiles": nt,
                        "postures": {k: sum(d.posture == k for d in dets) for k in ("standing", "sitting", "lying")}}
        ds.add_sample(s); summary.append(row)
    ds.save()
    out = BASE / "metrics" / "dd_test.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
