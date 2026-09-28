"""
bench_trt.py — NFR-V01 재측정: 최종 모델 → TensorRT FP16 엔진 → 1920×1080 한 장 처리 시간 (2026-09-28)

  엔진: imgsz [736,1280] · batch 4 (타일 4장을 한 번에) · FP16 — GPU 마다 다시 빌드한다
  재는 구간: 타일 자르기 → 엔진 추론 (전처리·후처리 포함 ultralytics predict) → 프레임 좌표로 합치기 → NMS 0.6
  입력: test_v2 영상 (1920×1080) · 앞 20장은 예열 · 평균 · P95
실행: python bench_trt.py --weights runs_person/soup_v7r2/weights/best.pt [--n 300]
"""
import argparse, json, time
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

BASE = Path(__file__).resolve().parent
TW, TH = 1280, 720


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--conf", type=float, default=0.15)
    args = ap.parse_args()
    w = Path(args.weights); eng = w.with_suffix(".engine")
    if not eng.exists():
        t = time.time()
        YOLO(str(w)).export(format="engine", half=True, imgsz=[736, 1280], batch=4, device=0, verbose=False)
        print(f"엔진 빌드 {time.time() - t:.0f} s → {eng}")
    model = YOLO(str(eng), task="detect")
    imgs = sorted((BASE / "data" / "test_v2" / "images").glob("*.jpg"))[:args.n + 20]
    ts = []
    for i, p in enumerate(imgs):
        img = cv2.imread(str(p))
        if img.shape[:2] != (1080, 1920):
            img = cv2.resize(img, (1920, 1080))
        t0 = time.perf_counter()
        h, wd = img.shape[:2]
        offs = [(x, y) for y in (0, h - TH) for x in (0, wd - TW)]
        res = model.predict([img[y:y + TH, x:x + TW] for x, y in offs], imgsz=[736, 1280], conf=args.conf,
                            batch=4, half=True, verbose=False)
        B, C = [], []
        for (ox, oy), r in zip(offs, res):
            if len(r.boxes):
                B.append(r.boxes.xyxy.cpu().numpy() + [ox, oy, ox, oy]); C.append(r.boxes.conf.cpu().numpy())
        if B:
            B = np.concatenate(B); C = np.concatenate(C)
            cv2.dnn.NMSBoxes([[float(a), float(b), float(c - a), float(d - b)] for a, b, c, d in B], C.tolist(), args.conf, 0.6)
        if i >= 20:
            ts.append((time.perf_counter() - t0) * 1000)
    ts = np.array(ts)
    out = {"모델": w.parent.parent.name, "장수": len(ts), "평균_ms": round(float(ts.mean()), 1),
           "P95_ms": round(float(np.percentile(ts, 95)), 1), "FPS": round(1000 / float(ts.mean()), 1),
           "구간": "타일 4장 자르기 + TensorRT FP16 [736,1280] batch 4 (ultralytics 전처리·후처리 포함) + 합치기 + NMS 0.6"}
    print(json.dumps(out, ensure_ascii=False))
    (BASE / "metrics" / f"nfr_v01_{out['모델']}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
