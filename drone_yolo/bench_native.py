"""
bench_native.py — 「타일 추론 덕에 기성 모델보다 좋은 것 아닌가」 확인 (11-04 · 사용자 질문)

  bench_all 은 모든 모델을 우리 운용 틀 (1920 → 1280×720 창 4장) 로 잼 → 통째 사진으로 학습된 기성 모델에 불리할 수 있음
  여기선 **창 없이 통째** (가로 1920 맞춘 그림 한 장 → imgsz 640 · 1280 · 1920) 로 다시 잼 · 채점은 bench_all.score 그대로
  짝    셋마다 bench_all 의 기성 최고 모델 + v12_f91_swa (우리도 통째로 · 참고)
  출력  metrics/bench_native.csv (model · mode · imgsz · set …)
실행  python bench_native.py
"""
import csv

import cv2
import numpy as np

from bench_all import BASE, EXTS, Model, score

OUT = BASE / "metrics" / "bench_native.csv"
BEST = {"test_sar": "aeronova", "test_obl": "aeronova", "test_v2": "visdrone_yolo11x", "test_kr": "aeronova",
        "test_ladd_h": "ap_ladd_26l", "test_visdrone": "visdrone_yolo26x", "test_nomad": "ap_mix_26l", "test_nii": "visdrone_yolov9e",
        "test_sard": "ap_sard_26l", "test_heridal": "ap_heridal_26l", "test_cy": "visdrone_yolo26x"}


class Full(Model):
    """창 없이 통째 — 가로 1920 맞춘 한 장을 imgsz 로"""
    def __call__(self, img):
        h, w = img.shape[:2]; s = 1920 / w
        im = cv2.resize(img, (1920, max(1, round(h * s))), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        r = self.m.predict(im, imgsz=self.imgsz, conf=0.01, classes=self.cls, half=True, verbose=False)[0]
        b, c = r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()
        return b / s, c


def main():
    jobs = {}
    for s, m in BEST.items():
        jobs.setdefault((m, EXTS[m]), []).append(s)
    jobs[("v12_f91_swa", BASE / "runs_person" / "v12_f91_swa" / "weights" / "best.pt")] = list(BEST)
    done = {(r["model"], int(r["imgsz"]), r["set"]) for r in csv.DictReader(open(OUT))} if OUT.exists() else set()
    new = not OUT.exists()
    f = open(OUT, "a", newline="")
    w = csv.DictWriter(f, fieldnames=["model", "mode", "imgsz", "set", "imgs", "n_gt", "AP50", "R@0.2", "P@0.2", "neg_imgs", "FP/neg@0.2", "ms"])
    if new:
        w.writeheader()
    for (name, path), sets in jobs.items():
        for z in (640, 1280, 1920):
            todo = [s for s in sets if (name, z, s) not in done]
            if not todo:
                continue
            m = Full(path, z)
            for s in todo:
                r = {"model": name, "mode": "full", "imgsz": z, "set": s} | score(m, BASE / "data" / s)
                w.writerow(r); f.flush(); print(name, z, s, r["AP50"], r["ms"], flush=True)
            del m


if __name__ == "__main__":
    main()
