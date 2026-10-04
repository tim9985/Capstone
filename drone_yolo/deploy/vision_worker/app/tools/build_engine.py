"""
build_engine.py — TensorRT 엔진 만들기 (GPU · TensorRT 버전마다 한 번)

  FP16 · imgsz [736,1280] · batch 4 (타일 4장) → <모델 폴더>/best_trt<버전>_<GPU>.engine
  worker 가 처음 뜰 때 자동으로도 만든다 (VISION_BUILD_ENGINE=1) — 미리 만들어 두면 첫 기동이 빨라진다
실행: python -m app.tools.build_engine --weights /data/models/soup_v7r2/best.pt
"""
import argparse
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    a = ap.parse_args()
    import tensorrt as trt
    import torch
    from ultralytics import YOLO
    w = Path(a.weights)
    gpu = torch.cuda.get_device_name(0).replace("NVIDIA ", "").replace("GeForce ", "").replace(" ", "")
    eng = w.parent / f"best_trt{trt.__version__}_{gpu}.engine"
    t = time.time()
    out = YOLO(str(w)).export(format="engine", half=True, imgsz=[736, 1280], batch=4, device=0, verbose=False)
    Path(out).replace(eng)
    print(f"엔진 {eng} · {time.time() - t:.0f} s")


if __name__ == "__main__":
    main()
