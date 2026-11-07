"""
gdino_eval.py — Grounding DINO (base · 개방형 · 「person.」) 를 같은 틀로 (11-07 · SOTA 비교) → runs_fair/<셋>/sota_gdino@<방식>.npz
  state 환경 (transformers 5.x) · fair_eval 과 같은 사진 순서 (gt.npz names) · 가로 1920 맞춤 → 4조각 1280×720 또는 통째 · NMS 0.6 · 확신 ≥ 0.05
  ⚠ 확신 = 상자 점수 (문장 「person」 일치) · 0.05 아래는 버림 (기성 · 우리 0.01 보다 높음 → 재현율 끝이 약간 잘릴 수 있음 · 보고서에 표기)
실행  /home/se/venvs/state/bin/python gdino_eval.py
"""
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

BASE = Path(__file__).resolve().parent
RUN = BASE / "runs_fair"
SETS = ["test_sar", "test_obl", "test_v2", "test_kr", "test_cy", "test_visdrone", "test_nomad", "test_ladd_h", "test_auair"]
MID = "IDEA-Research/grounding-dino-base"
proc = AutoProcessor.from_pretrained(MID)
model = AutoModelForZeroShotObjectDetection.from_pretrained(MID, torch_dtype=torch.float16).cuda().eval()


def tiles(W, H, tw=1280, th=720):
    xs = [0] if W <= tw else sorted(set(list(range(0, W - tw + 1, tw // 2)) + [W - tw]))
    ys = [0] if H <= th else sorted(set(list(range(0, H - th + 1, th // 2)) + [H - th]))
    return [(x, y) for y in ys for x in xs]


@torch.no_grad()
def detect(crops):
    ims = [Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for c in crops]
    inp = proc(images=ims, text=["person."] * len(ims), return_tensors="pt").to("cuda")
    inp["pixel_values"] = inp["pixel_values"].half()
    out = model(**inp)
    res = proc.post_process_grounded_object_detection(out, inp.input_ids, threshold=0.05, text_threshold=0.05,
                                                      target_sizes=[(c.shape[0], c.shape[1]) for c in crops])
    return [(r["boxes"].float().cpu().numpy(), r["scores"].float().cpu().numpy()) for r in res]


def run(img, full):
    h, w = img.shape[:2]; s = 1920 / w
    im = cv2.resize(img, (1920, max(1, round(h * s))), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    if full:
        b, c = detect([im])[0]; return b / s, c
    T = tiles(1920, im.shape[0]); res = detect([im[y:y + 720, x:x + 1280] for x, y in T])
    b = np.concatenate([r[0] + np.array([x, y, x, y], np.float32) for (x, y), r in zip(T, res)]); c = np.concatenate([r[1] for r in res])
    if len(b):
        keep = cv2.dnn.NMSBoxes([[float(a), float(q), float(e - a), float(d - q)] for a, q, e, d in b], c.tolist(), 0.05, 0.6)
        keep = np.array(keep).ravel().astype(int) if len(keep) else np.array([], int); b, c = b[keep], c[keep]
    return b / s, c


for s in SETS:
    d = RUN / s; names = list(np.load(d / "gt.npz")["names"]); t0 = time.time()
    for proto, full in (("tile", False), ("full", True)):
        out = d / f"sota_gdino@{proto}.npz"
        if out.exists():
            continue
        I, Bx, C = [], [], []
        for i, nm in enumerate(names):
            b, c = run(cv2.imread(str(BASE / "data" / s / "images" / nm)), full)
            I.append(np.full(len(c), i, np.int32)); Bx.append(np.asarray(b, np.float32).reshape(-1, 4)); C.append(np.asarray(c, np.float32))
        np.savez_compressed(out, img=np.concatenate(I), box=np.concatenate(Bx), conf=np.concatenate(C))
        print("done", s, proto, f"{time.time() - t0:.0f}s", flush=True)
