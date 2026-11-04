"""
fair_eval.py — 공정 비교 다시 (11-04 · 사용자 「공정하게 성능 평가 수행됐는지부터」) · 검출 저장만 · 분석은 fair_eval_report.py

  왜 다시  ① bench_all 표에 v12 가 학습에 쓴 셋 (SARD test · HERIDAL test = 평가 사진 자체 · NII-CU = v12 만 같은 장소) 이 섞임
           ② 기성 모델을 창 4장 (우리 틀) 로만 · 통째는 셋마다 1등만 다시 → 13개 전부 두 방식
           ③ 사진별 결과가 없어 차이의 신뢰구간 · 같은 재현율 헛탐지를 못 냄
  셋      처음 보는 곳 (v12 학습에 없음): UAV-SAR · Okutama · WiSARD (장소 분리) · AI-Hub (장소 분리) · Cyprus
          같은 데이터셋 다른 사진 (v12 와 그 셋 전용 기성 모두 학습에 씀): VisDrone val · NOMAD 배우 097 · 100 · LADD 뒤 30 %
          + AU-AIR 도로 (사람 없는 장 헛탐지)
  모델    우리 v12_f91_swa = 운용 틀 (가로 1920 → 1280×720 창 4장 · imgsz 1280)
          기성 13개 × { 창 4장 (bench_all 에서 그 셋에 좋았던 imgsz 640 · 1280) , 통째 1280 }
  방식    셋마다 사진을 한 번만 읽고 모든 모델을 돌림 · 확신 ≥ 0.01 · NMS 0.6 · FP16 · 장수 > 500 이면 고르게 500
  저장    runs_fair/<셋>/gt.npz (이름 · 크기 · 정답 박스) · runs_fair/<셋>/<모델>@<방식>.npz (사진 번호 · 박스 · 확신)  (git 밖)
실행  /home/se/miniconda3/envs/drone/bin/python fair_eval.py
"""
import csv, sys, time
from pathlib import Path

import cv2
import numpy as np

from bench_all import BASE, EXTS, MAXN, Model
from bench_native import Full
from eval_external_det import load_gt

OUT = BASE / "runs_fair"
SETS = ["test_sar", "test_obl", "test_v2", "test_kr", "test_cy", "test_visdrone", "test_nomad", "test_ladd_h", "test_auair"]


def tile_imgsz():
    """bench_all 에서 (기성, 셋) 마다 창 방식이 더 좋았던 imgsz · 없으면 640"""
    best = {}
    for r in csv.DictReader(open(BASE / "metrics" / "bench_all.csv")):
        if r["kind"] != "ext" or r["AP50"] == "":
            continue
        k, v = (r["model"], r["set"]), float(r["AP50"])
        if k not in best or v > best[k][0]:
            best[k] = (v, int(r["imgsz"]))
    return {k: z for k, (v, z) in best.items()}


def main():
    tz = tile_imgsz()
    ours = Model(BASE / "runs_person" / "v12_f91_swa" / "weights" / "best.pt", 1280)
    ext_full = {m: Full(p, 1280) for m, p in EXTS.items()}
    tiles = {}                                                     # (기성, imgsz) → Model
    for s in SETS:
        d = OUT / s; d.mkdir(parents=True, exist_ok=True)
        imgs = sorted((BASE / "data" / s / "images").glob("*.jpg"))
        if len(imgs) > MAXN:
            imgs = [imgs[i] for i in np.linspace(0, len(imgs) - 1, MAXN).astype(int)]
        runs = {"v12_f91_swa@tile1280": ours} | {f"{m}@full1280": f for m, f in ext_full.items()}
        for m, p in EXTS.items():
            z = tz.get((m, s), 640)
            if (m, z) not in tiles:
                tiles[(m, z)] = Model(p, z)
            runs[f"{m}@tile{z}"] = tiles[(m, z)]
        runs = {k: v for k, v in runs.items() if not (d / f"{k}.npz").exists()}
        if not runs:
            continue
        acc = {k: ([], [], []) for k in runs}; gts, sizes = [], []; t0 = time.time()
        for i, p in enumerate(imgs):
            img = cv2.imread(str(p)); ih, iw = img.shape[:2]
            gts.append(load_gt(p, iw, ih).astype(np.float32)); sizes.append((iw, ih))
            for k, m in runs.items():
                b, c = m(img)
                acc[k][0].append(np.full(len(c), i, np.int32)); acc[k][1].append(np.asarray(b, np.float32).reshape(-1, 4)); acc[k][2].append(np.asarray(c, np.float32))
            if i % 100 == 0:
                print(s, i, len(imgs), f"{time.time() - t0:.0f}s", flush=True)
        if not (d / "gt.npz").exists():
            np.savez_compressed(d / "gt.npz", names=np.array([p.name for p in imgs]), sizes=np.array(sizes, np.int32),
                                gt=np.concatenate(gts) if gts else np.zeros((0, 4), np.float32), gt_img=np.concatenate([np.full(len(g), i, np.int32) for i, g in enumerate(gts)]))
        for k, (ii, bb, cc) in acc.items():
            np.savez_compressed(d / f"{k}.npz", img=np.concatenate(ii), box=np.concatenate(bb), conf=np.concatenate(cc))
        print("done", s, len(runs), f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
