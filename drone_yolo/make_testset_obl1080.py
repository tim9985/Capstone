"""
make_testset_obl1080.py — test_obl 을 운용 해상도로 (11-08 · 성능 향상 검토) → data/test_obl1080 · data/test_obl1080_q50

  왜   지금 test_obl = Okutama 가 따로 준 1280×720 추출 프레임 (JPEG) 을 1.5 배 키워 씀 → 우리 카메라 (IMX415 1080p 원래 해상도) 보다 나쁜 조건
       Okutama 원본은 4K 영상 (전 시퀀스 있음 · 라벨도 4K 좌표) → **같은 시퀀스 · 같은 프레임** 을 4K → 1920×1080 면적 평균 축소로 다시 뽑음
  같게  장면 · 프레임 번호 · 정답 = test_obl 그대로 (YOLO 정규화 좌표라 해상도와 무관 · 라벨 파일 복사)
  확인  영상 전체를 160×90 으로 훑어 test_obl 원래 그림과 내용이 가장 비슷한 프레임 (±400 · 시퀀스 7개는 번호가 어긋남) · 어긋남 · 상관 기록
  두 판  test_obl1080 (JPEG 95 · 깨끗) · test_obl1080_q50 (JPEG 50 · 전송 압축 흉내 — 실기체 H.264 4~8 Mbps 쪽)
  평가  eval_test_v2.py --data data/test_obl1080 --tag test_obl1080 (기본 타일 4장 = 운용 방식)
  ⚠ Okutama 원본 · 파생물은 git 밖 (data/)
실행  /home/se/miniconda3/envs/drone/bin/python make_testset_obl1080.py
"""
import glob, json, shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

BASE = Path(__file__).resolve().parent
SRC = BASE / "data" / "test_obl"
RAW = BASE / "data" / "raw" / "okutama"
OUTS = {"test_obl1080": 95, "test_obl1080_q50": 50}


def main():
    want = defaultdict(list)
    for p in sorted((SRC / "images").glob("*.jpg")):
        _, seq, fr = p.stem.split("_"); want[seq].append(int(fr))
    for o in OUTS:
        for d in ("images", "labels"):
            (BASE / "data" / o / d).mkdir(parents=True, exist_ok=True)
    log = {}
    for seq, frs in sorted(want.items()):
        vids = [v for v in glob.glob(str(RAW / "**" / f"{seq}.*"), recursive=True) if v.endswith((".mp4", ".mov"))]
        if not vids:
            log[seq] = "영상 없음"; continue
        # 1차: 영상 전체를 작은 회색 그림으로 (160×90) → 원하는 프레임마다 내용이 가장 비슷한 영상 프레임 (번호가 어긋난 시퀀스 있음 · 11-08)
        cap = cv2.VideoCapture(vids[0]); th = []
        while True:
            ok, im = cap.read()
            if not ok:
                break
            g = cv2.cvtColor(cv2.resize(im, (160, 90), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32); th.append((g - g.mean()) / (g.std() + 1e-6))
        cap.release(); th = np.stack(th).reshape(len(th), -1)
        pick, offs, cors, prev = {}, [], [], 0
        for f in sorted(frs):
            r = cv2.cvtColor(cv2.resize(cv2.imread(str(SRC / "images" / f"okutama_{seq}_{f:05d}.jpg")), (160, 90), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32)
            r = ((r - r.mean()) / (r.std() + 1e-6)).ravel(); lo, hi = max(0, f + prev - 400), min(len(th), f + prev + 400)
            if hi <= lo:
                continue
            c = th[lo:hi] @ r / r.size; j = int(np.argmax(c)); k = lo + j
            pick[f] = k; prev = k - f; offs.append(k - f); cors.append(float(c[j]))
        # 2차: 고른 프레임만 원래 해상도로
        cap = cv2.VideoCapture(vids[0]); want_k = {k: f for f, k in pick.items()}; k = -1; done = 0
        while done < len(want_k):
            ok = cap.grab(); k += 1
            if not ok:
                break
            if k in want_k:
                ok, im = cap.retrieve(); f = want_k[k]; done += 1
                im = cv2.resize(im, (1920, 1080), interpolation=cv2.INTER_AREA); name = f"okutama_{seq}_{f:05d}"
                for o, q in OUTS.items():
                    cv2.imwrite(str(BASE / "data" / o / "images" / f"{name}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, q])
                    shutil.copy(SRC / "labels" / f"{name}.txt", BASE / "data" / o / "labels" / f"{name}.txt")
        cap.release()
        log[seq] = {"장수": len(cors), "프레임 어긋남 (중앙 · 최소 · 최대)": [int(np.median(offs)), int(np.min(offs)), int(np.max(offs))] if offs else None,
                    "상관 (중앙 · 최소)": [round(float(np.median(cors)), 3), round(float(np.min(cors)), 3)] if cors else None}
        print(seq, log[seq], flush=True)
    for o, q in OUTS.items():
        (BASE / "data" / o / "manifest.json").write_text(json.dumps(
            {"목적": "test_obl 과 같은 장면 · 프레임 · 정답을 4K 원본 → 1920×1080 (운용 해상도) 로", "JPEG": q, "원본": "Okutama 4K 영상 · 면적 평균 축소", "확인": log}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
