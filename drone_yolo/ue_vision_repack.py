"""
ue_vision_repack.py — 촬영 결과를 JPEG 85 로 다시 눌러 전송용 압축 (노트북 · 2026-10-14)
  사용자 scp 실측 343 KB/s → 서버 요청: JPEG 92 → 85 · 우선순위 순서로 하나씩 (촬영이 끝난 것부터)
  ue_flight/<run> → C:/work/ve/<run>_q85.tar.gz (원본 ue_flight/<run> 은 그대로)
실행 (conda drone — cv2): python ue_vision_repack.py S3_a16 S2_a16 S4_a16 S1_a16 S5_a16 ...
"""
import shutil
import tarfile
import sys
import time
from pathlib import Path

import cv2
import numpy as np

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

SRC = Path(__file__).resolve().parent / "ue_flight"
VE = Path("C:/work/ve")
Q = 85


def done(run):
    """run.json 은 촬영 마지막에 쓴다 → 있으면 촬영이 끝난 것"""
    return (SRC / run / "run.json").exists()


for run in sys.argv[1:]:
    while not done(run):
        time.sleep(20)
    tmp = VE / "pack" / run
    shutil.rmtree(tmp, ignore_errors=True)
    (tmp / "frames").mkdir(parents=True)
    for f in sorted((SRC / run / "frames").glob("*.jpg")):
        # cv2.imread/imwrite 는 Windows 한글 경로(바탕화면\캡스톤)를 못 연다 → 바이트로 읽고 쓴다
        img = cv2.imdecode(np.fromfile(str(f), np.uint8), cv2.IMREAD_COLOR)
        cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, Q])[1].tofile(str(tmp / "frames" / f.name))
    for f in ("flight.csv", "boxes.jsonl", "actors.json", "run.json"):
        shutil.copy2(SRC / run / f, tmp / f)
    out = VE / f"{run}_q{Q}.tar.gz"
    with tarfile.open(out, "w:gz") as tf:                  # Windows 기본 tar 는 --force-local 을 모른다 → 내장 tarfile
        tf.add(tmp, arcname=run)
    shutil.rmtree(tmp)
    print(f"{run}: {out.name} {out.stat().st_size / 1e6:.0f} MB", flush=True)
