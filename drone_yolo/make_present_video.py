"""
make_present_video.py — 발표 자료용 디지털관 영상 추론 결과 (10-23)

  영상 프레임마다 배포 탐지 (v12_f91_swa · TensorRT · 0.20 · 타일 4장) + 추적 (worker state.StateTracker · BoT-SORT · 문턱 0.15)
  → 박스 · 추적 번호 · 확신도 · 위쪽 띠 (모델 · 탐지 ms · 사람 수) 를 그려 mp4 로 저장 (git 밖 · 발표자료 폴더)
  ⚠ 정답 라벨 없음 → 정성 예시 · 휴대폰으로 건물 높은 층에서 찍은 영상 (드론 아님)
실행  /home/se/miniconda3/envs/drone/bin/python make_present_video.py <출력 폴더> [--start N] <영상 …>   (--start = 출력 번호 시작 · 기존 영상 안 덮음)
"""
import sys, time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / "deploy" / "vision_worker"))
from app.vision_core.detector import PersonDetector  # noqa: E402
from app.vision_core.state import StateTracker  # noqa: E402

PACK = BASE / "runs_person" / "_pack_v12_sim" / "v12_f91_swa"
POSTURE = BASE / "runs_person" / "_pack_v12" / "posture.json"
FONT = str(BASE.parent / "발표자료" / "_도구" / "NanumGothicBold.ttf")
PAL = [(214, 120, 42), (52, 104, 235), (122, 175, 27), (164, 58, 232), (32, 160, 237), (72, 72, 227), (180, 105, 0), (60, 170, 160)]   # BGR


def main(out, videos, start=1):
    det = PersonDetector(PACK / "best.pt", PACK / "best_trt11.3.0.99_RTX3090.engine", conf=0.20)
    f_big, f_sm = ImageFont.truetype(FONT, 34), ImageFont.truetype(FONT, 22)
    for k, v in enumerate(videos, start):
        cap = cv2.VideoCapture(str(v)); fps = cap.get(cv2.CAP_PROP_FPS) or 30
        W, H = int(cap.get(3)), int(cap.get(4))
        dst = out / f"디관_영상_{k:02d}_탐지.mp4"
        wr = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
        trk = StateTracker(str(POSTURE)); i = 0; ms_all = []; ids = set()
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            boxes, confs, ms = det.detect(fr); ms_all.append(ms)
            st = trk.update(i / fps, fr, [tuple(map(float, b)) for b in boxes], [float(c) for c in confs]) if len(boxes) else []
            for (x1, y1, x2, y2), c, (tid, _) in zip(boxes.astype(int), confs, st or [(None, None)] * len(boxes)):
                n = int(str(tid).split(".")[0]) if tid else 0
                col = PAL[n % len(PAL)] if tid else (0, 0, 255)
                cv2.rectangle(fr, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), col, 3)
                lab = f"#{n} {c:.2f}" if tid else f"{c:.2f}"
                (tw, th), _ = cv2.getTextSize(lab, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(fr, (x1 - 3, y1 - th - 10), (x1 - 3 + tw + 8, y1 - 3), col, -1)
                cv2.putText(fr, lab, (x1 + 1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                if tid:
                    ids.add(n)
            band = fr[:62].copy(); fr[:62] = (band * 0.35).astype(np.uint8)
            img = Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)); d = ImageDraw.Draw(img)
            d.text((20, 12), f"탐지 박스 {len(boxes)}개", font=f_big, fill=(255, 255, 255))
            d.text((270, 20), f"v12_f91_swa · TensorRT FP16 · 임계값 0.20 · 탐지 {ms:.0f} ms/장 · 추적 BoT-SORT (# = 추적 번호)",
                   font=f_sm, fill=(230, 230, 230))
            wr.write(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)); i += 1
        wr.release()
        m = sorted(ms_all[3:]) or ms_all
        print(dst.name, i, "장", "탐지 ms 중앙", round(m[len(m) // 2], 1), "추적 번호", len(ids))


if __name__ == "__main__":
    o = Path(sys.argv[1]); o.mkdir(parents=True, exist_ok=True)
    a = sys.argv[2:]; st = 1
    if a and a[0] == "--start":
        st, a = int(a[1]), a[2:]
    main(o, [Path(p) for p in a], st)
