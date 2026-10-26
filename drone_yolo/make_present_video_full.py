"""
make_present_video_full.py — 발표 자료용 디지털관 영상 「비전 임무 전체」 판 (10-26)

  make_present_video.py (탐지만) 와 같은 탐지 위에 worker 모듈을 그대로 얹는다
    탐지 v12_f91_swa · TensorRT · 0.20 · 타일 4장 → 추적 state.StateTracker (BoT-SORT · 문턱 0.15)
    → 자세 (박스 모양 V3 · 누움 / 앉음 / 서기) · 움직임 (1초마다 배경 맞춤으로 카메라 움직임 뺌 · 이동 / 정지 s) · 등급 (state.grade)
    → 상의 색 (color.ColorProfiler · 같은 추적끼리 ColorAccumulator 누적 · 3장마다 한 번 관측)
  박스 색 = 등급 (URGENT 빨강 · HIGH 주황 · CHECK 노랑 · NORMAL 초록) · 흰색 = 추적 3 s 미만 (grade 규칙상 CHECK · 관찰 중으로 표시)
  ⚠ 정답 라벨 없음 → 정성 예시 · 휴대폰으로 건물 높은 층에서 찍은 영상 (드론 아님) · 관절점 자세 (FlyPose) 는 배포에 없어 안 씀
실행  /home/se/miniconda3/envs/drone/bin/python make_present_video_full.py <출력 폴더> [--start N] <영상 …>
"""
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / "deploy" / "vision_worker"))
from app.vision_core.detector import PersonDetector  # noqa: E402
from app.vision_core.state import StateTracker  # noqa: E402
from app.vision_core.color import ColorProfiler, ColorAccumulator, frame_stats, KO  # noqa: E402

PACK = BASE / "runs_person" / "_pack_v12_sim" / "v12_f91_swa"
POSTURE = BASE / "runs_person" / "_pack_v12" / "posture.json"
FONT = str(BASE.parent / "발표자료" / "_도구" / "NanumGothicBold.ttf")
GRADE_COL = {"URGENT": (230, 30, 30), "HIGH": (250, 130, 20), "CHECK": (240, 210, 40), "NORMAL": (60, 200, 110), "관찰 중": (255, 255, 255)}   # RGB
POSE_KO = {"lying": "누움", "sitting": "앉음", "standing": "서기"}
SWATCH = {"red": (220, 40, 40), "orange": (245, 140, 30), "yellow": (240, 210, 40), "green": (40, 160, 70), "blue": (40, 100, 220),
          "navy": (30, 40, 110), "purple": (130, 60, 170), "pink": (240, 140, 180), "white": (245, 245, 245), "gray": (140, 140, 140),
          "black": (20, 20, 20), "brown": (120, 75, 40)}
COLOR_EVERY = 3        # 같은 추적의 상의 색 관측 간격 (장)
MIN_OBS_SHOW = 3       # 누적 관측이 이만큼 모이면 색 표시


TOP = 92               # 위쪽 띠 높이 (라벨이 띠에 안 들어가게)
LINE_H = 21


def _ov(a, b):
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))


def rect_gap(a, b):
    dx = max(b[0] - a[2], a[0] - b[2], 0); dy = max(b[1] - a[3], a[1] - b[3], 0)
    return (dx * dx + dy * dy) ** 0.5


def place(items, W, H, top, prev):
    """라벨 자리 고르기 — 박스 둘레 자리 후보 (가까운 것부터) 중 비용이 가장 낮은 곳
    비용 = 다른 라벨과 겹침 ×4 + 모든 사람 박스 (여백 6 px) 와 겹침 ×6 + 박스에서 거리 ×2 · 같은 추적의 앞 프레임 자리는 −250 (깜빡임 줄임)
    prev = {추적 번호: 박스 왼쪽 아래 기준 라벨 위치} — 갱신한다"""
    boxes = [(b[0] - 6, b[1] - 6, b[2] + 6, b[3] + 6) for b, *_ in items]
    order = sorted(range(len(items)), key=lambda j: (items[j][3] is None, items[j][0][1]))   # 추적된 사람 먼저 · 위쪽부터
    placed, res = [], [None] * len(items)
    for j in order:
        (x1, y1, x2, y2), col, lines, key, w, h = items[j]
        cands = []
        for dd in (6, 34, 70, 115, 170):
            cands += [(x1 - 3, y2 + dd), (x1 - 3, y1 - dd - h), (x2 + dd, y1), (x1 - dd - w, y1),
                      (x2 + dd, y2 + dd - h / 2), (x1 - dd - w, y2 + dd - h / 2), (x2 + dd, y1 - dd - h / 2), (x1 - dd - w, y1 - dd - h / 2)]
        if key in prev:
            cands.insert(0, (x1 + prev[key][0], y2 + prev[key][1]))
        best = None
        for c, (cx, cy) in enumerate(cands):
            cx = min(max(cx, 0), W - w); cy = min(max(cy, top + 2), H - h)
            r = (cx, cy, cx + w, cy + h)
            cost = 4 * sum(_ov(r, q) for q in placed) + 6 * sum(_ov(r, q) for q in boxes) + 2 * rect_gap(r, (x1, y1, x2, y2))
            if c == 0 and key in prev:
                cost -= 250
            if best is None or cost < best[0]:
                best = (cost, r)
        r = best[1]
        placed.append(r)
        if key:
            prev[key] = (r[0] - x1, r[1] - y2)
        res[j] = ((x1, y1, x2, y2), col, lines, key, (r[0], r[1], w, h))
    return res


def main(out, videos, start=1):
    det = PersonDetector(PACK / "best.pt", PACK / "best_trt11.3.0.99_RTX3090.engine", conf=0.20)
    prof = ColorProfiler()
    f_big, f_sm, f_lab = (ImageFont.truetype(FONT, n) for n in (32, 20, 17))
    for k, v in enumerate(videos, start):
        cap = cv2.VideoCapture(str(v)); fps = cap.get(cv2.CAP_PROP_FPS) or 30
        W, H = int(cap.get(3)), int(cap.get(4))
        dst = out / f"디관_영상_{k:02d}_전체임무.mp4"
        wr = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
        trk = StateTracker(str(POSTURE)); acc = {}; i = 0; grades = {}; lab_prev = {}
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            boxes, confs, ms = det.detect(fr)
            st = trk.update(i / fps, fr, [tuple(map(float, b)) for b in boxes], [float(c) for c in confs]) if len(boxes) else []
            stats = frame_stats(fr) if len(boxes) and i % COLOR_EVERY == 0 else None
            img = Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
            ov = Image.new("RGBA", img.size, (0, 0, 0, 0)); do = ImageDraw.Draw(ov)
            do.rectangle([0, 0, W, TOP], fill=(0, 0, 0, 150))
            d = ImageDraw.Draw(img)
            labels = []
            for b, c, (tid, s) in zip(boxes, confs, st or [(None, None)] * len(boxes)):
                x1, y1, x2, y2 = (float(t) for t in b)
                if tid and stats is not None:
                    acc.setdefault(tid, ColorAccumulator()).add(prof.profile(fr, (x1, y1, x2, y2), stats=stats))
                g = "관찰 중" if not tid or not s or not s.get("tracked") or s["observed_s"] < 3 else s["grade"]
                if tid and g != "관찰 중":
                    grades[tid] = g
                col = GRADE_COL[g]
                d.rectangle([x1 - 3, y1 - 3, x2 + 3, y2 + 3], outline=col, width=3)
                l1 = f"#{str(tid).split('.')[0]} {c:.2f}" if tid else f"{c:.2f}"
                l2 = ""
                if s:
                    p = s["posture"]; l2 = f"{POSE_KO[p['label']]} {p['p'][p['label']]:.2f}"
                    m = s["motion"]
                    if m["measured"]:
                        l2 += " · 이동" if m["moving"] else (f" · 정지 {m['still_s']}s" if m["still_s"] else " · 느림")
                cres = acc[tid].result() if tid in acc else None
                labels.append((x1, y1, x2, y2, col, g, l1, l2, cres, str(tid).split(".")[0] if tid else None))
            img = Image.alpha_composite(img.convert("RGBA"), ov).convert("RGB"); d = ImageDraw.Draw(img)
            items = []
            for x1, y1, x2, y2, col, g, l1, l2, cres, key in labels:
                lines = [(f"{l1}  {g}" if g != "관찰 중" else l1, None)]
                if l2:
                    lines.append((l2, None))
                if cres and cres["status"] == "ok" and cres["n_obs"] >= MIN_OBS_SHOW:
                    lines.append((f"상의 {KO[cres['top'][0][0]]} {cres['top'][0][1]:.2f}", cres["top"][0][0]))
                elif key:                                        # 추적 중 · 상의 화소가 충분한 관측이 아직 3번 미만
                    lines.append(("상의 판정 중", None))
                tw = max(d.textlength(t, font=f_lab) + (22 if sw else 0) for t, sw in lines) + 14
                items.append(((x1, y1, x2, y2), col, lines, key, tw, LINE_H * len(lines) + 6))
            for (x1, y1, x2, y2), col, lines, key, (lx, ly, lw, lh) in place(items, W, H, TOP, lab_prev):
                gap = rect_gap((lx, ly, lx + lw, ly + lh), (x1, y1, x2, y2))
                if gap > 10:                                     # 떨어진 라벨 → 박스와 연결선
                    bx, by = min(max((lx + lw / 2), x1), x2), min(max((ly + lh / 2), y1), y2)
                    qx, qy = min(max(bx, lx), lx + lw), min(max(by, ly), ly + lh)
                    d.line([(bx, by), (qx, qy)], fill=col, width=2)
                d.rectangle([lx, ly, lx + lw, ly + lh], fill=(20, 20, 20))
                d.rectangle([lx, ly, lx + 4, ly + lh], fill=col)
                for j, (t, sw) in enumerate(lines):
                    tx = lx + 10
                    if sw:
                        d.rectangle([tx, ly + 6 + LINE_H * j, tx + 15, ly + 21 + LINE_H * j], fill=SWATCH[sw], outline=(255, 255, 255)); tx += 22
                    d.text((tx, ly + 3 + LINE_H * j), t, font=f_lab, fill=(255, 255, 255))
            d.text((20, 8), f"사람 후보 {len(boxes)}", font=f_big, fill=(255, 255, 255))
            d.text((210, 15), f"탐지 v12_f91_swa · 임계값 0.20 · {ms:.0f} ms/장  |  추적 BoT-SORT  |  자세 = 박스 모양  |  움직임 = 카메라 움직임 뺀 속도  |  상의 색 12색 누적",
                   font=f_sm, fill=(235, 235, 235))
            x = 20
            d.text((x, 56), "박스 색 = 위급 등급", font=f_sm, fill=(235, 235, 235)); x += 190
            for g in ("URGENT", "HIGH", "CHECK", "NORMAL", "관찰 중"):
                d.rectangle([x, 60, x + 18, 78], fill=GRADE_COL[g]); x += 24
                t = g if g != "관찰 중" else "관찰 중 (추적 3 s 미만)"
                d.text((x, 56), t, font=f_sm, fill=(235, 235, 235)); x += d.textlength(t, font=f_sm) + 22
            d.text((x + 10, 56), "· 순위 보조용 — 최종 판단은 운영자", font=f_sm, fill=(190, 190, 190))
            wr.write(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)); i += 1
        wr.release()
        from collections import Counter
        cols = Counter(a.result()["top"][0][0] for a in acc.values() if a.result()["status"] == "ok" and a.result()["n_obs"] >= MIN_OBS_SHOW)
        print(dst.name, i, "장 · 추적", len({str(t).split('.')[0] for t in acc} | {str(t).split('.')[0] for t in grades}),
              "· 마지막 등급", dict(Counter(grades.values())), "· 상의 1순위", dict(cols), "· 카메라 보정 실패", trk.stats["homography_fail"], "/", trk.stats["ticks"])


if __name__ == "__main__":
    o = Path(sys.argv[1]); o.mkdir(parents=True, exist_ok=True)
    a = sys.argv[2:]; st = 1
    if a and a[0] == "--start":
        st, a = int(a[1]), a[2:]
    main(o, [Path(p) for p in a], st)
