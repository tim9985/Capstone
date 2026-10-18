"""
make_present_flow.py — 발표 자료용 UE 시연 흐름도 한 장 (10-18)

  위 줄 (명령 →): 관제 웹 → 백엔드 → SIM 연결기 → UE 시뮬 · 아래 줄 (결과 ←): 백엔드 → 비전 → 관제 웹 · 다시 운영자로
실행  /home/se/miniconda3/envs/drone/bin/python make_present_flow.py <그림 폴더>
"""
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

BASE = Path(__file__).resolve().parent
FB = str(BASE.parent / "발표자료" / "_도구" / "NanumGothicBold.ttf")
FR = str(BASE.parent / "발표자료" / "_도구" / "NanumGothic.ttf")
BG, INK, INK2 = (252, 252, 251), (11, 11, 11), (82, 81, 78)
KIND = {"web": ((42, 120, 214), "관제 (프론트)"), "srv": ((70, 70, 72), "백엔드 서버"),
        "sim": ((235, 104, 52), "시뮬레이션 (노트북)"), "vis": ((27, 150, 110), "비전 (서버 GPU)")}
W, H = 2400, 1350


def f(n, bold=True):
    return ImageFont.truetype(FB if bold else FR, n)


def box(d, x, y, w, h, kind, num, title, lines):
    col = KIND[kind][0]
    d.rounded_rectangle([x, y, x + w, y + h], radius=26, fill=(255, 255, 255), outline=col, width=6)
    d.rounded_rectangle([x, y, x + w, y + 78], radius=26, fill=col)
    d.rectangle([x, y + 52, x + w, y + 78], fill=col)
    d.ellipse([x + 18, y + 12, x + 72, y + 66], fill=(255, 255, 255))
    d.text((x + 45, y + 39), str(num), font=f(34), fill=col, anchor="mm")
    d.text((x + 88, y + 39), title, font=f(36), fill=(255, 255, 255), anchor="lm")
    for i, t in enumerate(lines):
        d.text((x + 30, y + 112 + i * 50), t, font=f(31, False), fill=INK)


def arrow(d, p, q, label=None, lpos="above", col=INK2):
    (x1, y1), (x2, y2) = p, q
    d.line([p, q], fill=col, width=7)
    import math
    a = math.atan2(y2 - y1, x2 - x1); L = 30
    d.polygon([(x2, y2), (x2 - L * math.cos(a - 0.45), y2 - L * math.sin(a - 0.45)), (x2 - L * math.cos(a + 0.45), y2 - L * math.sin(a + 0.45))], fill=col)
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        fnt = f(27)
        if lpos == "above":
            d.text((mx, my - 16), label, font=fnt, fill=INK, anchor="md")
        elif lpos == "below":
            d.text((mx, my + 16), label, font=fnt, fill=INK, anchor="ma")
        elif lpos == "right":
            d.text((mx + 22, my), label, font=fnt, fill=INK, anchor="lm")
        else:
            d.text((mx - 22, my), label, font=fnt, fill=INK, anchor="rm")


def main(out):
    im = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(im)
    d.text((70, 60), "UE 시연 흐름 — 운영자 명령에서 후보 카드까지", font=f(58), fill=INK)
    # 범례
    x = 70
    for k, (col, name) in KIND.items():
        d.rounded_rectangle([x, 160, x + 36, 196], radius=8, fill=col)
        d.text((x + 50, 178), name, font=f(28, False), fill=INK2, anchor="lm"); x += 90 + d.textlength(name, font=f(28, False))
    bw, bh = 440, 330
    xs = [60, 660, 1260, 1860]
    y1, y2 = 280, 880
    box(d, xs[0], y1, bw, bh, "web", 1, "관제 웹", ["로그인", "수색 임무 · 구역 · 경로", "인상착의 (예: 초록 상의)", "비행 명령"])
    box(d, xs[1], y1, bw, bh, "srv", 2, "백엔드", ["임무 · 경로 저장", "권한 · 로그 확인", "SIM 작업 배정"])
    box(d, xs[2], y1, bw, bh, "sim", 3, "SIM 연결기", ["작업 받아 실행", "이륙 · 경로 비행 · 착륙", "사진 + 위치 · 자세 업로드"])
    box(d, xs[3], y1, bw + 40, bh, "sim", 4, "UE 시뮬", ["금오공대 캠퍼스 3D", "드론 비행 (16~20 m · 45°)", "카메라 1080p 촬영"])
    box(d, xs[2], y2, bw, bh, "srv", 5, "백엔드", ["프레임 · 위치 저장", "분석 작업 배정", "관측 → 후보 카드"])
    box(d, xs[1], y2, bw, bh, "vis", 6, "비전", ["사람 탐지 (≈ 20 ms)", "추적 · 같은 사람 묶기", "상태 등급 · 상의 색", "좌표 (발끝 × 지형)"])
    box(d, xs[0], y2, bw, bh, "web", 7, "관제 웹", ["지도 위 후보 카드", "위급 · 인상착의 순 정렬", "스냅샷 · 알림"])
    my = y1 + bh / 2
    arrow(d, (xs[0] + bw + 8, my), (xs[1] - 10, my), "임무 · 명령", "above")
    arrow(d, (xs[1] + bw + 8, my), (xs[2] - 10, my), "비행 작업", "above")
    arrow(d, (xs[2] + bw + 8, my - 30), (xs[3] - 10, my - 30), "API 제어", "above")
    arrow(d, (xs[3] - 8, my + 30), (xs[2] + bw + 10, my + 30), "사진 · 위치", "below")
    arrow(d, (xs[2] + bw / 2, y1 + bh + 8), (xs[2] + bw / 2, y2 - 10), "사진 + 위치 · 자세 업로드", "right")
    my2 = y2 + bh / 2
    arrow(d, (xs[2] - 8, my2), (xs[1] + bw + 10, my2), "분석 작업", "above")
    arrow(d, (xs[1] - 8, my2), (xs[0] + bw + 10, my2), "후보 · 상태 · 색", "below")
    arrow(d, (xs[0] + bw / 2, y2 - 8), (xs[0] + bw / 2, y1 + bh + 10), "운영자 판단 · 다시 보기", "right", (42, 120, 214))
    d.text((70, H - 70), "사람이 있으면 사진을 받은 뒤 약 0.6 s 안에 관제 화면에 후보로 뜸 (10-17 실연 · 300장 모두 분석)", font=f(30, False), fill=INK2)
    im.save(out / "00_UE_시연_흐름도.png")


if __name__ == "__main__":
    o = Path(sys.argv[1]); o.mkdir(parents=True, exist_ok=True); main(o)
