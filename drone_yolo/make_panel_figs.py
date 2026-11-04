"""
make_panel_figs.py — 엔지니어링 페어 판넬 (책자 · A4 한 쪽 · 그림 가로 8~9 cm) 비전 그림 (11-04 · 마감 이번 주말 · 실기체 전 → 시뮬레이션)

  ①  판넬_비전_1_위급순서.png  UE S3 (16 m · 45°) 한 장면 · worker 1.3 출력 그대로 (등급 · 멈춘 시간) · 위 = 전체 화면 · 아래 = 4명 확대
       장면 = 누운 사람이 「높음」 인 장면 중 걷는 사람 「보통」 · 오탐 0 (S3_a16 v13 run 00373) · 표시는 run JSON 값 그대로 (손으로 안 바꿈)
  ③  판넬_비전_2_숫자.png      공정 비교 (metrics/fair_eval.json · 처음 보는 5곳 · 같은 도로 헛탐지) + 처리 시간 (profile_worker.json)
  인쇄 기준  가로 2400 px (9 cm 에 약 680 dpi) · 글씨 ≥ 54 px (9 cm 인쇄 시 ≥ 7 pt)
실행  /home/se/miniconda3/envs/drone/bin/python make_panel_figs.py <S3_a16 v13 run json>
"""
import json, sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

BASE = Path(__file__).resolve().parent
TOOLS = BASE.parent / "발표자료" / "_도구"
OUT = BASE.parent / "발표자료" / "엔지니어링페어_비전" / "판넬"
FB, FR = str(TOOLS / "NanumGothicBold.ttf"), str(TOOLS / "NanumGothic.ttf")
INK, INK2, BG = (11, 11, 11), (82, 81, 78), (255, 255, 255)
OURS, EXT = (42, 120, 214), (161, 159, 152)
GRADE = {"URGENT": ("긴급", (220, 40, 40), (255, 255, 255)), "HIGH": ("높음", (245, 130, 30), INK),
         "CHECK": ("확인", (240, 205, 50), INK), "NORMAL": ("보통", (70, 190, 110), INK)}
POSE = {"lying": "누워서", "crouching": "웅크린 채", "sitting": "앉은 채", "standing": "선 채", "walking": "걷는 중", "crawling": "기는 중"}


def F(n, bold=True):
    return ImageFont.truetype(FB if bold else FR, n)


def tw(dr, s, f):
    return dr.textbbox((0, 0), s, font=f)[2]


def fig1(run_json, frame="00373", picks=("Person_Lying_02", "Person_Crouching_01", "Person_Lying_01", "Person_Walking_01")):
    d = json.load(open(run_json)); A = d["actors"]
    fr = [x for x in d["frames"] if x["frame"] == frame][0]
    obs = {}
    for o in fr["obs"]:
        a = o.get("actor")
        if a and (a not in obs or o["conf"] > obs[a]["conf"]):
            obs[a] = o
    src = Image.open(BASE / "ue_flight" / "S3_a16" / "frames" / f"{frame}.jpg").convert("RGB")
    W = 2400; s = W / src.width; top = src.resize((W, round(src.height * s)), Image.LANCZOS); dr = ImageDraw.Draw(top)
    # 전체 화면 위 박스 · 번호 (4명만 · 나머지 사람도 박스는 그림 = 실제 출력)
    for a, o in obs.items():
        if a in picks:
            continue
        x0, y0, x1, y1 = [v * s for v in o["bbox"]]
        dr.rectangle([x0 - 6, y0 - 6, x1 + 6, y1 + 6], outline=GRADE[o["grade"]][1], width=6)
    for k, a in enumerate(picks, 1):
        o = obs[a]; x0, y0, x1, y1 = [v * s for v in o["bbox"]]; col = GRADE[o["grade"]][1]
        dr.rectangle([x0 - 8, y0 - 8, x1 + 8, y1 + 8], outline=col, width=9)
        f = F(64); bx, by = x1 + 14, y0 - 30
        dr.rounded_rectangle([bx, by, bx + 70, by + 78], 12, fill=col); dr.text((bx + 35, by + 39), str(k), font=f, fill=GRADE[o["grade"]][2], anchor="mm")
    tag = "시뮬레이션 · 고도 16 m · 카메라 45°"; f = F(54)
    dr.rounded_rectangle([24, 24, 24 + tw(dr, tag, f) + 48, 24 + 86], 16, fill=(0, 0, 0, 160)); dr.text((48, 67), tag, font=f, fill=(255, 255, 255), anchor="lm")
    # 아래 확대 4칸
    n = len(picks); gap = 24; cw = (W - gap * (n - 1)) // n; ch = round(cw * 0.75); lab_h = 230
    canvas = Image.new("RGB", (W, top.height + gap + ch + lab_h), BG); canvas.paste(top, (0, 0)); dc = ImageDraw.Draw(canvas)
    for k, a in enumerate(picks):
        o = obs[a]; x0, y0, x1, y1 = o["bbox"]; cx, cy = (x0 + x1) / 2, (y0 + y1) / 2; half_w = max(x1 - x0, (y1 - y0) / 0.75) * 1.4
        half_w = max(half_w, 90); box = (cx - half_w, cy - half_w * 0.75, cx + half_w, cy + half_w * 0.75)
        tile = src.crop(tuple(round(v) for v in box)).resize((cw, ch), Image.LANCZOS)
        X = k * (cw + gap); Y = top.height + gap; canvas.paste(tile, (X, Y))
        name, col, fg = GRADE[o["grade"]]
        dc.rectangle([X, Y, X + cw - 1, Y + ch - 1], outline=col, width=10)
        dc.rounded_rectangle([X + 14, Y + 14, X + 92, Y + 100], 12, fill=col); dc.text((X + 53, Y + 57), str(k + 1), font=F(64), fill=fg, anchor="mm")
        dc.rectangle([X, Y + ch, X + cw - 1, Y + ch + 96], fill=col); dc.text((X + cw / 2, Y + ch + 48), name, font=F(70), fill=fg, anchor="mm")
        pose = POSE.get(A[a]["pose_raw"], ""); st = o["still_s"]
        line = f"{pose} {st}초 멈춤" if A[a]["pose_raw"] in ("lying", "crouching", "sitting") else pose
        if A[a]["pose_raw"] == "lying" and o["grade"] == "CHECK":
            line = f"누운 지 {st}초 (지켜봄)"
        dc.text((X + cw / 2, Y + ch + 96 + 66), line, font=F(56, bold=False), fill=INK, anchor="mm")
    OUT.mkdir(parents=True, exist_ok=True); canvas.save(OUT / "판넬_비전_1_위급순서.png", dpi=(680, 680))
    print("①", canvas.size, {a: (obs[a]["grade"], obs[a]["still_s"]) for a in picks})


def fig2():
    FE = json.load(open(BASE / "metrics" / "fair_eval.json")); PW = json.load(open(BASE / "metrics" / "profile_worker.json"))
    cm = FE["confusion_U"]; R_o = cm["ours"][0] / sum(cm["ours"][:2]); R_e = cm["ext"][0] / sum(cm["ext"][:2])
    hp = FE["height_pooled_U"]; s_o, s_e = hp["ours"]["0-24"]["R"], hp["ext"]["0-24"]["R"]; ms = PW["stages"]["탐지"]["median_ms"]
    W, H = 2400, 1320; im = Image.new("RGB", (W, H), BG); dr = ImageDraw.Draw(im)
    cols = [(60, 760), (840, 1560), (1640, 2340)]
    # 1 · 2 = 막대 쌍 · 3 = 큰 숫자
    for (x0, x1), (title, sub, a, b) in zip(cols[:2], [("찾은 사람", "헛탐지가 같을 때", R_o, R_e), ("작은 사람", "키 24 px 미만", s_o, s_e)]):
        dr.text((x0, 70), title, font=F(84), fill=INK); dr.text((x0, 180), sub, font=F(58, False), fill=INK2)
        base = 1010; top = 330; hmax = base - top - 70; bw = 230
        for k, (v, col, lab) in enumerate(((a, OURS, "우리"), (b, EXT, "공개 모델"))):
            bx = x0 + 40 + k * (bw + 110); h = hmax * v
            dr.rounded_rectangle([bx, base - h, bx + bw, base], 18, fill=col)
            dr.text((bx + bw / 2, base - h - 12), f"{v * 100:.0f}%", font=F(76 if k == 0 else 66), fill=INK if k == 0 else INK2, anchor="mb")
            dr.text((bx + bw / 2, base + 24), lab, font=F(56, k == 0), fill=INK if k == 0 else INK2, anchor="mt")
        dr.line([x0, base, x1 - 40, base], fill=(200, 199, 192), width=4)
    x0 = cols[2][0]
    dr.text((x0, 70), "처리 속도", font=F(84), fill=INK); dr.text((x0, 180), "1080p 한 장에서 사람 찾기", font=F(58, False), fill=INK2)
    dr.text((x0, 560), f"{ms / 1000:.2f}초", font=F(230), fill=OURS, anchor="lm")
    dr.text((x0, 790), "초당 약 25장", font=F(64), fill=INK)
    dr.text((x0, 880), "추적 · 위급도까지 포함", font=F(56, False), fill=INK2)
    dr.text((60, H - 60), "학습에 안 쓴 장소 5곳 · 공개 모델 13개 중 장소마다 가장 좋은 것과 같은 조건 비교", font=F(50, False), fill=INK2, anchor="lm")
    im.save(OUT / "판넬_비전_2_숫자.png", dpi=(680, 680)); print("③", im.size, round(R_o, 3), round(R_e, 3), s_o, s_e, ms)


if __name__ == "__main__":
    fig1(sys.argv[1]); fig2()
