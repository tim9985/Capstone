"""
make_panel_figs.py — 엔지니어링 페어 판넬 (책자 · A4 한 쪽 · 그림 가로 8~9 cm) 비전 그림 (11-04 · 마감 이번 주말 · 실기체 전 → 시뮬레이션)

  11-04 2판 (사용자 「AI 느낌 · 뭐가 뭘 뜻하는지 모름」) — 대시보드 꾸밈 (둥근 딱지 · 색 띠 · 큰 숫자) 빼고 보고서 그림처럼
  ①  판넬_비전_1_위급순서.png  왼쪽 = 드론 화면 전체 (가 · 나 · 다 · 라) + 읽는 법 · 오른쪽 = 찾은 사람 4명을 한 줄씩 말로
       UE S3 (16 m · 45°) worker 1.3 출력 그대로 (등급 · 멈춘 시간 · S3_a16 v13 run 00373) · 손으로 바꾼 값 없음
  ②  판넬_비전_2_비교.png      공개 모델과 같은 조건 비교 표 (metrics/fair_eval.json) + 처리 시간 (profile_worker.json)
  인쇄 기준  가로 2400 px ≈ 9 cm · 본문 글씨 ≥ 56 px (≈ 6.5 pt)
실행  /home/se/miniconda3/envs/drone/bin/python make_panel_figs.py <S3_a16 v13 run json>
"""
import json, sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

BASE = Path(__file__).resolve().parent
TOOLS = BASE.parent / "발표자료" / "_도구"
OUT = BASE.parent / "발표자료" / "엔지니어링페어_비전" / "판넬"
FB, FR = str(TOOLS / "NanumGothicBold.ttf"), str(TOOLS / "NanumGothic.ttf")
INK, INK2, RULE, BG = (20, 20, 20), (90, 90, 90), (200, 200, 200), (255, 255, 255)
# 상자 색 = 관제 화면 · worker 표시와 같은 위급도 색
GRADE = {"URGENT": ("긴급", (215, 40, 40)), "HIGH": ("높음", (240, 125, 20)), "CHECK": ("확인", (230, 190, 30)), "NORMAL": ("보통", (50, 160, 80))}
TAG = "가나다라"


def F(n, bold=False):
    return ImageFont.truetype(FB if bold else FR, n)


def describe(pose, grade, still):
    """사람마다 두 줄 — 무엇을 봤나 / 그래서 어떻게 판단했나 (값은 worker 출력)"""
    if pose == "lying" and grade == "CHECK":
        return f"방금 누움 ({still}초째)", "→ 확인 (10초 지켜봄)"
    what = {"lying": f"누운 채 {still}초째 멈춤", "crouching": f"웅크린 채 {still}초째 멈춤", "sitting": f"앉은 채 {still}초째 멈춤",
            "walking": "걷고 있음", "standing": "서 있음", "crawling": "기어가는 중"}[pose]
    return what, f"→ 위급도 {GRADE[grade][0]}"


def fig1(run_json, frame="00373", picks=("Person_Lying_02", "Person_Crouching_01", "Person_Lying_01", "Person_Walking_01")):
    d = json.load(open(run_json)); A = d["actors"]
    fr = [x for x in d["frames"] if x["frame"] == frame][0]
    obs = {}
    for o in fr["obs"]:
        a = o.get("actor")
        if a and (a not in obs or o["conf"] > obs[a]["conf"]):
            obs[a] = o
    src = Image.open(BASE / "ue_flight" / "S3_a16" / "frames" / f"{frame}.jpg").convert("RGB")
    W, H = 2400, 1250; LW = 1320; im = Image.new("RGB", (W, H), BG); dr = ImageDraw.Draw(im)
    # 왼쪽 위 — 드론 화면 전체
    s = LW / src.width; ov = src.resize((LW, round(src.height * s)), Image.LANCZOS); od = ImageDraw.Draw(ov)
    for a, o in obs.items():
        x0, y0, x1, y1 = [v * s for v in o["bbox"]]; col = GRADE[o["grade"]][1]
        od.rectangle([x0 - 4, y0 - 4, x1 + 4, y1 + 4], outline=col, width=5 if a in picks else 3)
    for k, a in enumerate(picks):
        x0, y0, x1, y1 = [v * s for v in obs[a]["bbox"]]; t = TAG[k]; f = F(46, True)
        tx, ty = (x0 - 64, y0 - 20) if a == "Person_Lying_02" else (x1 + 10, y0 - 46)
        od.rectangle([tx, ty, tx + 54, ty + 54], fill=BG, outline=INK, width=2); od.text((tx + 27, ty + 27), t, font=f, fill=INK, anchor="mm")
    od.rectangle([0, 0, 380, 64], fill=BG); od.text((16, 32), "시뮬레이션 화면", font=F(44, True), fill=INK, anchor="lm")
    im.paste(ov, (0, 0)); dr.rectangle([0, 0, LW - 1, ov.height - 1], outline=INK, width=2)
    # 왼쪽 아래 — 읽는 법
    y = ov.height + 46
    dr.text((0, y), "읽는 법", font=F(60, True), fill=INK); y += 92
    dr.text((0, y), "네모는 프로그램이 찾은 사람, 네모 색은 위급도입니다.", font=F(52), fill=INK); y += 86
    order = ["URGENT", "HIGH", "CHECK", "NORMAL"]; note = {"URGENT": "가장 먼저", "HIGH": "먼저 확인", "CHECK": "지켜보는 중", "NORMAL": "움직이는 사람"}
    for i, g in enumerate(order):
        cx = (i % 2) * 660; cy = y + (i // 2) * 84
        dr.rectangle([cx, cy + 8, cx + 52, cy + 60], outline=GRADE[g][1], width=8)
        dr.text((cx + 76, cy + 34), f"{GRADE[g][0]} ({note[g]})", font=F(52), fill=INK, anchor="lm")
    y += 2 * 84 + 26
    dr.text((0, y), "위급도는 자세와, 움직이지 않은 시간으로 정합니다.", font=F(52), fill=INK2)
    # 오른쪽 — 찾은 사람 한 줄씩
    X0 = LW + 60; rh = (H - 20) // len(picks); cs = rh - 44
    for k, a in enumerate(picks):
        o = obs[a]; Y = k * rh + 10; col = GRADE[o["grade"]][1]
        x0, y0, x1, y1 = o["bbox"]; cx, cy = (x0 + x1) / 2, (y0 + y1) / 2; r = max(x1 - x0, y1 - y0) * 0.8 + 16
        crop = src.crop((round(cx - r), round(cy - r), round(cx + r), round(cy + r))).resize((cs, cs), Image.LANCZOS)
        im.paste(crop, (X0, Y + 12)); dr.rectangle([X0, Y + 12, X0 + cs - 1, Y + 12 + cs - 1], outline=col, width=7)
        tx = X0 + cs + 34
        l1, l2 = describe(A[a]["pose_raw"], o["grade"], o["still_s"])
        dr.text((tx, Y + rh / 2 - 44), f"{TAG[k]}   {l1}", font=F(54, True), fill=INK, anchor="lm")
        dr.text((tx, Y + rh / 2 + 44), l2, font=F(54), fill=INK, anchor="lm")
        if k < len(picks) - 1:
            dr.line([X0, Y + rh, W, Y + rh], fill=RULE, width=2)
    OUT.mkdir(parents=True, exist_ok=True); im.save(OUT / "판넬_비전_1_위급순서.png", dpi=(680, 680))
    print("①", im.size, {a: (obs[a]["grade"], obs[a]["still_s"]) for a in picks})


def fig2():
    FE = json.load(open(BASE / "metrics" / "fair_eval.json")); PW = json.load(open(BASE / "metrics" / "profile_worker.json"))
    cm = FE["confusion_U"]; R_o = cm["ours"][0] / sum(cm["ours"][:2]); R_e = cm["ext"][0] / sum(cm["ext"][:2])
    hp = FE["height_pooled_U"]; s_o, s_e = hp["ours"]["0-24"]["R"], hp["ext"]["0-24"]["R"]; ms = PW["stages"]["탐지"]["median_ms"]
    S = PW["stages"]; pipe = S["탐지"]["median_ms"] + S["추적 + 상태"]["median_ms"] + S["좌표 (사람 전부)"]["median_ms"]
    OURS, EXT = (31, 58, 95), (190, 190, 190)
    W, H = 2400, 1060; im = Image.new("RGB", (W, H), BG); dr = ImageDraw.Draw(im)
    dr.text((0, 14), "공개된 AI 모델과 같은 조건에서 비교", font=F(64, True), fill=INK)
    dr.text((0, 106), "학습에 쓰지 않은 장소 5곳의 드론 사진으로, 공개 모델 13개 중 장소마다 가장 잘한 모델과 비교", font=F(48), fill=INK2)
    # 범례
    ly = 196
    for i, (c, t) in enumerate(((OURS, "우리 팀"), (EXT, "공개 모델 (장소마다 가장 잘한 것)"))):
        lx = i * 420
        dr.rectangle([lx, ly + 10, lx + 44, ly + 54], fill=c); dr.text((lx + 62, ly + 32), t, font=F(48), fill=INK, anchor="lm")
    dr.line([0, 286, W, 286], fill=INK, width=3)
    LBL = 980; BX0, BX1 = LBL + 20, W - 260
    rows = [("실제 사람을 찾아낸 비율", "잘못 찾는 횟수(헛탐지)를 같게 맞춘 뒤 비교", R_o, R_e),
            ("멀리 있어 아주 작게 보이는 사람", "화면 세로 길이의 2%보다 작게 보이는 사람", s_o, s_e)]
    y = 316
    for title, note, a, b in rows:
        dr.text((0, y + 8), title, font=F(56, True), fill=INK); dr.text((0, y + 90), note, font=F(44), fill=INK2)
        for k, (v, c) in enumerate(((a, OURS), (b, EXT))):
            by = y + 14 + k * 92; bw = (BX1 - BX0) * v
            dr.rectangle([BX0, by, BX0 + bw, by + 64], fill=c)
            dr.text((BX0 + bw + 22, by + 32), f"{v * 100:.0f}%", font=F(56, k == 0), fill=INK, anchor="lm")
        y += 250; dr.line([0, y - 22, W, y - 22], fill=RULE, width=2)
    dr.text((0, y + 8), "영상 한 장 처리 시간", font=F(56, True), fill=INK)
    dr.text((0, y + 90), "1080p 화면 한 장에서 사람을 찾는 데 걸린 시간", font=F(44), fill=INK2)
    dr.text((BX0, y + 30), f"{ms / 1000:.2f}초", font=F(64, True), fill=INK)
    dr.text((BX0 + 270, y + 44), f"(자세와 위급도 판단까지 하면 1초에 약 {1000 / pipe:.0f}장)", font=F(48), fill=INK2)
    dr.line([0, y + 170, W, y + 170], fill=INK, width=3)
    im.save(OUT / "판넬_비전_2_비교.png", dpi=(680, 680)); print("②", im.size, round(R_o, 3), round(R_e, 3), s_o, s_e, ms, round(pipe, 1))


if __name__ == "__main__":
    fig1(sys.argv[1]); fig2()
    old = OUT / "판넬_비전_2_숫자.png"
    if old.exists():
        old.rename(OUT / "지난판_판넬_비전_2_숫자.png")
