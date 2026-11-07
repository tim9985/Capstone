"""
make_panel_tables.py — 엔지니어링 페어 판넬 표 후보 (11-07 · 사용자 「기성 모델 11개쯤과 비교 · 데이터셋별 P · R · AUROC · 여러 후보 · 무채색」)

  원자료  metrics/fair_metrics.json (fair_metrics.py · 검출 = fair_eval.py) · posture_fair.json · profile_worker.json · 상의 색 결정 노트 값
  모델    우리 v12 (운용 4조각 · 문턱 0.20) + 기성 13개 (각각 · 평가셋마다 크기 · 방식 중 AP50 최고)
  평가셋  모든 모델이 처음 보는 3곳 (UAV-SAR · Okutama · Cyprus) · 우리만 같은 데이터셋 다른 장소 학습 2곳 (WiSARD · AI-Hub — 평가 장소는 학습에서 뺌 · make_v12_data)
          · 같은 데이터셋 다른 사진 3곳 (VisDrone · NOMAD · LADD) · SARD · HERIDAL · NII-CU 는 평가 사진 · 장소가 학습과 겹쳐 뺌 (11-07 검토 반영)
  후보    A 전체 AP50 표 · B 지표별 평균 표 · C 데이터셋별 상세 · D 순위 점그림 · E 우리 모델 성능 카드
  출력    발표자료/엔지니어링페어_비전/판넬/후보_{A..E}_*.png (git 밖) · 회색조 · 우리 행만 진한 테두리
실행  /home/se/miniconda3/envs/drone/bin/python make_panel_tables.py
"""
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

BASE = Path(__file__).resolve().parent
TOOLS = BASE.parent / "발표자료" / "_도구"
OUT = BASE.parent / "발표자료" / "엔지니어링페어_비전" / "판넬"
FB, FR = str(TOOLS / "NanumGothicBold.ttf"), str(TOOLS / "NanumGothic.ttf")
INK, INK2, MUTE, RULE, BG = (24, 24, 24), (95, 95, 95), (150, 150, 150), (215, 215, 215), (255, 255, 255)
ACC = (40, 52, 66)                       # 우리 행 테두리 · 점 (짙은 회청)
M = json.load(open(BASE / "metrics" / "fair_metrics.json"))
PF = json.load(open(BASE / "metrics" / "posture_fair.json"))
PW = json.load(open(BASE / "metrics" / "profile_worker.json"))
U = ["test_sar", "test_obl", "test_cy"]; W = ["test_v2", "test_kr"]; B = ["test_visdrone", "test_nomad", "test_ladd_h"]; SETS = U + W + B
GROUPS = ((U, "모든 모델이 처음 보는 데이터셋"), (W, "같은 데이터셋 · 안 쓴 장소 (우리만 다른 장소 학습)"), (B, "같은 데이터셋 · 다른 사진"))
SN = {"test_sar": ("UAV-SAR", "드론 수색"), "test_v2": ("WiSARD", "숲"), "test_obl": ("Okutama", "비스듬히"), "test_kr": ("AI-Hub", "한국"),
      "test_cy": ("Cyprus", "아주 작은 사람"), "test_visdrone": ("VisDrone", "도시"), "test_nomad": ("NOMAD", "가림"), "test_ladd_h": ("LADD", "숲 수색")}
MODELS = [("v12_f91_swa", "우리 팀 (YOLO11m)"), ("aeronova", "AeroNova"), ("visdrone_yolov9e", "VisDrone YOLOv9e"), ("visdrone_yolo11x", "VisDrone YOLO11x"),
          ("visdrone_yolo26x", "VisDrone YOLO26x"), ("ap_mix_26l", "AerialPerson 혼합"), ("ap_sard_26l", "SARD 전용 L"), ("ap_sard_26m", "SARD 전용 M"),
          ("ap_heridal_26l", "HERIDAL 전용 L"), ("ap_heridal_26m", "HERIDAL 전용 M"), ("ap_nomad_26l", "NOMAD 전용 L"), ("ap_nomad_26m", "NOMAD 전용 M"),
          ("ap_ladd_26l", "LADD 전용 L"), ("ap_ladd_26m", "LADD 전용 M")]
TRAINED = {"visdrone": {"test_visdrone"}, "ap_mix": {"test_nomad", "test_ladd_h"}, "ap_nomad": {"test_nomad"}, "ap_ladd": {"test_ladd_h"}}
OURS = "v12_f91_swa"


def F(n, b=False):
    return ImageFont.truetype(FB if b else FR, n)


def pick(model, s):
    """평가셋 s 에서 모델의 기록 — 우리 = 운용 4조각 · 기성 = AP50 최고 방식 · P · R · F1 은 같은 방식의 최적 F1 문턱"""
    cand = {t: v for t, v in M[s].items() if t.split("@")[0] == model}
    if model == OURS:
        t = "v12_f91_swa@tile1280"; return cand[t] | {"tag": t}
    t = max(cand, key=lambda k: cand[k]["AP50"]); return cand[t] | {"tag": t}


def trained(model, s):
    return any(model.startswith(k) and s in v for k, v in TRAINED.items())


def avg(model, key, sets=U):
    return sum(pick(model, s)[key] for s in sets) / len(sets)


def gray(v, lo=0.0, hi=1.0):
    """값 → 회색 칸 (높을수록 진함) · 글자색"""
    x = max(0.0, min(1.0, (v - lo) / (hi - lo))); g = round(246 - 150 * x)
    return (g, g, g), (BG if g < 140 else INK)


def title(dr, t, sub, y=10):
    dr.text((0, y), t, font=F(64, True), fill=INK); dr.text((0, y + 92), sub, font=F(46), fill=INK2)


def notes(dr, y, lines):
    for k, t in enumerate(lines):
        dr.text((0, y + k * 60), t, font=F(40), fill=INK2)


COMMON = ["AP50: 박스가 사람과 절반 이상 겹치면 맞음으로 셈, 1이 만점 · 평가셋마다 최대 500장, 모든 모델 같은 채점 · 같은 사진이 학습에 들어간 평가셋은 없음",
          "우리 팀 = 실제 운용 방식(1080p를 4조각, 문턱 0.20) 그대로 · 기성 모델 = 크기와 처리 방식 중 가장 잘 나온 값",
          "SARD · HERIDAL · NII-CU 는 우리 학습에 같은 사진이나 장소가 들어 있어 뺐음 · * = 그 데이터셋으로 학습한 기성 모델 · 우리 운용값 (문턱 0.20) 은 성능 카드에"]


# ───────── A. 전체 AP50 표 (회색 칸) ─────────
def cand_a():
    W, NC, AC, RH, top = 3000, 620, 300, 84, 380; cw = (W - NC - AC) / len(SETS)
    order = [OURS] + sorted([m for m, _ in MODELS if m != OURS], key=lambda m: -avg(m, "AP50"))
    H = top + RH * len(order) + 260; im = Image.new("RGB", (W, H), BG); dr = ImageDraw.Draw(im)
    title(dr, "공개 사람 탐지 모델 13개와 데이터셋별 비교", "AP50 (1이 만점) · 칸이 진할수록 높음 · 굵은 글씨 = 그 평가셋 최고")
    y0 = 200; dr.line([0, y0, W, y0], fill=INK, width=4)
    gx = lambda i: NC + cw * i
    for ss, t in GROUPS:
        i0 = SETS.index(ss[0]); x0, x1 = gx(i0), gx(i0 + len(ss))
        dr.text(((x0 + x1) / 2, y0 + 34), t, font=F(34, True), fill=INK, anchor="mm"); dr.line([x0 + 10, y0 + 62, x1 - 10, y0 + 62], fill=INK2, width=2)
    for i, s in enumerate(SETS):
        a, b = SN[s]; cx = gx(i) + cw / 2
        dr.text((cx, y0 + 100), a, font=F(42, True), fill=INK, anchor="mm"); dr.text((cx, y0 + 146), b, font=F(36), fill=INK2, anchor="mm")
    dr.text((W - AC / 2, y0 + 100), "처음 보는", font=F(42, True), fill=INK, anchor="mm"); dr.text((W - AC / 2, y0 + 146), "3곳 평균", font=F(42, True), fill=INK, anchor="mm")
    dr.text((10, y0 + 122), "모델", font=F(44, True), fill=INK, anchor="lm"); dr.line([0, top - 6, W, top - 6], fill=INK, width=3)
    best = {s: max(pick(m, s)["AP50"] for m, _ in MODELS) for s in SETS}; bavg = max(avg(m, "AP50") for m, _ in MODELS)
    name = dict(MODELS)
    for r, m in enumerate(order):
        y = top + r * RH
        for i, s in enumerate(SETS):
            v = pick(m, s)["AP50"]; fill, fg = gray(v)
            dr.rectangle([gx(i) + 3, y + 3, gx(i + 1) - 3, y + RH - 3], fill=fill)
            isb = round(v, 2) == round(best[s], 2)
            dr.text((gx(i) + cw / 2, y + RH / 2), f"{v:.2f}" + ("*" if trained(m, s) else ""), font=F(44, isb), fill=fg, anchor="mm")
        a = avg(m, "AP50"); fill, fg = gray(a)
        dr.rectangle([W - AC + 12, y + 3, W - 3, y + RH - 3], fill=fill)
        dr.text((W - AC / 2 + 6, y + RH / 2), f"{a:.2f}", font=F(46, round(a, 2) == round(bavg, 2)), fill=fg, anchor="mm")
        dr.text((10, y + RH / 2), name[m], font=F(44, m == OURS), fill=INK, anchor="lm")
        if m == OURS:
            dr.rectangle([0, y, W - 1, y + RH], outline=ACC, width=6)
    yb = top + RH * len(order) + 10; dr.line([0, yb, W, yb], fill=INK, width=4)
    notes(dr, yb + 30, COMMON)
    im.save(OUT / "후보_A_전체_AP50_표.png", dpi=(450, 450)); return im.size


# ───────── B. 지표별 평균 (처음 보는 5곳) — 칸 안 막대 ─────────
def cand_b():
    cols = [("AP50", "AP50", 1.0, "사람 찾기 종합"), ("AP50_95", "AP50-95", 1.0, "박스 정확도까지"), ("P_best", "정밀도 P", 1.0, "찾은 것 중 맞은 비율"),
            ("R_best", "재현율 R", 1.0, "실제 사람 중 찾은 비율"), ("F1_best", "F1", 1.0, "P · R 균형"), ("road_fp_best", "도로 헛탐지", None, "건/장 · 낮을수록 좋음")]
    order = [OURS] + sorted([m for m, _ in MODELS if m != OURS], key=lambda m: -avg(m, "AP50"))
    W, NC, RH, top = 3000, 640, 84, 330; cw = (W - NC) / len(cols)
    H = top + RH * len(order) + 300; im = Image.new("RGB", (W, H), BG); dr = ImageDraw.Draw(im)
    title(dr, "지표별 비교 — 모든 모델이 처음 보는 데이터셋 3곳 평균", "P · R · F1 은 모델마다 F1 이 가장 높은 문턱에서 · 도로 헛탐지는 같은 문턱에서 사람 없는 도로 사진 한 장당 잘못 찾은 수")
    y0 = 196; dr.line([0, y0, W, y0], fill=INK, width=4)
    for j, (_, h, _, sub) in enumerate(cols):
        cx = NC + cw * j + cw / 2; dr.text((cx, y0 + 40), h, font=F(44, True), fill=INK, anchor="mm"); dr.text((cx, y0 + 92), sub, font=F(34), fill=INK2, anchor="mm")
    dr.text((10, y0 + 66), "모델", font=F(44, True), fill=INK, anchor="lm"); dr.line([0, top - 6, W, top - 6], fill=INK, width=3)
    name = dict(MODELS); vals = {(m, k): avg(m, k) for m in order for k, *_ in cols if k != "road_fp_best"}
    for m in order:
        rf = [pick(m, s)["road_fp_best"] for s in U]; vals[(m, "road_fp_best")] = sum(x for x in rf if x is not None) / max(1, sum(x is not None for x in rf))
    fpmax = max(vals[(m, "road_fp_best")] for m in order)
    for r, m in enumerate(order):
        y = top + r * RH; dr.text((10, y + RH / 2), name[m], font=F(44, m == OURS), fill=INK, anchor="lm")
        for j, (k, _, hi, _) in enumerate(cols):
            v = vals[(m, k)]; x0 = NC + cw * j + 20; bwmax = cw - 190
            frac = (v / hi) if hi else (v / fpmax); col = ACC if m == OURS else (150, 150, 150)
            dr.rectangle([x0, y + 24, x0 + bwmax * min(1, frac), y + RH - 24], fill=col)
            best = (min if k == "road_fp_best" else max)(vals[(mm, k)] for mm in order)
            dr.text((x0 + bwmax + 16, y + RH / 2), f"{v:.2f}", font=F(42, abs(v - best) < 1e-9), fill=INK, anchor="lm")
        if m == OURS:
            dr.rectangle([0, y, W - 1, y + RH], outline=ACC, width=6)
        dr.line([0, y + RH, W, y + RH], fill=RULE, width=1)
    yb = top + RH * len(order) + 10; dr.line([0, yb, W, yb], fill=INK, width=4)
    notes(dr, yb + 30, COMMON[:2] + ["3곳 = UAV-SAR · Okutama · Cyprus (어느 모델도 학습에 안 쓴 데이터셋) · 도로 헛탐지 = AU-AIR 사람 없는 사진 233장"])
    im.save(OUT / "후보_B_지표별_평균_표.png", dpi=(450, 450)); return im.size


# ───────── C. 데이터셋별 상세 (2 × 4 작은 표) ─────────
def cand_c():
    W = 3000; cols, rows_n = 4, 2; pw = (W - 60 * (cols - 1)) / cols; ph = 640; top = 200
    H = top + rows_n * (ph + 60) + 230; im = Image.new("RGB", (W, H), BG); dr = ImageDraw.Draw(im)
    title(dr, "데이터셋별 상세 — 우리 팀과 그 데이터셋 상위 기성 모델 4개", "AP50 · 정밀도 P · 재현율 R · F1 (P · R · F1 = 모델마다 F1 최고 문턱) · 굵은 글씨 = 그 칸 최고")
    name = dict(MODELS)
    for k, s in enumerate(SETS):
        X = (k % cols) * (pw + 60); Y = top + (k // cols) * (ph + 60)
        a, b = SN[s]; n_gt = pick(OURS, s)["n_gt"]
        dr.text((X, Y), f"{a}", font=F(48, True), fill=INK); dr.text((X + dr.textlength(a, font=F(48, True)) + 18, Y + 8), f"{b} · 사람 {n_gt:,}명", font=F(36), fill=INK2)
        dr.line([X, Y + 72, X + pw, Y + 72], fill=INK, width=3)
        ms = [OURS] + sorted([m for m, _ in MODELS if m != OURS], key=lambda m: -pick(m, s)["AP50"])[:4]
        heads = ["AP50", "P", "R", "F1"]; keys = ["AP50", "P_best", "R_best", "F1_best"]; nc = pw * 0.52; cw = (pw - nc) / 4
        for j, h in enumerate(heads):
            dr.text((X + nc + cw * j + cw / 2, Y + 108), h, font=F(36, True), fill=INK2, anchor="mm")
        best = {kk: max(pick(m, s)[kk] for m in ms) for kk in keys}
        for r, m in enumerate(ms):
            yy = Y + 150 + r * 92; nm = name[m].replace(" (YOLO11m)", "")
            if m == OURS:
                dr.rectangle([X, yy - 4, X + pw, yy + 80], outline=ACC, width=5)
            dr.text((X + 12, yy + 38), nm + ("*" if trained(m, s) else ""), font=F(33, m == OURS), fill=INK, anchor="lm")
            for j, kk in enumerate(keys):
                v = pick(m, s)[kk]; dr.text((X + nc + cw * j + cw / 2, yy + 38), f"{v:.2f}", font=F(37, abs(v - best[kk]) < 1e-9), fill=INK, anchor="mm")
        dr.line([X, Y + 150 + 5 * 92 + 6, X + pw, Y + 150 + 5 * 92 + 6], fill=INK, width=2)
        if k == 0:
            dr.text((X, Y - 0), "", font=F(10))
    yb = top + rows_n * (ph + 60) - 20
    dr.text((0, yb - 10), "UAV-SAR · Okutama · Cyprus = 모든 모델이 처음 봄 · WiSARD · AI-Hub = 우리만 같은 데이터셋의 다른 장소 학습 · VisDrone · NOMAD · LADD = 같은 데이터셋 다른 사진", font=F(40, True), fill=INK)
    notes(dr, yb + 60, COMMON)
    im.save(OUT / "후보_C_데이터셋별_상세.png", dpi=(450, 450)); return im.size


# ───────── D. 순위 점그림 (처음 보는 5곳) ─────────
def cand_d():
    order = sorted([m for m, _ in MODELS], key=lambda m: -avg(m, "AP50")); name = dict(MODELS)
    W, NC, RH, top = 3000, 620, 92, 300; X0, X1 = NC + 40, W - 120
    H = top + RH * len(order) + 330; im = Image.new("RGB", (W, H), BG); dr = ImageDraw.Draw(im)
    title(dr, "모든 모델이 처음 보는 데이터셋 3곳의 AP50 — 모델 순위", "막대 = 3곳 평균 · 점 = 평가셋 하나 (모양으로 구분) · 오른쪽일수록 좋음")
    xs = lambda v: X0 + (X1 - X0) * v
    shapes = {"test_sar": "o", "test_obl": "^", "test_cy": "x"}
    for t in [0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        dr.line([xs(t), top - 20, xs(t), top + RH * len(order)], fill=RULE, width=2); dr.text((xs(t), top - 40), f"{t:.1f}", font=F(36), fill=INK2, anchor="mm")
    dr.line([xs(0.8), top - 20, xs(0.8), top + RH * len(order)], fill=INK2, width=3)
    dr.text((xs(0.8) + 10, top + RH * len(order) + 30), "요구 기준 0.80", font=F(36), fill=INK2)

    def mark(x, y, sh, col, r=13):
        if sh == "o": dr.ellipse([x - r, y - r, x + r, y + r], outline=col, width=5)
        elif sh == "s": dr.rectangle([x - r, y - r, x + r, y + r], outline=col, width=5)
        elif sh == "^": dr.polygon([(x, y - r - 2), (x - r - 2, y + r), (x + r + 2, y + r)], outline=col, width=5)
        elif sh == "d": dr.polygon([(x, y - r - 3), (x - r - 3, y), (x, y + r + 3), (x + r + 3, y)], outline=col, width=5)
        else: dr.line([x - r, y - r, x + r, y + r], fill=col, width=5); dr.line([x - r, y + r, x + r, y - r], fill=col, width=5)
    for r, m in enumerate(order):
        y = top + r * RH + RH / 2; col = ACC if m == OURS else (120, 120, 120)
        dr.text((10, y), name[m], font=F(42, m == OURS), fill=INK, anchor="lm")
        a = avg(m, "AP50"); dr.rectangle([X0, y - 10, xs(a), y + 10], fill=(225, 225, 225) if m != OURS else (170, 180, 190))
        for s in U:
            mark(xs(pick(m, s)["AP50"]), y, shapes[s], col)
        dr.text((xs(a) + 0, y - 38), f"{a:.2f}", font=F(34, m == OURS), fill=INK, anchor="mm")
    ly = top + RH * len(order) + 100; lx = 10
    for s in U:
        mark(lx + 16, ly + 20, shapes[s], INK); dr.text((lx + 46, ly + 20), f"{SN[s][0]} ({SN[s][1]})", font=F(38), fill=INK, anchor="lm"); lx += 700
    notes(dr, ly + 80, COMMON[:2])
    im.save(OUT / "후보_D_순위_점그림.png", dpi=(450, 450)); return im.size


# ───────── E. 우리 모델 성능 카드 ─────────
def cand_e():
    W = 3000; im = Image.new("RGB", (W, 1900), BG); dr = ImageDraw.Draw(im)
    title(dr, "우리 팀 비전 성능 한눈에", "탐지 = 실제 운용 방식 (1080p 4조각 · 문턱 0.20) · 자세 · 처리 시간 · 상의 색은 아래")
    heads = ["평가셋", "AP50", "AP50-95", "정밀도 P", "재현율 R", "F1", "사람 수"]; cwid = [760, 340, 340, 340, 340, 340, 540]
    xs = [sum(cwid[:i]) for i in range(len(cwid) + 1)]; y = 200; dr.line([0, y, W, y], fill=INK, width=4)
    for j, h in enumerate(heads):
        dr.text(((xs[j] + 20) if j == 0 else (xs[j] + xs[j + 1]) / 2, y + 50), h, font=F(44, True), fill=INK, anchor="lm" if j == 0 else "mm")
    y += 100; dr.line([0, y, W, y], fill=INK, width=2)
    for ss, grp in GROUPS:
        dr.text((20, y + 34), grp, font=F(38, True), fill=INK2, anchor="lm"); y += 68
        for s in ss:
            d = pick(OURS, s); P, R = d["P20"], d["R20"]; F1 = 2 * P * R / (P + R) if P + R else 0
            vals = [f"{SN[s][0]} ({SN[s][1]})", f"{d['AP50']:.2f}", f"{d['AP50_95']:.2f}", f"{P:.2f}", f"{R:.2f}", f"{F1:.2f}", f"{d['n_gt']:,}명"]
            for j, v in enumerate(vals):
                if 1 <= j <= 5:
                    fill, fg = gray(float(v)); dr.rectangle([xs[j] + 8, y + 6, xs[j + 1] - 8, y + 70], fill=fill)
                else:
                    fg = INK
                dr.text(((xs[j] + 20) if j == 0 else (xs[j] + xs[j + 1]) / 2, y + 38), v, font=F(42), fill=fg, anchor="lm" if j == 0 else "mm")
            y += 78
        dr.line([0, y + 4, W, y + 4], fill=RULE, width=2); y += 14
    dr.line([0, y, W, y], fill=INK, width=4); y += 50
    po, pa, pl = PF["okutama_obl"], PF["archangel_real"], PF["pipeline_okutama"]
    S = PW["stages"]; pipe = S["탐지"]["median_ms"] + S["추적 + 상태"]["median_ms"] + S["좌표 (사람 전부)"]["median_ms"]
    box = [("누운 사람 가려내기 (AUROC)", [f"자세 단계 · Okutama {po['AUROC']:.2f} [{po['CI'][0]:.2f}–{po['CI'][1]:.2f}]",
                                         f"자세 단계 · Archangel {pa['AUROC']:.2f} [{pa['CI'][0]:.2f}–{pa['CI'][1]:.2f}]",
                                         f"탐지까지 이은 실제 영상 {pl['AUROC']:.2f} [{pl['CI'][0]:.2f}–{pl['CI'][1]:.2f}]"]),
           ("처리 속도 · 헛탐지", [f"1080p 한 장 사람 찾기 {S['탐지']['median_ms']:.1f} ms (기준 30 ms 이하)",
                                 f"자세 · 위급도까지 {pipe:.0f} ms, 1초에 약 {1000 / pipe:.0f}장",
                                 f"사람 없는 도로 헛탐지 {pick(OURS, 'test_sar')['road_fp20']:.2f} 건/장 (기준 1 이하)"]),
           ("상의 색 12가지 (처음 본 PETA)", ["손으로 정한 색 규칙 0.27", "그림 · 글 함께 배운 모델 (SigLIP 2) 0.56", "추가 학습 없음 · 사람 한 명 3 ms"])]
    bw = (W - 80) / 3
    for k, (h, lines) in enumerate(box):
        X = k * (bw + 40); dr.rectangle([X, y, X + bw, y + 330], outline=INK2, width=2)
        dr.text((X + 30, y + 50), h, font=F(44, True), fill=INK, anchor="lm")
        for i, t in enumerate(lines):
            dr.text((X + 30, y + 130 + i * 66), t, font=F(38), fill=INK, anchor="lm")
    y += 380
    notes(dr, y, ["AUROC: 누운 사람을 다른 자세보다 위로 올리는 능력 (0.5 = 아무렇게나, 1 = 완벽) · [ ] = 95 % 구간",
                  "P · R · F1 = 문턱 0.20 (운용값) · AP50-95 = 겹침 기준을 0.50~0.95 로 바꿔 가며 잰 평균 (박스를 얼마나 정확히 그리나)"])
    im = im.crop((0, 0, W, y + 140)); im.save(OUT / "후보_E_우리_성능_카드.png", dpi=(450, 450)); return im.size


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for f in (cand_a, cand_b, cand_c, cand_d, cand_e):
        print(f.__name__, f())
