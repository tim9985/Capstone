"""
make_fair_figs.py — 엔지니어링 페어 (11-25) 포스터 · 팜플렛용 비전 그림 (10-30)

  F1 비전 흐름 (드론 영상 → 찾기 → 따라가기 → 위급도 → 위치 · 인상착의 → 관제)        PIL · 3600×1200
  F2 핵심 숫자 카드 6장 (일반 관람객 눈높이)                                              PIL · 3000×1700
  F3 상의 색 맞히기 — 규칙 → SigLIP 2 (6개 평가셋)                                          matplotlib
  F4 같은 오탐 기준에서 놓친 사람 줄이기 (이전 모델 vs 지금 · metrics/thr_sweep.json)       matplotlib
  색 = dataviz 기본 범주 1 · 2번 (#2a78d6 이전 · #eb6834 지금 · 검증 통과) · 바탕 #fcfcfb · 글자 #0b0b0b / #52514e
  ⚠ 수치는 10-30 기준 — 실기체 수치는 11-21 동결 뒤 다시 그림 · 공개 데이터셋 사진은 넣지 않음 (약관)
실행  /home/se/miniconda3/envs/drone/bin/python make_fair_figs.py <그림 폴더>
"""
import json, math, sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

BASE = Path(__file__).resolve().parent
TOOLS = BASE.parent / "발표자료" / "_도구"
FB, FR = str(TOOLS / "NanumGothicBold.ttf"), str(TOOLS / "NanumGothic.ttf")
BG, INK, INK2, GRID = (252, 252, 251), (11, 11, 11), (82, 81, 78), "#e6e5df"
C1, C2 = "#2a78d6", "#eb6834"
RGB1, RGB2 = (42, 120, 214), (235, 104, 52)


def f(n, bold=True):
    return ImageFont.truetype(FB if bold else FR, n)


# ───────────────────────── F1 흐름 ─────────────────────────
STEPS = [
    ("드론 영상", ["카메라 1920×1080", "짐벌 45° · 16~20 m", "실시간 전송"]),
    ("사람 찾기", ["YOLO11m (직접 학습)", "누운 · 가려진 사람 포함", "한 장 20 ms"]),
    ("따라가기", ["같은 사람 = 같은 번호", "드론 흔들림 보정", "BoT-SORT"]),
    ("위급도", ["누움 · 멈춘 시간", "→ 긴급 · 높음 · 확인 · 보통", "순위만 올림"]),
    ("위치 · 인상착의", ["발끝 → 지형 → 위경도", "상의 색 12가지", "SigLIP 2 (학습 없이)"]),
    ("관제 화면", ["후보 카드 · 사진 · 지도", "최종 판단은 운영자", ""]),
]


def fig_flow(out):
    W, H = 3600, 960
    im = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(im)
    d.text((90, 70), "비전이 하는 일 — 드론 영상에서 「도움이 필요한 사람」 을 찾아 순위대로", font=f(78), fill=INK)
    d.text((92, 175), "모든 탐지를 거르지 않고 관제 화면에 올리되, 쓰러져 오래 멈춘 사람을 맨 위로 — 최종 판단은 사람이 한다", font=f(44, False), fill=INK2)
    n = len(STEPS); gap = 70; bw = (W - 180 - gap * (n - 1)) // n; by, bh = 320, 470
    for i, (t, lines) in enumerate(STEPS):
        x = 90 + i * (bw + gap)
        col = RGB2 if i == 3 else RGB1
        d.rounded_rectangle([x, by, x + bw, by + bh], radius=34, fill=(255, 255, 255), outline=col, width=7)
        d.rounded_rectangle([x, by, x + bw, by + 120], radius=34, fill=col); d.rectangle([x, by + 86, x + bw, by + 120], fill=col)
        d.ellipse([x + 22, by + 22, x + 98, by + 98], fill=(255, 255, 255))
        d.text((x + 60, by + 60), str(i + 1), font=f(48), fill=col, anchor="mm")
        d.text((x + 115, by + 60), t, font=f(50 if len(t) < 6 else 42), fill=(255, 255, 255), anchor="lm")
        for k, l in enumerate(lines):
            if l:
                d.text((x + 34, by + 170 + k * 88), l, font=f(38, False), fill=INK)
        if i < n - 1:
            ax, ay = x + bw + 8, by + bh // 2
            d.line([(ax, ay), (ax + gap - 18, ay)], fill=INK2, width=9)
            d.polygon([(ax + gap - 6, ay), (ax + gap - 34, ay - 18), (ax + gap - 34, ay + 18)], fill=INK2)
    d.text((90, by + bh + 50), "사람 찾기 20 ms · 전 과정 38~51 ms (초당 약 20장) · 서버 GPU (RTX 3090) · 관제 경유 실연: 받은 뒤 분석까지 중앙 0.58 초", font=f(40, False), fill=INK2)
    im.save(out / "F1_비전_흐름.png")


# ───────────────────────── F2 숫자 카드 ─────────────────────────
CARDS = [
    ("20.6 ms", "사진 한 장에서 사람 찾기", "1920×1080 · 타일 4장 · TensorRT · 기준 ≤ 30 ms", RGB1),
    ("+29 %p", "작은 사람을 더 찾음", "같은 헛탐지 기준 · 키 24 px 미만 · 0.26 → 0.55", RGB1),
    ("0.3~0.6 m", "사람 위치 오차 (중앙값)", "시뮬레이션 비행 · 지형 반영 · 기준 10~15 m", RGB1),
    ("0.92", "누운 사람 가려내기 (AUROC)", "실제 드론 영상 (Okutama) · 찾은 사람 중 · 드론 움직임 보정", RGB1),
    ("1.8배", "상의 색 맞히기", "6개 평가셋 평균 0.34 → 0.61 (색 균형 정확도) · 추가 학습 없이", RGB1),
    ("≤ 0.55", "사람 없는 장면 헛탐지 (건/장)", "숲 · 도로 · 시뮬레이션 · 기준 ≤ 1", RGB1),
]


def fig_cards(out):
    W, H = 3000, 1380
    im = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(im)
    d.text((80, 60), "숫자로 보는 비전", font=f(76), fill=INK)
    d.text((82, 165), "학습에 쓰지 않은 장소 · 시뮬레이션 · 실제 드론 영상으로 잰 값 (10-30) — 실기체 수치는 11월 비행 뒤 추가", font=f(38, False), fill=INK2)
    cw, ch, gx, gy = 900, 490, 60, 55; x0, y0 = 80, 270
    for i, (big, title, sub, col) in enumerate(CARDS):
        r, c = divmod(i, 3); x, y = x0 + c * (cw + gx), y0 + r * (ch + gy)
        d.rounded_rectangle([x, y, x + cw, y + ch], radius=30, fill=(255, 255, 255), outline=(225, 224, 218), width=4)
        d.rectangle([x, y + 30, x + 14, y + ch - 30], fill=col)
        d.text((x + 60, y + 60), title, font=f(46), fill=INK)
        d.text((x + 56, y + 150), big, font=f(150), fill=col)
        # 부제 줄바꿈 (폭 맞춤)
        words, lines, cur = sub.split(" "), [], ""
        for w_ in words:
            t = (cur + " " + w_).strip()
            if d.textlength(t, font=f(34, False)) > cw - 110:
                lines.append(cur); cur = w_
            else:
                cur = t
        lines.append(cur)
        for k, l in enumerate(lines[:3]):
            d.text((x + 60, y + 360 + k * 50), l, font=f(34, False), fill=INK2)
    im.save(out / "F2_숫자_카드.png")


# ───────────────────────── F3 · F4 그래프 ─────────────────────────
def charts(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm
    from matplotlib.patches import Patch
    for ff in (FR, FB):
        fm.fontManager.addfont(ff)
    plt.rcParams.update({"font.family": "NanumGothic", "axes.unicode_minus": False, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
                         "axes.edgecolor": "#c9c8c2", "axes.labelcolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e", "text.color": "#0b0b0b"})
    # F3 상의 색
    sets = ["드론\n(UAV-Human)", "드론\n(NOMAD)", "지상 CCTV\n(PETA)", "디지털관\n사진", "디지털관\n영상", "시뮬레이션\n20 m"]
    # 결정 노트 「상의 색 SigLIP 2 제로샷」 표 · UE 는 잔디 문장 넣은 최종판 (0.521 → 0.536 · _학습 큐 G2) · N 은 잔디 +0.2 %p (표기 0.42 같음)
    old = [0.286, 0.336, 0.270, 0.314, 0.422, 0.393]
    new = [0.512, 0.422, 0.560, 0.744, 0.885, 0.536]
    fig, ax = plt.subplots(figsize=(14, 6.8), dpi=220)
    w = 0.36
    for i in range(len(sets)):
        ax.bar(i - w / 2 - 0.012, old[i], w, color=C1, zorder=3)
        ax.bar(i + w / 2 + 0.012, new[i], w, color=C2, zorder=3)
        ax.text(i - w / 2 - 0.012, old[i] + 0.015, f"{old[i]:.2f}", ha="center", va="bottom", fontsize=13, color="#52514e")
        ax.text(i + w / 2 + 0.012, new[i] + 0.015, f"{new[i]:.2f}", ha="center", va="bottom", fontsize=14, fontweight="bold")
    ax.set_xticks(range(len(sets))); ax.set_xticklabels(sets, fontsize=13.5)
    ax.set_ylim(0, 1.05); ax.set_ylabel("색 균형 정확도 (12색 · 색마다 정확도 평균)", fontsize=13)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(handles=[Patch(color=C1, label="이전 — 손으로 정한 색 규칙"), Patch(color=C2, label="지금 — SigLIP 2 (그림 · 글 함께 배운 모델 · 추가 학습 없이)")],
              loc="upper left", frameon=False, fontsize=13, ncol=2, bbox_to_anchor=(0, 1.08))
    fig.suptitle("상의 색 맞히기 — 6개 평가셋 모두 올라감 (처음 보는 평가셋 3개 포함)", x=0.02, ha="left", fontsize=18, fontweight="bold", y=0.99)
    fig.text(0.02, 0.012, "「빨간 옷을 입은 사람」 처럼 찾는 색으로 후보 순위를 올리는 데 씀 (거르지 않음) · 사람 한 명 3 ms (GPU) · 풀밭 초록 · 검정↔남색 보정 포함 · 디지털관 2개는 라벨 검수 전 (잠정)",
             fontsize=11.5, color="#52514e", va="bottom")
    fig.subplots_adjust(left=0.07, right=0.98, top=0.82, bottom=0.17)
    fig.savefig(out / "F3_상의색_개선.png"); plt.close(fig)
    # F4 재현율
    D = json.load(open(BASE / "metrics" / "thr_sweep.json"))
    o_, n_ = D["soup_v7r2"], D["v12_f91_swa"]
    cats = ["숲 (가림 많음)", "숲 · 작은 사람\n(키 24 px 미만)", "비스듬 45°", "한국 수직"]
    o = [o_["test_v2"]["0.4"]["R"], o_["test_v2"]["0.4"]["크기별_R"]["0-24"], o_["test_obl"]["0.4"]["R"], o_["test_kr"]["0.4"]["R"]]
    n = [n_["test_v2"]["0.2"]["R"], n_["test_v2"]["0.2"]["크기별_R"]["0-24"], n_["test_obl"]["0.2"]["R"], n_["test_kr"]["0.2"]["R"]]
    fig, ax = plt.subplots(figsize=(12, 6.6), dpi=220)
    for i in range(len(cats)):
        ax.bar(i - w / 2 - 0.012, o[i], w, color=C1, zorder=3)
        ax.bar(i + w / 2 + 0.012, n[i], w, color=C2, zorder=3)
        ax.text(i - w / 2 - 0.012, o[i] + 0.015, f"{o[i]:.2f}", ha="center", va="bottom", fontsize=13, color="#52514e")
        ax.text(i + w / 2 + 0.012, n[i] + 0.015, f"{n[i]:.2f}", ha="center", va="bottom", fontsize=14, fontweight="bold")
        ax.text(i, max(o[i], n[i]) + 0.14, f"+{(n[i] - o[i]) * 100:.1f} %p", ha="center", fontsize=15, fontweight="bold", color="#0b0b0b")
    ax.set_xticks(range(len(cats))); ax.set_xticklabels(cats, fontsize=13.5)
    ax.set_ylim(0, 1.3); ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0]); ax.set_ylabel("재현율 (찾은 사람 비율)", fontsize=13)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(handles=[Patch(color=C1, label="이전 모델"), Patch(color=C2, label="지금 모델 (데이터 다시 짬 · 문턱 0.20)")],
              loc="upper left", frameon=False, fontsize=13, ncol=2, bbox_to_anchor=(0, 1.07))
    fig.suptitle("놓치는 사람 줄이기 — 헛탐지 기준 (사람 없는 도로 ≤ 1 건/장) 은 같게", x=0.02, ha="left", fontsize=18, fontweight="bold", y=0.99)
    fig.text(0.02, 0.012, "학습에 쓰지 않은 장소로만 잼 (숲 WiSARD · 비스듬 Okutama · 한국 AI-Hub) · 수색에선 놓치는 것이 헛탐지보다 위험 → 헛탐지 기준을 지키는 가장 낮은 문턱",
             fontsize=11.5, color="#52514e", va="bottom")
    fig.subplots_adjust(left=0.08, right=0.98, top=0.83, bottom=0.17)
    fig.savefig(out / "F4_재현율_향상.png"); plt.close(fig)


if __name__ == "__main__":
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    fig_flow(out); fig_cards(out); charts(out)
    print("그림", sorted(p.name for p in out.glob("F*.png")))
