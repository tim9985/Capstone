"""
make_present_charts.py — 발표 자료용 그래프 2개 (10-17)

  10 탐지 AP50: 우리 (학습에 없던 장소) vs 논문 기준선 (같은 데이터 안에서 나눔) · 요구 기준 0.80
  11 프레임 간격 영향 (UE 8회 · metrics/ue_frame_interval.json): 추적 IDF1 · 누운 사람 URGENT/HIGH 도달
  색 = dataviz 기준 팔레트 1 · 2번 (#2a78d6 · #eb6834 · 검증 통과 · 밝은 면 #fcfcfb)
실행  /home/se/miniconda3/envs/drone/bin/python make_present_charts.py <그림 폴더>
"""
import json, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

BASE = Path(__file__).resolve().parent
FONTS = BASE.parent / "발표자료" / "_도구"
for f in ("NanumGothic.ttf", "NanumGothicBold.ttf"):
    fm.fontManager.addfont(str(FONTS / f))
plt.rcParams.update({"font.family": "NanumGothic", "axes.unicode_minus": False, "figure.facecolor": "#fcfcfb",
                     "axes.facecolor": "#fcfcfb", "axes.edgecolor": "#c9c8c2", "axes.labelcolor": "#52514e",
                     "xtick.color": "#52514e", "ytick.color": "#52514e", "text.color": "#0b0b0b"})
S1, S2, INK2, GRID = "#2a78d6", "#eb6834", "#52514e", "#e6e5df"


def chart_ap50(out):
    cats = ["숲 (WiSARD)", "비스듬 45° (Okutama)", "한국 수직 (AI-Hub)"]
    ours = [0.659, 0.616, 1.000]
    paper = [0.892, 0.723, None]
    fig, ax = plt.subplots(figsize=(12, 6.2), dpi=150)
    x = range(len(cats)); w = 0.36
    for i in x:
        ax.bar(i - w / 2 - 0.01, ours[i], w, color=S1, zorder=3)
        ax.text(i - w / 2 - 0.01, ours[i] + 0.015, f"{ours[i]:.3f}", ha="center", va="bottom", fontsize=13, color="#0b0b0b")
        if paper[i] is not None:
            ax.bar(i + w / 2 + 0.01, paper[i], w, color=S2, zorder=3)
            ax.text(i + w / 2 + 0.01, paper[i] + 0.015, f"{paper[i]:.3f}", ha="center", va="bottom", fontsize=13, color="#0b0b0b")
        else:
            ax.text(i + w / 2 + 0.01, 0.04, "논문\n기준선\n없음", ha="center", va="bottom", fontsize=11, color=INK2)
    ax.axhline(0.80, color="#0b0b0b", lw=1.2, zorder=4)
    ax.text(-0.53, 0.812, "요구 기준 0.80", ha="left", va="bottom", fontsize=12, color="#0b0b0b")
    ax.set_xlim(-0.6, 2.6)
    ax.set_xticks(list(x)); ax.set_xticklabels(cats, fontsize=13)
    ax.set_ylim(0, 1.08); ax.set_ylabel("사람 AP50", fontsize=13)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=S1, label="우리 v12_f91_swa — 학습에 없던 장소로 평가"),
                       Patch(color=S2, label="논문 기준선 — 같은 데이터 안에서 나눠 평가")],
              loc="upper left", frameon=False, fontsize=12, ncol=2, bbox_to_anchor=(0, 1.1))
    fig.suptitle("탐지 정확도 — 같은 데이터라도 「처음 보는 장소」 로 재면 낮게 나온다", x=0.02, ha="left", fontsize=17, fontweight="bold", y=0.99)
    fig.text(0.02, 0.01, "논문: WiSARD (YOLOv5 · 가시광 · 60/20/20 분할) · Okutama-Action (SSD · 사람 탐지 mAP@0.5 · 시험 10편) — 모델 · 해상도가 달라 직접 비교가 아닌 참고\n"
             "우리: 경계 합치기 · 숲은 라벨 구멍 무시 영역 · 한국은 가림이 적어 쉬운 편 · 초기 모델 (09-23) 은 같은 장소 0.86 ↔ 처음 보는 장소 0.22",
             fontsize=10.5, color=INK2, va="bottom")
    fig.subplots_adjust(left=0.07, right=0.98, top=0.83, bottom=0.17)
    fig.savefig(out / "10_탐지_AP50_평가방식.png"); plt.close(fig)


def chart_interval(out):
    d = json.load(open(BASE / "metrics" / "ue_frame_interval.json"))
    labels = ["0.1 s", "0.2 s", "0.5 s", "1 s", "2 s"]
    idf1 = [r["idf1"] for r in d]
    lying = [int(r["lying_urgent"].split("/")[0]) for r in d]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5.6), dpi=150)
    for ax in (a1, a2):
        ax.axvspan(-0.4, 1.4, color="#eeede7", zorder=0)
        ax.axvline(2, color=S2, lw=1.6, zorder=2)
        ax.grid(axis="y", color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_xticks(range(5)); ax.set_xticklabels(labels, fontsize=12); ax.set_xlim(-0.4, 4.4)
        ax.set_xlabel("프레임 간격 (UE 8회 · 같은 영상을 솎아서)", fontsize=12)
    a1.plot(range(5), idf1, color=S1, lw=2, marker="o", ms=9, zorder=3)
    for i, v in enumerate(idf1):
        a1.text(i + (0.14 if i == 2 else 0), v + 0.025, f"{v:.2f}", ha="left" if i == 2 else "center", fontsize=12)
    a1.set_ylim(0, 0.6); a1.set_title("추적 IDF1 (높을수록 같은 사람을 끝까지 같은 ID 로)", fontsize=13, loc="left")
    a2.bar(range(5), lying, 0.5, color=S1, zorder=3)
    for i, v in enumerate(lying):
        a2.text(i + (0.1 if i == 2 else 0), v + 0.1, f"{v}/10", ha="left" if i == 2 else "center", fontsize=12)
    a2.set_ylim(0, 4); a2.set_yticks([0, 1, 2, 3, 4]); a2.set_title("누운 사람이 URGENT/HIGH 에 오른 수 (10명 중)", fontsize=13, loc="left")
    a1.text(0.5, 0.56, "필요 0.1~0.2 s", ha="center", fontsize=11.5, color=INK2)
    a1.text(2.08, 0.56, "실연 (10-17)\n0.5 s 간격", ha="left", va="top", fontsize=11.5, color=S2)
    fig.suptitle("프레임 간격이 길어지면 추적 · 상태 판단이 무너진다", x=0.02, ha="left", fontsize=17, fontweight="bold", y=0.99)
    fig.text(0.02, 0.01, "탐지 · 좌표 · 상의 색은 간격 영향이 작다 (색 1순위 0.39~0.46 · 질의 대상 MATCH 0.58~0.74) — 상태 (멈춘 시간) 와 추적은 0.1~0.2 s 가 필요",
             fontsize=10.5, color=INK2, va="bottom")
    fig.subplots_adjust(left=0.06, right=0.98, top=0.82, bottom=0.17, wspace=0.18)
    fig.savefig(out / "11_프레임간격_영향.png"); plt.close(fig)


if __name__ == "__main__":
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    chart_ap50(out); chart_interval(out)
