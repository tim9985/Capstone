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



def chart_threshold(out):
    """12 임계값 — 낮추면 덜 놓치고 (미탐 ↓) 헛것이 늘어난다 (오탐 ↑) · 운용 0.20 = 오탐 기준을 지키는 가장 낮은 값 근처"""
    d = json.load(open(BASE / "metrics" / "thr_sweep.json"))["v12_f91_swa"]
    th = [0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6]
    S3, S4 = "#1baf7a", "#4a3aa7"
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 6.4), dpi=150)
    for ax in (a1, a2):
        ax.axvline(0.20, color="#0b0b0b", lw=1.6, zorder=4)
        ax.grid(color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_xlim(0.04, 0.62); ax.set_xlabel("임계값 (확신도 문턱)", fontsize=13)
    lines = (("test_v2", "숲", S1), ("test_obl", "비스듬 45°", S2), ("test_kr", "한국 수직", S3))
    for key, name, col in lines:
        miss = [1 - d[key][str(t)]["R"] for t in th]
        a1.plot(th, miss, color=col, lw=2.2, marker="o", ms=6, zorder=3)
        a1.text(0.605, miss[-1], name, color="#0b0b0b", fontsize=12, va="center")
    a1.set_ylim(0, 0.5); a1.set_ylabel("놓친 사람 비율 (1 - 재현율)", fontsize=13)
    a1.set_title("미탐 — 임계값을 높일수록 더 놓친다", fontsize=14, loc="left")
    m20 = 1 - d["test_v2"]["0.2"]["R"]
    a1.annotate(f"숲 0.20 에서 {m20:.2f}", (0.20, m20), (0.30, m20 - 0.12), fontsize=11.5,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=1))
    for key, name, col, field in (("test_auair", "도로 장면 (사람 없음)", S4, "음성장_오탐"), ("test_v2", "숲 장면 (사람 없음)", S1, "음성장_오탐")):
        fp = [d[key][str(t)][field] for t in th]
        a2.plot(th, fp, color=col, lw=2.2, marker="o", ms=6, zorder=3, label=name)
    a2.legend(loc="upper right", frameon=False, fontsize=12)
    a2.axvspan(0.175, 0.62, color="#eeede7", zorder=0)
    a2.text(0.40, 1.55, "오탐 기준 통과 구간", fontsize=11.5, color=INK2, ha="center")
    a2.axhline(1.0, color=S2, lw=1.4, zorder=2)
    a2.text(0.045, 1.03, "기준 ≤ 1 건/장", color=S2, fontsize=11.5, va="bottom")
    a2.set_ylim(0, 3.2); a2.set_ylabel("사람 없는 장면 오탐 (건/장)", fontsize=13)
    a2.set_title("오탐 — 임계값을 낮출수록 헛것이 는다", fontsize=14, loc="left")
    for ax in (a1, a2):
        ax.text(0.205, ax.get_ylim()[1] * 0.97, "운용 0.20", fontsize=12, fontweight="bold", va="top")
    fig.suptitle("임계값 = 「이 확신도 이상이면 사람」 · 우리는 오탐 기준을 지키는 가장 낮은 값으로 → 놓치지 않는 쪽", x=0.02, ha="left",
                 fontsize=16, fontweight="bold", y=0.985)
    fig.text(0.02, 0.012, "← 낮추면: 덜 놓침 · 헛것 ↑      높이면: 헛것 ↓ · 더 놓침 →      ·  추적은 0.15 부터 시작 · 상태 · 상의 색은 후보를 지우지 않고 순위만 바꿈 (미탐 우선)\n"
             "v12_f91_swa · 장소 분리 평가셋 (숲 WiSARD · 비스듬 Okutama · 한국 AI-Hub) · 도로 = AU-AIR 사람 없는 장 795장 · 숲 = 사람 없는 장 41장",
             fontsize=10.5, color=INK2, va="bottom")
    fig.subplots_adjust(left=0.06, right=0.93, top=0.84, bottom=0.17, wspace=0.28)
    fig.savefig(out / "12b_임계값_미탐_오탐.png"); plt.close(fig)


def chart_recall(out):
    """12 재현율이 올라갔다 — ① 같은 오탐 기준 (도로 ≤ 1 건/장) 에서 이전 모델 vs 지금 ② 임계값을 낮춰 재현율을 올림"""
    D = json.load(open(BASE / "metrics" / "thr_sweep.json"))
    old, new = D["soup_v7r2"], D["v12_f91_swa"]
    ot, nt = "0.4", "0.2"                                   # 도로 오탐 ≤ 1 을 지키는 운용값 (soup_v7r2 0.40 · v12 0.20)
    cats = ["숲", "비스듬 45°", "한국 수직", "숲 · 작은 사람\n(키 24 px 미만)"]
    o = [old["test_v2"][ot]["R"], old["test_obl"][ot]["R"], old["test_kr"][ot]["R"], old["test_v2"][ot]["크기별_R"]["0-24"]]
    n = [new["test_v2"][nt]["R"], new["test_obl"][nt]["R"], new["test_kr"][nt]["R"], new["test_v2"][nt]["크기별_R"]["0-24"]]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 6.6), dpi=150, gridspec_kw={"width_ratios": [1.25, 1]})
    w = 0.36
    for i in range(4):
        a1.bar(i - w / 2 - 0.01, o[i], w, color="#b9b8b2", zorder=3)
        a1.bar(i + w / 2 + 0.01, n[i], w, color=S1, zorder=3)
        a1.text(i - w / 2 - 0.01, o[i] + 0.015, f"{o[i]:.2f}", ha="center", fontsize=12, color=INK2)
        a1.text(i + w / 2 + 0.01, n[i] + 0.015, f"{n[i]:.2f}", ha="center", fontsize=13, fontweight="bold")
        a1.text(i, max(o[i], n[i]) + 0.09, f"+{(n[i] - o[i]) * 100:.1f} %p", ha="center", fontsize=13, color=S1, fontweight="bold")
    a1.set_xticks(range(4)); a1.set_xticklabels(cats, fontsize=12.5); a1.set_ylim(0, 1.18); a1.set_ylabel("재현율 (찾은 사람 비율)", fontsize=13)
    from matplotlib.patches import Patch
    a1.legend(handles=[Patch(color="#b9b8b2", label="이전 모델 soup_v7r2 · 임계값 0.40"), Patch(color=S1, label="지금 v12_f91_swa · 임계값 0.20")],
              loc="upper left", frameon=False, fontsize=12, ncol=1)
    a1.set_title("같은 오탐 기준 (사람 없는 도로 ≤ 1 건/장) 에서 재현율", fontsize=14, loc="left")
    th = [0.1, 0.125, 0.15, 0.175, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6]
    for key, name, col in (("test_v2", "숲", S1), ("test_obl", "비스듬 45°", S2)):
        r = [new[key][str(t)]["R"] for t in th]
        a2.plot(th, r, color=col, lw=2.4, marker="o", ms=6, zorder=3)
        a2.text(0.605, r[-1], name, fontsize=12, va="center")
        r5, r2 = new[key]["0.5"]["R"], new[key]["0.2"]["R"]
        a2.plot([0.5], [r5], marker="o", ms=13, mfc="white", mec=col, mew=2.5, zorder=4)
        a2.plot([0.2], [r2], marker="o", ms=13, color=col, zorder=4)
        a2.text(0.33 if key == "test_v2" else 0.215, r2 + 0.018 if key == "test_v2" else r2 - 0.07, f"0.5 → 0.2 : +{(r2 - r5) * 100:.1f} %p", color=col, fontsize=12.5, fontweight="bold")
    a2.axvline(0.20, color="#0b0b0b", lw=1.6); a2.text(0.205, 0.80, "운용 0.20", fontsize=12, fontweight="bold", va="top")
    a2.axvspan(0.09, 0.175, color="#f3e3da", zorder=0); a2.text(0.132, 0.515, "오탐\n기준 초과", fontsize=11, color=S2, ha="center")
    a2.set_xlim(0.09, 0.62); a2.set_ylim(0.50, 0.81); a2.set_xlabel("임계값 (확신도 문턱)", fontsize=13); a2.set_ylabel("재현율", fontsize=13)
    a2.set_title("임계값을 낮춰 재현율을 올림 (v12)", fontsize=14, loc="left")
    for ax in (a1, a2):
        ax.grid(axis="y", color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.suptitle("재현율이 올라갔다 — 데이터 재구성 + 임계값 0.20 (오탐 기준을 지키는 가장 낮은 값)", x=0.02, ha="left", fontsize=16, fontweight="bold", y=0.985)
    fig.text(0.02, 0.012, "임계값 = 「이 확신도 이상이면 사람」 · 낮추면 덜 놓치지만 헛것이 는다 → 도로 오탐 ≤ 1 건/장을 지키는 가장 낮은 값을 고름 (이전 모델은 0.40 까지 올려야 했음)\n"
             "장소 분리 평가셋 (숲 WiSARD · 비스듬 Okutama · 한국 AI-Hub) · IoU 0.5 · 타일 4장 · 경계 합치기",
             fontsize=10.5, color=INK2, va="bottom")
    fig.subplots_adjust(left=0.06, right=0.95, top=0.85, bottom=0.17, wspace=0.22)
    fig.savefig(out / "12_재현율_향상.png"); plt.close(fig)



if __name__ == "__main__":
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    chart_ap50(out); chart_interval(out); chart_threshold(out); chart_recall(out)
