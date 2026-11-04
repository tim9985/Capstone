"""
make_fair_summary.py — 엔지니어링 페어 「결과 몇 장」 (11-04 · 사용자 「성능 결과가 너무 많아 한 장 · 두세 장으로」)

  1_한눈에        기성 모델과 공정 비교 (AP50 · 95 % 구간 · 같은 재현율 도로 헛탐지) + 핵심 숫자 4개
  2_탐지          정밀도-재현율 곡선 3곳 · 같은 헛탐지에서 사람 키별 재현율 · 혼동표
  3_상태_색_속도  누움 ROC (정답 박스 · 탐지까지) · 자세 혼동 · 상의 색 · 단계별 처리 시간
  원자료  metrics/fair_eval.json (fair_eval_report.py) · posture_fair.json · profile_worker.json · 상의 색 결정 노트 표 (아래 COLOR)
  출력    발표자료/엔지니어링페어_비전/그림/요약_{1,2,3}_*.png (git 밖) · 데이터셋 사진은 넣지 않음 (숫자 · 그래프만)
실행  /home/se/miniconda3/envs/drone/bin/python make_fair_summary.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch
import numpy as np

BASE = Path(__file__).resolve().parent
TOOLS = BASE.parent / "발표자료" / "_도구"
OUT = BASE.parent / "발표자료" / "엔지니어링페어_비전" / "그림"
for f in ("NanumGothic.ttf", "NanumGothicBold.ttf"):
    fm.fontManager.addfont(str(TOOLS / f))
SURF, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
OURS, EXT, ORG = "#2a78d6", "#a19f98", "#eb6834"           # 우리 = 파랑 · 기성 = 회색 (배경 역할) · 둘째 계열 = 주황 (검증 통과 쌍)
GOOD, BAD = "#0ca30c", "#d03b3b"                            # 판정 (항상 글자와 함께)
plt.rcParams.update({"font.family": "NanumGothic", "axes.unicode_minus": False, "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                     "font.size": 11})
FE = json.load(open(BASE / "metrics" / "fair_eval.json"))
PF = json.load(open(BASE / "metrics" / "posture_fair.json"))
PW = json.load(open(BASE / "metrics" / "profile_worker.json"))
EXTN = {"aeronova": "AeroNova", "visdrone_yolo11x": "VisDrone-11x", "visdrone_yolo26x": "VisDrone-26x", "visdrone_yolov9e": "VisDrone-v9e",
        "ap_mix_26l": "AerialPerson 혼합", "ap_ladd_26l": "LADD 전용", "ap_ladd_26m": "LADD 전용 m", "ap_nomad_26l": "NOMAD 전용", "ap_nomad_26m": "NOMAD 전용 m",
        "ap_heridal_26l": "HERIDAL 전용", "ap_heridal_26m": "HERIDAL 전용 m", "ap_sard_26l": "SARD 전용", "ap_sard_26m": "SARD 전용 m"}
# 상의 색 — 결정 노트 「상의 색 SigLIP 2 제로샷」 표 (색 균형 1순위) · PETA 짝 95 % = color_lab_confirm.json (B1 SigLIP 상의 대 V10)
COLOR = [("PETA 지상 CCTV (처음 봄 · 1,606장)", 0.270, 0.560, "main"), ("드론 UAV-Human (방법 고를 때 씀)", 0.286, 0.512, "sel"),
         ("드론 NOMAD (방법 고를 때 씀)", 0.336, 0.422, "sel"), ("시뮬레이션 20 m", 0.393, 0.536, "sel"),
         ("디지털관 사진 (잠정 라벨)", 0.314, 0.744, "prov"), ("디지털관 영상 (잠정 라벨)", 0.422, 0.885, "prov")]
PETA_CI = (0.258, 0.316)


def ename(tag):
    m, p = tag.split("@"); return f"{EXTN.get(m, m)} · {'통째' if p.startswith('full') else '4조각'} {p[4:]}"


def title(fig, eyebrow, head, sub):
    fig.text(0.045, 0.952, eyebrow, fontsize=11, color=INK2)
    fig.text(0.045, 0.905, head, fontsize=25, fontweight="bold", va="center")
    fig.text(0.045, 0.858, sub, fontsize=13, color=INK2, va="center")


def clean(ax, grid="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(length=0, labelsize=10)
    if grid:
        ax.grid(axis=grid, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)


def pct(v):
    return f"{v * 100:.0f} %"


def sets_in(group):
    S = [(k, v) for k, v in FE["sets"].items() if v["group"] == group]
    return sorted(S, key=lambda kv: -kv[1]["diff"]["AP50"])


# ───────────────────────── 1. 한눈에 ─────────────────────────
def fig1():
    fig = plt.figure(figsize=(16, 9), dpi=200)
    U, Bs = sets_in("U"), sets_in("B")
    wins = sum(v["diff"]["verdict"] == "앞섬" for _, v in U); loses = sum(v["diff"]["verdict"] == "뒤짐" for _, v in U); ties = len(U) - wins - loses
    more = sum((v["ext"]["R_eq_fp"] or 0) < v["ours"]["R020"] for _, v in U + Bs)
    title(fig, "자율 정찰 드론 · 비전 파트 · 엔지니어링 페어 2026",
          "공개 모델 13개와 같은 조건에서 겨뤘습니다",
          f"처음 보는 {len(U)}곳: AP 로 {wins}곳 확실히 앞섬 · {ties}곳 구분 안 됨 · {'뒤진 곳 없음' if not loses else f'{loses}곳 뒤짐'}   |   헛탐지를 같게 맞추면 {len(U) + len(Bs)}곳 중 {more}곳에서 더 많이 찾음")
    ax = fig.add_axes([0.045, 0.215, 0.565, 0.6]); ax.set_xlim(0, 1); ax.axis("off")
    rows = [("h", "처음 보는 곳 — 우리도, 기성도 학습에 안 쓴 장소")] + [("s", kv) for kv in U] + [("h", "같은 데이터셋의 다른 사진 — 우리와 그 셋 전용 기성 모델이 모두 학습에 씀")] + [("s", kv) for kv in Bs]
    n = len(rows); ax.set_ylim(n + 0.15, -1.2)
    X0, X1, LO, HI = 0.30, 0.80, 0.40, 1.0                    # AP 0.40~1.00 → 축 0.30~0.80
    xa = lambda v: X0 + (v - LO) / (HI - LO) * (X1 - X0)
    for t in np.arange(0.4, 1.01, 0.1):
        ax.plot([xa(t)] * 2, [-0.55, n - 0.5], color=GRID, lw=0.8, zorder=0)
        ax.text(xa(t), -0.75, f"{t:.1f}", ha="center", va="bottom", fontsize=10, color=INK2)
    ax.text(X0, -1.15, "AP50 (사람을 빠짐없이 · 정확히 찾는 정도 · 1 = 완벽)", fontsize=10.5, color=INK2, va="bottom")
    ax.text(0.995, -1.15, "우리 - 기성 [95 % 구간]", fontsize=10.5, color=INK2, ha="right", va="bottom")
    ys = {}
    for i, (kind, val) in enumerate(rows):
        if kind == "h":
            ax.text(0, i + 0.05, val, fontsize=11.5, fontweight="bold", color=INK2, va="center"); continue
        s, r = val; ys[s] = i; o, e, d = r["ours"], r["ext"], r["diff"]
        ax.text(0, i - 0.12, r["label"], fontsize=13, fontweight="bold", va="center")
        ax.text(0, i + 0.24, f"{r['imgs']}장 · {r['n_gt']:,}명", fontsize=9.5, color=INK2, va="center")
        a, b = sorted([o["AP50"], e["AP50"]])
        ax.plot([xa(a), xa(b)], [i, i], color=OURS if o["AP50"] >= e["AP50"] else EXT, lw=3, alpha=0.35, solid_capstyle="round", zorder=2)
        for k, v in r["ours"]["repeats"].items():
            if k != "v12_f91_swa":
                ax.plot([xa(v)] * 2, [i - 0.17, i + 0.17], color=OURS, lw=1.4, alpha=0.55, zorder=3)
        ax.scatter([xa(e["AP50"])], [i], s=95, color=EXT, edgecolor=SURF, linewidth=1.5, zorder=4)
        ax.scatter([xa(o["AP50"])], [i], s=130, color=OURS, edgecolor=SURF, linewidth=1.5, zorder=5)
        lab_e = ename(e["tag"]); side = -1 if e["AP50"] < o["AP50"] else 1
        ax.text(xa(e["AP50"]), i + 0.36, lab_e, fontsize=8.6, color=INK2, ha="center", va="center")
        col = GOOD if d["verdict"] == "앞섬" else (BAD if d["verdict"] == "뒤짐" else INK2)
        ax.text(0.995, i - 0.12, f"{d['AP50'] * 100:+.1f} %p", fontsize=13, fontweight="bold", ha="right", va="center", color=INK)
        ax.text(0.995, i + 0.24, f"[{d['CI'][0] * 100:+.1f}, {d['CI'][1] * 100:+.1f}] · {d['verdict']}", fontsize=9.5, ha="right", va="center", color=col)
    # 범례 (직접)
    ly = n - 0.1
    ax.scatter([xa(0.42)], [ly], s=110, color=OURS); ax.text(xa(0.42) + 0.012, ly, "우리 v12 (운용 그대로)", va="center", fontsize=10)
    ax.plot([xa(0.64)] * 2, [ly - 0.15, ly + 0.15], color=OURS, lw=1.4, alpha=0.55); ax.text(xa(0.64) + 0.008, ly, "같은 설정 다시 학습 2개", va="center", fontsize=10)
    ax.scatter([xa(0.86)], [ly], s=85, color=EXT); ax.text(xa(0.86) + 0.012, ly, "그 셋 기성 최고", va="center", fontsize=10)
    # 오른쪽 — 같은 헛탐지에서 찾은 비율
    ax2 = fig.add_axes([0.665, 0.215, 0.295, 0.6]); ax2.set_xlim(0, 1); ax2.set_ylim(n + 0.15, -1.2); ax2.axis("off")
    fpo = FE["fp_ours_road"]; Z0, Z1 = 0.0, 0.62; xz = lambda v: Z0 + v * (Z1 - Z0)
    for t in (0, 0.25, 0.5, 0.75, 1.0):
        ax2.plot([xz(t)] * 2, [-0.55, n - 0.5], color=GRID, lw=0.8, zorder=0); ax2.text(xz(t), -0.75, f"{t * 100:.0f}", ha="center", va="bottom", fontsize=10, color=INK2)
    ax2.text(0, -1.15, f"찾은 사람 % — 도로 헛탐지를 우리와 같게 ({fpo:.2f}건/장) 맞췄을 때", fontsize=10.5, color=INK2, va="bottom")
    for s_, i in ys.items():
        r = FE["sets"][s_]; ro, re = r["ours"]["R020"], (r["ext"]["R_eq_fp"] or 0.0)
        ax2.plot([xz(min(ro, re)), xz(max(ro, re))], [i, i], color=OURS if ro >= re else EXT, lw=3, alpha=0.35, solid_capstyle="round", zorder=2)
        ax2.scatter([xz(re)], [i], s=95, color=EXT, edgecolor=SURF, linewidth=1.5, zorder=4)
        ax2.scatter([xz(ro)], [i], s=130, color=OURS, edgecolor=SURF, linewidth=1.5, zorder=5)
        ax2.text(1.0, i - 0.12, f"{(ro - re) * 100:+.0f} %p", ha="right", va="center", fontsize=12.5, fontweight="bold")
        eq = r["ext"]["road_fp_eq_recall"]
        sub = ("같은 수를 찾으면 기성 헛탐지 " + (f"×{eq / fpo:.0f}" if eq / fpo >= 10 else f"×{eq / fpo:.1f}")) if eq else "기성은 우리만큼 못 찾음"
        ax2.text(1.0, i + 0.24, sub, ha="right", va="center", fontsize=8.6, color=INK2)
    # 핵심 숫자
    S = PW["stages"]; pipe = S["탐지"]["median_ms"] + S["추적 + 상태"]["median_ms"] + S["좌표 (사람 전부)"]["median_ms"]
    pl = PF["pipeline_okutama"]
    kpis = [(f"{S['탐지']['median_ms']:.0f} ms", "1080p 한 장 사람 찾기 (기준 ≤ 30)", f"추적 · 자세 · 위치까지 약 {pipe:.0f} ms · 초당 약 {1000 / pipe:.0f}장"),
            (f"{fpo:.2f}건", "사람 없는 도로 헛탐지 / 장 (기준 ≤ 1)", "AU-AIR 도로 · 확신 0.20"),
            (f"{pl['AUROC']:.2f}", "누운 사람을 위로 올리는 순위 (AUROC)", f"실제 드론 영상 {pl['videos']}편 · 탐지까지 이어서 · [{pl['CI'][0]:.2f}, {pl['CI'][1]:.2f}]"),
            ("0.27 → 0.56", "상의 색 맞히기 (12색 · 처음 본 셋)", "PETA 1,606장 · 규칙 → SigLIP 2 · 추가 학습 없음")]
    for k, (big, lab, sub) in enumerate(kpis):
        x = 0.045 + k * 0.235
        fig.add_artist(plt.Line2D([x, x + 0.215], [0.168, 0.168], color=INK, lw=1.6, transform=fig.transFigure))
        fig.text(x, 0.128, big, fontsize=24, fontweight="bold", va="center")
        fig.text(x, 0.087, lab, fontsize=11, va="center")
        fig.text(x, 0.062, sub, fontsize=9.3, color=INK2, va="center")
    fig.text(0.045, 0.012, "평가셋마다 최대 500장 · 모든 모델 같은 채점 (IoU 0.5 · 101점 AP) · 기성 = 공개 가중치 13개 × 2가지 추론 방식 중 그 평가셋에서 가장 높은 것 (기성에 유리하게 고름) · "
             "95 % 구간 = 연속 사진 20장 묶음 짝 부트스트랩\n빠진 셋: SARD · HERIDAL (우리 학습에 평가 사진 포함) · NII-CU (우리만 같은 장소 학습)",
             fontsize=8.6, color=INK2, va="bottom")
    fig.savefig(OUT / "요약_1_한눈에.png"); plt.close(fig)


# ───────────────────────── 2. 탐지 ─────────────────────────
def fig2():
    fig = plt.figure(figsize=(16, 9), dpi=200)
    hp = FE["height_pooled_U"]; cm = FE["confusion_U"]
    R_o = cm["ours"][0] / (cm["ours"][0] + cm["ours"][1]); R_e = cm["ext"][0] / (cm["ext"][0] + cm["ext"][1])
    title(fig, "탐지 — 처음 보는 5곳을 합쳐서", "같은 헛탐지 수준에서 사람을 더 많이 찾습니다",
          f"도로 헛탐지를 우리와 같게 ({FE['fp_ours_road']:.2f}건/장) 맞추면 찾은 사람 비율 우리 {pct(R_o)} · 기성 {pct(R_e)} · 차이는 작은 사람에서 가장 큼")
    pick = ["test_sar", "test_v2", "test_obl"]
    for j, s in enumerate(pick):
        r = FE["sets"][s]; ax = fig.add_axes([0.045 + j * 0.318, 0.47, 0.27, 0.32]); clean(ax, grid="both")
        for who, col, lw in (("ext", EXT, 2.2), ("ours", OURS, 2.6)):
            p = np.array(r[who]["pr"]); ax.plot(p[:, 0], p[:, 1], color=col, lw=lw, solid_capstyle="round", zorder=3)
        ax.scatter([r["ours"]["R020"]], [r["ours"]["P020"]], s=55, color=OURS, edgecolor=SURF, zorder=5)
        ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.05); ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
        ax.set_title(r["label"], loc="left", fontsize=14, fontweight="bold", pad=8)
        ax.text(0.03, 0.17, f"우리 AP {r['ours']['AP50']:.2f}", fontsize=11, fontweight="bold", transform=ax.transAxes)
        ax.text(0.03, 0.07, f"{ename(r['ext']['tag'])} {r['ext']['AP50']:.2f}", fontsize=9.6, color=INK2, transform=ax.transAxes)
        if j == 0:
            ax.set_ylabel("정밀도 (맞은 비율)", fontsize=10.5)
        ax.set_xlabel("재현율 (찾은 비율)", fontsize=10.5)
    # 사람 키별 재현율 (같은 헛탐지)
    ax = fig.add_axes([0.045, 0.125, 0.5, 0.24]); clean(ax)
    keys = ["0-24", "24-48", "48-96", "96-inf"]; labs = ["24 px 미만", "24~48 px", "48~96 px", "96 px 이상"]
    x = np.arange(len(keys)); w = 0.34
    for k, (who, col) in enumerate((("ext", EXT), ("ours", OURS))):
        v = [hp[who].get(kk, {}).get("R") or 0 for kk in keys]
        ax.bar(x + (k - 0.5) * (w + 0.03), v, w, color=col, zorder=3)
        for xi, vi in zip(x, v):
            ax.text(xi + (k - 0.5) * (w + 0.03), vi + 0.02, pct(vi), ha="center", va="bottom", fontsize=10.5, fontweight="bold" if who == "ours" else None, color=INK if who == "ours" else INK2)
    ax.set_xticks(x); ax.set_xticklabels([f"{l}\n{hp['ours'].get(k, {}).get('n', 0):,}명" for l, k in zip(labs, keys)], fontsize=10.5)
    ax.set_ylim(0, 1.12); ax.set_yticks([0, 0.5, 1]); ax.set_yticklabels(["0", "50 %", "100 %"])
    ax.set_title("사람 키별 찾은 비율 — 도로 헛탐지를 같게 맞춘 문턱에서", loc="left", fontsize=13, fontweight="bold", pad=8)
    ax.text(0.99, 1.02, "■ 우리   ■ 셋마다 기성 최고", transform=ax.transAxes, ha="right", fontsize=1)
    ax.scatter([], [], color=OURS, marker="s", s=60, label="우리"); ax.scatter([], [], color=EXT, marker="s", s=60, label="셋마다 기성 최고")
    ax.legend(loc="upper left", frameon=False, fontsize=10, ncol=2, bbox_to_anchor=(0, 1.0))
    # 혼동표
    ax = fig.add_axes([0.6, 0.11, 0.36, 0.28]); ax.set_xlim(0, 10); ax.set_ylim(0, 5.4); ax.axis("off")
    ax.text(0, 5.25, "혼동표 — 도로 헛탐지를 같게 맞춘 문턱 · 처음 보는 5곳 합", fontsize=13, fontweight="bold", va="top")
    for k, (who, name, col) in enumerate((("ours", "우리", OURS), ("ext", "셋마다 기성 최고", EXT))):
        TP, FN, FP = cm[who]; ox = 1.6 + k * 4.3; c = 1.8
        ax.text(ox + c, 4.25, name, ha="center", fontsize=11.5, fontweight="bold")
        ax.text(ox + c / 2, 3.85, "사람이라 함", ha="center", fontsize=9.5, color=INK2); ax.text(ox + c * 1.5 + 0.1, 3.85, "못 찾음", ha="center", fontsize=9.5, color=INK2)
        mx = max(TP, FN, FP)
        for (cx, cy, v, lab) in ((0, 0, TP, "찾음"), (1, 0, FN, "놓침"), (0, 1, FP, "헛탐지"), (1, 1, None, "")):
            X, Y = ox + cx * (c + 0.1), 3.65 - (cy + 1) * (c * 0.92)
            alpha = 0 if v is None else 0.12 + 0.75 * v / mx
            ax.add_patch(FancyBboxPatch((X, Y), c, c * 0.86, boxstyle="round,pad=0,rounding_size=0.12", fc=col if v is not None else SURF, alpha=alpha if v is not None else 1,
                                        ec=GRID, lw=1))
            if v is not None:
                ax.text(X + c / 2, Y + c * 0.52, f"{v:,}", ha="center", va="center", fontsize=13, fontweight="bold", color=INK)
                ax.text(X + c / 2, Y + c * 0.22, lab, ha="center", va="center", fontsize=9.5, color=INK2)
            else:
                ax.text(X + c / 2, Y + c * 0.4, "셀 수 없음", ha="center", va="center", fontsize=9, color=MUTED)
        if k == 0:
            ax.text(ox - 0.15, 3.65 - c * 0.92 + c * 0.43, "실제 사람", ha="right", va="center", fontsize=10)
            ax.text(ox - 0.15, 3.65 - 2 * c * 0.92 + c * 0.43, "배경", ha="right", va="center", fontsize=10)
    fig.text(0.045, 0.015, "정밀도-재현율 곡선: 오른쪽 위일수록 좋음 · 점 = 우리 운용 문턱 0.20 · 처음 보는 5곳 = UAV-SAR · WiSARD 숲 · Okutama 비스듬 · AI-Hub 한국 · Cyprus\n기성 문턱은 AU-AIR 도로 헛탐지가 우리 (문턱 0.20) 와 같아지는 값 · 사람 키 = 가로 1920 으로 맞춘 그림에서 · 혼동표 헛탐지 = 평가셋 안 (라벨 없는 사람 · 물체 포함)",
             fontsize=8.8, color=INK2, va="bottom")
    fig.savefig(OUT / "요약_2_탐지.png"); plt.close(fig)


# ───────────────────────── 3. 상태 · 색 · 속도 ─────────────────────────
def fig3():
    fig = plt.figure(figsize=(16, 9), dpi=200)
    pl = PF["pipeline_okutama"]
    title(fig, "탐지 다음 — 위급도 · 인상착의 · 실시간", "누운 사람을 위로, 옷 색으로 찾기, 실시간으로",
          "자세는 거르지 않고 순위만 바꿈 (분류하면 앉음 · 서기가 섞임) · 색은 추가 학습 없이 · 모든 단계가 한 장 약 40 ms")
    # ROC
    ax = fig.add_axes([0.045, 0.47, 0.25, 0.33]); clean(ax, grid="both")
    ax.plot([0, 1], [0, 1], color=AXIS, lw=1.2, ls=(0, (4, 3)))
    for s, col, nm in (("okutama_obl", OURS, "Okutama 실제 드론"), ("archangel_real", ORG, "Archangel 실사람")):
        p = np.array(PF[s]["roc"]); ax.plot(p[:, 0], p[:, 1], color=col, lw=2.4)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02); ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel("다른 자세를 누움으로 고른 비율", fontsize=10); ax.set_ylabel("누운 사람을 고른 비율", fontsize=10)
    ax.set_title("누움 vs 나머지 ROC — 자세 단계 (정답 박스)", loc="left", fontsize=12.5, fontweight="bold", pad=8)
    for k, (s, col, nm) in enumerate((("okutama_obl", OURS, "Okutama 실제 드론"), ("archangel_real", ORG, "Archangel 실사람 (처음 봄)"))):
        v = PF[s]; y = 0.30 - k * 0.11
        ax.plot([0.38, 0.45], [y, y], color=col, lw=2.4, transform=ax.transAxes)
        ax.text(0.47, y, f"{nm} {v['AUROC']:.2f}", transform=ax.transAxes, va="center", fontsize=10, fontweight="bold")
        ax.text(0.47, y - 0.05, f"[{v['CI'][0]:.2f}, {v['CI'][1]:.2f}] · {v['n']:,}명", transform=ax.transAxes, va="center", fontsize=8.6, color=INK2)
    fig.text(0.045, 0.385, f"탐지까지 이은 실제 영상 {pl['videos']}편: AUROC {pl['AUROC']:.2f} [{pl['CI'][0]:.2f}, {pl['CI'][1]:.2f}] \n누운 사람 중 상태가 잡힌 비율 {pct(pl['lying_caught'])} → 실기체로 개선",
             fontsize=9.6, color=INK2)
    # 자세 혼동 (Okutama)
    ax = fig.add_axes([0.355, 0.47, 0.25, 0.33]); ax.axis("off"); ax.set_xlim(0, 4.2); ax.set_ylim(0, 4.3)
    cmx = np.array(PF["okutama_obl"]["confusion"]); cls = PF["okutama_obl"]["classes"]
    ax.text(0, 4.25, "자세 혼동행렬 — Okutama (행 = 실제)", fontsize=12.5, fontweight="bold", va="top")
    ramp = matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#eef4fc", "#86b6ef", "#2a78d6", "#184f95"])
    for i in range(3):
        ax.text(1.05, 3.0 - i, cls[i], ha="right", va="center", fontsize=10)
        tot = cmx[i].sum()
        for j in range(3):
            sh = cmx[i, j] / tot; X, Y = 1.15 + j * 1.0, 2.55 - i
            ax.add_patch(FancyBboxPatch((X, Y), 0.94, 0.9, boxstyle="round,pad=0,rounding_size=0.06", fc=ramp(sh), ec=INK if i == j else "none", lw=1.2))
            ax.text(X + 0.47, Y + 0.55, pct(sh), ha="center", va="center", fontsize=12, fontweight="bold", color=SURF if sh > 0.55 else INK)
            ax.text(X + 0.47, Y + 0.25, f"{cmx[i, j]:,}", ha="center", va="center", fontsize=8.6, color=SURF if sh > 0.55 else INK2)
        ax.text(1.62 + i * 1.0, 3.6, cls[i], ha="center", fontsize=10)
    fig.text(0.355, 0.385, "누운 사람은 93 % 누움으로 · 앉음 · 서기는 많이 섞임\n→ 자세로 거르지 않고 위급 순서만 바꿈", fontsize=9.6, color=INK2)
    # 상의 색
    ax = fig.add_axes([0.665, 0.47, 0.295, 0.33]); ax.set_xlim(0, 1); ax.set_ylim(len(COLOR) - 0.4, -1.3); ax.axis("off")
    ax.text(0, -1.25, "상의 색 맞히기 (12색 · 색마다 정확도 평균)", fontsize=12.5, fontweight="bold", va="top")
    X0, X1 = 0.56, 0.92; xa = lambda v: X0 + v * (X1 - X0)
    for t in (0, 0.5, 1):
        ax.plot([xa(t)] * 2, [-0.25, len(COLOR) - 0.5], color=GRID, lw=0.8, zorder=0); ax.text(xa(t), -0.3, f"{t:g}", ha="center", va="bottom", fontsize=9, color=INK2)
    for i, (lab, a, b, kind) in enumerate(COLOR):
        main = kind == "main"
        ax.text(0, i, lab, fontsize=10.2 if main else 9.4, fontweight="bold" if main else None, va="center", color=INK if main else INK2)
        ax.plot([xa(a), xa(b)], [i, i], color=OURS, lw=2.6 if main else 1.6, alpha=0.5 if main else 0.3, solid_capstyle="round")
        ax.scatter([xa(a)], [i], s=60 if main else 34, color=EXT, zorder=3); ax.scatter([xa(b)], [i], s=95 if main else 45, color=OURS, alpha=1 if main else 0.55, zorder=4, edgecolor=SURF)
        ax.text(1.0, i, f"×{b / a:.1f}", ha="right", va="center", fontsize=10.5 if main else 9.4, fontweight="bold" if main else None, color=INK if main else INK2)
    fig.text(0.665, 0.385, f"PETA: 규칙 0.27 → SigLIP 2 0.56 · 차이 95 % [+{PETA_CI[0] * 100:.0f}, +{PETA_CI[1] * 100:.0f}] %p\n회색 점 = 이전 색 규칙 · 파란 점 = 지금",
             fontsize=9.6, color=INK2)
    # 처리 시간
    ax = fig.add_axes([0.2, 0.12, 0.55, 0.22]); clean(ax, grid="x")
    S = PW["stages"]
    st = [("사람 찾기 (4조각 · TensorRT)", S["탐지"], OURS), ("추적 · 자세 · 멈춤 · 등급", S["추적 + 상태"], EXT), ("상의 색 SigLIP 2 · 1명", PW["siglip_per_person"], EXT),
          ("위치 (위경도)", S["좌표 (사람 전부)"], EXT)]
    y = np.arange(len(st))[::-1]
    for yi, (lab, v, col) in zip(y, st):
        ax.barh(yi, v["median_ms"], 0.5, color=col, zorder=3)
        ax.plot([v["median_ms"], v["p95_ms"]], [yi, yi], color=INK2, lw=1.3, zorder=4); ax.plot([v["p95_ms"]] * 2, [yi - 0.15, yi + 0.15], color=INK2, lw=1.3, zorder=4)
        ax.text(v["p95_ms"] + 2, yi, f"{v['median_ms']:.1f} ms  (P95 {v['p95_ms']:.0f})", va="center", fontsize=10.5)
    ax.set_yticks(y); ax.set_yticklabels([s[0] for s in st], fontsize=10.5)
    ax.axvline(30, color=BAD, lw=1.2, ls=(0, (4, 3))); ax.text(31, y[0] + 0.42, "탐지 기준 30 ms", color=BAD, fontsize=9.5, va="center")
    ax.set_xlim(0, 130); ax.set_xlabel("한 장 처리 시간 (ms · 막대 = 중앙값 · 선 끝 = P95) · RTX 3090 · 배포 코드 그대로", fontsize=10)
    ax.set_title("단계별 처리 시간", loc="left", fontsize=12.5, fontweight="bold", pad=8)
    M = PW["memory"]
    fig.text(0.77, 0.22, f"GPU 메모리 {M['+탐지 (TensorRT)']['gpu_mb'] / 1024:.1f}~{M['끝']['gpu_mb'] / 1024:.1f} GB\nRAM {M['+탐지 (TensorRT)']['rss_mb'] / 1024:.1f}~{M['끝']['rss_mb'] / 1024:.1f} GB\n(worker 한 개 · 최소 사양 8 GB 안)",
             fontsize=11, va="center", linespacing=1.6)
    fig.text(0.045, 0.015, "자세 = 박스 가로세로 비 · 크기 2특징 로지스틱 (SARD · NOMAD 로 학습 · Okutama · Archangel 은 처음 봄 · 기어감 뺌 · 무릎 → 앉음) · 95 % = 영상 · 회차 묶음 부트스트랩 · "
             "색 평가셋 중 디지털관 2개는 검수 전 잠정 라벨 · 처리 시간 = UE 비행 300장 (사람 중앙 1명)", fontsize=8.6, color=INK2, va="bottom")
    fig.savefig(OUT / "요약_3_상태_색_속도.png"); plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig1(); fig2(); fig3()
    print("ok", OUT)
