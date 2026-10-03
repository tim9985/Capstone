"""
state_trackfix.py — 추적 고치기 Q1 판정: 누운 사람의 ID 바뀜을 막으면 요구조 점수가 오르나 (2026-10-03)

  배경 (10-03 진단 · state_diag.py): 탐지 박스 한 장 누움 AUROC 0.983 ↔ 5초 중앙값 0.800 — 누움 70 표본 중 66 이 추적 ID 바뀜
        원인 후보: BoT-SORT 기본 new_track_thresh 0.25 > 탐지 운용 임계 0.15 → 확신도 0.15~0.25 인 누운 사람은 새 추적을 못 만들고 주변 추적에 붙는다
  후보 (state_pipeline.py --dump 로 각각 실행 · 탐지 soup_v9x2 · 자세 box)
    V0 기본 botsort.yaml                       (현행)
    V1 botsort_t015.yaml                       추적 문턱 0.15 (= 탐지 운용 임계 · NFR-V05)
    V2 V1 + --reset-jump                       1초 이동 > 3 몸높이/초면 이력 비움 · 자세 누적은 최근 5초 안 표본만
    V3 V2 + ReID (botsort_t015_reid.yaml)      탐지기 특징으로 겉모습이 다르면 안 붙인다
  자세 규칙 (표본에 남긴 최근 이력으로 다시 계산 · 무동작 항은 실행 값 그대로)
    agg 실행 값 (V0·V1 최근 5표본 중앙값 · V2·V3 최근 5초 중앙값) · single 지금 한 장 · mean 최근 5초 평균 · max3 최근 3초 최댓값
  판정 (돌리기 전에 정함 · 볼트 _학습 큐)
    Okutama 비스듬 23편을 이름순 짝수(A 12편) · 홀수(B 11편)로 나눔 — 연기 실행에서 본 1.1.1 은 A
    A 에서 (후보 × 규칙) 중 요구조 점수 누움 AUROC 최고를 고르고 → B 에서 현행(V0 · agg) 과 영상 블록 짝 부트스트랩 2,000회
    채택 = B 에서 점수 누움 AUROC 차 95 % 구간 > 0 **그리고** 누움 잡힌 비율 차가 유의하게 나쁘지 않음
출력: metrics/state_trackfix.json (고른 것 · 채택 · 전체 후보 관찰)
실행: python state_trackfix.py              (기본 태그 q1_v0 ~ q1_v3)
      python state_trackfix.py eval <태그> <규칙>   (다른 실행 하나를 같은 규칙으로 평가 — 예: YOLO26 파이프라인)
"""
import collections
import glob
import json
import os
import sys

import numpy as np

from okutama_motion import FPS, LAB, NADIR, OK, load_boxes
from posture_v2 import auc

BASE = os.path.dirname(os.path.abspath(__file__))
VARIANTS = {"V0": ("q1_v0", "botsort.yaml", False), "V1": ("q1_v1", "configs/trackers/botsort_t015.yaml", False),
            "V2": ("q1_v2", "configs/trackers/botsort_t015.yaml", True), "V3": ("q1_v3", "configs/trackers/botsort_t015_reid.yaml", True)}
RULES = ("agg", "single", "mean", "max3")


def videos():
    frame_dir = {os.path.basename(d): d for d in glob.glob(f"{OK}/Drone*/*/Extracted-Frames-1280x720/*")}
    vids = [v for v in sorted(os.path.basename(p)[:-4] for p in glob.glob(f"{LAB}/*.txt")) if v in frame_dir and v not in NADIR]
    return vids, vids[0::2], vids[1::2]


def gt_lying(vids):
    return {v: sum(1 for f, d in load_boxes(f"{LAB}/{v}.txt").items() if f % FPS == 0 for b in d.values() if b[4] == "Lying") for v in vids}


def rescore(S, rule):
    """규칙별 (누움 · 앉음) → 요구조 점수 = 0.5 누움 + 0.2 앉음 + 무동작 항 (실행 값)."""
    out = []
    for s in S:
        still = s["score"] - 0.5 * s["agg"][0] - 0.2 * s["agg"][1]
        if rule == "agg":
            ly, si = s["agg"][0], s["agg"][1]
        elif rule == "single":
            ly, si = s["p"][0], s["p"][1]
        else:
            w = 5 if rule == "mean" else 3
            idx = [i for i, f in enumerate(s["h_fr"]) if f > s["fr"] - w * FPS]
            f = np.mean if rule == "mean" else np.max
            ly, si = float(f([s["h_ly"][i] for i in idx])), float(f([s["h_si"][i] for i in idx]))
        out.append(0.5 * ly + 0.2 * si + still)
    return np.array(out)


def per_video(S, sc):
    """영상별 배열 — 누움 여부 · 점수 · 추적 ID (부트스트랩에서 영상째 다시 뽑기 위해)."""
    d = {}
    vid = np.array([s["vid"] for s in S]); y = np.array([s["pose"] == "Lying" for s in S]); t = np.array([s["t"] for s in S])
    for v in np.unique(vid):
        m = vid == v; d[v] = (y[m], sc[m], t[m])
    return d


def measures_pv(pv, pick, gl):
    """점수 누움 AUROC · 누움 잡힌 비율 · 추적 단위 AUROC — pick (영상 목록 · 중복 허용) 기준."""
    ys, xs, ks = [], [], []
    for j, v in enumerate(pick):
        if v in pv:
            y, x, t = pv[v]; ys.append(y); xs.append(x); ks.append(j * 1_000_000 + t)
    if not ys:
        return {"점수AUROC": float("nan"), "잡힌비율": float("nan"), "추적AUROC": float("nan")}
    y, x, k = np.concatenate(ys), np.concatenate(xs), np.concatenate(ks)
    uk, inv = np.unique(k, return_inverse=True)
    tp = np.full(len(uk), -np.inf); np.maximum.at(tp, inv, x)
    tl = np.zeros(len(uk), bool); np.logical_or.at(tl, inv, y)
    return {"점수AUROC": auc(x[y], x[~y]) if y.any() and (~y).any() else float("nan"),
            "잡힌비율": float(y.sum() / max(sum(gl[v] for v in pick), 1)),
            "추적AUROC": auc(tp[tl], tp[~tl]) if tl.any() and (~tl).any() else float("nan")}


def measures(S, sc, vids, gl):
    return measures_pv(per_video(S, sc), sorted(vids), gl)


def paired_videos(Sa, sa, Sb, sb, vids, gl, B=2000):
    """B − A · 영상 블록 부트스트랩 (같은 영상 묶음으로 두 실행을 같이 다시 뽑는다)."""
    rng = np.random.default_rng(0); vids = sorted(vids)
    pa, pb = per_video(Sa, sa), per_video(Sb, sb)
    base_a, base_b = measures_pv(pa, vids, gl), measures_pv(pb, vids, gl)
    boot = collections.defaultdict(list)
    for _ in range(B):
        pick = list(rng.choice(vids, len(vids)))
        a, b = measures_pv(pa, pick, gl), measures_pv(pb, pick, gl)
        for k in a:
            if not (np.isnan(a[k]) or np.isnan(b[k])):
                boot[k].append(b[k] - a[k])
    return {k: {"차": round(base_b[k] - base_a[k], 3), "95%": [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)]}
            for k, v in boot.items()}


def load(tag):
    return json.load(open(os.path.join(BASE, "runs_state", f"samples_{tag}.json")))


def main():
    vids, A, B = videos()
    gl = gt_lying(vids)
    print(f"영상 {len(vids)} · A {len(A)} (누움 정답 {sum(gl[v] for v in A)}) · B {len(B)} (누움 정답 {sum(gl[v] for v in B)})")
    S = {k: load(v[0]) for k, v in VARIANTS.items() if os.path.exists(os.path.join(BASE, "runs_state", f"samples_{v[0]}.json"))}
    res = {"분할": {"A": A, "B": B}, "후보": {}}
    for k, s in S.items():
        for r in RULES:
            sc = rescore(s, r)
            res["후보"][f"{k}|{r}"] = {"A": {m: round(x, 3) for m, x in measures(s, sc, set(A), gl).items()},
                                     "B (관찰)": {m: round(x, 3) for m, x in measures(s, sc, set(B), gl).items()},
                                     "전체 (관찰)": {m: round(x, 3) for m, x in measures(s, sc, set(vids), gl).items()}}
            print(f"  {k} {r:6s} A {res['후보'][f'{k}|{r}']['A']}", flush=True)
    order = [f"{k}|{r}" for k in VARIANTS if k in S for r in RULES]
    best = max(order, key=lambda c: (res["후보"][c]["A"]["점수AUROC"], -order.index(c)))
    bk, br = best.split("|")
    print(f"→ A 에서 고름: {bk} · {br}")
    pb = paired_videos(S["V0"], rescore(S["V0"], "agg"), S[bk], rescore(S[bk], br), B, gl)
    adopt = pb["점수AUROC"]["95%"][0] > 0 and not (pb["잡힌비율"]["95%"][1] < 0)
    res.update({"고른 것": {"후보": bk, "규칙": br, "tracker": VARIANTS[bk][1], "reset": VARIANTS[bk][2]},
                "B 판정 (고른 것 − 현행 V0·agg)": pb, "채택": bool(adopt)})
    print(f"   B 판정 {pb} → {'채택' if adopt else '기각'}")
    json.dump(res, open(os.path.join(BASE, "metrics", "state_trackfix.json"), "w"), ensure_ascii=False, indent=1)


def eval_one(tag, rule):
    vids, A, B = videos(); gl = gt_lying(vids); s = load(tag)
    sc = rescore(s, rule)
    out = {"전체": measures(s, sc, set(vids), gl), "A": measures(s, sc, set(A), gl), "B": measures(s, sc, set(B), gl)}
    print(json.dumps({k: {m: round(x, 3) for m, x in v.items()} for k, v in out.items()}, ensure_ascii=False))
    return out


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "eval":
        eval_one(sys.argv[2], sys.argv[3])
    else:
        main()
