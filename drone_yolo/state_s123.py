"""
state_s123.py — S1 · S2 · S3: 진짜 추적 (worker state.py) 으로 이력 쓰는 결론 다시 보기 (10-08 · 판정 기준은 _학습 큐 「10-08 계획」)

  입력  runs_state/worker_samples_s123.json — state_worker_check.py --all (soup_v9x2 · 타일 탐지 · 비스듬 23편 · 1초 표본 + 추적 이력 h)
  점수  0.5 × 누움 + 0.2 × 앉음 + (이동 중이면 0 · 아니면 0.3 × min(무동작, 20)/20)  — worker 와 같은 식
  S1  자세 항 = 지금 한 장 (single · 현행) vs 최근 w 초 중앙값 (agg3 · agg5) — A (짝수 12) 에서 w 고름 → B (홀수 11) 짝 판정
  S2  자세 항 = A4' (박스 2특징 + 지난 3초 log(h/w) 최소 · 최대 · 흔들림 · Archangel 로 맞춤 · posture_a5 와 같음) vs single — B 짝 판정
  S3  관찰 — 자세별 무동작 분포 · 추적 단위 AUROC (추적 최고 점수 vs 누운 적 있음) · 확정 추적 비율
  판정  영상 블록 짝 부트스트랩 2,000회 · 채택 = 점수 누움 AUROC 차 95 % 구간 > 0 (잡힌 비율은 표본이 같아 변하지 않음)
출력: metrics/state_s123.json · metrics/AUTO_RESULT_d.md (덧붙임)
"""
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from posture_v2 import auc
from state_trackfix import gt_lying, videos

BASE = Path(__file__).resolve().parent
REP = BASE / "metrics" / "AUTO_RESULT_d.md"


def score(ly, si, still, moving):
    return 0.5 * ly + 0.2 * si + (0 if moving else 0.3 * min(still, 20) / 20)


def terms(s, rule, a4=None):
    if rule == "a4":
        return a4
    if rule == "single" or not s["h"]:
        return s["p"]
    if rule.startswith("agg"):
        w = int(rule[3:]); t = s["h"][-1][0]
        win = [x for x in s["h"] if x[0] > t - w + 1e-6]
        return [float(np.median([x[1] for x in win])), float(np.median([x[2] for x in win]))]
    return a4


def main():
    from posture_a5 import fit_arc, wfeat
    S = json.load(open(BASE / "runs_state" / "worker_samples_s123.json"))
    vids, A, B = videos()
    sc_m, m4 = fit_arc()["A4'"]
    X = []
    for s in S:
        hist = [(x[3], x[4]) for x in s["h"]][-3:] or [tuple(s["wh"])]   # 확정 전 탐지 — 이력 없음 → 지금 박스만 (Archangel 학습의 추적 첫 표본과 같음)
        X.append(wfeat(hist[-1], hist)[1])
    P4 = {}
    idx4 = [i for i, x in enumerate(X) if x is not None]
    if idx4:
        pr = m4.predict_proba(sc_m.transform(np.array([X[i] for i in idx4])))
        for i, p in zip(idx4, pr):
            P4[i] = [float(p[0]), float(p[1])]
    y = np.array([s["pose"] == "Lying" for s in S]); vid = np.array([s["vid"] for s in S])
    trk = np.array([f"{s['vid']}#{s['track']}" if s.get("tracked") and s["track"] else f"{s['vid']}#u{i}" for i, s in enumerate(S)])
    SC = {}
    for rule in ("single", "agg3", "agg5", "a4"):
        SC[rule] = np.array([score(*terms(s, rule, P4[i] if rule == "a4" else None), s["still"], s["moving"]) for i, s in enumerate(S)])

    def blk(vs):
        return np.concatenate([np.where(vid == v)[0] for v in vs]) if len(vs) else np.array([], int)

    def m(rule, idx):
        sc, l = SC[rule][idx], y[idx]
        tk = trk[idx]; ut, inv = np.unique(tk, return_inverse=True)
        tmax = np.full(len(ut), -np.inf); np.maximum.at(tmax, inv, sc)
        tl = np.zeros(len(ut), bool); np.logical_or.at(tl, inv, l)
        return {"점수 누움AUROC": auc(sc[l], sc[~l]) if l.any() and (~l).any() else float("nan"),
                "추적 AUROC": auc(tmax[tl], tmax[~tl]) if tl.any() and (~tl).any() else float("nan")}

    res = {"표본": {"전체": len(S), "누움": int(y.sum()), "확정 추적 비율": round(float(np.mean([s.get("tracked", True) for s in S])), 3),
                   "A4' 계산된 표본": len(P4)}, "A": {}, "B": {}}
    iA, iB = blk(A), blk(B)
    for r in SC:
        res["A"][r] = {k: round(float(v), 4) for k, v in m(r, iA).items()}
        res["B"][r] = {k: round(float(v), 4) for k, v in m(r, iB).items()}
    pick = max(("agg3", "agg5"), key=lambda r: res["A"][r]["점수 누움AUROC"])
    rng = np.random.default_rng(0); Bv = np.array(B)

    def paired(base, cand):
        d = defaultdict(list)
        for _ in range(2000):
            idx = blk(list(rng.choice(Bv, len(Bv))))
            if not y[idx].any():
                continue
            a, b = m(base, idx), m(cand, idx)
            for k in a:
                if not (math.isnan(a[k]) or math.isnan(b[k])):
                    d[k].append(b[k] - a[k])
        return {k: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)] for k, v in d.items()}

    res["S1"] = {"A 에서 고름": pick, "짝 B (single → 고른 것)": paired("single", pick)}
    res["S1"]["채택"] = bool(res["S1"]["짝 B (single → 고른 것)"].get("점수 누움AUROC", [0])[0] > 0)
    res["S2"] = {"짝 B (single → A4p)": paired("single", "a4")}
    res["S2"]["채택"] = bool(res["S2"]["짝 B (single → A4p)"].get("점수 누움AUROC", [0])[0] > 0)
    st = defaultdict(list)
    for s in S:
        if s.get("tracked", True):
            st[s["pose"]].append((s["still"], s["moving"], s["measured"]))
    res["S3 무동작 (확정 추적)"] = {p: {"n": len(v), "무동작 중앙": float(np.median([a for a, _, _ in v])),
                                     "≥10초": round(float(np.mean([a >= 10 for a, _, _ in v])), 3),
                                     "이동 중": round(float(np.mean([b for _, b, _ in v])), 3),
                                     "속도 잰 비율": round(float(np.mean([c for _, _, c in v])), 3)} for p, v in st.items()}
    res["S3 누움 잡힌 비율"] = {"A": round(int(y[iA].sum()) / max(sum(gt_lying(A).values()), 1), 3),
                              "B": round(int(y[iB].sum()) / max(sum(gt_lying(B).values()), 1), 3)}
    (BASE / "metrics" / "state_s123.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    L = [f"\n## S1 · S2 · S3 — 진짜 추적으로 다시 (soup_v9x2 · 타일 · 비스듬 23편 · {res['표본']})",
         "| 규칙 | A 점수 AUROC | B 점수 AUROC | B 추적 AUROC |", "|---|---|---|---|"]
    for r in SC:
        L.append(f"| {r} | {res['A'][r]['점수 누움AUROC']} | {res['B'][r]['점수 누움AUROC']} | {res['B'][r]['추적 AUROC']} |")
    L.append(f"- S1 A 에서 {pick} 고름 · 짝 B {res['S1']['짝 B (single → 고른 것)']} → {'✅ 채택' if res['S1']['채택'] else '❌ 현행 유지'}")
    L.append(f"- S2 짝 B {res['S2']['짝 B (single → A4p)']} → {'✅ 채택' if res['S2']['채택'] else '❌ 현행 유지'}")
    L.append(f"- S3 무동작 {json.dumps(res['S3 무동작 (확정 추적)'], ensure_ascii=False)} · 잡힌 비율 {res['S3 누움 잡힌 비율']}")
    with open(REP, "a") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
