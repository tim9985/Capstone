"""
posture_a5.py — A5: A4 (방위 변화) 를 실전 파이프라인 (탐지 박스 · BoT-SORT 추적 · Okutama 비스듬) 에서 확인 (2026-10-07)
판정 기준은 _학습 큐 「10-07 A5」 에 미리 적음

  입력  runs_state/samples_a5.json — state_pipeline.py (soup_v9x2 · botsort_t015 · --reset-jump · --posture=box · --dump)
        1초 표본마다 지금 박스 (h_wh 마지막) 와 같은 추적의 지난 표본 박스 (h_wh · ID 바뀜이면 끊김)
  판정기 (모두 Archangel-Real 전체로 맞춤 · Okutama 는 평가만 · 텔레메트리 없음 → φ · 크기×거리는 못 씀)
        B0   지금 (SARD + NOMAD 박스 2특징) = 표본의 p
        A3-0 박스 2특징 · Archangel 로 맞춤
        A4'  박스 2특징 + 지난 3 초 (1초 간격 3표본 · 현재 포함) log(h/w) 최소 · 최대 · 표준편차
             Archangel 학습도 같은 간격으로 (0.5 초 크롭 중 10 프레임 간격만)
  지표  (Okutama 판정 영상 B = 이름순 홀수 11편 · Q1 과 같은 분할 · A 는 참고)
        자세만: 누움 AUROC (P(누움) · 누움 vs 나머지)
        점수: 0.5·누움 + 0.2·(앉음 · 무릎 자리) + 0.3·min(무동작, 20)/20 → 누움 AUROC · 추적 단위 AUROC
        서기 → 누움 (가장 높은 확률) 비율
        짝 = 영상 블록 부트스트랩 2,000회

실행: /home/se/venvs/state/bin/python posture_a5.py
출력: metrics/posture_a5.json · metrics/AUTO_RESULT_archangel.md (덧붙임)
"""
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from posture_archangel_eval import load as load_arc
from posture_v2 import auc
from state_trackfix import videos

BASE = Path(__file__).resolve().parent
REP = BASE / "metrics" / "AUTO_RESULT_archangel.md"
POSE = {"Lying": 0, "Sitting": 1, "Standing": 2, "Walking": 2, "Running": 2}


def wfeat(cur, hist):
    lr = math.log(max(cur[1], 1) / max(cur[0], 1)); ls = math.log(max(cur[0], cur[1], 1))
    h = [math.log(max(b, 1) / max(a, 1)) for a, b in hist]
    return [lr, ls], [lr, ls, min(h), max(h), float(np.std(h))]


def fit_arc():
    R = load_arc()
    tr = defaultdict(list)
    for r in R:
        tr[(r["place"], r["track"])].append(r)
    X2, X5, y = [], [], []
    for rows in tr.values():
        rows.sort(key=lambda r: int(r["frame"]))
        fr = {int(r["frame"]): r for r in rows}
        for r in rows:
            f = int(r["frame"])
            hist = [(fr[g]["w"], fr[g]["h"]) for g in (f - 20, f - 10, f) if g in fr]       # 1초 간격 3표본 (현재 포함)
            a, b = wfeat((r["w"], r["h"]), hist)
            X2.append(a); X5.append(b); y.append(r["y"])
    out = {}
    for name, X in (("A3-0", np.array(X2)), ("A4'", np.array(X5))):
        sc = StandardScaler().fit(X)
        out[name] = (sc, LogisticRegression(max_iter=3000, class_weight="balanced").fit(sc.transform(X), np.array(y)))
    return out


def main():
    M = fit_arc()
    S = [s for s in json.load(open(BASE / "runs_state" / "samples_a5.json")) if s.get("pose") in POSE and s.get("h_wh")]
    vids, A, B = videos()
    X2, X5 = [], []
    for s in S:
        hw = [tuple(x) for x in s["h_wh"]]
        a, b = wfeat(hw[-1], hw[-3:])
        X2.append(a); X5.append(b)
    P = {"B0": np.array([s["p"] for s in S])}
    for name, X in (("A3-0", np.array(X2)), ("A4'", np.array(X5))):
        sc, m = M[name]; P[name] = m.predict_proba(sc.transform(X))         # lying · kneeling(앉음 자리) · standing
    y = np.array([POSE[s["pose"]] for s in S]); ly = y == 0
    still = np.array([min(s["still"], 20) / 20 for s in S])
    vid = np.array([s["vid"] for s in S]); trk = np.array([f"{s['vid']}#{s['t']}" for s in S])
    score = {k: 0.5 * p[:, 0] + 0.2 * p[:, 1] + 0.3 * still for k, p in P.items()}

    def meas(k, idx):
        p, sc = P[k][idx], score[k][idx]; l = ly[idx]; st = y[idx] == 2
        tk = trk[idx]; ut, inv = np.unique(tk, return_inverse=True)
        tmax = np.full(len(ut), -np.inf); np.maximum.at(tmax, inv, sc)
        tl = np.zeros(len(ut), bool); np.logical_or.at(tl, inv, l)
        return {"자세 누움AUROC": auc(p[l, 0], p[~l, 0]), "점수 누움AUROC": auc(sc[l], sc[~l]),
                "추적 AUROC": auc(tmax[tl], tmax[~tl]) if tl.any() and (~tl).any() else float("nan"),
                "서기→누움": float(np.mean(p[st].argmax(1) == 0)), "누움→서기": float(np.mean(p[l].argmax(1) == 2))}

    def block(vs):
        return np.concatenate([np.where(vid == v)[0] for v in vs]) if vs else np.array([], int)

    res = {"표본": {"전체": int(len(S)), "누움": int(ly.sum())}, "A": {}, "B": {}, "짝 B": {}}
    for split, vs in (("A", A), ("B", B)):
        idx = block(vs)
        for k in P:
            res[split][k] = {m: round(float(v), 4) for m, v in meas(k, idx).items()}
    rng = np.random.default_rng(0); Bv = np.array(B)
    for k in ("A3-0", "A4'"):
        d = defaultdict(list)
        for _ in range(2000):
            idx = block(list(rng.choice(Bv, len(Bv))))
            if not ly[idx].any():
                continue
            ma, mb = meas("B0", idx), meas(k, idx)
            for m in ("자세 누움AUROC", "점수 누움AUROC", "추적 AUROC"):
                if not (math.isnan(ma[m]) or math.isnan(mb[m])):
                    d[m].append(mb[m] - ma[m])
        res["짝 B"][f"B0 → {k}"] = {m: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)] for m, v in d.items()}
    (BASE / "metrics" / "posture_a5.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    L = [f"\n## A5 — 실전 파이프라인 (soup_v9x2 · Q1 추적 · Okutama 비스듬 · 1초 표본 {res['표본']})",
         "| 판정기 | B 자세 누움 AUROC | B 점수 누움 AUROC | B 추적 AUROC | B 서기→누움 | B 누움→서기 | (A 점수 AUROC) |", "|---|---|---|---|---|---|---|"]
    for k in P:
        b, a = res["B"][k], res["A"][k]
        L.append(f"| {k} | {b['자세 누움AUROC']} | {b['점수 누움AUROC']} | {b['추적 AUROC']} | {b['서기→누움']} | {b['누움→서기']} | {a['점수 누움AUROC']} |")
    for k, v in res["짝 B"].items():
        L.append(f"- 짝 B {k}: {v}")
    with open(REP, "a") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
