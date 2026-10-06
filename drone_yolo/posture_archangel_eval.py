"""
posture_archangel_eval.py — 자세 판정기 개선 A3 · A4 · A6 (Archangel-Real · 2026-10-07)
판정 기준은 obsidian 「11 자세 판별 - 방식 · 기하 원리 · 오차 · 개선」 6절 · _학습 큐 「10-07 A」 에 미리 적음

  공통  data/pose_eval/archangel_real/crops.csv (A1) · 자세 lying · kneeling · standing (기어감 제외)
        나누기 = 촬영 회차 (AA_BP_<회차>) 단위 5겹 (같은 사람 · 장소가 학습/평가에 같이 안 들어감) → 겹 밖 예측 (OOF)
        점수 = P(누움) · 주 지표 = 누움 vs 서기 AUROC · 누움→서기 오판 (가장 높은 확률) · 방위 묶음 (A2 와 같은 정의)
        짝 비교 = 회차 블록 부트스트랩 1,000회 (같은 크롭 · 같은 정답)
  B0   지금 판정기 (SARD + NOMAD 로 맞춘 박스 2특징) — 학습 안 함
  A3-0 박스 2특징을 Archangel 로 다시 맞춤 (데이터 효과)
  A3   + 내려다보는 각 φ = atan(고도/반경) · 화면 세로 위치 · log(긴 변 px × 기울기 거리) (실제 크기 대용) · log(h/w)·φ   (G1 · G2)
  A4   A3 + 같은 추적의 지난 3 초 (현재 포함 6표본) log(h/w) 최소 · 최대 · 표준편차 (G3 · 방위 변화 · 과거만 씀)
  A6   DINOv2 (vit_base_patch14 · 224) 특징 + 로지스틱 — Archangel 로 학습 → Archangel OOF · **Okutama 비스듬** (장소 완전 분리) 판정
       + 박스 애매 구간 (Okutama h/w 1~3 · 1080p) 만 외형으로 바꾸는 규칙 (G5)

실행: /home/se/venvs/state/bin/python posture_archangel_eval.py a3|a4|a6|all
출력: metrics/posture_archangel_<단계>.json · metrics/AUTO_RESULT_archangel.md (덧붙임)
"""
import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from posture_fusion import CLS, box_feat, rows_of
from posture_v2 import auc

BASE = Path(__file__).resolve().parent
ARC = BASE / "data" / "pose_eval" / "archangel_real"
OKU = BASE / "data" / "pose_eval" / "okutama_obl"
REP = BASE / "metrics" / "AUTO_RESULT_archangel.md"
CL = ("lying", "kneeling", "standing")
BINS = [(0, 15), (15, 30), (30, 45), (45, 60), (60, 91)]


def load():
    R = [r for r in csv.DictReader(open(ARC / "crops.csv")) if r["pose"] in CL]
    for r in R:
        r["w"], r["h"] = float(r["w1080"]), float(r["h1080"])
        r["alt"], r["radius"] = int(r["alt"]), int(r["radius"])
        r["phi"] = math.atan2(r["alt"], r["radius"])
        r["rng"] = math.hypot(r["alt"], r["radius"])
        r["y"] = CL.index(r["pose"])
        r["sess"] = re.match(r"AA_BP_(\d+)_", r["place"]).group(1)
        r["phase"] = 360.0 * int(r["frame"]) / max(int(r["frame_n"]), 1)
        r["lr"] = math.log(max(r["h"], 1) / max(r["w"], 1))
    # 방위 (A2 와 같은 정의) — 누운 사람 추적마다 세로/가로가 가장 큰 위상에서의 차
    tr = defaultdict(list)
    for r in R:
        tr[(r["place"], r["track"])].append(r)
    for rows in tr.values():
        rows.sort(key=lambda r: int(r["frame"]))
        top = max(rows, key=lambda r: r["lr"])["phase"]
        for i, r in enumerate(rows):
            r["az"] = abs((r["phase"] - top + 90) % 180 - 90) if r["pose"] == "lying" and len(rows) >= 8 else None
            win = [x["lr"] for x in rows[max(0, i - 5):i + 1]]            # 지난 3 초 (0.5 초 × 6) · 과거만
            r["win"] = (min(win), max(win), float(np.std(win)), len(win))
    return R


def feats(R, kind):
    lr = np.array([r["lr"] for r in R]); ls = np.log([max(r["w"], r["h"], 1) for r in R])
    X = [lr, ls]
    if kind in ("a3", "a4"):
        phi = np.array([r["phi"] for r in R]); yb = np.array([float(r["y_bottom"]) - 0.5 for r in R])
        size = np.log([max(r["w"], r["h"], 1) * r["rng"] for r in R])
        X += [phi, yb, size, lr * phi]
    if kind == "a4":
        W = np.array([r["win"] for r in R])
        X += [W[:, 0], W[:, 1], W[:, 2], np.log(W[:, 3])]
    return np.c_[tuple(X)] if len(X) > 1 else np.array(X).T


def oof(R, X, groups, C=1.0):
    y = np.array([r["y"] for r in R]); P = np.zeros((len(R), 3))
    for tr, te in GroupKFold(5).split(X, y, groups):
        sc = StandardScaler().fit(X[tr])
        m = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(sc.transform(X[tr]), y[tr])
        P[te] = m.predict_proba(sc.transform(X[te]))
    return P


def b0(R):
    parts = [rows_of(CLS / p / "train") for p in ("sard", "nomad")]
    m = LogisticRegression(max_iter=2000, class_weight="balanced").fit(box_feat(np.concatenate([p[2] for p in parts])),
                                                                        np.concatenate([p[1] for p in parts]))
    P3 = m.predict_proba(box_feat(np.array([[r["w"], r["h"]] for r in R])))        # lying · sitting · standing
    return np.c_[P3[:, 0], P3[:, 1], P3[:, 2]]                                      # 가운데 = 앉음 (무릎 자리)


def score(R, P, name):
    ly = [i for i, r in enumerate(R) if r["pose"] == "lying"]; st = [i for i, r in enumerate(R) if r["pose"] == "standing"]
    pred = P.argmax(1)
    out = {"방법": name, "누움AUROC": round(auc(P[ly, 0], P[st, 0]), 4),
           "누움→서기": round(float(np.mean(pred[ly] == 2)), 3), "서기→누움": round(float(np.mean(pred[st] == 0)), 3),
           "방위별 누움→서기": {}}
    for a, b in BINS:
        sel = [i for i in ly if R[i]["az"] is not None and a <= R[i]["az"] < b]
        out["방위별 누움→서기"][f"{a}~{b}°"] = round(float(np.mean(pred[sel] == 2)), 3) if sel else None
    return out


def paired(R, Pa, Pb, rng, n=1000):
    """회차 블록 부트스트랩 — 누움 AUROC 차 (b − a) · 방위 0~30° 누움→서기 차"""
    sess = np.array([r["sess"] for r in R]); us = np.unique(sess); idx = {s: np.where(sess == s)[0] for s in us}
    y = np.array([r["pose"] for r in R]); az = np.array([r["az"] if r["az"] is not None else -1 for r in R])
    da, dm = [], []
    for _ in range(n):
        s = np.concatenate([idx[k] for k in rng.choice(us, len(us))])
        ly, st = s[y[s] == "lying"], s[y[s] == "standing"]
        if len(ly) and len(st):
            da.append(auc(Pb[ly, 0], Pb[st, 0]) - auc(Pa[ly, 0], Pa[st, 0]))
        a30 = ly[(az[ly] >= 0) & (az[ly] < 30)]
        if len(a30):
            dm.append(float(np.mean(Pb[a30].argmax(1) == 2) - np.mean(Pa[a30].argmax(1) == 2)))
    ci = lambda v: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
    return {"누움AUROC 차 (95%)": ci(da), "방위 0~30° 누움→서기 차 (95%)": ci(dm)}


def report(lines):
    with open(REP, "a") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


def run_a34(kind):
    R = load(); groups = [r["sess"] for r in R]; rng = np.random.default_rng(0)
    P0 = b0(R); P30 = oof(R, feats(R, "a2"), groups); Pk = oof(R, feats(R, kind), groups)
    res = {"B0": score(R, P0, "B0 지금 판정기"), "A3-0": score(R, P30, "A3-0 박스 2특징 · Archangel 로 맞춤"),
           kind.upper(): score(R, Pk, {"a3": "A3 + φ · 화면 위치 · 크기×거리", "a4": "A4 = A3 + 지난 3 초 비 최소·최대·흔들림"}[kind]),
           f"짝 B0 → {kind.upper()}": paired(R, P0, Pk, rng), f"짝 A3-0 → {kind.upper()}": paired(R, P30, Pk, rng)}
    if kind == "a4":
        P3 = oof(R, feats(R, "a3"), groups); res["짝 A3 → A4"] = paired(R, P3, Pk, rng)
    (BASE / "metrics" / f"posture_archangel_{kind}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    L = [f"\n## {kind.upper()} ({len(R):,} 크롭 · 회차 5겹 OOF)", "| 방법 | 누움 AUROC | 누움→서기 | 서기→누움 | 방위 0~15° | 15~30° | 30~45° | 45~60° | 60~90° |", "|---|---|---|---|---|---|---|---|---|"]
    for k in ("B0", "A3-0", kind.upper()):
        v = res[k]; b = v["방위별 누움→서기"]
        L.append(f"| {v['방법']} | {v['누움AUROC']} | {v['누움→서기']} | {v['서기→누움']} | " + " | ".join(str(b[x]) for x in b) + " |")
    for k in res:
        if k.startswith("짝"):
            L.append(f"- {k}: {res[k]}")
    report(L)


def run_a6():
    import torch
    from posture_foundation import Dino
    R = load(); groups = [r["sess"] for r in R]; rng = np.random.default_rng(0)
    dino = Dino()
    F = dino.image([str(ARC / r["file"]) for r in R])
    y = np.array([r["y"] for r in R])
    Cs = (0.1, 0.3, 1.0)
    best = max(Cs, key=lambda C: float(np.mean(oof(R, F, groups, C).argmax(1) == y)))
    P6 = oof(R, F, groups, best); P0 = b0(R)
    res = {"C": best, "Archangel B0": score(R, P0, "B0 지금 판정기"), "Archangel A6": score(R, P6, f"A6 DINOv2 · Archangel 겹 밖 (C={best})"),
           "짝 B0 → A6 (Archangel)": paired(R, P0, P6, rng)}
    # Okutama 비스듬 — 장소 완전 분리 · 누움 vs 나머지 (앉음 포함)
    sc = StandardScaler().fit(F); m = LogisticRegression(C=best, max_iter=3000, class_weight="balanced").fit(sc.transform(F), y)
    O = list(csv.DictReader(open(OKU / "crops.csv")))
    Fo = dino.image([str(OKU / r["file"]) for r in O]); Po = m.predict_proba(sc.transform(Fo))
    wh = np.array([[float(r["w1080"]) * 1.5, float(r["h1080"]) * 1.5] for r in O])     # Okutama 는 720p px → ×1.5
    parts = [rows_of(CLS / p / "train") for p in ("sard", "nomad")]
    mb = LogisticRegression(max_iter=2000, class_weight="balanced").fit(box_feat(np.concatenate([p[2] for p in parts])), np.concatenate([p[1] for p in parts]))
    Pb = mb.predict_proba(box_feat(wh))
    ratio = wh[:, 1] / wh[:, 0]; amb = (ratio >= 1) & (ratio <= 3)
    mix = np.where(amb, Po[:, 0], Pb[:, 0])
    pose = np.array([r["pose"] for r in O]); ly = pose == "lying"
    place = np.array([r["place"] for r in O]); up = np.unique(place); pidx = {p: np.where(place == p)[0] for p in up}

    def ok_ci(sa, sb):
        d = []
        for _ in range(1000):
            s = np.concatenate([pidx[k] for k in rng.choice(up, len(up))])
            l = ly[s]
            if l.any() and (~l).any():
                d.append(auc(sb[s][l], sb[s][~l]) - auc(sa[s][l], sa[s][~l]))
        return [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)]
    res["Okutama 비스듬"] = {
        "n": {k: int((pose == k).sum()) for k in ("lying", "sitting", "standing")},
        "누움AUROC 박스 (B0)": round(auc(Pb[ly, 0], Pb[~ly, 0]), 4), "누움AUROC A6 DINOv2": round(auc(Po[ly, 0], Po[~ly, 0]), 4),
        "누움AUROC G5 (애매 구간만 A6)": round(auc(mix[ly], mix[~ly]), 4),
        "애매 구간 (h/w 1~3) 크롭 수 · 그중 누움": [int(amb.sum()), int((amb & ly).sum())],
        "짝 B0 → A6 (95%)": ok_ci(Pb[:, 0], Po[:, 0]), "짝 B0 → G5 (95%)": ok_ci(Pb[:, 0], mix)}
    (BASE / "metrics" / "posture_archangel_a6.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    o = res["Okutama 비스듬"]; a, b = res["Archangel B0"], res["Archangel A6"]
    report([f"\n## A6 DINOv2 외형 (C={best})",
            f"- Archangel 겹 밖: 누움 AUROC B0 {a['누움AUROC']} → A6 {b['누움AUROC']} · 누움→서기 {a['누움→서기']} → {b['누움→서기']} · 방위별 A6 {b['방위별 누움→서기']}",
            f"- 짝 (Archangel): {res['짝 B0 → A6 (Archangel)']}",
            f"- **Okutama 비스듬** (장소 분리 · n {o['n']}): 누움 AUROC 박스 {o['누움AUROC 박스 (B0)']} · A6 {o['누움AUROC A6 DINOv2']} · G5 {o['누움AUROC G5 (애매 구간만 A6)']} · 애매 구간 {o['애매 구간 (h/w 1~3) 크롭 수 · 그중 누움']}",
            f"- 짝 (Okutama): B0 → A6 {o['짝 B0 → A6 (95%)']} · B0 → G5 {o['짝 B0 → G5 (95%)']}"])


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("a3", "all"): run_a34("a3")
    if what in ("a4", "all"): run_a34("a4")
    if what in ("a6", "all"): run_a6()
