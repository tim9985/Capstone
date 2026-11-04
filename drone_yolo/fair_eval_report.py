"""
fair_eval_report.py — runs_fair 검출 → 공정 비교 수치 (11-04) → metrics/fair_eval.json · 표 출력

  기성 최고   셋마다 기성 13개 × {창 4장 · 통째 1280} 중 그 셋 AP50 가 가장 높은 것 (평가셋으로 고름 → 기성에 유리 · 보수적)
  차이 구간   짝 블록 부트스트랩 2,000회 — 블록 = 같은 묶음 (이름 끝 번호 앞) 안 연속 20장 · 같은 사진 묶음으로 두 모델 AP 를 다시 셈
  헛탐지      ① 같은 재현율: 기성 문턱을 그 셋에서 우리 재현율 (@0.20) 과 같아지게 낮춤 → 그 문턱의 AU-AIR 사람 없는 장 헛탐지
              ② 같은 헛탐지: 기성 문턱을 도로 헛탐지가 우리 (@0.20) 와 같아지게 → 그 문턱의 재현율 · 사람 키별 재현율
  사람 키     가로 1920 으로 맞춘 그림에서의 px (모든 모델이 보는 크기 · fair_viz_data 의 1080/세로 는 4:3 그림에서 작게 셈 → 고침)
  셋 묶음     처음 보는 곳 U · 같은 데이터셋 다른 사진 B (보고서에 따로)
실행  /home/se/miniconda3/envs/drone/bin/python fair_eval_report.py
"""
import csv, json
from collections import defaultdict
from pathlib import Path

import numpy as np

from eval_external_det import ap101, iou_mat

BASE = Path(__file__).resolve().parent
RUN = BASE / "runs_fair"
OURS = "v12_f91_swa@tile1280"
U = {"test_sar": "UAV-SAR", "test_v2": "WiSARD 숲", "test_obl": "Okutama 비스듬", "test_kr": "AI-Hub 한국", "test_cy": "Cyprus"}
B = {"test_visdrone": "VisDrone", "test_nomad": "NOMAD", "test_ladd_h": "LADD"}
LABEL = U | B
HB = [(0, 24), (24, 48), (48, 96), (96, 1e9)]
RNG = np.random.default_rng(0)
NB = 2000


def load(s, tag):
    g = np.load(RUN / s / "gt.npz"); d = np.load(RUN / s / f"{tag}.npz")
    n = len(g["names"]); out = []
    for i in range(n):
        gt = g["gt"][g["gt_img"] == i]; m = d["img"] == i; box, cf = d["box"][m], d["conf"][m]
        o = np.argsort(-cf); box, cf = box[o], cf[o]; M = iou_mat(gt, box); used = np.zeros(len(gt), bool); best = np.zeros(len(gt)); tp = np.zeros(len(cf), bool)
        for j in range(len(cf)):
            if len(gt):
                c = np.where((M[:, j] >= 0.5) & ~used)[0]
                if len(c):
                    k = c[np.argmax(M[c, j])]; used[k] = True; best[k] = cf[j]; tp[j] = True
        w = g["sizes"][i][0]
        out.append({"conf": cf, "tp": tp, "n": len(gt), "best": best, "h": (gt[:, 3] - gt[:, 1]) * 1920 / w if len(gt) else np.zeros(0)})
    return out, list(g["names"])


def ap_of(per, idx):
    S = np.concatenate([per[i]["conf"] for i in idx]); T = np.concatenate([per[i]["tp"] for i in idx]); n = sum(per[i]["n"] for i in idx)
    return ap101(S, T, n) if n else float("nan")


def blocks(names):
    grp = defaultdict(list)
    for i, nm in enumerate(names):
        st = nm.rsplit(".", 1)[0]; grp[st.rsplit("_", 1)[0] if "_" in st else "all"].append(i)
    out = []
    for g in grp.values():
        out += [g[k:k + 20] for k in range(0, len(g), 20)]
    return out


def recall_at(per, t):
    n = sum(p["n"] for p in per); return float(sum((p["best"] >= t).sum() for p in per) / n) if n else float("nan")


def prec_at(per, t):
    T = np.concatenate([p["tp"][p["conf"] >= t] for p in per]); return float(T.mean()) if len(T) else 1.0


def fp_neg(per, t):
    neg = [p for p in per if p["n"] == 0]; return float(sum((p["conf"] >= t).sum() for p in neg) / len(neg))


def pr_curve(per):
    S = np.concatenate([p["conf"] for p in per]); T = np.concatenate([p["tp"] for p in per]); n = sum(p["n"] for p in per)
    o = np.argsort(-S); tp = np.cumsum(T[o]); fp = np.cumsum(~T[o]); rec = tp / n; prec = tp / np.maximum(tp + fp, 1)
    env = np.maximum.accumulate(prec[::-1])[::-1]; pts = []
    for r in np.linspace(0, rec[-1], 60):
        pts.append([round(float(r), 4), round(float(env[min(np.searchsorted(rec, r), len(env) - 1)]), 4)])
    return pts


def thr_for(fn, target, lo=0.01, hi=0.99, dec=True):
    """fn(t) 가 t 에 따라 줄어드는 양 (재현율 · 헛탐지) → fn(t) ≥ target 인 가장 큰 t (없으면 None)"""
    ts = np.round(np.arange(lo, hi + 1e-9, 0.005), 3); ok = [t for t in ts if fn(t) >= target - 1e-9]
    return float(max(ok)) if ok else None


def main():
    road = {}
    tags_road = [p.stem for p in (RUN / "test_auair").glob("*@*.npz")]
    for t in tags_road:
        road[t] = load("test_auair", t)[0]
    fp_ours = fp_neg(road[OURS], 0.20) if OURS in road else None
    rep = {}
    for r in csv.DictReader(open(BASE / "metrics" / "bench_all.csv")):
        if r["model"] in ("v12_f91_swa", "v12_f92_swa", "v12_f93_swa") and r["AP50"]:
            rep.setdefault(r["set"], {})[r["model"]] = float(r["AP50"])
    res = {"fp_ours_road": None if fp_ours is None else round(fp_ours, 3), "sets": {}}
    pool = {"ours": defaultdict(lambda: [0, 0]), "ext": defaultdict(lambda: [0, 0])}
    cm = {"U": {"ours": [0, 0, 0], "ext": [0, 0, 0]}}
    for s in LABEL:
        if not (RUN / s / "gt.npz").exists():
            continue
        tags = [p.stem for p in (RUN / s).glob("*@*.npz")]
        P = {t: load(s, t) for t in tags}
        names = P[OURS][1]; per = {t: v[0] for t, v in P.items()}; allidx = list(range(len(names)))
        aps = {t: ap_of(per[t], allidx) for t in tags}
        ext_tags = [t for t in tags if not t.startswith("v12_")]
        best = max(ext_tags, key=lambda t: aps[t])
        bl = blocks(names); diffs = []; ours_bs = []
        for _ in range(NB):
            idx = [i for b in RNG.integers(0, len(bl), len(bl)) for i in bl[b]]
            a, e = ap_of(per[OURS], idx), ap_of(per[best], idx); diffs.append(a - e); ours_bs.append(a)
        lo, hi = np.percentile(diffs, [2.5, 97.5]); olo, ohi = np.percentile(ours_bs, [2.5, 97.5])
        R_o, P_o = recall_at(per[OURS], 0.20), prec_at(per[OURS], 0.20)
        e_road = road.get(best)
        # ① 같은 재현율 → 기성 도로 헛탐지
        t_eq_r = thr_for(lambda t: recall_at(per[best], t), R_o)
        fp_eq_r = fp_neg(e_road, t_eq_r) if (e_road is not None and t_eq_r is not None) else None
        # ② 같은 도로 헛탐지 → 기성 재현율
        t_eq_fp = None
        if e_road is not None and fp_ours is not None:
            ok = [t for t in np.round(np.arange(0.01, 0.995, 0.005), 3) if fp_neg(e_road, t) <= fp_ours + 1e-9]
            t_eq_fp = float(min(ok)) if ok else 1.0             # 어떤 문턱에서도 우리 헛탐지까지 못 내려감 → 같은 조건이면 찾는 사람 0
        R_e_fp = recall_at(per[best], t_eq_fp) if t_eq_fp is not None else None
        hb = {}
        for a, b in HB:
            k = f"{a}-{int(b) if b < 1e8 else 'inf'}"
            for who, pp, t in (("ours", per[OURS], 0.20), ("ext", per[best], t_eq_fp)):
                if t is None:
                    continue
                h = np.concatenate([p["h"] for p in pp]); bb = np.concatenate([p["best"] for p in pp]); m = (h >= a) & (h < b)
                hb.setdefault(k, {})[who] = {"n": int(m.sum()), "R": round(float((bb[m] >= t).mean()), 4) if m.any() else None}
                if s in U:
                    pool[who][k][0] += int(m.sum()); pool[who][k][1] += int((bb[m] >= t).sum())
        if s in U:
            for who, pp, t in (("ours", per[OURS], 0.20), ("ext", per[best], t_eq_fp)):
                if t is None:
                    continue
                n = sum(p["n"] for p in pp); TP = int(sum((p["best"] >= t).sum() for p in pp)); FPc = int(sum(((~p["tp"]) & (p["conf"] >= t)).sum() for p in pp))
                cm["U"][who][0] += TP; cm["U"][who][1] += n - TP; cm["U"][who][2] += FPc
        rec = {"label": LABEL[s], "group": "U" if s in U else "B", "imgs": len(names), "n_gt": sum(p["n"] for p in per[OURS]), "blocks": len(bl),
               "ours": {"AP50": round(aps[OURS], 4), "CI": [round(olo, 4), round(ohi, 4)], "R020": round(R_o, 4), "P020": round(P_o, 4),
                        "repeats": rep.get(s, {}), "pr": pr_curve(per[OURS])},
               "ext": {"tag": best, "AP50": round(aps[best], 4), "R020": round(recall_at(per[best], 0.20), 4), "pr": pr_curve(per[best]),
                       "t_eq_recall": t_eq_r, "road_fp_eq_recall": None if fp_eq_r is None else round(fp_eq_r, 3), "max_recall": round(recall_at(per[best], 0.01), 4),
                       "t_eq_fp": t_eq_fp, "R_eq_fp": None if R_e_fp is None else round(R_e_fp, 4), "road_fp_020": None if e_road is None else round(fp_neg(e_road, 0.20), 3)},
               "diff": {"AP50": round(aps[OURS] - aps[best], 4), "CI": [round(lo, 4), round(hi, 4)],
                        "verdict": "앞섬" if lo > 0 else ("뒤짐" if hi < 0 else "구분 안 됨")},
               "top_ext": sorted([[t, round(aps[t], 4)] for t in ext_tags], key=lambda x: -x[1])[:5], "height": hb}
        res["sets"][s] = rec
        print(f"{LABEL[s]:14s} 우리 {aps[OURS]:.3f} [{olo:.3f},{ohi:.3f}] | 기성 {best} {aps[best]:.3f} | 차 {aps[OURS]-aps[best]:+.3f} [{lo:+.3f},{hi:+.3f}] {rec['diff']['verdict']}"
              f" | 같은 재현율 {R_o:.2f} → 기성 문턱 {t_eq_r} 도로 {fp_eq_r} | 같은 헛탐지 {fp_ours} → 기성 문턱 {t_eq_fp} 재현율 {R_e_fp}", flush=True)
    res["height_pooled_U"] = {who: {k: {"n": v[0], "R": round(v[1] / v[0], 4) if v[0] else None} for k, v in d.items()} for who, d in pool.items()}
    res["confusion_U"] = cm["U"]
    json.dump(res, open(BASE / "metrics" / "fair_eval.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps({"height_pooled_U": res["height_pooled_U"], "confusion_U": res["confusion_U"], "fp_ours_road": res["fp_ours_road"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
