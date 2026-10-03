"""
posture_v2.py — 자세 판별 학습 계획 P0 · P1 · P2 · P3 판정 · P4 판정 · 보고 (2026-10-03)

계획 → obsidian 「11 서버 학습 계획/08 자세 판별 학습 계획」 (판정 기준은 그 노트에 돌리기 전에 적었다)

  p3      특징 캐시 (DINOv2 · SigLIP 2 제로샷) → 비스듬 · 수직 판정기 고르기 → Okutama · AI-Hub 산악5 판정
          비스듬  조합 {box · box+relh · box+sig · box+relh+sig · box+dino · box+relh+dino · dino}
                  × 학습 출처 {SN = SARD+NOMAD · SNC = +C2A}
                  고르기 = **출처 하나 빼기** (SARD 빼고 → SARD 에서 · NOMAD 빼고 → NOMAD 에서)
                          점수 = 누움 AUROC · 앉음 vs 서기 AUROC (있는 것만) 평균 — Okutama 는 안 본다
          수직    {dino · box+dino} × {A = AI-Hub 학습 5곳 · AC = +C2A} — 산악8 (val) 매크로 F1 로 고름
          판정    P1 = box+relh vs box (같은 출처 · Okutama 비스듬 앉음 F1 짝 구간)
                  P2 = 고른 조합의 SNC vs SN (출처 빼기 점수 · Okutama 관찰)
                  P3 = 고른 경로 판정 vs 10-01 기준선 (비스듬 box·SN · 수직 dino·A) 짝 구간
                  P0 = 같은 추적 3 · 5초 누적 vs 한 장 (Okutama 비스듬 · 수직)
          출력    metrics/posture_v2_p3.json · weights/posture_v2_p3.pkl · runs_posture/p3_probs.npz
  judge4  P4 (posture_ft.py 3회 반복) vs P3 — 3회 모두 짝 구간 > 0 (한 평가셋 이상) · 평균 차 > 반복 폭
          · 어느 평가셋도 유의하게 나빠지지 않음 · 앉음 재현율 ≥ 0.2 → 채택이면 weights/posture_v2_best.txt = ft_s0
  report  metrics/AUTO_RESULT_posture.md

규칙 · 함정
  · Okutama crops.csv 의 w1080 · h1080 은 **1280×720 px 그대로** (make_posture_crops 가 1.0 배율로 적음) → 여기서 ×1.5
    (10-01 posture_fusion 은 이걸 몰랐다 — 박스 크기 특징만 영향 · 세로/가로 비는 그대로)
  · C2A 상대 키는 결측 처리 — 긴 변 ≥ 24 px 로 고른 탓에 다른 사람보다 늘 크다 (전 자세 중앙값 1.52)
  · 로지스틱 C 고정 (특징 조합마다 고르지 않는다) — 박스·relh·sig 만 1.0 · dino 포함 0.3
실행 (state 환경 · GPU): /home/se/venvs/state/bin/python posture_v2.py p3 | judge4 | report  [--smoke]
"""
import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scipy.stats import rankdata

from eval_posture import CLASSES, metrics


def auc(pos, neg):
    """동점 평균 순위 AUROC — eval_posture.auc 는 동점을 순서대로 매겨 확률이 전부 같으면 0 이 나온다."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return float("nan")
    r = rankdata(np.r_[pos, neg])
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))

BASE = Path(__file__).resolve().parent
CLS, EVAL = BASE / "data" / "pose_cls", BASE / "data" / "pose_eval"
FEAT = BASE / "data" / "pose_feat"
RUNS = BASE / "runs_posture"
SETS = {"sard": CLS / "sard" / "train", "nomad": CLS / "nomad" / "train", "c2a": CLS / "c2a" / "train",
        "aihub": CLS / "aihub" / "train", "aihub_val": CLS / "aihub" / "val",
        "aihub_test": EVAL / "aihub_test", "okutama_obl": EVAL / "okutama_obl", "okutama_nadir": EVAL / "okutama_nadir"}
EVAL_SETS = ("aihub_test", "okutama_obl", "okutama_nadir")
VIEW = {"aihub_test": "nadir", "okutama_obl": "oblique", "okutama_nadir": "nadir"}
OBL_KINDS = ("box", "box+relh", "box+sig", "box+relh+sig", "box+dino", "box+relh+dino", "dino")
OBL_SRC = {"SN": ("sard", "nomad"), "SNC": ("sard", "nomad", "c2a")}
NAD_KINDS = ("dino", "box+dino")
NAD_SRC = {"A": ("aihub",), "AC": ("aihub", "c2a")}
BASELINE = {"oblique": ("box", "SN"), "nadir": ("dino", "A")}
SMOKE = "--smoke" in sys.argv


# ── 데이터 ────────────────────────────────────────────────────────────────
def load_set(name):
    root = SETS[name]
    rows = list(csv.DictReader(open(root / "crops.csv")))
    if SMOKE:
        rows = rows[::max(1, len(rows) // 300)]
    rel = {}
    if (root / "relh.csv").exists() and name != "c2a":
        rel = {r["file"]: r for r in csv.DictReader(open(root / "relh.csv"))}
    k = 1.5 if name.startswith("okutama") else 1.0
    D = {"name": name, "files": [str(root / r["file"]) for r in rows],
         "y": np.array([CLASSES.index(r["pose"]) for r in rows]),
         "w": np.array([float(r["w1080"]) for r in rows]) * k, "h": np.array([float(r["h1080"]) for r in rows]) * k,
         "place": np.array([r["place"] if r["source"] == "okutama" else str(r["alt"] or r["place"]) for r in rows])}
    rh, rl, miss = [], [], []
    for r in rows:
        q = rel.get(r["file"])
        ok = q is not None and int(q["n_ref"]) > 0
        rh.append(float(q["rel_h"]) if ok else 1.0); rl.append(float(q["rel_long"]) if ok else 1.0); miss.append(0.0 if ok else 1.0)
    D["relh"] = np.c_[np.log(np.clip(rh, 0.05, 20)), np.log(np.clip(rl, 0.05, 20)), miss]
    if name.startswith("okutama"):
        parts = [Path(f).stem.split("_") for f in D["files"]]       # oku_<영상>_<프레임>_<추적 ID>
        D["fr"] = np.array([int(p[2]) for p in parts]); D["tid"] = np.array([int(p[3]) for p in parts])
    w, h = np.maximum(D["w"], 1), np.maximum(D["h"], 1)
    nadir = VIEW.get(name) == "nadir" or name.startswith("aihub")
    D["geo"] = np.maximum(w, h) / np.minimum(w, h) if nadir else -h / w   # eval_posture.load 와 같은 박스 모양 기준선
    return D


def features(sets):
    """DINOv2 특징 · SigLIP 2 제로샷 확률 — data/pose_feat/<묶음>.npz 캐시."""
    FEAT.mkdir(parents=True, exist_ok=True)
    need = [D for D in sets.values() if SMOKE or not (FEAT / f"{D['name']}.npz").exists()]
    got = {}
    if need:
        from posture_foundation import Dino, Siglip
        dino = Dino()
        got = {D["name"]: {"dino": dino.image(D["files"]).astype(np.float16)} for D in need}
        del dino
        sig = Siglip(); T = sig.text()
        for D in need:
            z = 100 * (sig.image(D["files"]) @ T.T); z = np.exp(z - z.max(1, keepdims=True))
            got[D["name"]]["sig"] = (z / z.sum(1, keepdims=True)).astype(np.float32)
            if not SMOKE:
                np.savez(FEAT / f"{D['name']}.npz", files=np.array(D["files"]), **got[D["name"]])
        del sig
    for D in sets.values():
        if SMOKE and D["name"] in got:
            D["dino"], D["sig"] = got[D["name"]]["dino"].astype(np.float32), got[D["name"]]["sig"]
            continue
        z = np.load(FEAT / f"{D['name']}.npz")
        assert list(z["files"]) == D["files"], f"특징 캐시가 크롭 목록과 다름: {D['name']} — data/pose_feat 를 지우고 다시"
        D["dino"], D["sig"] = z["dino"].astype(np.float32), z["sig"]


def X_of(D, kind):
    cols = []
    for part in kind.split("+"):
        if part == "box":
            w, h = np.maximum(D["w"], 1), np.maximum(D["h"], 1)
            cols.append(np.c_[np.log(h / w), np.log(np.maximum(w, h))])
        elif part == "relh":
            cols.append(D["relh"])
        elif part == "sig":
            cols.append(np.log(np.clip(D["sig"], 1e-4, 1)))
        elif part == "dino":
            cols.append(D["dino"])
    return np.concatenate(cols, 1)


def fit(sets, srcs, kind):
    X = np.concatenate([X_of(sets[s], kind) for s in srcs]); y = np.concatenate([sets[s]["y"] for s in srcs])
    C = 0.3 if "dino" in kind else 1.0
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000, class_weight="balanced")).fit(X, y)


def proba(clf, D, kind):
    p = clf.predict_proba(X_of(D, kind)); out = np.zeros((len(p), 3))
    for j, c in enumerate(clf.classes_):
        out[:, c] = p[:, j]
    return out


# ── 지표 ──────────────────────────────────────────────────────────────────
def sel_score(y, p):
    """출처 빼기 점수 — 누움 AUROC · 앉음 vs 서기 AUROC (둘 다 있을 때) 평균."""
    parts = [auc(p[y == 0, 0], p[y != 0, 0])]
    if (y == 1).any() and (y == 2).any():
        s = p[:, 1] / np.maximum(p[:, 1] + p[:, 2], 1e-9)
        parts.append(auc(s[y == 1], s[y == 2]))
    return float(np.mean(parts)), [round(v, 3) for v in parts]


def f1s(y, p):
    pr = p.argmax(1); out = []
    for i in range(3):
        tp = ((pr == i) & (y == i)).sum(); r = tp / max((y == i).sum(), 1); q = tp / max((pr == i).sum(), 1)
        out.append(2 * r * q / max(r + q, 1e-9))
    return out


def stats(y, p):
    f = f1s(y, p)
    return {"누움AUROC": auc(p[y == 0, 0], p[y != 0, 0]), "매크로F1": float(np.mean(f)), "앉음F1": f[1]}


def paired(y, pa, pb, block, n=1000):
    """B − A 의 블록 부트스트랩 95 % 구간 (누움 AUROC · 매크로 F1 · 앉음 F1)."""
    rng = np.random.default_rng(0); ub = np.unique(block); idx = {b: np.where(block == b)[0] for b in ub}
    base = {k: stats(y, pb)[k] - stats(y, pa)[k] for k in ("누움AUROC", "매크로F1", "앉음F1")}
    boot = {k: [] for k in base}
    for _ in range(n):
        s = np.concatenate([idx[b] for b in rng.choice(ub, len(ub))])
        yy = y[s]
        if not (yy == 0).any() or (yy == 0).all():
            continue
        a, b = stats(yy, pa[s]), stats(yy, pb[s])
        for k in boot:
            boot[k].append(b[k] - a[k])
    return {k: {"차": round(base[k], 3), "95%": [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)]}
            for k, v in boot.items()}


def aggregate(D, p, win):
    """같은 추적 (영상 · ID) 의 최근 win 초 로그 확률 평균 — 인과적 (미래 안 봄)."""
    lp = np.log(np.clip(p, 1e-6, 1)); out = np.empty_like(p)
    key = np.array([f"{a}|{b}" for a, b in zip(D["place"], D["tid"])])
    for k in np.unique(key):
        ii = np.where(key == k)[0]; ii = ii[np.argsort(D["fr"][ii])]; fr = D["fr"][ii]
        for j, i in enumerate(ii):
            m = ii[(fr > fr[j] - win * 30) & (fr <= fr[j])]
            z = lp[m].mean(0); z = np.exp(z - z.max()); out[i] = z / z.sum()
    return out


def full_metrics(D, p):
    return metrics(D["y"], p, D["geo"], D["place"], None if SMOKE else np.random.default_rng(0))


# ── P3 ────────────────────────────────────────────────────────────────────
def p3():
    sets = {n: load_set(n) for n in SETS}
    print("크롭", {n: len(D["y"]) for n, D in sets.items()}, flush=True)
    features(sets)
    res = {"고르기_비스듬_출처빼기": {}, "고르기_수직_산악8": {}}
    # 비스듬 — 출처 하나 빼기
    for v, srcs in OBL_SRC.items():
        for kind in OBL_KINDS:
            parts, det = [], {}
            for hold in ("sard", "nomad"):
                tr = [s for s in srcs if s != hold]
                sc, pp = sel_score(sets[hold]["y"], proba(fit(sets, tr, kind), sets[hold], kind))
                parts.append(sc); det[f"{hold} 빼고"] = pp
            res["고르기_비스듬_출처빼기"][f"{kind}|{v}"] = {"점수": round(float(np.mean(parts)), 4), **det}
            print(f"  비스듬 {kind:14s} {v:3s} 점수 {np.mean(parts):.4f} {det}", flush=True)
    ob = max(res["고르기_비스듬_출처빼기"], key=lambda k: res["고르기_비스듬_출처빼기"][k]["점수"])
    ob_kind, ob_v = ob.split("|")
    # 수직 — 산악8
    for v, srcs in NAD_SRC.items():
        for kind in NAD_KINDS:
            f = float(np.mean(f1s(sets["aihub_val"]["y"], proba(fit(sets, srcs, kind), sets["aihub_val"], kind))))
            res["고르기_수직_산악8"][f"{kind}|{v}"] = round(f, 4)
            print(f"  수직 {kind:9s} {v:2s} 산악8 매크로 F1 {f:.4f}", flush=True)
    nd = max(res["고르기_수직_산악8"], key=res["고르기_수직_산악8"].get)
    nd_kind, nd_v = nd.split("|")
    res["고른 것"] = {"oblique": [ob_kind, ob_v], "nadir": [nd_kind, nd_v]}
    print(f"→ 비스듬 {ob_kind} · {ob_v} / 수직 {nd_kind} · {nd_v}", flush=True)

    models = {"oblique": (ob_kind, ob_v, fit(sets, OBL_SRC[ob_v], ob_kind)),
              "nadir": (nd_kind, nd_v, fit(sets, NAD_SRC[nd_v], nd_kind))}
    base = {"oblique": ("box", "SN", fit(sets, OBL_SRC["SN"], "box")), "nadir": ("dino", "A", fit(sets, NAD_SRC["A"], "dino"))}
    probs, res["판정"] = {}, {}
    for s in EVAL_SETS:
        D, view = sets[s], VIEW[s]
        pn, pb = proba(models[view][2], D, models[view][0]), proba(base[view][2], D, base[view][0])
        probs[f"{s}|p3"], probs[f"{s}|base"] = pn, pb
        r = {"P3 고른 판정": full_metrics(D, pn), "10-01 기준선": full_metrics(D, pb),
             "P3 − 기준선 (짝)": paired(D["y"], pb, pn, D["place"])}
        if view == "oblique":                                        # P1 · P2 관찰 (같은 출처끼리)
            pbx = proba(fit(sets, OBL_SRC[ob_v], "box"), D, "box"); prh = proba(fit(sets, OBL_SRC[ob_v], "box+relh"), D, "box+relh")
            r["P1 box+relh − box (짝)"] = paired(D["y"], pbx, prh, D["place"])
            other = "SN" if ob_v == "SNC" else "SNC"
            po = proba(fit(sets, OBL_SRC[other], ob_kind), D, ob_kind)
            r["P2 관찰 SNC − SN (짝)"] = paired(D["y"], po, pn, D["place"]) if ob_v == "SNC" else paired(D["y"], pn, po, D["place"])
            r["모든 조합 (관찰 · 고르기에 안 씀)"] = {f"{k}|{v}": {kk: round(vv, 3) for kk, vv in stats(D["y"], proba(fit(sets, OBL_SRC[v], k), D, k)).items()}
                                         for v in OBL_SRC for k in OBL_KINDS}
        if s.startswith("okutama"):                                  # P0 추적 누적
            for win in (3, 5):
                pa = aggregate(D, pn, win); probs[f"{s}|p3_agg{win}"] = pa
                r[f"P0 누적 {win}초 − 한 장 (짝)"] = paired(D["y"], pn, pa, D["place"])
            r["P0 누적 5초 지표"] = full_metrics(D, probs[f"{s}|p3_agg5"])
        res["판정"][s] = r
        m = r["P3 고른 판정"]
        print(f"\n== {s} ({view}) P3 누움 AUROC {m['누움AUROC_분류기']} · 매크로 F1 {m['매크로F1']} · 재현율 {m['재현율']}"
              f"\n   vs 기준선 {r['P3 − 기준선 (짝)']}", flush=True)
        for k in r:
            if k.startswith(("P0", "P1", "P2")) and "지표" not in k:
                print(f"   {k}: {r[k]}", flush=True)
    if SMOKE:
        print("연기 실행 끝 (저장 안 함)"); return
    RUNS.mkdir(exist_ok=True)
    np.savez(RUNS / "p3_probs.npz", **probs)
    pickle.dump({k: {"kind": v[0], "sources": v[1], "clf": v[2]} for k, v in models.items()}, open(BASE / "weights" / "posture_v2_p3.pkl", "wb"))
    (BASE / "metrics" / "posture_v2_p3.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float))
    (BASE / "weights" / "posture_v2_best.txt").write_text("p3\n")


# ── P4 판정 ───────────────────────────────────────────────────────────────
def judge4():
    sets = {s: load_set(s) for s in EVAL_SETS}
    P3 = np.load(RUNS / "p3_probs.npz")
    seeds = sorted(RUNS.glob("ft_s*/probs.npz"))
    if not seeds:
        print("P4 결과 없음 — P3 유지"); return
    res, ok_all, better_any = {"반복": [p.parent.name for p in seeds]}, True, False
    for s in EVAL_SETS:
        D, a = sets[s], P3[f"{s}|p3"]
        per = [paired(D["y"], a, np.load(p)[s], D["place"]) for p in seeds]
        st = [stats(D["y"], np.load(p)[s]) for p in seeds]
        sit_rec = [float((np.load(p)[s].argmax(1)[D["y"] == 1] == 1).mean()) for p in seeds]
        r = {"짝 (P4 − P3)": per, "앉음 재현율": [round(v, 3) for v in sit_rec]}
        for k in ("누움AUROC", "매크로F1"):
            vals = [x[k] for x in st]; band = max(vals) - min(vals); mean_d = float(np.mean([x[k]["차"] for x in per]))
            all_pos = all(x[k]["95%"][0] > 0 for x in per); any_neg = any(x[k]["95%"][1] < 0 for x in per)
            r[k] = {"반복 값": [round(v, 3) for v in vals], "반복 폭": round(band, 3), "평균 차": round(mean_d, 3),
                    "3회 모두 > 0": all_pos, "유의하게 나쁨": any_neg}
            better_any |= all_pos and mean_d > band
            ok_all &= not any_neg
        ok_all &= min(sit_rec) >= 0.2
        res[s] = r
        print(f"== {s}: {json.dumps({k: v for k, v in r.items() if k != '짝 (P4 − P3)'}, ensure_ascii=False)}", flush=True)
    res["판정"] = "P4 채택 (ft_s0)" if (better_any and ok_all) else "P3 유지"
    print("→", res["판정"])
    (BASE / "metrics" / "posture_v2_judge4.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float))
    (BASE / "weights" / "posture_v2_best.txt").write_text("ft_s0\n" if res["판정"].startswith("P4") else "p3\n")


# ── 보고 ──────────────────────────────────────────────────────────────────
def report():
    L = ["# 자세 판별 P0~P5 자동 결과 (`chain_p.sh`)", ""]
    j = json.loads((BASE / "metrics" / "posture_v2_p3.json").read_text())
    L += [f"- 고른 판정: 비스듬 **{' · '.join(j['고른 것']['oblique'])}** · 수직 **{' · '.join(j['고른 것']['nadir'])}**", "",
          "| 평가셋 | P3 누움 AUROC | 매크로 F1 | 기준선 누움 AUROC | 매크로 F1 | P3−기준선 누움 (95 %) | 매크로 F1 (95 %) |", "|---|---|---|---|---|---|---|"]
    for s, r in j["판정"].items():
        a, b, d = r["P3 고른 판정"], r["10-01 기준선"], r["P3 − 기준선 (짝)"]
        L.append(f"| {s} | {a['누움AUROC_분류기']} | {a['매크로F1']} | {b['누움AUROC_분류기']} | {b['매크로F1']} | "
                 f"{d['누움AUROC']['차']} {d['누움AUROC']['95%']} | {d['매크로F1']['차']} {d['매크로F1']['95%']} |")
    L += ["", "| 평가셋 | 항목 | 누움 AUROC 차 (95 %) | 매크로 F1 차 | 앉음 F1 차 |", "|---|---|---|---|---|"]
    for s, r in j["판정"].items():
        for k, d in r.items():
            if k.startswith(("P0", "P1", "P2")) and "지표" not in k:
                L.append(f"| {s} | {k} | {d['누움AUROC']['차']} {d['누움AUROC']['95%']} | {d['매크로F1']['차']} {d['매크로F1']['95%']} | {d['앉음F1']['차']} {d['앉음F1']['95%']} |")
    jp = BASE / "metrics" / "posture_v2_judge4.json"
    if jp.exists():
        L += ["", f"- P4 판정: **{json.loads(jp.read_text())['판정']}** → `metrics/posture_v2_judge4.json`"]
    for f in sorted((BASE / "metrics").glob("state_pipeline_v9x2_pose_*.json")):
        d = json.loads(f.read_text()); pm = d.get("자세 지표 (탐지 박스)", {})
        L.append(f"- P5 {f.stem}: 점수 누움 AUROC {d['누움 가려내기 AUROC (1초 표본)']['요구조 점수']} · 추적 {d['추적 단위 누운 적 있음 AUROC']['최고 점수']}"
                 f" · 자세(탐지 박스) {pm}")
    out = BASE / "metrics" / "AUTO_RESULT_posture.md"
    out.write_text("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    {"p3": p3, "judge4": judge4, "report": report}[sys.argv[1]]()
