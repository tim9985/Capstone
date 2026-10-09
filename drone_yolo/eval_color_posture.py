"""
eval_color_posture.py — C4: 색 판정 영역을 자세로 고르면 인상착의 비교가 나아지는가 (10-09 · 판정 기준은 _학습 큐 「10-09 C4 · K1」)

  배경  ColorProfiler 는 pose="lying" 이면 상의 띠 (박스 위 15~60 %) 대신 박스 전체 ∩ 사람 마스크를 본다
        C1 (eval_appearance) 은 정답 자세를 넘겨 쟀지만 worker 는 자세를 안 넘긴다 → C1 은 worker 보다 유리한 조건이었다
  조건  ① none   자세 없음 (worker 지금)
        ② gt     정답 자세 (NOMAD 활동 → 서기 · 누움 · C1 조건) = 자세 · 몸통 위치를 더 알 때의 상한
        ③ v3     worker 표시 자세 V3 (Archangel 실제 + UE 로 학습 — NOMAD 밖) 의 1순위가 누움이면 누움 = 공정한 예측
        ④ b0     worker 점수 자세 B0 누움 확률 ≥ 0.5 (SARD + NOMAD 로 학습 → NOMAD 에서는 부풀려짐 · 참고)
  데이터 C1 과 같은 NOMAD 배우 42명 캐시 (eval_color · 1080p 크롭) · 배우 × 거리 × 자세 묶음 누적 = 후보 하나 · similar=0
  지표  (묶음, 질의 색) 쌍 점수 AUROC · 자기 색 MATCH 율 · 다른 색 오일치 (닮은 색 제외) — 전체 · 서기 · 누움 · 거리별
        배우 블록 짝 부트스트랩 1,000회 (③−① · ②−① · ④−①) · 자세 판정기 자체 정확도 (누움 재현율 · 서기를 누움으로)
실행: /home/se/miniconda3/envs/drone/bin/python eval_color_posture.py [--cache …]   → metrics/eval_color_posture.json
"""
import argparse
import json
import pickle
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / "deploy" / "vision_worker"))
from app.vision_core.appearance import SIMILAR, AppearanceComparator   # noqa: E402
from app.vision_core.color import ColorAccumulator, ColorProfiler      # noqa: E402
from app.vision_core.state import load_posture                         # noqa: E402
from eval_appearance import jobs_keys                                  # noqa: E402
from eval_color import MAP, PRIMARY                                    # noqa: E402
from posture_v2 import auc                                             # noqa: E402

CONDS = ("none", "gt", "v3", "b0")


def _prof(item):
    if item is None:
        return None
    p = ColorProfiler()
    return p.profile(item["crop"], item["box"], pose=None, stats=item["stats"]), p.profile(item["crop"], item["box"], pose="lying", stats=item["stats"])


def metrics(rows, sel=None):
    """rows: (배우, 거리, 묶음 자세, 양성, 닮은색 음성, 점수, 판정) · sel: 행 고르기"""
    r = [x for x in rows if sel is None or sel(x)]
    if not r:
        return None
    y = np.array([x[3] for x in r]); near = np.array([x[4] for x in r]); s = np.array([x[5] for x in r]); v = np.array([x[6] for x in r])
    return {"pairs": len(r), "AUROC": round(float(auc(s[y], s[~y])), 4),
            "own_MATCH": round(float(np.mean(v[y] == "MATCH")), 3), "own_UNDETERMINED": round(float(np.mean(v[y] == "UNDETERMINED")), 3),
            "false_MATCH_excl_similar": round(float(np.mean(v[~y & ~near] == "MATCH")), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/tmp/claude-1001/-home-se-JupyterLAB/027ea9bd-abdf-476f-ad0d-25f3e2ea7ffc/scratchpad/eval_color_cache.pkl")
    ap.add_argument("--posture", default=str(BASE / "runs_state" / "posture.json"))
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()
    keys, truth = jobs_keys(12)
    items = pickle.load(open(a.cache, "rb"))
    assert len(items) == len(keys)
    score_m, disp_m = load_posture(a.posture)
    with Pool(12) as pool:
        profs = pool.map(_prof, items, chunksize=32)
    # 크롭마다 예측 자세 (1080p 박스 w · h)
    acc = {c: defaultdict(ColorAccumulator) for c in CONDS}
    post = defaultdict(lambda: [0, 0])            # (모델, 정답) → [누움 예측 수, 전체]
    for k, it, pr in zip(keys, items, profs):
        if it is None or pr is None:
            continue
        x1, y1, x2, y2 = it["box"]; wh = [[x2 - x1, y2 - y1]]
        v3_lying = int(np.argmax(disp_m.proba(wh)[0])) == 0
        b0_lying = float(score_m.proba(wh)[0][0]) >= 0.5
        gt_lying = k[2] == "lying"
        for name, flag in (("v3", v3_lying), ("b0", b0_lying)):
            post[(name, k[2])][0] += flag; post[(name, k[2])][1] += 1
        for c, lying in (("none", False), ("gt", gt_lying), ("v3", v3_lying), ("b0", b0_lying)):
            acc[c][k].add(pr[1] if lying else pr[0])
    queries = sorted({PRIMARY[t] for t in truth.values() if t in PRIMARY})
    comp = AppearanceComparator(similar=0.0)
    rows = {c: [] for c in CONDS}
    for c in CONDS:
        for k, A in acc[c].items():
            r = A.result(); actor, dist, pose = k; t = truth[actor]; near_own = SIMILAR.get(PRIMARY[t], set())
            for q in queries:
                m = comp.compare(r, {"upper": [q]})
                rows[c].append((actor, dist, pose, q in MAP[t], q in near_own and q not in MAP[t], m["score"], m["verdict"]))
    res = {"groups": len(acc["none"]), "queries": queries,
           "posture_model_on_NOMAD": {f"{m} 누움 예측 · 정답 {g}": f"{v[0]}/{v[1]} ({v[0] / max(v[1], 1):.3f})" for (m, g), v in sorted(post.items())},
           "conditions": {}}
    for c in CONDS:
        res["conditions"][c] = {"all": metrics(rows[c]), "standing": metrics(rows[c], lambda x: x[2] == "standing"),
                                "lying": metrics(rows[c], lambda x: x[2] == "lying"),
                                "by_distance": {d: metrics(rows[c], lambda x, d=d: x[1] == d) for d in sorted({x[1] for x in rows[c]})}}
    # 배우 블록 짝 부트스트랩 (같은 행 순서 — 조건마다 묶음 · 질의 순서가 같다)
    keyrow = lambda r: (r[0], r[1], r[2])
    idx_of = {c: {} for c in CONDS}
    for c in CONDS:
        cnt = defaultdict(int)
        for i, r in enumerate(rows[c]):
            kq = (keyrow(r), cnt[keyrow(r)]); cnt[keyrow(r)] += 1; idx_of[c][kq] = i
    common = [kq for kq in idx_of["none"] if all(kq in idx_of[c] for c in CONDS)]
    actors = sorted({kq[0][0] for kq in common}); by_actor = defaultdict(list)
    for j, kq in enumerate(common):
        by_actor[kq[0][0]].append(j)
    arr = {c: (np.array([rows[c][idx_of[c][kq]][3] for kq in common]), np.array([rows[c][idx_of[c][kq]][5] for kq in common]),
               np.array([rows[c][idx_of[c][kq]][2] for kq in common])) for c in CONDS}
    rng = np.random.default_rng(0); diffs = defaultdict(list)
    for _ in range(a.boot):
        s = np.concatenate([by_actor[u] for u in rng.choice(actors, len(actors))])
        for c in ("gt", "v3", "b0"):
            for part, pm in (("all", None), ("standing", "standing"), ("lying", "lying")):
                yb, sb, pb = arr["none"][0][s], arr["none"][1][s], arr["none"][2][s]
                yc, sc = arr[c][0][s], arr[c][1][s]
                msk = np.ones(len(s), bool) if pm is None else (pb == pm)
                if yb[msk].any() and (~yb[msk]).any():
                    diffs[(c, part)].append(auc(sc[msk][yc[msk]], sc[msk][~yc[msk]]) - auc(sb[msk][yb[msk]], sb[msk][~yb[msk]]))
    res["AUROC_diff_vs_none_95"] = {f"{c} {p}": [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)] for (c, p), v in sorted(diffs.items())}
    lo_all = res["AUROC_diff_vs_none_95"]["v3 all"][0]; hi_st = res["AUROC_diff_vs_none_95"]["v3 standing"][1]
    res["C4 판정 (③ v3 채택)"] = bool(lo_all > 0 and hi_st >= 0)
    up = res["conditions"]["gt"]["all"]["AUROC"] - res["conditions"]["none"]["all"]["AUROC"]
    res["상한 (② − ①) AUROC"] = round(up, 4); res["색 쪽 관절점 트랙"] = "계속 검토" if up >= 0.02 else "접음 (상한 < 0.02)"
    (BASE / "metrics" / "eval_color_posture.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
