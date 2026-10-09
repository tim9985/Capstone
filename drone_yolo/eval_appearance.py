"""
eval_appearance.py — C1: 인상착의 비교 (worker app/vision_core/appearance.py · C-0705) 검증 (10-09 · 판정 기준은 _학습 큐 「10-09 C」)

  데이터  eval_color.py 와 같은 NOMAD 배우 42명 (상의 색 정답) · 같은 뽑기 (--n 12) · 캐시된 크롭 (git 밖)
  색 판정 worker 기본값 ColorProfiler() · 배우 × 거리 × 자세 묶음마다 ColorAccumulator 누적 = 후보 하나
  질의    정답 대표색에 들어 있는 색 (MAP) = 양성 · 그 밖 색 = 음성 (질의 색 = 배우들의 대표색 9종)
  지표    (묶음, 질의) 쌍의 점수 AUROC — similar=0.5 vs 0 · 배우 블록 부트스트랩 1,000회 · MATCH 율 · 오일치율 (닮은 색 포함 · 제외) · 거리별
실행: /home/se/miniconda3/envs/drone/bin/python eval_appearance.py [--cache <eval_color 캐시>]
출력: metrics/eval_appearance.json
"""
import argparse
import csv
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / "deploy" / "vision_worker"))
from app.vision_core.appearance import SIMILAR, AppearanceComparator   # noqa: E402
from app.vision_core.color import ColorAccumulator, ColorProfiler      # noqa: E402
from eval_color import MAP, POSE, PRIMARY                               # noqa: E402
from make_posture_crops import NOMAD, _nomad_act                        # noqa: E402
from posture_v2 import auc                                              # noqa: E402


def jobs_keys(n):
    """eval_color.main 과 같은 순서 (캐시와 맞춘다)"""
    truth = {r["actor"]: r["upper_color"].strip().lower() for r in csv.DictReader(open(BASE / "metrics" / "nomad_actor_selection.csv", encoding="utf-8-sig"))}
    acts = {x["id"]: x["labels"] for x in json.load(open(NOMAD / "activityLabels.json"))}
    groups = defaultdict(list)
    for r in json.load(open(NOMAD / "annotations.json")):
        actor_s, dist_s, f_s = r["file_name"][:-4].split("_")
        if actor_s not in truth or truth[actor_s] not in MAP:
            continue
        pose = POSE.get(_nomad_act(acts, int(actor_s[5:]), dist_s[1:], int(f_s[1:])))
        boxes = [b["bbox"] for b in r["annotations"] if int(b.get("visibility", 100)) >= 50]
        if pose is None or len(boxes) != 1:
            continue
        groups[(actor_s, dist_s, pose)].append(1)
    keys = []
    for k, v in sorted(groups.items()):
        idx = np.linspace(0, len(v) - 1, min(n, len(v))).round().astype(int)
        keys += [k] * len(set(idx))
    return keys, truth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--cache", default="/tmp/claude-1001/-home-se-JupyterLAB/027ea9bd-abdf-476f-ad0d-25f3e2ea7ffc/scratchpad/eval_color_cache.pkl")
    a = ap.parse_args()
    keys, truth = jobs_keys(a.n)
    items = pickle.load(open(a.cache, "rb"))
    assert len(items) == len(keys), f"캐시 {len(items)} ≠ 뽑기 {len(keys)} — eval_color 와 같은 --n 인지 확인"
    prof = ColorProfiler()
    acc = defaultdict(ColorAccumulator)
    for k, it in zip(keys, items):
        if it is not None:
            acc[k].add(prof.profile(it["crop"], it["box"], pose=it["pose"], stats=it["stats"]))
    queries = sorted({PRIMARY[t] for t in truth.values() if t in PRIMARY})
    rows = []                                                   # (배우, 거리, 양성, 닮은색 음성, 질의, 점수0, 점수.5, 판정.5)
    C0, C5 = AppearanceComparator(similar=0.0), AppearanceComparator(similar=0.5)
    for k, A in acc.items():
        r = A.result(); actor, dist, pose = k; t = truth[actor]
        near_own = SIMILAR.get(PRIMARY[t], set())
        for q in queries:
            m0, m5 = C0.compare(r, {"upper": [q]}), C5.compare(r, {"upper": [q]})
            rows.append((actor, dist, q in MAP[t], q in near_own and q not in MAP[t], q, m0["score"], m5["score"], m5["verdict"]))
    act = np.array([x[0] for x in rows]); y = np.array([x[2] for x in rows]); near = np.array([x[3] for x in rows])
    s0 = np.array([x[5] for x in rows]); s5 = np.array([x[6] for x in rows]); v = np.array([x[7] for x in rows]); dist = np.array([x[1] for x in rows])
    res = {"묶음": len(acc), "질의 색": queries, "쌍": len(rows), "양성 쌍": int(y.sum()),
           "AUROC similar=0": round(float(auc(s0[y], s0[~y])), 4), "AUROC similar=0.5": round(float(auc(s5[y], s5[~y])), 4)}
    ua = np.unique(act); idx = {u: np.where(act == u)[0] for u in ua}; rng = np.random.default_rng(0); d = []
    for _ in range(1000):
        s = np.concatenate([idx[u] for u in rng.choice(ua, len(ua))])
        if y[s].any() and (~y[s]).any():
            d.append(auc(s5[s][y[s]], s5[s][~y[s]]) - auc(s0[s][y[s]], s0[s][~y[s]]))
    res["AUROC 차 (0.5 − 0) 95%"] = [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)]
    res["similar 채택"] = bool(res["AUROC 차 (0.5 − 0) 95%"][0] > 0)
    res["판정 (similar=0.5)"] = {"자기 색 MATCH 율": round(float(np.mean(v[y] == "MATCH")), 3),
                                 "자기 색 MATCH+PARTIAL": round(float(np.mean(np.isin(v[y], ["MATCH", "PARTIAL"]))), 3),
                                 "자기 색 판정 불가": round(float(np.mean(v[y] == "UNDETERMINED")), 3),
                                 "다른 색 오일치율 (전체)": round(float(np.mean(v[~y] == "MATCH")), 3),
                                 "다른 색 오일치율 (닮은 색 제외)": round(float(np.mean(v[~y & ~near] == "MATCH")), 3),
                                 "닮은 색에 PARTIAL": round(float(np.mean(v[near] == "PARTIAL")), 3) if near.any() else None}
    res["거리별 AUROC (0.5)"] = {dd: round(float(auc(s5[(dist == dd) & y], s5[(dist == dd) & ~y])), 4) for dd in sorted(set(dist))}
    res["거리별 자기 색 MATCH 율"] = {dd: round(float(np.mean(v[(dist == dd) & y] == "MATCH")), 3) for dd in sorted(set(dist))}
    (BASE / "metrics" / "eval_appearance.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
