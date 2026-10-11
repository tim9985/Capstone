"""
diag_small.py — S 팔 설명용 표 (판정 아님 · chain_s.sh S4) → 마크다운을 표준 출력으로

  모델마다 (soup_v7r2 대비) test_v2 장소별 AP50 (edge 합치기) · 크기별 재현율@0.15 (test_v2 · test_obl)
  입력: runs_person/dumps/<모델>_<tag>.npz (dump_dets.py)
실행: /home/se/miniconda3/envs/drone/bin/python diag_small.py v11s_f71 v11s_f72 [soup_v11s]
"""
import sys
from collections import defaultdict

import numpy as np

from analyze_dets import ap_at, match
from diag_tile_merge import run

BINS = [(0, 24), (24, 36), (36, 50), (50, 80), (80, 1e9)]


def size_recall(dets, gts, op=0.15):
    hit = defaultdict(lambda: [0, 0])
    for i, gt in enumerate(gts):
        b, s = dets[i]; k = s >= op; _, used = match(gt, b[k], s[k], 0.5)
        L = np.maximum(gt[:, 2] - gt[:, 0], gt[:, 3] - gt[:, 1]) if len(gt) else np.zeros(0)
        for lo, hi in BINS:
            sel = (L >= lo) & (L < hi); hit[(lo, hi)][0] += int(used[sel].sum()); hit[(lo, hi)][1] += int(sel.sum())
    return [hit[x][0] / max(hit[x][1], 1) for x in BINS]


def main():
    models = ["soup_v7r2"] + sys.argv[1:]
    print("\n## S4 설명용 — 장소별 · 크기별 (판정 아님)\n")
    rows = {}
    for m in models:
        d, g, names = run(m, "test_v2", "edge")
        places = sorted({n.rsplit("_", 1)[0] for n in names})
        pa = [ap_at(d, g, 0.5, sel=lambda i, p=p: names[i].rsplit("_", 1)[0] == p) for p in places]
        do, go, _ = run(m, "test_obl", "base")
        rows[m] = (pa, size_recall(d, g), size_recall(do, go))
    short = [p.split("_")[1] + ("_" + p[-4:] if p[-4:].isdigit() else "") for p in places]
    print("| 모델 | " + " | ".join(f"v2 {s}" for s in short) + " |\n|---|" + "---|" * len(short))
    for m, (pa, _, _) in rows.items():
        print(f"| {m} | " + " | ".join(f"{x:.3f}" for x in pa) + " |")
    lab = [f"{lo}-{hi if hi < 1e9 else '∞'}" for lo, hi in BINS]
    for k, name in ((1, "test_v2 (edge)"), (2, "test_obl")):
        print(f"\n크기별 재현율@0.15 — {name}\n\n| 모델 | " + " | ".join(lab) + " |\n|---|" + "---|" * len(lab))
        for m, r in rows.items():
            print(f"| {m} | " + " | ".join(f"{x:.3f}" for x in r[k]) + " |")


if __name__ == "__main__":
    main()
