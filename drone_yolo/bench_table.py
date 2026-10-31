"""
bench_table.py — metrics/bench_all.csv → 표 (마크다운 · 볼트에 붙임) (11-01)
  기성 모델은 imgsz 1280 · 640 중 셋마다 좋은 쪽 · ⚠ = 우리도 그 셋을 학습에 씀 (SARD · HERIDAL 전부)
실행  python bench_table.py > metrics/bench_all.md
"""
import csv
from collections import defaultdict
from pathlib import Path

from bench_all import EXTS, OURS, SETS

BASE = Path(__file__).resolve().parent
LABEL = {"test_sar": "UAV-SAR", "test_obl": "Okutama", "test_v2": "WiSARD", "test_kr": "AI-Hub", "test_ladd_h": "LADD 뒤30%", "test_visdrone": "VisDrone val",
         "test_nomad": "NOMAD 96~", "test_nii": "NII-CU", "test_auair": "AU-AIR", "test_sard": "SARD ⚠", "test_heridal": "HERIDAL ⚠", "test_cy": "Cyprus"}
OWN = {"ap_heridal": "test_heridal", "ap_ladd": "test_ladd_h", "ap_nomad": "test_nomad", "ap_sard": "test_sard", "visdrone": "test_visdrone"}

R = defaultdict(dict)
for r in csv.DictReader(open(BASE / "metrics" / "bench_all.csv")):
    if r["AP50"] == "":
        continue
    k = (r["model"], r["set"]); v = (float(r["AP50"]), r)
    if k not in R or v[0] > R[k][0]:
        R[k] = v
sets = [s for s in SETS if any((m, s) in R for m in OURS + list(EXTS))]


def row(m, mark=""):
    cells = []
    for s in sets:
        v = R.get((m, s))
        own = any(m.startswith(p) and OWN[p] == s for p in OWN)
        cells.append("—" if not v else (f"**{v[0]:.3f}**" if own else f"{v[0]:.3f}") + (f" ({v[1]['imgsz']})" if v and v[1]["kind"] == "ext" else ""))
    return f"| {mark}{m} | " + " | ".join(cells) + " |"


print("AP50 (IoU 0.5) · 같은 추론 틀 (가로 1920 → 1280×720 창 · NMS 0.6) · 기성 모델 괄호 = 고른 imgsz · **굵게** = 그 데이터셋 전용 모델의 자기 셋\n")
print("| 모델 | " + " | ".join(LABEL[s] for s in sets) + " |"); print("|---|" + "---|" * len(sets))
for m in OURS:
    if any((m, s) in R for s in sets):
        print(row(m, "우리 "))
for m in EXTS:
    if any((m, s) in R for s in sets):
        print(row(m, "기성 "))
fp = [(m, R[(m, "test_auair")][1]) for m in OURS + list(EXTS) if (m, "test_auair") in R]
if fp:
    print("\nAU-AIR 사람 없는 장 오탐 @0.20 (건/장): " + " · ".join(f"{m} {r['FP/neg@0.2']}" for m, r in fp))
