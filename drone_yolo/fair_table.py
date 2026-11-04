"""
fair_table.py — 기성 모델 × 데이터셋 AP50 표 (11-04 · 사용자 「각 기성모델과 데이터셋별 비교 표」) → metrics/fair_table.json
  runs_fair 검출 (fair_eval.py) 에서 (모델, 방식, 셋) AP50 → 기성은 계열마다 크기 (l · m) · 방식 (4조각 · 통째 1280) 중 그 셋 최고 · 우리는 운용 4조각 그대로
  셋 = fair_eval_report 의 처음 보는 5곳 + 같은 데이터셋 3곳 (SARD · HERIDAL · NII-CU 는 우리 학습과 겹쳐 뺌)
"""
import json
from fair_eval_report import BASE, RUN, U, B, load, ap_of
FAM = {"우리 팀 (v12)": ["v12_f91_swa"], "AeroNova": ["aeronova"], "AerialPerson 혼합": ["ap_mix_26l"],
       "HERIDAL 전용": ["ap_heridal_26l", "ap_heridal_26m"], "LADD 전용": ["ap_ladd_26l", "ap_ladd_26m"], "NOMAD 전용": ["ap_nomad_26l", "ap_nomad_26m"],
       "SARD 전용": ["ap_sard_26l", "ap_sard_26m"], "VisDrone YOLOv9e": ["visdrone_yolov9e"], "VisDrone YOLO11x": ["visdrone_yolo11x"], "VisDrone YOLO26x": ["visdrone_yolo26x"]}
TRAINED = {"AerialPerson 혼합": {"test_nomad", "test_ladd_h"}, "LADD 전용": {"test_ladd_h"}, "NOMAD 전용": {"test_nomad"},
           "VisDrone YOLOv9e": {"test_visdrone"}, "VisDrone YOLO11x": {"test_visdrone"}, "VisDrone YOLO26x": {"test_visdrone"},
           "우리 팀 (v12)": {"test_visdrone", "test_nomad", "test_ladd_h"}}
out = {"sets": {**U, **B}, "groups": {"U": list(U), "B": list(B)}, "rows": {}}
for s in list(U) + list(B):
    tags = [p.stem for p in (RUN / s).glob("*@*.npz")]; n = None
    aps = {}
    for t in tags:
        per, names = load(s, t); aps[t] = ap_of(per, range(len(names)))
    for fam, ms in FAM.items():
        cand = [(v, t) for t, v in aps.items() if t.split("@")[0] in ms and (fam != "우리 팀 (v12)" or t.endswith("tile1280"))]
        v, t = max(cand)
        out["rows"].setdefault(fam, {})[s] = {"AP50": round(v, 4), "tag": t, "trained": s in TRAINED.get(fam, set())}
    print(s, {f: out["rows"][f][s]["AP50"] for f in FAM}, flush=True)
json.dump(out, open(BASE / "metrics" / "fair_table.json", "w"), ensure_ascii=False, indent=1)
