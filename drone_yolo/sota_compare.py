"""
sota_compare.py — 우리 v12 대 일반 탐지 SOTA · 항공 기성 (11-07 · 사용자 「철저하게 냉정하게」) → metrics/sota_compare.json · 표 출력

  모델   우리 v12 (운용 4조각 그대로) · SOTA (YOLO11x · RT-DETR-x · YOLO-World · Grounding DINO) · 항공 기성 최고 2개 (AeroNova · VisDrone-v9e)
         SOTA · 기성 = 평가셋마다 방식 (4조각 1280 · 640 · 통째) 중 AP50 최고 (평가셋으로 고름 → 상대에 유리)
  지표   AP50 · AP50-95 · 최적 F1 의 P · R · 같은 도로 헛탐지 (AU-AIR 233장 · 우리 0.20 의 값) 에서 재현율
  판정   짝 블록 부트스트랩 2,000회 (연속 20장 묶음) — 평가셋마다 · 처음 보는 3곳 합 (세 셋 AP 평균의 차)
"""
import json
import numpy as np
from eval_external_det import ap101
from fair_eval_report import BASE, RUN, blocks, load, ap_of, recall_at, fp_neg

U3 = ["test_sar", "test_obl", "test_cy"]; W2 = ["test_v2", "test_kr"]; B3 = ["test_visdrone", "test_nomad", "test_ladd_h"]
SETS = U3 + W2 + B3
OURS = "v12_f91_swa@tile1280"
RIVALS = {"YOLO11m COCO (우리 출발점)": "coco_yolo11m", "YOLO11x (COCO)": "sota_yolo11x", "RT-DETR-x (COCO)": "sota_rtdetrx", "YOLO-World v2 (개방형)": "sota_yoloworld",
          "Grounding DINO (개방형)": "sota_gdino", "AeroNova (항공)": "aeronova", "VisDrone YOLOv9e (항공)": "visdrone_yolov9e"}
RNG = np.random.default_rng(0); NB = 2000


def main():
    road = {p.stem: load("test_auair", p.stem)[0] for p in (RUN / "test_auair").glob("*@*.npz")}
    fpo = fp_neg(road[OURS], 0.20)
    data, out = {}, {"road_negatives": sum(p["n"] == 0 for p in road[OURS]), "ours_road_fp": round(fpo, 3), "sets": {}, "pooled_U3": {}}
    for s in SETS:
        tags = [p.stem for p in (RUN / s).glob("*@*.npz")]; P = {t: load(s, t) for t in tags}
        names = P[OURS][1]; per = {t: v[0] for t, v in P.items()}; idx = list(range(len(names)))
        aps = {t: ap_of(per[t], idx) for t in tags}
        pick = {r: max([t for t in tags if t.split("@")[0] == m], key=lambda t: aps[t]) for r, m in RIVALS.items() if any(t.split("@")[0] == m for t in tags)}
        data[s] = (per, blocks(names), pick)
        rec = {"ours": round(aps[OURS], 4), "rivals": {}}
        bl = blocks(names); draws = [[i for b in RNG.integers(0, len(bl), len(bl)) for i in bl[b]] for _ in range(NB)]
        oa = np.array([ap_of(per[OURS], d) for d in draws])
        for r, t in pick.items():
            ra = np.array([ap_of(per[t], d) for d in draws]); lo, hi = np.percentile(oa - ra, [2.5, 97.5])
            Rr = None
            if t in road:
                ok = [th for th in np.round(np.arange(0.01, 0.995, 0.005), 3) if fp_neg(road[t], th) <= fpo + 1e-9]
                Rr = round(recall_at(per[t], min(ok)), 4) if ok else 0.0
            rec["rivals"][r] = {"tag": t, "AP50": round(aps[t], 4), "diff": round(aps[OURS] - aps[t], 4), "CI": [round(lo, 4), round(hi, 4)],
                                "verdict": "우리 앞섬" if lo > 0 else ("우리 뒤짐" if hi < 0 else "구분 안 됨"), "R_at_our_road_fp": Rr}
        rec["ours_R_at_020"] = round(recall_at(per[OURS], 0.20), 4)
        out["sets"][s] = rec; print(s, rec["ours"], {r: (v["AP50"], v["verdict"]) for r, v in rec["rivals"].items()}, flush=True)
    # 처음 보는 3곳 합 — 셋마다 묶음 다시 뽑아 세 AP 평균의 차
    for r in RIVALS:
        if not all(r in data[s][2] for s in U3):
            continue
        diffs = []; om = []; rm = []
        for _ in range(NB):
            a = b = 0.0
            for s in U3:
                per, bl, pick = data[s]; d = [i for k in RNG.integers(0, len(bl), len(bl)) for i in bl[k]]
                a += ap_of(per[OURS], d); b += ap_of(per[pick[r]], d)
            diffs.append((a - b) / 3)
        mo = np.mean([out["sets"][s]["ours"] for s in U3]); mr = np.mean([out["sets"][s]["rivals"][r]["AP50"] for s in U3])
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        out["pooled_U3"][r] = {"ours": round(mo, 4), "rival": round(mr, 4), "diff": round(mo - mr, 4), "CI": [round(lo, 4), round(hi, 4)],
                               "verdict": "우리 앞섬" if lo > 0 else ("우리 뒤짐" if hi < 0 else "구분 안 됨")}
        print("U3", r, out["pooled_U3"][r], flush=True)
    json.dump(out, open(BASE / "metrics" / "sota_compare.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
