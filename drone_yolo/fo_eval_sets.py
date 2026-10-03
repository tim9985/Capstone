"""
fo_eval_sets.py — 평가셋 세 개에 모델 예측을 얹어 FiftyOne 으로 본다 (2026-10-04)

  test_obl (Okutama 비스듬 · 단일 추론 · NMS 0.7) · test_v2 (WiSARD 숲 · 타일 4장 · NMS 0.6)
  test_kr (AI-Hub 산악5 · 타일 4장 · NMS 0.6) — 판정 스크립트와 같은 추론 규칙
  예측은 analyze_models.predict_cached 의 캐시(runs_person/<모델>/preds_<평가셋>.npz · conf ≥ 0.01)를 쓰고,
  없으면 만든다. 앱에는 운용 임계 conf ≥ 0.15 만 올린다.
  FiftyOne 평가(IoU 0.5) → 박스마다 tp/fp/fn · 장마다 eval_<모델>_tp/fp/fn

실행: python fo_eval_sets.py soup_v7r2 [test_obl test_v2 test_kr]
보기: 브라우저 http://<서버>:5151 → 데이터셋 고르기 (앱은 start_fiftyone_app.sh 로 늘 떠 있다)
"""
import sys
from pathlib import Path

import fiftyone as fo
import numpy as np

import analyze_models as A
import diag_misses as D

BASE = Path(__file__).resolve().parent
A.SETS.setdefault("test_kr", ("tile", 0.6))
OP_CONF = 0.15


def build(model, tag):
    recs = A.predict_cached(model, tag)
    op = A.SETS[tag][1]; key = f"eval_{model}"
    ds = fo.load_dataset(tag) if fo.dataset_exists(tag) else fo.Dataset(tag, persistent=True)
    if not len(ds):
        ds.add_samples([fo.Sample(filepath=str((BASE / "data").resolve() / tag / "images" / f"{r[0]}.jpg")) for r in recs])
    ds.compute_metadata()
    by = {Path(s.filepath).stem: s for s in ds}
    field = "predictions" if model == "soup_v7r2" else f"pred_{model}"
    for stem, G, B, C in recs:
        s = by[stem]; w, h = s.metadata.width, s.metadata.height
        k = D.nms(B, C, op); B2, C2 = B[k], C[k]; hi = C2 >= OP_CONF
        box = lambda b: [b[0] / w, b[1] / h, (b[2] - b[0]) / w, (b[3] - b[1]) / h]
        if "ground_truth" not in s or s.ground_truth is None:
            s["ground_truth"] = fo.Detections(detections=[fo.Detection(label="person", bounding_box=box(g)) for g in G])
        s[field] = fo.Detections(detections=[fo.Detection(label="person", bounding_box=box(b), confidence=float(c))
                                             for b, c in zip(B2[hi], C2[hi])])
        s.save()
    if key in ds.list_evaluations():
        ds.delete_evaluation(key)
    r = ds.evaluate_detections(field, gt_field="ground_truth", eval_key=key, iou=0.5, compute_mAP=True, max_preds=300)
    tp, fp, fn = (sum(ds.values(f"{key}_{t}")) for t in ("tp", "fp", "fn"))
    print(f"{tag:9s} {model}: 장 {len(ds):,} · 정답 {tp + fn:,} · 찾음 {tp:,} · 오탐 {fp:,} · 놓침 {fn:,}"
          f" · 재현율@0.15 {tp / max(tp + fn, 1):.3f} · 정밀도 {tp / max(tp + fp, 1):.3f}", flush=True)


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "soup_v7r2"
    for tag in (sys.argv[2:] or ["test_obl", "test_v2", "test_kr"]):
        build(model, tag)
