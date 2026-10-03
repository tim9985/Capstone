"""
fo_all_sets.py — 서버에 있는 원천 데이터셋마다 모델 예측을 얹어 FiftyOne 으로 본다 (2026-10-04)

  평가셋 3개(test_obl · test_v2 · test_kr)는 fo_eval_sets.py. 여기는 그 밖의 원천 — 학습에 넣은 크롭(1280×720)과
  안 쓴 데이터(C2A · UE · test_nii)까지. 데이터셋 이름 = 원천 이름 (`aihub` · `wisard` …)
  ⚠ 학습에 쓴 장면은 성적이 부풀려진다 — 장마다 태그로 구분: `학습` (train_v7) · `검증` (val_v6b) · `미사용`
  추론 규칙: 1920 장 → 타일 4장 + NMS 0.6 (판정과 같음) · 1280×720 크롭 → 한 장 + NMS 0.7 (test_obl 과 같음)
            · 더 작은 장(C2A) → 긴 변을 32 배수로 올린 크기로 한 장 (1280 으로 키우지 않는다)
  예측 캐시 runs_person/<모델>/preds_src_<원천>.npz (conf ≥ 0.01) · 앱에는 conf ≥ 0.15 만

실행: python fo_all_sets.py [모델=soup_v7r2] [원천 …]   (원천을 안 주면 전부 · 끝난 원천은 캐시로 건너뜀)
뺀 것: det_fov* · det_budget · det_fullframe (같은 원천을 화각·고도로 다시 자른 시험용 사본 ~7만 장)
"""
import glob
import math
import sys
from pathlib import Path

import fiftyone as fo
import numpy as np
from fiftyone import ViewField as F

import diag_misses as D

BASE = Path(__file__).resolve().parent
DATA = (BASE / "data").resolve()
TW, TH = 1280, 720
OP_CONF = 0.15
SRC = {  # 원천 → 이미지 폴더 glob (DATA 기준)
    "aihub": ["det_aihub/images/*/"],
    "wisard": ["det/wisard/images/*/"],
    "nomad": ["det/nomad_*/images/*/"],
    "nomad_v9": ["det_v6/nomad_v9/images/*/"],
    "sard": ["det_v6/sard/images/*/"],
    "heridal": ["det_v6/heridal/images/*/"],
    "nii": ["det_v6/nii/images/*/"],
    "test_nii": ["test_nii/images/"],
    "unicamp": ["det_v6/unicamp/images/*/"],
    "visdrone": ["det_v6/visdrone/images/*/"],
    "nadir_rot": ["det_v6/nadir_rot/images/*/"],
    "neg": ["det_neg/images/*/", "det_v6/uc_neg/images/*/"],
    "c2a": ["raw/C2A/new_dataset3/*/images/"],
    "ue_level01": ["pose_cls/ue_level01/images/"],
}


def files_of(src):
    return sorted(f for g in SRC[src] for d in glob.glob(str(DATA / g))
                  for f in glob.glob(d.rstrip("/") + "/*") if f.lower().endswith((".jpg", ".jpeg", ".png")))


def predict_cached(model_name, src, files):
    f = BASE / "runs_person" / model_name / f"preds_src_{src}.npz"
    if f.exists():
        return list(np.load(f, allow_pickle=True)["recs"])
    import cv2
    from ultralytics import YOLO
    model = YOLO(str(BASE / "runs_person" / model_name / "weights" / "best.pt"))
    recs = []
    for i, p in enumerate(files):
        img = cv2.imread(p); h, w = img.shape[:2]
        if w >= 1920:
            offs = [(x, y) for y in (0, h - TH) for x in (0, w - TW)]; tiles = [img[y:y + TH, x:x + TW] for x, y in offs]
            sz, nms = 1280, 0.6
        else:
            offs, tiles = [(0, 0)], [img]
            sz, nms = min(1280, max(320, 32 * math.ceil(max(w, h) / 32))), 0.7
        res = model.predict(tiles, imgsz=sz, conf=0.01, iou=0.9, max_det=1000, batch=len(tiles), half=True, verbose=False)
        B, C = [], []
        for (ox, oy), r in zip(offs, res):
            if len(r.boxes):
                B.append(r.boxes.xyxy.cpu().numpy() + np.array([ox, oy, ox, oy], np.float32)); C.append(r.boxes.conf.cpu().numpy())
        B = np.concatenate(B) if B else np.zeros((0, 4), np.float32); C = np.concatenate(C) if C else np.zeros(0, np.float32)
        recs.append((p, D.gt_boxes(Path(p), w, h), B, C, nms))
        if i % 1000 == 0:
            print(f"   {src} {i:,}/{len(files):,}", flush=True)
    np.savez_compressed(f, recs=np.array(recs, dtype=object))
    return recs


def usage_sets():
    rd = lambda n: {str(Path(l.strip()).resolve()) for l in open(BASE / "configs" / "lists" / n) if l.strip()}
    return rd("train_v7.txt") | rd("train_v7_r2.txt"), rd("val_v6b.txt")


def build(model_name, src, train, val):
    files = files_of(src)
    recs = predict_cached(model_name, src, files)
    key = f"eval_{model_name}"; field = "predictions" if model_name == "soup_v7r2" else f"pred_{model_name}"
    if fo.dataset_exists(src):
        ds = fo.load_dataset(src)
    else:
        ds = fo.Dataset(src, persistent=True)
        ds.add_samples([fo.Sample(filepath=r[0], tags=[Path(r[0]).parent.name,
                                                       "학습" if r[0] in train else "검증" if r[0] in val else "미사용"])
                        for r in recs])
    ds.compute_metadata()
    by = {s.filepath: s for s in ds}
    for p, G, B, C, nms in recs:
        s = by[p]; w, h = s.metadata.width, s.metadata.height
        k = D.nms(B, C, nms); B2, C2 = B[k], C[k]; hi = C2 >= OP_CONF
        box = lambda b: [b[0] / w, b[1] / h, (b[2] - b[0]) / w, (b[3] - b[1]) / h]
        if not s.has_field("ground_truth") or s["ground_truth"] is None:
            s["ground_truth"] = fo.Detections(detections=[fo.Detection(label="person", bounding_box=box(g)) for g in G])
        s[field] = fo.Detections(detections=[fo.Detection(label="person", bounding_box=box(b), confidence=float(c))
                                             for b, c in zip(B2[hi], C2[hi])])
        s.save()
    if key in ds.list_evaluations():
        ds.delete_evaluation(key)
    ds.evaluate_detections(field, gt_field="ground_truth", eval_key=key, iou=0.5, max_preds=300)
    views = {"1_놓침_많은_순": ds.sort_by(f"{key}_fn", reverse=True),
             "2_오탐_많은_순": ds.sort_by(f"{key}_fp", reverse=True),
             "3_놓친_사람만": ds.filter_labels("ground_truth", F(key) == "fn").filter_labels("predictions", F(key) == "fp", only_matches=False),
             "4_사람없는_장_오탐": ds.match(F("ground_truth.detections").length() == 0).match(F(f"{key}_fp") > 0),
             "5_학습에_안_쓴_장": ds.match_tags("미사용")}
    for n, v in views.items():
        if ds.has_saved_view(n):
            ds.delete_saved_view(n)
        ds.save_view(n, v)
    out = []
    for tag in ("학습", "검증", "미사용"):
        v = ds.match_tags(tag)
        if not len(v):
            continue
        tp, fp, fn = (sum(v.values(f"{key}_{t}")) for t in ("tp", "fp", "fn"))
        out.append(f"{tag} {len(v):,}장 정답 {tp + fn:,} 재현율 {tp / max(tp + fn, 1):.3f} 정밀도 {tp / max(tp + fp, 1):.3f} 오탐 {fp:,}")
    print(f"{src:10s} | " + " · ".join(out), flush=True)


if __name__ == "__main__":
    args = sys.argv[1:]
    model_name = args.pop(0) if args and args[0] not in SRC else "soup_v7r2"
    train, val = usage_sets()
    for src in (args or list(SRC)):
        build(model_name, src, train, val)
