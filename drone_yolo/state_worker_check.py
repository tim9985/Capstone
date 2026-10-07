"""
state_worker_check.py — V4 판정: worker 상태 모듈 (app/vision_core/state.py) 이 연구 파이프라인 (Q1 V2·single) 과 같은 성능인가 (10-07)

  기준 (돌리기 전에 _학습 큐 에 적음): Okutama 비스듬 B 11편 · 1초 표본 점수 누움 AUROC ≥ 0.90 그리고 누움 잡힌 비율 ≥ 0.50
  흐름  프레임 3장마다 (초당 10장) → 1920×1080 으로 키움 → worker 타일 탐지 (PersonDetector) → StateTracker.update (시각 = 프레임/30)
        → 1초마다 (프레임 % 30 == 0) 이번에 갱신된 상태를 정답 박스 (×1.5 · IoU ≥ 0.5) 의 자세와 맞춤
  출력  metrics/state_worker_check_<태그>.json · runs_state/worker_samples_<태그>.json (1초 표본 · git 밖 — Okutama 파생)
  --trackbox   원인 분리용 — 정답 맞추기에 추적기 (칼만) 박스 (연구 채점과 같음)
  --native     원인 분리용 — 추적 · 움직임 보정도 720p 원본 영상으로 (연구와 완전히 같은 조건)
  --fullframe  원인 분리용 — 탐지만 연구 방식 (720p 전체 화면 · imgsz 1280 · conf 0.15 · NMS 0.6 · FP16) · 박스 ×1.5 로 1080p 에 맞춰 같은 상태 모듈에 넣음
실행: /home/se/miniconda3/envs/drone/bin/python state_worker_check.py --weights=runs_person/soup_v9x2/weights/best.pt --tag=v9x2 --posture=<posture.json> [--limit=1] [--fullframe]
"""
import collections
import glob
import json
import os
import sys
import time

import cv2
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE, "deploy", "vision_worker"))
from app.vision_core.detector import PersonDetector          # noqa: E402
from app.vision_core.state import StateTracker               # noqa: E402
from okutama_motion import FPS, LAB, OK, load_boxes          # noqa: E402
from okutama_motion_det import iou                           # noqa: E402
from posture_v2 import auc                                   # noqa: E402
from state_trackfix import gt_lying, videos                   # noqa: E402

UP = 1.5                                                     # 720p → 1080p
FULL = "--fullframe" in sys.argv
TBOX = "--trackbox" in sys.argv                              # 정답 맞추기에 추적기 박스 (state_pipeline 과 같은 채점)
NATIVE = "--native" in sys.argv                              # 추적 · 상태도 720p 원본 영상으로 (연구와 같은 조건 · --fullframe 과 같이)


def arg(k, d=None):
    return next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith(f"--{k}=")), d)


def run(vid, fdir, det, posture):
    gt = load_boxes(f"{LAB}/{vid}.txt")
    st = StateTracker(posture)
    n = len(glob.glob(f"{fdir}/*.jpg"))
    S, ms = [], []
    for fr in range(0, n, 3):
        img = cv2.imread(f"{fdir}/{fr}.jpg")
        if img is None:
            continue
        small = img
        img = cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_LINEAR)
        t0 = time.perf_counter()
        if FULL:
            r = det.predict(small, conf=0.15, iou=0.6, imgsz=1280, half=True, verbose=False)[0]
            boxes, confs = r.boxes.xyxy.cpu().numpy() * UP, r.boxes.conf.cpu().numpy()
        else:
            boxes, confs, _ = det.detect(img)
        kept = [(tuple(float(v) for v in b), float(c)) for b, c in zip(boxes, confs)
                if min(b[2], 1920) > max(b[0], 0) and min(b[3], 1080) > max(b[1], 0)]
        kb = [(max(0.0, b[0]), max(0.0, b[1]), min(1920.0, b[2]), min(1080.0, b[3])) for b, _ in kept]
        if NATIVE:                                            # 720p 원본 영상 · 720p 박스 (상태 모듈이 1080p 로 환산)
            out = st.update(fr / FPS, small, [tuple(v / UP for v in b) for b in kb], [c for _, c in kept])
        else:
            out = st.update(fr / FPS, img, kb, [c for _, c in kept])
        ms.append((time.perf_counter() - t0) * 1000)
        if fr % FPS:
            continue
        g = {k: (v[0] * UP, v[1] * UP, v[2] * UP, v[3] * UP, v[4]) for k, v in gt.get(fr, {}).items()}
        for j, (b, (_, c), (tkey, s)) in enumerate(zip(kb, kept, out)):
            if s is None or abs(s["updated_at_s"] - fr / FPS) > 1e-6:
                continue
            if TBOX and j in st.track_boxes:
                b = tuple(v * (UP if NATIVE else 1) for v in st.track_boxes[j])
            best, pose = 0.5, None
            for (x1, y1, x2, y2, p) in g.values():
                v = iou(b, (x1, y1, x2, y2))
                if v >= best:
                    best, pose = v, p
            if pose:
                S.append({"vid": vid, "fr": fr, "track": tkey, "pose": pose, "score": s["score"], "grade": s["grade"], "conf": c,
                          "lying_p": s["score_terms"]["lying"], "display": s["posture"]["label"], "still": s["motion"]["still_s"],
                          "moving": s["motion"]["moving"], "measured": s["motion"]["measured"]})
    return S, st.stats, ms


def boot(S, vids, f, reps=1000, seed=0):
    rng = np.random.default_rng(seed); by = collections.defaultdict(list)
    for s in S:
        by[s["vid"]].append(s)
    vals = []
    for _ in range(reps):
        pick = [s for v in rng.choice(vids, len(vids)) for s in by[v]]
        y = np.array([s["pose"] == "Lying" for s in pick]); x = np.array([f(s) for s in pick])
        if y.any() and (~y).any():
            vals.append(auc(x[y], x[~y]))
    return [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)] if vals else None


def main():
    weights, tag, posture = arg("weights"), arg("tag", "check"), arg("posture")
    _, _, B = videos()
    B = B[: int(arg("limit", len(B)))]
    frame_dir = {os.path.basename(d): d for d in glob.glob(f"{OK}/Drone*/*/Extracted-Frames-1280x720/*")}
    wpath = os.path.join(BASE, weights) if not os.path.isabs(weights) else weights
    if FULL:
        from ultralytics import YOLO
        det = YOLO(wpath)
    else:
        det = PersonDetector(wpath, conf=0.15)
    allS, stats, ms = [], collections.Counter(), []
    for v in B:
        S, stt, m = run(v, frame_dir[v], det, posture)
        allS += S; stats.update(stt); ms += m
        print(f"{v}: 표본 {len(S)} · 누움 {sum(s['pose'] == 'Lying' for s in S)} · {stt}", flush=True)
    y = np.array([s["pose"] == "Lying" for s in allS]); sc = np.array([s["score"] for s in allS]); cf = np.array([s["conf"] for s in allS])
    gl = sum(gt_lying(B).values())
    tab = collections.defaultdict(collections.Counter)
    for s in allS:
        tab[s["pose"]][s["grade"]] += 1
    disp = collections.defaultdict(collections.Counter)
    for s in allS:
        disp[{"Lying": "lying", "Sitting": "sitting"}.get(s["pose"], "standing")][s["display"]] += 1
    au = round(float(auc(sc[y], sc[~y])), 4) if y.any() and (~y).any() else float("nan")
    caught = round(int(y.sum()) / max(gl, 1), 3)
    res = {"태그": tag, "가중치": weights, "탐지 방식": "전체 화면 720p (연구 방식)" if FULL else "타일 4장 1080p (worker)",
           "추적 영상": "720p 원본" if NATIVE else "1080p (키움)", "채점 박스": "추적기 박스" if TBOX else "탐지 박스", "영상 (B)": len(B), "1초 표본": len(allS), "누움 표본": int(y.sum()), "누움 정답 1초 표본": gl,
           "점수 누움 AUROC": au, "95% (영상 블록)": boot(allS, B, lambda s: s["score"]),
           "탐지 확신도 AUROC (기준선)": round(float(auc(cf[y], cf[~y])), 4) if y.any() and (~y).any() else None, "누움 잡힌 비율": caught,
           "판정 (≥0.90 · ≥0.50)": "PASS" if au >= 0.90 and caught >= 0.50 else "FAIL",
           "Q1 V2·single (연구 · 참고)": {"점수 누움 AUROC": 0.953, "누움 잡힌 비율": 0.573},
           "정답 자세 × 등급": {k: dict(v) for k, v in tab.items()}, "정답 자세 × 표시 자세 (V3)": {k: dict(v) for k, v in disp.items()},
           "추적 · 갱신": dict(stats), "탐지+상태 ms (중앙 · P95)": [round(float(np.median(ms)), 1), round(float(np.percentile(ms, 95)), 1)]}
    os.makedirs(os.path.join(BASE, "runs_state"), exist_ok=True)
    json.dump(allS, open(os.path.join(BASE, "runs_state", f"worker_samples_{tag}.json"), "w"))
    json.dump(res, open(os.path.join(BASE, "metrics", f"state_worker_check_{tag}.json"), "w"), ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
