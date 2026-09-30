"""
okutama_motion_det.py — 움직임 판정을 **탐지 박스 + 추적**으로 다시 (상태 인지 B1 · 2026-09-30)

왜
  okutama_motion.py 는 정답 박스로 이동/정지 AUROC 0.93 (3초 창). 운용에서는 탐지 박스가 흔들리고
  추적 ID 가 끊긴다 — 실제 파이프라인에서도 되는지 본다.

방법 (Okutama 비스듬 25편 · 1280×720 추출 프레임)
  soup_v7r2 (단일 1280 · conf 0.15 · NMS 0.6 · FP16) + BoT-SORT 추적 · 초당 10장 (3프레임마다)
  1초 간격 (t, t+30) 에서 같은 추적 ID 의 발끝점 → 배경 호모그래피로 드론 움직임을 뺀 이동량 / 몸 높이
  자세 = 그 시각 정답 박스와 IoU ≥ 0.5 로 짝지은 라벨 (짝이 없으면 뺀다)

실행: python okutama_motion_det.py [--limit 3]    출력: metrics/okutama_motion_det.json
"""
import collections
import glob
import json
import os
import sys

import cv2
import numpy as np

from okutama_motion import FPS, LAB, NADIR, OK, POSE, auc, homography, load_boxes

BASE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(BASE, "runs_person", "soup_v7r2", "weights", "best.pt")
OUT = os.path.join(BASE, "metrics", "okutama_motion_det.json")
STEP = 3                                                  # 30 fps → 초당 10장
JUMP = 3.0                                                # 사람이 1초에 몸 높이 3배 넘게 움직이면 ID 바뀜


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / max((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter, 1e-9)


def match_pose(box, gt):
    """추적 박스 ↔ 정답 박스 IoU 최대 (≥0.5) 의 자세."""
    best, pose = 0.5, None
    for (x1, y1, x2, y2, p) in gt.values():
        v = iou(box, (x1, y1, x2, y2))
        if v >= best:
            best, pose = v, p
    return pose


def main():
    limit = next((int(a.split("=")[1]) for a in sys.argv[1:] if a.startswith("--limit=")), None)
    from ultralytics import YOLO
    model = YOLO(WEIGHTS)
    frame_dir = {os.path.basename(d): d for d in glob.glob(f"{OK}/Drone*/*/Extracted-Frames-1280x720/*")}
    orb, bf = cv2.ORB_create(3000), cv2.BFMatcher(cv2.NORM_HAMMING)
    comp = collections.defaultdict(list); seq = collections.defaultdict(dict)
    stat = collections.Counter()
    vids = [v for v in sorted(os.path.basename(p)[:-4] for p in glob.glob(f"{LAB}/*.txt"))
            if v in frame_dir and v not in NADIR][:limit]
    for vid in vids:
        gt = load_boxes(f"{LAB}/{vid}.txt")
        n = len(glob.glob(f"{frame_dir[vid]}/*.jpg"))
        tracks = {}                                           # 프레임 → {ID: 박스}
        for k, fr in enumerate(range(0, n, STEP)):
            img = cv2.imread(f"{frame_dir[vid]}/{fr}.jpg")
            if img is None:
                continue
            r = model.track(img, persist=k > 0, tracker="botsort.yaml", conf=0.15, iou=0.6, imgsz=1280,
                            half=True, verbose=False)[0]
            if fr % FPS == 0 and r.boxes.id is not None:
                tracks[fr] = {int(i): tuple(map(float, b)) for i, b in zip(r.boxes.id.tolist(), r.boxes.xyxy.tolist())}
        for fr in sorted(tracks):
            a, b = tracks[fr], tracks.get(fr + FPS)
            if not b or fr not in gt or (fr + FPS) not in gt:
                continue
            common = [t for t in a if t in b]
            if not common:
                continue
            ia = cv2.imread(f"{frame_dir[vid]}/{fr}.jpg", 0); ib = cv2.imread(f"{frame_dir[vid]}/{fr + FPS}.jpg", 0)
            fake = lambda d: {t: (*v, None) for t, v in d.items()}          # homography() 는 5-튜플 박스
            H, _ = homography(ia, ib, fake(a), fake(b), orb, bf)
            if H is None:
                stat["정합 실패"] += 1
                continue
            for t in common:
                pa, pb = match_pose(a[t], gt[fr]), match_pose(b[t], gt[fr + FPS])
                if pa is None or pa != pb:
                    stat["자세 짝 없음"] += 1
                    continue
                x1, y1, x2, y2 = a[t]; u1, v1, u2, v2 = b[t]
                fa, fb = np.array([(x1 + x2) / 2, y2]), np.array([(u1 + u2) / 2, v2])
                wa = cv2.perspectiveTransform(fa.reshape(1, 1, 2).astype(np.float32), H).ravel()
                s = float(np.linalg.norm(fb - wa) / (((y2 - y1) + (v2 - v1)) / 2))
                comp[pa].append(s); seq[(vid, t)][fr] = (pa, s); stat["표본"] += 1
        # 정답 박스 수 대비 추적으로 잡힌 비율 (1초 표본 기준)
        stat["정답 1초 표본"] += sum(len(gt[f]) for f in gt if f % FPS == 0)
        print(f"{vid}: 누적 표본 {stat['표본']}", flush=True)

    w3 = collections.defaultdict(list); w3m = collections.defaultdict(list)
    for d in seq.values():
        for fr, (p, v) in d.items():
            nxt = [d.get(fr + k * FPS) for k in (1, 2)]
            if all(x is not None and x[0] == p for x in nxt):
                vals = [v, nxt[0][1], nxt[1][1]]
                w3[p].append(sum(vals) / 3)
                ok = [x for x in vals if x <= JUMP]                  # 몸 높이 3배/초 초과 = 추적 ID 바뀜으로 본다
                if len(ok) >= 2:
                    w3m[p].append(float(np.median(ok)))
    jumps = sum(v > JUMP for L in comp.values() for v in L) / max(1, sum(len(L) for L in comp.values()))
    still, move = ("Lying", "Sitting", "Standing"), ("Walking", "Running")
    pool = lambda d, ks: sum((d[k] for k in ks), [])
    q = lambda x, p: round(float(np.percentile(x, p)), 3) if len(x) else None
    res = {"영상": len(vids), **stat,
           "자세별 (몸 높이/초)": {k: {"1초 p50": q(comp[k], 50), "3초 창 p50": q(w3[k], 50), "3초 창 p90": q(w3[k], 90),
                                   "n": len(comp[k])} for k in POSE},
           "이동 vs 정지 AUROC": {"1초": auc(pool(comp, move), pool(comp, still)),
                               "3초 창": auc(pool(w3, move), pool(w3, still)),
                               "3초 창 중앙값 · 튐 제외": auc(pool(w3m, move), pool(w3m, still))},
           "튐 (>3 몸높이/초) 비율": round(float(jumps), 3),
           "3초 창 중앙값 p50": {k: q(w3m[k], 50) for k in POSE},
           "3초 창 중앙값 임계 0.25": {"정지를 정지로": round(float(np.mean(np.array(pool(w3m, still)) < 0.25)), 3),
                                    "이동을 정지로 (오인)": round(float(np.mean(np.array(pool(w3m, move)) < 0.25)), 3)}}
    json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
