"""
kp_resolution.py — K1: 관절점 해상도 필요조건 — 사람이 작으면 몸통 4점 (어깨 · 골반) 이 잡히는가 (10-09 · 판정 기준은 _학습 큐 「10-09 C4 · K1」)

  데이터  NOMAD a10 원본 5472×3078 (항공 비스듬 · 사람 키 ~300 px) · 한 사람만 있는 장 · 서기 (Walking) · 누움 (Laying) 무작위 (시드 0)
  가짜 정답  원본 크롭에서 yolo11m-pose 가 그 사람 (IoU ≥ 0.3) 의 몸통 4점을 모두 확신 ≥ 0.7 로 찾은 경우만 · 남은 비율을 자세별로 적는다
  크기  H = 박스 긴 변 (서기 = 키 · 누움 = 몸 길이 — 운용 표 「누움 가로」 와 같은 뜻 · 자세 판정기 특징 log max(w,h) 와도 같음)
  줄이기  사람 긴 변이 H px (1080p 기준) 가 되도록 원본 크롭을 줄임 (INTER_AREA = 그 거리에서 찍힌 것과 같은 정보량)
          → worker 후보 크롭처럼 다시 키움 (사람 키 256 px · INTER_LINEAR) → 모델 추론 (imgsz 640 · conf 0.1) · IoU 최대 사람
  지표  몸통 4점 PCK@0.1 (박스 긴 변의 10 % 안 · 관절 확신 ≥ 0.5) · 좌우 무시 PCK · 4점 모두 맞음 · 사람 찾음 — H × 모델 × 자세
  판정  쓸 수 있는 키 = PCK ≥ 0.8 인 가장 작은 H · > 81 px (재관측 12 m) 이면 관절점 트랙 중단 · ≤ 61 px 이면 K1b
        yolo11s-pose = 정답과 다른 모델 (공정) · yolo11m-pose = 정답을 만든 모델 (자기 일치 → 낙관)
실행: /home/se/miniconda3/envs/drone/bin/python kp_resolution.py [--n 300]   → metrics/kp_resolution.json (수치만 · NOMAD 파생 그림 없음)
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from eval_color import POSE
from make_posture_crops import NOMAD, _nomad_act

BASE = Path(__file__).resolve().parent
WDIR = BASE.parent / "data" / "raw" / "pose_weights"
TORSO = (5, 6, 11, 12)                 # COCO: 왼어깨 · 오른어깨 · 왼골반 · 오른골반
HEIGHTS = (30, 45, 60, 80, 120)
UP_H = 256


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u if u > 0 else 0.0


def best_person(res, box):
    """추론 결과에서 박스와 IoU 최대인 사람 → (관절 17×2, 확신 17) 또는 None"""
    if res.keypoints is None or res.boxes is None or len(res.boxes) == 0:
        return None
    b = res.boxes.xyxy.cpu().numpy(); j = max(range(len(b)), key=lambda i: iou(b[i], box))
    if iou(b[j], box) < 0.3:
        return None
    k = res.keypoints
    return k.xy.cpu().numpy()[j], (k.conf.cpu().numpy()[j] if k.conf is not None else np.ones(17))


def pick(n, seed=0, existing=False):
    acts = {x["id"]: x["labels"] for x in json.load(open(NOMAD / "activityLabels.json"))}
    pool = defaultdict(list)
    for r in json.load(open(NOMAD / "annotations.json")):
        actor_s, dist_s, f_s = r["file_name"][:-4].split("_")
        if dist_s != "a10":
            continue
        pose = POSE.get(_nomad_act(acts, int(actor_s[5:]), dist_s[1:], int(f_s[1:])))
        boxes = [b["bbox"] for b in r["annotations"] if int(b.get("visibility", 100)) >= 50]
        if pose is None or len(boxes) != 1:
            continue
        if existing and not (NOMAD / "images" / actor_s / f"{actor_s}_{dist_s}" / r["file_name"]).exists():
            continue
        pool[pose].append((NOMAD / "images" / actor_s / f"{actor_s}_{dist_s}" / r["file_name"], boxes[0], pose))
    rnd = random.Random(seed); out = []
    for pose in ("standing", "lying"):
        out += rnd.sample(pool[pose], min(n // 2, len(pool[pose])))
    return out, {k: len(v) for k, v in pool.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--existing", action="store_true", help="서버에 있는 원본에서만 뽑기 (NOMAD 60/100명 · 보조 실행)")
    a = ap.parse_args()
    models = {m: YOLO(str(WDIR / f"{m}.pt")) for m in ("yolo11m-pose", "yolo11s-pose")}
    gt_model = models["yolo11m-pose"]
    jobs, pool_sizes = pick(a.n, existing=a.existing)
    kept = defaultdict(lambda: [0, 0]); stats = defaultdict(lambda: defaultdict(list))
    native_h = []
    for path, (x, y, w, h), pose in jobs:
        img = cv2.imread(str(path))
        if img is None:
            continue
        m = 0.5 * max(w, h)
        X1, Y1 = int(max(0, x - m)), int(max(0, y - m)); X2, Y2 = int(min(img.shape[1], x + w + m)), int(min(img.shape[0], y + h + m))
        crop = img[Y1:Y2, X1:X2]; box = np.array([x - X1, y - Y1, x - X1 + w, y - Y1 + h], float)
        kept[pose][1] += 1
        r = gt_model.predict(crop, imgsz=640, conf=0.1, verbose=False, half=True)[0]
        g = best_person(r, box)
        if g is None or (g[1][list(TORSO)] < 0.7).any():
            continue
        L = max(w, h)
        kept[pose][0] += 1; gxy = g[0]; native_h.append(L)
        for H in HEIGHTS:
            s = H / L
            small = cv2.resize(crop, (max(1, round(crop.shape[1] * s)), max(1, round(crop.shape[0] * s))), interpolation=cv2.INTER_AREA)
            f = UP_H / H
            up = cv2.resize(small, (max(1, round(small.shape[1] * f)), max(1, round(small.shape[0] * f))), interpolation=cv2.INTER_LINEAR)
            k_up = up.shape[1] / crop.shape[1]                # 원본 크롭 → 키운 크롭 배율
            for mn, model in models.items():
                rr = model.predict(up, imgsz=640, conf=0.1, verbose=False, half=True)[0]
                p = best_person(rr, box * k_up)
                key = (mn, H, pose)
                if p is None:
                    stats[key]["found"].append(0); stats[key]["pck"] += [0] * 4; stats[key]["pck_lr"] += [0] * 4; stats[key]["all4"].append(0)
                    continue
                pxy, pc = p[0] / k_up, p[1]
                tol = 0.1 * L
                ok = [bool(pc[i] >= 0.5 and np.linalg.norm(pxy[i] - gxy[i]) <= tol) for i in TORSO]
                okl = []
                for pair in ((5, 6), (11, 12)):                 # 좌우 무시 — 두 점 짝을 바꿔도 맞으면 맞음
                    d1 = [pc[i] >= 0.5 and np.linalg.norm(pxy[i] - gxy[i]) <= tol for i in pair]
                    d2 = [pc[pair[1 - k]] >= 0.5 and np.linalg.norm(pxy[pair[1 - k]] - gxy[pair[k]]) <= tol for k in (0, 1)]
                    okl += d1 if sum(d1) >= sum(d2) else d2
                stats[key]["found"].append(1); stats[key]["pck"] += ok; stats[key]["pck_lr"] += okl; stats[key]["all4"].append(int(all(ok)))
    res = {"data": "NOMAD a10 원본 5472×3078 · 한 사람 장", "pool": pool_sizes,
           "pseudo_gt_kept": {k: f"{v[0]}/{v[1]}" for k, v in kept.items()},
           "native_person_h_px": {"median": float(np.median(native_h)) if native_h else None},
           "table": {}}
    for mn in models:
        for pose in ("standing", "lying", "all"):
            row = {}
            for H in HEIGHTS:
                keys = [(mn, H, p) for p in ("standing", "lying")] if pose == "all" else [(mn, H, pose)]
                agg = defaultdict(list)
                for k in keys:
                    for f in ("found", "pck", "pck_lr", "all4"):
                        agg[f] += stats[k][f]
                if agg["found"]:
                    row[H] = {f: round(float(np.mean(agg[f])), 3) for f in ("found", "pck", "pck_lr", "all4")} | {"n": len(agg["found"])}
            res["table"][f"{mn} {pose}"] = row
    usable = {}
    for mn in models:
        row = res["table"][f"{mn} all"]
        ok = [row.get(H, {}).get("pck", 0) >= 0.8 for H in HEIGHTS]     # 그 크기 이상이 모두 통과하는 가장 작은 H
        usable[mn] = next((H for i, H in enumerate(HEIGHTS) if all(ok[i:])), None)
    res["usable_height_px (PCK≥0.8)"] = usable
    best = min([v for v in usable.values() if v is not None], default=None)
    res["K1 판정"] = ("중단 (어느 모델도 120 px 까지 PCK 0.8 못 미침)" if best is None else
                     "중단 (> 81 px · 재관측 12 m 로도 모자람)" if best > 81 else
                     "K1b 로 (≤ 61 px)" if best <= 61 else "조건부 (재관측 8~12 m)")
    (BASE / "metrics" / ("kp_resolution_existing.json" if a.existing else "kp_resolution.json")).write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
