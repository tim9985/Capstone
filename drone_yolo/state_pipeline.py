"""
state_pipeline.py — 요구조자 상태 인지 v1: 추적 → 자세 · 무동작 → 등급 · 점수 → 후보 순위 (2026-10-01)

  계획 → obsidian 「06 요구조자 상태 인지 계획」 · 원칙: **후보를 거르지 않고 순위만 바꾼다**
  흐름 (Okutama 비스듬 영상 · 1280×720 · 30 fps)
    ① 탐지 soup_v7r2 (conf 0.15 · NMS 0.6 · FP16) + BoT-SORT 추적 · 초당 10장 (3 프레임마다)
    ② 1초마다 배경 호모그래피로 드론 움직임을 빼고 발끝점 이동량 → 몸 높이/초 (okutama_motion_det 와 같은 식)
    ③ 자세 = 비스듬 박스 모양 판정 (3자세 로지스틱 · SARD + NOMAD 크롭으로 맞춤 · posture_fusion 「비스듬판정_box」)
    ④ 추적별 상태 (1초마다): 누움 · 앉음 확률 (최근 5초 중앙값) · 속도 (최근 3초 중앙값 · 3 몸높이/초 초과 = ID 튐 → 뺀다)
                             · 무동작 시간 (속도 < 0.25 가 이어진 초) · 관측 시간
    ⑤ 등급 (규칙 v1) — ❔ 관측 < 3초 · 🔴 누움 ≥ 0.7 & 무동작 ≥ 10초 · 🟠 누움+앉음 ≥ 0.7 & 무동작 ≥ 10초
                       · ❔ 누움 0.3~0.7 · ⚪ 나머지 (이동 · 서 있음)
       점수 = 0.5 × 누움 + 0.2 × 앉음 + 0.3 × min(무동작, 20초)/20  (이동 중이면 무동작 0)
  평가: 1초 표본마다 추적 박스 ↔ 정답 박스 (IoU ≥ 0.5) 자세로
    · 누움 표본을 점수로 가려내는 AUROC (기준선 = 탐지 확신도) · 정답 자세 × 등급 표
    · 추적 단위: 추적 최고 점수로 "누운 적 있는 사람" 가려내기 AUROC
  ⚠ Okutama 는 공원에서 배우가 연출한 행동 · 정답 박스로 자세를 붙인다 · 낙상 없음
  자세 판정기 갈아끼우기 (10-03 · P5): --posture=box (v1 · 기본) | p3 | ft_s0 → posture_runtime.py
    · 1초 표본마다 한 장 확률 · 5초 중앙값 확률을 남기고 탐지 박스 기준 자세 지표 (누움 AUROC · 매크로 F1 · 앉음 F1) 를 낸다
    · --dump → runs_state/samples_<tag>.json (1초 표본 전부 · 맞은 정답 ID · 최근 5표본 누움 확률 · git 밖 — Okutama 파생물)
실행: python state_pipeline.py [--limit=3] [--video=1.1.1] [--weights=runs_person/soup_v9x2/weights/best.pt] [--tag=v9x2] [--no-render] [--posture=p3]
출력: metrics/state_pipeline[_<tag>].json · runs_state/<영상>.mp4 (git 밖 · Okutama 파생물)
"""
import collections
import csv
import glob
import json
import os
import sys

import cv2
import numpy as np
from okutama_motion import FPS, LAB, NADIR, OK, auc, homography, load_boxes
from okutama_motion_det import iou

BASE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(BASE, "runs_person", "soup_v7r2", "weights", "best.pt")
OUT = os.path.join(BASE, "metrics", "state_pipeline.json")
STEP, JUMP, STILL = 3, 3.0, 0.25
TO1080 = 1080 / 720                                          # Okutama 1280×720 → 1080p 환산 (학습 크롭과 같은 단위)
COLOR = {"🔴": (0, 0, 255), "🟠": (0, 140, 255), "❔": (200, 200, 0), "⚪": (200, 200, 200)}
TXT = {"🔴": "URGENT", "🟠": "HIGH", "❔": "CHECK", "⚪": "normal"}


POSE3 = {"Lying": 0, "Sitting": 1}                           # 나머지 (서기 · 걷기 · 달리기) = 2


def match_gt(box, gt):
    """추적 박스 ↔ 정답 박스 IoU 최대 (≥0.5) 의 (자세, 정답 ID) — match_pose 와 같은 규칙 + ID."""
    best, out = 0.5, (None, None)
    for gid, (x1, y1, x2, y2, p) in gt.items():
        v = iou(box, (x1, y1, x2, y2))
        if v >= best:
            best, out = v, (p, gid)
    return out


def grade(st):
    ly, si, still, obs, mv = st["lying"], st["sitting"], st["still"], st["obs"], st["moving"]
    score = 0.5 * ly + 0.2 * si + (0 if mv else 0.3 * min(still, 20) / 20)
    if obs < 3:
        g = "❔"
    elif ly >= 0.7 and still >= 10:
        g = "🔴"
    elif ly + si >= 0.7 and still >= 10:
        g = "🟠"
    elif 0.3 <= ly < 0.7 and not mv:
        g = "❔"
    else:
        g = "⚪"
    return g, round(score, 3)


def run_video(vid, fdir, model, rt, orb, bf, render=False):
    from ultralytics import YOLO  # noqa: F401  (model 은 밖에서 만든다)
    gt = load_boxes(f"{LAB}/{vid}.txt")
    n = len(glob.glob(f"{fdir}/*.jpg"))
    hist = collections.defaultdict(lambda: {"ly": [], "si": [], "stp": [], "frs": [], "sp": [], "seen": 0, "still": 0, "conf": []})
    rt.reset()
    prev = None                                               # (fr, 박스 dict, 회색 영상)
    samples, writer = [], None
    for k, fr in enumerate(range(0, n, STEP)):
        img = cv2.imread(f"{fdir}/{fr}.jpg")
        if img is None:
            continue
        r = model.track(img, persist=k > 0, tracker="botsort.yaml", conf=0.15, iou=0.6, imgsz=1280, half=True, verbose=False)[0]
        cur = {}
        if r.boxes.id is not None:
            cur = {int(i): (tuple(map(float, b)), float(c)) for i, b, c in zip(r.boxes.id.tolist(), r.boxes.xyxy.tolist(), r.boxes.conf.tolist())}
        if fr % FPS == 0:                                     # ── 1초마다 상태 갱신 ──
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            H = None
            if prev is not None and prev[0] == fr - FPS:
                fake = lambda d: {t: (*v[0], None) for t, v in d.items()}
                H, _ = homography(prev[2], gray, fake(prev[1]), fake(cur), orb, bf)
            P = rt.predict(img, cur)
            for t, (b, c) in cur.items():
                s = hist[t]; s["seen"] += 1; s["conf"].append(c)
                p = P[t]
                s["ly"].append(p[0]); s["si"].append(p[1]); s["stp"].append(p[2]); s["frs"].append(fr)
                if H is not None and t in prev[1]:
                    a = prev[1][t][0]
                    fa = np.float32([[(a[0] + a[2]) / 2, a[3]]]).reshape(1, 1, 2)
                    wa = cv2.perspectiveTransform(fa, H).ravel()
                    sp = float(np.linalg.norm(np.array([(b[0] + b[2]) / 2, b[3]]) - wa) / max(((a[3] - a[1]) + (b[3] - b[1])) / 2, 1))
                    if sp <= JUMP:
                        s["sp"].append(sp)
                med = float(np.median(s["sp"][-3:])) if s["sp"] else None
                moving = med is not None and med >= 0.4
                s["still"] = s["still"] + 1 if (med is not None and med < STILL) else 0
                st = {"lying": float(np.median(s["ly"][-5:])), "sitting": float(np.median(s["si"][-5:])),
                      "still": s["still"], "obs": s["seen"], "moving": moving}
                g, score = grade(st)
                s["last"] = (g, score, st)
                pose, gid = match_gt(b, gt.get(fr, {}))
                if pose:
                    samples.append({"vid": vid, "t": t, "fr": fr, "pose": pose, "grade": g, "score": score,
                                    "conf": c, "lying_p": round(st["lying"], 3), "still": st["still"],
                                    "p": [round(float(v), 4) for v in p],
                                    "agg": [round(st["lying"], 4), round(st["sitting"], 4), round(float(np.median(s["stp"][-5:])), 4)],
                                    "gt": gid, "h_ly": [round(float(v), 3) for v in s["ly"][-5:]], "h_fr": s["frs"][-5:]})
            prev = (fr, cur, gray)
        if render:
            if writer is None:
                os.makedirs(os.path.join(BASE, "runs_state"), exist_ok=True)
                writer = cv2.VideoWriter(os.path.join(BASE, "runs_state", f"{vid}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                         10, (img.shape[1], img.shape[0]))
            for t, (b, c) in cur.items():
                g, score, st = hist[t].get("last", ("❔", 0, {"lying": 0, "still": 0}))
                x1, y1, x2, y2 = map(int, b)
                cv2.rectangle(img, (x1, y1), (x2, y2), COLOR[g], 2)
                cv2.putText(img, f"#{t} {TXT[g]} {score:.2f} lie{st['lying']:.1f} still{st['still']}s", (x1, max(12, y1 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR[g], 1, cv2.LINE_AA)
            writer.write(img)
    if writer is not None:
        writer.release()
    ranking = sorted(((t, *s["last"][:2]) for t, s in hist.items() if "last" in s), key=lambda x: -x[2])[:5]
    return samples, ranking


def main():
    limit = next((int(a.split("=")[1]) for a in sys.argv[1:] if a.startswith("--limit=")), None)
    only = next((a.split("=")[1] for a in sys.argv[1:] if a.startswith("--video=")), None)
    weights = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--weights=")), WEIGHTS)
    tag = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--tag=")), None)
    out = OUT.replace(".json", f"_{tag}.json") if tag else OUT
    posture = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--posture=")), "box")
    render_ok = "--no-render" not in sys.argv
    from ultralytics import YOLO
    frame_dir = {os.path.basename(d): d for d in glob.glob(f"{OK}/Drone*/*/Extracted-Frames-1280x720/*")}
    vids = [v for v in sorted(os.path.basename(p)[:-4] for p in glob.glob(f"{LAB}/*.txt")) if v in frame_dir and v not in NADIR]
    if only:
        vids = [only]
    vids = vids[:limit]
    # 데모 영상: 누움 정답이 가장 많은 비스듬 영상 하나
    demo = max(vids, key=lambda v: sum(p[4] == "Lying" for f in load_boxes(f"{LAB}/{v}.txt").values() for p in f.values()))
    from posture_runtime import Runtime
    rt = Runtime(posture); orb, bf = cv2.ORB_create(3000), cv2.BFMatcher(cv2.NORM_HAMMING)
    allS, ranks = [], {}
    gt_lying = sum(1 for v in vids for f, d in load_boxes(f"{LAB}/{v}.txt").items() if f % FPS == 0 for b in d.values() if b[4] == "Lying")
    for v in vids:
        model = YOLO(weights)                                 # 영상마다 추적 상태를 새로
        S, rk = run_video(v, frame_dir[v], model, rt, orb, bf, render=(render_ok and v == demo))
        allS += S; ranks[v] = rk
        print(f"{v}: 표본 {len(S)} · 상위 {rk[:3]}", flush=True)
    y = np.array([s["pose"] == "Lying" for s in allS]); sc = np.array([s["score"] for s in allS]); cf = np.array([s["conf"] for s in allS])
    still_gt = np.array([s["pose"] in ("Lying", "Sitting") for s in allS])
    tab = collections.defaultdict(collections.Counter)
    for s in allS:
        tab[s["pose"]][s["grade"]] += 1
    # 추적 단위 — 추적 최고 점수 vs "누운 적 있음"
    per = collections.defaultdict(lambda: [0.0, False, 0.0])
    for s in allS:
        k = (s["vid"], s["t"]); per[k][0] = max(per[k][0], s["score"]); per[k][1] |= s["pose"] == "Lying"; per[k][2] = max(per[k][2], s["conf"])
    tp = np.array([v[0] for v in per.values()]); tl = np.array([v[1] for v in per.values()]); tc = np.array([v[2] for v in per.values()])
    from posture_v2 import stats
    y3 = np.array([POSE3.get(s["pose"], 2) for s in allS])
    pose_m = {"n": {c: int((y3 == i).sum()) for i, c in enumerate(("lying", "sitting", "standing"))}}
    for k, f in (("한 장", "p"), ("5초 중앙값", "agg")):
        pr = np.array([s[f] for s in allS]) if allS else np.zeros((0, 3))
        pose_m[k] = {kk: round(vv, 3) for kk, vv in stats(y3, pr).items()} if len(pr) else {}
    res = {"탐지 가중치": os.path.relpath(weights, BASE), "자세 판정기": posture, "자세 지표 (탐지 박스)": pose_m,
           "영상": len(vids), "1초 표본": len(allS),
           "누움 1초 표본 (탐지·추적으로 잡힌 것)": int(y.sum()), "누움 정답 1초 표본": int(gt_lying),
           "누움 잡힌 비율": round(int(y.sum()) / max(gt_lying, 1), 3), "데모": f"runs_state/{demo}.mp4" if render_ok else None,
           "누움 가려내기 AUROC (1초 표본)": {"요구조 점수": round(auc(sc[y], sc[~y]), 3), "탐지 확신도(기준선)": round(auc(cf[y], cf[~y]), 3)},
           "누움·앉음 가려내기 AUROC": {"요구조 점수": round(auc(sc[still_gt], sc[~still_gt]), 3), "탐지 확신도": round(auc(cf[still_gt], cf[~still_gt]), 3)},
           "추적 단위 누운 적 있음 AUROC": {"최고 점수": round(auc(tp[tl], tp[~tl]), 3), "최고 확신도": round(auc(tc[tl], tc[~tl]), 3),
                                        "추적 수": int(len(tp)), "누운 추적": int(tl.sum())},
           "정답 자세 × 등급 (1초 표본)": {p: dict(c) for p, c in tab.items()},
           "영상별 상위 5 (ID · 등급 · 점수)": ranks}
    json.dump(res, open(out, "w"), ensure_ascii=False, indent=1)
    if "--dump" in sys.argv:
        os.makedirs(os.path.join(BASE, "runs_state"), exist_ok=True)
        json.dump(allS, open(os.path.join(BASE, "runs_state", f"samples_{tag or 'default'}.json"), "w"), default=int)
    print(json.dumps({k: v for k, v in res.items() if k != "영상별 상위 5 (ID · 등급 · 점수)"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
