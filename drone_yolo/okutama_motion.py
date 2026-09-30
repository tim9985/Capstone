"""
okutama_motion.py — 드론 움직임을 빼면 "사람이 움직이는지" 를 잴 수 있나 (2026-09-30)

왜
  9.30 설계 발표 피드백: 자세 + 이동 속도 변화로 요구조자를 판단해 보라.
  드론이 움직이면 화면 속 사람 위치는 드론 움직임까지 섞인다 — 보정 없이 되는지 · 보정하면 되는지 잰다.

방법 (Okutama 정답 박스 · 1280×720 추출 프레임 · 30 fps)
  1초 간격 프레임 쌍마다 사람 박스를 가린 배경에서 ORB 특징 → RANSAC 호모그래피
  발끝점(박스 아래 가운데)을 호모그래피로 옮긴 위치와 1초 뒤 실제 위치의 차이 = 사람 자신의 움직임
  몸 높이(박스 높이)로 나눠 크기 영향을 뺀다 · 3초 창 = 같은 자세로 이어진 1초 쌍 3개 평균
  박스 모양으로 자세를 가를 수 있는지도 시점별로 본다 — 세로/가로(h/w) · 길쭉함(긴 변/짧은 변)

한계
  정답 박스 기준 (탐지 박스는 흔들림이 더 크다) · 배우가 대본대로 연기 (낙상 없음)

실행: python okutama_motion.py            출력: metrics/okutama_motion.json
"""
import collections
import glob
import json
import os
import re

import cv2
import numpy as np

OK = "/home/se/JupyterLAB/Capstone/data/raw/okutama"
LAB = f"{OK}/Labels/MultiActionLabels/3840x2160"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "metrics", "okutama_motion.json")
POSE = ("Lying", "Sitting", "Standing", "Walking", "Running")
FPS, S = 30, 1 / 3.0                      # 라벨 4K → 추출 프레임 1280×720
# 하향 90° 15편 — 볼트 「Okutama-Action」 okutama_90 (박스 h/w 중앙값으로 가름)
NADIR = set("2.2.2 1.1.8 1.1.4 2.2.10 2.2.5 2.2.1 1.1.11 1.1.7 2.2.7 1.1.5 2.2.4 2.2.3 1.1.9 2.2.6 2.2.8".split())
QUOTED = re.compile(r'"([^"]*)"')


def auc(pos, neg):
    """pos 가 neg 보다 클 확률 (순위 기반 AUROC)."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    r = np.argsort(np.argsort(np.r_[pos, neg])) + 1
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def load_boxes(path):
    """프레임 → {트랙: (x1, y1, x2, y2, 자세)} · 1280 좌표 · lost 제외."""
    boxes = collections.defaultdict(dict)
    for ln in open(path):
        p = ln.split()
        if len(p) < 10 or int(p[6]):
            continue
        tid, x1, y1, x2, y2, fr = map(int, p[:6])
        pose = next((a for a in QUOTED.findall(ln)[1:] if a in POSE), None)
        if pose:
            boxes[fr][tid] = (x1 * S, y1 * S, x2 * S, y2 * S, pose)
    return boxes


def homography(ia, ib, a, b, orb, bf):
    """사람 박스를 가린 배경으로 t → t+1초 호모그래피 · 실패하면 None."""
    masks = []
    for img, bx in ((ia, a), (ib, b)):
        m = np.full(img.shape, 255, np.uint8)
        for (x1, y1, x2, y2, _) in bx.values():
            cv2.rectangle(m, (int(x1) - 8, int(y1) - 8), (int(x2) + 8, int(y2) + 8), 0, -1)
        masks.append(m)
    ka, da = orb.detectAndCompute(ia, masks[0])
    kb, db = orb.detectAndCompute(ib, masks[1])
    if da is None or db is None:
        return None, None
    good = [x for x, y in (p for p in bf.knnMatch(da, db, k=2) if len(p) == 2) if x.distance < 0.75 * y.distance]
    if len(good) < 40:
        return None, None
    pa = np.float32([ka[x.queryIdx].pt for x in good])
    pb = np.float32([kb[x.trainIdx].pt for x in good])
    H, inl = cv2.findHomography(pa, pb, cv2.RANSAC, 3.0)
    if H is None or inl.sum() < 30:
        return None, None
    keep = inl.ravel() == 1
    return H, float(np.median(np.linalg.norm(pb[keep] - pa[keep], axis=1)))


def main():
    frame_dir = {os.path.basename(d): d for d in glob.glob(f"{OK}/Drone*/*/Extracted-Frames-1280x720/*")}
    orb, bf = cv2.ORB_create(3000), cv2.BFMatcher(cv2.NORM_HAMMING)
    raw, comp = collections.defaultdict(list), collections.defaultdict(list)
    seq = collections.defaultdict(dict)                     # (영상, 트랙) → {프레임: (자세, 보정 이동량)}
    asp = {"oblique": collections.defaultdict(list), "nadir": collections.defaultdict(list)}
    elo = {"oblique": collections.defaultdict(list), "nadir": collections.defaultdict(list)}   # 긴 변 / 짧은 변
    cam, pairs, skipped = [], 0, 0

    for path in sorted(glob.glob(f"{LAB}/*.txt")):
        vid = os.path.basename(path)[:-4]
        if vid not in frame_dir:
            continue
        boxes = load_boxes(path)
        if not boxes:
            continue
        frs = sorted(boxes)
        for fr in range(frs[0], frs[-1] - FPS + 1, FPS):
            a, b = boxes.get(fr), boxes.get(fr + FPS)
            if not a or not b:
                continue
            for (x1, y1, x2, y2, pose) in a.values():
                w, h = max(x2 - x1, 1), max(y2 - y1, 1)
                g = "nadir" if vid in NADIR else "oblique"
                asp[g][pose].append(h / w)
                elo[g][pose].append(max(w, h) / min(w, h))
            common = [t for t in a if t in b and a[t][4] == b[t][4]]
            if not common:
                continue
            ia = cv2.imread(f"{frame_dir[vid]}/{fr}.jpg", 0)
            ib = cv2.imread(f"{frame_dir[vid]}/{fr + FPS}.jpg", 0)
            if ia is None or ib is None:
                continue
            H, bg = homography(ia, ib, a, b, orb, bf)
            if H is None:
                skipped += 1
                continue
            pairs += 1
            cam.append(bg)
            for t in common:
                x1, y1, x2, y2, pose = a[t]
                u1, v1, u2, v2, _ = b[t]
                fa, fb = np.array([(x1 + x2) / 2, y2]), np.array([(u1 + u2) / 2, v2])
                h = ((y2 - y1) + (v2 - v1)) / 2
                wa = cv2.perspectiveTransform(fa.reshape(1, 1, 2).astype(np.float32), H).ravel()
                raw[pose].append(float(np.linalg.norm(fb - fa) / h))
                comp[pose].append(float(np.linalg.norm(fb - wa) / h))
                seq[(vid, t)][fr] = (pose, comp[pose][-1])

    w3 = collections.defaultdict(list)
    for d in seq.values():
        for fr, (p, v) in d.items():
            nxt = [d.get(fr + k * FPS) for k in (1, 2)]
            if all(n is not None and n[0] == p for n in nxt):
                w3[p].append((v + nxt[0][1] + nxt[1][1]) / 3)

    q = lambda x, p: round(float(np.percentile(x, p)), 3) if len(x) else None
    still, move = ("Lying", "Sitting", "Standing"), ("Walking", "Running")
    pool = lambda d, ks: sum((d[k] for k in ks), [])
    res = {
        "프레임 쌍": pairs, "정합 실패": skipped,
        "배경 1초 이동 px (1280)": {"p50": q(cam, 50), "p90": q(cam, 90)},
        "자세별 1초 이동량 (몸 높이/초)": {k: {"보정 전 p50": q(raw[k], 50), "보정 후 p50": q(comp[k], 50),
                                          "보정 후 p90": q(comp[k], 90), "3초 창 p50": q(w3[k], 50),
                                          "3초 창 p90": q(w3[k], 90), "n": len(comp[k])} for k in POSE},
        "이동 vs 정지 AUROC": {"보정 전 1초": auc(pool(raw, move), pool(raw, still)),
                             "보정 후 1초": auc(pool(comp, move), pool(comp, still)),
                             "보정 후 3초 창": auc(pool(w3, move), pool(w3, still))},
        "3초 창 정지 임계": {str(th): {"정지를 정지로": round(float(np.mean(np.array(pool(w3, still)) < th)), 3),
                                  "이동을 정지로 (오인)": round(float(np.mean(np.array(pool(w3, move)) < th)), 3)}
                           for th in (0.2, 0.25, 0.3)},
        "박스 h/w": {},
    }
    for g, A in asp.items():
        if not (A["Lying"] and A["Standing"]):
            continue
        neg = lambda x: [-v for v in x]
        res["박스 h/w"][g] = {**{f"{k} p50": q(A[k], 50) for k in POSE},
                             "누움 vs 서기·걷기 AUROC": auc(neg(A["Lying"]), neg(A["Standing"] + A["Walking"])),
                             "누움 vs 앉기 AUROC": auc(neg(A["Lying"]), neg(A["Sitting"])),
                             "길쭉함 누움 p50": q(elo[g]["Lying"], 50), "길쭉함 서기 p50": q(elo[g]["Standing"], 50),
                             "길쭉함 누움 vs 서기·걷기 AUROC": auc(elo[g]["Lying"], elo[g]["Standing"] + elo[g]["Walking"]),
                             "누움 박스 n": len(A["Lying"])}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
