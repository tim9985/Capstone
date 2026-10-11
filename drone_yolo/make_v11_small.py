"""
make_v11_small.py — S 팔 데이터: 야생 지형의 작은 사람 (10-11 · v7 넘기)

  근거 (10-11 진단 · 볼트 「v7 넘기 - 파이프라인 검토」)
    · v7 학습 사람 중 24 px 미만 9 % — 거의 VisDrone (도시 거리) · 야생 지형 출처 (WiSARD · HERIDAL · SARD) 는 0~3 %
      (WiSARD 는 94 px 규칙으로 키웠고 · HERIDAL · SARD 는 31~94 px 로 맞춤)
    · 평가셋 작은 사람: test_v2 24 px 미만 21 % (Karen 30 % · AP 0.23) · test_obl 24~36 px 34 %
    · FlyPose 검출기 (TinyPerson 등 11종) 가 작은 사람에서 이김 — Karen +12 · Carnation_0025 +7 %p · obl 36 px 미만 재현율 +3~5 %p
  방법  1280×720 캔버스를 2×2 칸 (640×360) 으로 나누고, 칸마다 원본 한 장의 16:9 창을 **줄여** 넣는다
        (INTER_AREA 축소 = 더 높이서 찍은 것과 같은 화질 · 키우기 없음)
        칸의 사람 목표 크기 t ~ 로그 균등 [10, 36] px · 배율 s = t / (그 장 사람 긴 변 중앙값)
        창이 원본보다 커야 하면 원본 전체 (s = max(640/W, 360/H)) — 그때 사람은 목표보다 크다 (통계에 남김)
        출처 3곳 같은 비율: HERIDAL (4000×3000 · CC BY 3.0) · SARD (1920×1080 · CC BY 4.0) · WiSARD (v7 이 쓴 비행만 · 평가셋 장소 Karen · Carnation 제외)
        라벨: 칸 밖으로 잘린 사람은 보이는 넓이 ≥ 40 % 일 때만 보이는 부분 · 가로 · 세로 2 px 미만은 버림
  ⚠ WiSARD · HERIDAL · SARD 파생물 — git 에 올리지 않는다 (data/ 는 gitignore)
출력  data/det_v11/small/{images,labels}/train · configs/lists/small_v11.txt · metrics/v11_small_stats.json
실행: /home/se/miniconda3/envs/drone/bin/python make_v11_small.py [--n 3000]
"""
import argparse
import json
import math
import random
import re
import zlib
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

BASE = Path(__file__).resolve().parent
D = BASE / "data"
OUT = D / "det_v11" / "small"
LISTS = BASE / "configs" / "lists"
CW, CH, QW, QH = 1280, 720, 640, 360
T_LO, T_HI = 10.0, 36.0
TEST_PLACES = ("Karen", "Carnation")                 # test_v2 장소 — 절대 쓰지 않는다


def yolo_boxes(txt, W, H):
    out = []
    if not Path(txt).exists():
        return out
    for ln in open(txt, encoding="utf-8", errors="replace"):
        q = ln.split()
        if len(q) != 5:
            continue
        try:
            _, cx, cy, bw, bh = map(float, q)
        except ValueError:
            continue
        out.append(((cx - bw / 2) * W, (cy - bh / 2) * H, (cx + bw / 2) * W, (cy + bh / 2) * H))
    return out


def sources():
    her = [(str(p), str(D / "raw/HERIDAL/yolo_labels" / f"{p.stem}.txt"))
           for p in sorted((D / "raw/HERIDAL/heridal_keras_retinanet_voc/JPEGImages").glob("*.jpg"))
           if (D / "raw/HERIDAL/yolo_labels" / f"{p.stem}.txt").exists()]
    sroot = D / "raw/sard2/search-and-rescue-2"
    sard = [(str(p), str(sroot / sp / "labels" / f"{p.stem}.txt")) for sp in ("train", "valid", "test")
            for p in sorted((sroot / sp / "images").glob("*")) if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    # WiSARD — v7 학습 목록에 크롭이 있는 비행만 (평가셋 장소 제외)
    v7 = {re.sub(r"_(c\d+|n)$", "", Path(l.strip()).stem) for l in open(LISTS / "train_v7.txt") if "/det_fov/wisard/" in l}
    wis, flights = [], Counter()
    for fd in sorted((D / "raw/WiSARD").iterdir()):
        if not fd.is_dir() or "_VIS" not in fd.name or any(t in fd.name for t in TEST_PLACES):
            continue
        fr = [p for p in sorted(fd.glob("*.jpg")) if p.with_suffix(".txt").exists() and p.with_suffix(".txt").stat().st_size > 0]
        if not any(p.stem in v7 for p in fr):
            continue
        flights[fd.name] = len(fr); wis += [(str(p), str(p.with_suffix(".txt"))) for p in fr]
    return {"heridal": her, "sard": sard, "wisard": wis}, flights


def quadrant(src, r):
    img_p, lab_p = src
    img = cv2.imread(img_p)
    if img is None:
        return None
    H, W = img.shape[:2]
    b = np.array(yolo_boxes(lab_p, W, H), np.float32).reshape(-1, 4)
    if not len(b):
        return None
    ref = float(np.median(np.maximum(b[:, 2] - b[:, 0], b[:, 3] - b[:, 1])))
    t = math.exp(r.uniform(math.log(T_LO), math.log(T_HI)))
    s = max(t / max(ref, 1.0), QW / W, QH / H)
    ww, wh = QW / s, QH / s
    a = b[r.randrange(len(b))]
    cx = (a[0] + a[2]) / 2 + r.uniform(-0.3, 0.3) * ww; cy = (a[1] + a[3]) / 2 + r.uniform(-0.3, 0.3) * wh
    x0 = min(max(cx - ww / 2, 0), W - ww); y0 = min(max(cy - wh / 2, 0), H - wh)
    X0, Y0, X1, Y1 = int(round(x0)), int(round(y0)), int(round(x0 + ww)), int(round(y0 + wh))
    q = cv2.resize(img[Y0:Y1, X0:X1], (QW, QH), interpolation=cv2.INTER_AREA)
    sx, sy = QW / (X1 - X0), QH / (Y1 - Y0)
    rows, px = [], []
    for x1, y1, x2, y2 in b:
        cx1, cy1, cx2, cy2 = max(x1, X0), max(y1, Y0), min(x2, X1), min(y2, Y1)
        if cx2 <= cx1 or cy2 <= cy1 or (cx2 - cx1) * (cy2 - cy1) < 0.4 * (x2 - x1) * (y2 - y1):
            continue
        bx1, by1, bx2, by2 = (cx1 - X0) * sx, (cy1 - Y0) * sy, (cx2 - X0) * sx, (cy2 - Y0) * sy
        if bx2 - bx1 < 2 or by2 - by1 < 2:
            continue
        rows.append((bx1, by1, bx2, by2)); px.append(max(bx2 - bx1, by2 - by1))
    return (q, rows, px, t) if rows else None


def canvas(job):
    k, picks = job
    r = random.Random(zlib.crc32(f"v11s{k}".encode()))
    cv = np.zeros((CH, CW, 3), np.uint8); lines, px, tg, used = [], [], [], []
    for qi, (name, src) in enumerate(picks):
        res = None
        for _ in range(3):                                   # 사람이 칸 안에 안 들어오면 다시
            res = quadrant(src, r)
            if res:
                break
        if not res:
            return None
        q, rows, p, t = res; ox, oy = (qi % 2) * QW, (qi // 2) * QH
        cv[oy:oy + QH, ox:ox + QW] = q; px += p; tg.append(t); used.append(name)
        for x1, y1, x2, y2 in rows:
            x1, y1, x2, y2 = x1 + ox, y1 + oy, x2 + ox, y2 + oy
            lines.append(f"0 {(x1 + x2) / 2 / CW:.6f} {(y1 + y2) / 2 / CH:.6f} {(x2 - x1) / CW:.6f} {(y2 - y1) / CH:.6f}")
    out = OUT / "images" / "train" / f"small_{k:05d}.jpg"
    cv2.imwrite(str(out), cv, [cv2.IMWRITE_JPEG_QUALITY, 92])
    (OUT / "labels" / "train" / f"small_{k:05d}.txt").write_text("\n".join(lines) + "\n")
    return str(out), px, tg, used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    a = ap.parse_args()
    src, flights = sources()
    (OUT / "images" / "train").mkdir(parents=True, exist_ok=True); (OUT / "labels" / "train").mkdir(parents=True, exist_ok=True)
    rng = random.Random(11); names = sorted(src)
    jobs = []
    for k in range(a.n):
        picks = []
        for _ in range(4):
            nm = rng.choice(names); picks.append((nm, rng.choice(src[nm])))
        jobs.append((k, picks))
    with Pool(8) as pool:
        res = [x for x in pool.imap_unordered(canvas, jobs, chunksize=16) if x]
    res.sort()
    (LISTS / "small_v11.txt").write_text("\n".join(x[0] for x in res) + "\n")
    px = np.array([p for x in res for p in x[1]]); tg = np.array([t for x in res for t in x[2]])
    bins = [0, 12, 16, 24, 36, 50, 1e9]; h = np.histogram(px, bins=bins)[0]
    st = {"캔버스": len(res), "사람": int(len(px)), "출처 칸 수": dict(Counter(n for x in res for n in x[3])),
          "WiSARD 비행 (v7 과 같음 · 평가셋 장소 제외)": len(flights), "WiSARD 프레임 후보": sum(flights.values()),
          "HERIDAL 장": len(src["heridal"]), "SARD 장": len(src["sard"]),
          "사람 px 분포 (긴 변)": {f"{bins[i]}-{bins[i + 1] if bins[i + 1] < 1e9 else '∞'}": round(v / len(px), 3) for i, v in enumerate(h)},
          "사람 px 사분위": np.percentile(px, [5, 25, 50, 75, 95]).round(1).tolist(),
          "목표 t 중앙": round(float(np.median(tg)), 1)}
    (BASE / "metrics" / "v11_small_stats.json").write_text(json.dumps(st, ensure_ascii=False, indent=1))
    print(json.dumps(st, ensure_ascii=False))


if __name__ == "__main__":
    main()
