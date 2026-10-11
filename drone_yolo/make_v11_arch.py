"""
make_v11_arch.py — A 팔 데이터: Archangel-Real 을 탐지 학습용으로 (10-12 · v7 넘기)

  근거  운용 조건 (고도 16~20 m · 45° 비스듬) 에 가장 가까운 공개 실사람 데이터 — 고도 15~50 m · 원 궤도 360° · 서기 · 걷기 · 무릎 · 누움 · 기어감
        탐지 학습에는 한 번도 안 씀 (자세 A1~A6 에만) · test_obl (Okutama 비스듬) 의 작은 사람 24~36 px 재현율 0.59 가 가장 큰 구멍
  원본  data/raw/archangel/real/*.mp4 (1304×978 · 10 fps) · Archangel_annotations.zip (Anno.Ai JSON · 클립 <영상>-<a>-<b>.mp4 → 영상 프레임 = a − 1 + position)
  뽑기  주석이 있는 프레임만 · 영상마다 1초 (10 프레임) 간격 · 그 프레임의 **사람 주석 전부** (추적별로 솎지 않음 — 솎으면 사람이 빠진 라벨이 된다)
        원래 크기 그대로 1280×720 창 (가로 1304 → 거의 전부 · 세로 978 → 사람 하나를 기준으로 위치 지터)
        라벨: 창 밖으로 잘린 사람은 보이는 넓이 ≥ 40 % 일 때만 · 가로 · 세로 2 px 미만 버림 · 차량은 라벨 없음 (배경)
        영상당 상한 --per-video (기본 40) → 약 3,000장
  ⚠ Archangel 파생물 — git 에 올리지 않는다 (data/ 는 gitignore)
출력  data/det_v11/arch/{images,labels}/train · configs/lists/arch_v11.txt · metrics/v11_arch_stats.json
실행: /home/se/miniconda3/envs/drone/bin/python make_v11_arch.py [--per-video 40]
"""
import argparse
import json
import random
import re
import zipfile
import zlib
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

BASE = Path(__file__).resolve().parent
RAW = Path("/home/se/JupyterLAB/Capstone/data/raw/archangel")
VID = RAW / "real"
OUT = BASE / "data" / "det_v11" / "arch"
LISTS = BASE / "configs" / "lists"
CW, CH = 1280, 720
STEP = 10
PERSON = ("person- standing", "person- walking", "person- lying down", "person- kneeling", "person- crawling")
NAME = re.compile(r"AA_BP_(\d+)_([\d-]+)_(\d+)_(\d+)_(?:(moving)_)?\d+_\d+_EO")


def load_ann():
    """영상 → {프레임: [(x1, y1, x2, y2, 자세)]} · 주석 클립 범위 (프레임 집합)"""
    boxes, covered = defaultdict(lambda: defaultdict(list)), defaultdict(set)
    with zipfile.ZipFile(RAW / "Archangel_annotations.zip") as z:
        for n in z.namelist():
            if not n.endswith(".json"):
                continue
            d = json.loads(z.read(n))
            for v in d["data"]:
                m = re.match(r"(.+_EO)-(\d+)-(\d+)\.mp4$", v["name"])
                vid, a, b = m.group(1), int(m.group(2)), int(m.group(3))
                covered[vid].update(range(a - 1, b))
                for o in v["objectOccurrences"]:
                    lab = o["labels"][0]["name"] if o["labels"] else o["name"]
                    if lab not in PERSON:
                        continue
                    for t in o["tracks"]:
                        for an in t["annotations"]:
                            c = an["coords"]
                            boxes[vid][a - 1 + an["position"]].append((c["xMin"], c["yMin"], c["xMax"], c["yMax"], lab.split("- ")[1]))
    return boxes, covered


def one_video(job):
    vid, frames, per_video = job
    r = random.Random(zlib.crc32(vid.encode()))
    pick = sorted(frames)[::STEP]
    if len(pick) > per_video:
        pick = sorted(r.sample(pick, per_video))
    want = set(pick); out = []
    cap = cv2.VideoCapture(str(VID / f"{vid}.mp4")); fi = 0; last = max(want) if want else -1
    while fi <= last:                                    # H.264 탐색은 프레임이 틀어진다 → 순차 디코딩
        ok, img = cap.read()
        if not ok:
            break
        if fi in want:
            H, W = img.shape[:2]; b = frames[fi]
            a = b[r.randrange(len(b))]
            x0 = int(min(max((a[0] + a[2]) / 2 - CW / 2 + r.uniform(-0.3, 0.3) * CW, 0), W - CW))
            y0 = int(min(max((a[1] + a[3]) / 2 - CH / 2 + r.uniform(-0.3, 0.3) * CH, 0), H - CH))
            crop = img[y0:y0 + CH, x0:x0 + CW]
            lines, px, poses = [], [], []
            for x1, y1, x2, y2, pose in b:
                cx1, cy1, cx2, cy2 = max(x1, x0), max(y1, y0), min(x2, x0 + CW), min(y2, y0 + CH)
                if cx2 <= cx1 or cy2 <= cy1 or (cx2 - cx1) * (cy2 - cy1) < 0.4 * (x2 - x1) * (y2 - y1):
                    continue
                X1, Y1, X2, Y2 = cx1 - x0, cy1 - y0, cx2 - x0, cy2 - y0
                if X2 - X1 < 2 or Y2 - Y1 < 2:
                    continue
                lines.append(f"0 {(X1 + X2) / 2 / CW:.6f} {(Y1 + Y2) / 2 / CH:.6f} {(X2 - X1) / CW:.6f} {(Y2 - Y1) / CH:.6f}")
                px.append(max(X2 - X1, Y2 - Y1)); poses.append(pose)
            if lines:
                stem = f"arch_{vid}_{fi:05d}"
                cv2.imwrite(str(OUT / "images" / "train" / f"{stem}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
                (OUT / "labels" / "train" / f"{stem}.txt").write_text("\n".join(lines) + "\n")
                out.append((str(OUT / "images" / "train" / f"{stem}.jpg"), px, poses))
        fi += 1
    cap.release()
    return vid, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-video", type=int, default=40)
    a = ap.parse_args()
    boxes, covered = load_ann()
    (OUT / "images" / "train").mkdir(parents=True, exist_ok=True); (OUT / "labels" / "train").mkdir(parents=True, exist_ok=True)
    jobs = [(v, dict(boxes[v]), a.per_video) for v in sorted(boxes) if (VID / f"{v}.mp4").exists()]
    with Pool(8) as pool:
        res = list(pool.imap_unordered(one_video, jobs))
    files = sorted(f for _, out in res for f, _, _ in out)
    (LISTS / "arch_v11.txt").write_text("\n".join(files) + "\n")
    px = np.array([p for _, out in res for _, ps, _ in out for p in ps])
    poses = Counter(p for _, out in res for _, _, ps in out for p in ps)
    alts = Counter(int(NAME.match(v).group(3)) for v, out in res for _ in out)
    bins = [0, 24, 36, 50, 80, 1e9]; h = np.histogram(px, bins=bins)[0]
    st = {"영상": len(jobs), "주석 영상": len(boxes), "크롭": len(files), "사람": int(len(px)), "자세": dict(poses),
          "고도별 크롭": dict(sorted(alts.items())),
          "사람 px 분포 (긴 변)": {f"{bins[i]}-{bins[i + 1] if bins[i + 1] < 1e9 else '∞'}": round(v / len(px), 3) for i, v in enumerate(h)},
          "사람 px 사분위": np.percentile(px, [5, 25, 50, 75, 95]).round(1).tolist()}
    (BASE / "metrics" / "v11_arch_stats.json").write_text(json.dumps(st, ensure_ascii=False, indent=1))
    print(json.dumps(st, ensure_ascii=False))


if __name__ == "__main__":
    main()
