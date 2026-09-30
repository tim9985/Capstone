"""
make_posture_crops.py — 자세 분류 크롭 (상태 인지 A2 · 2026-09-30)

왜
  09-30 발표 피드백으로 자세를 다시 본다 → obsidian 「06 요구조자 상태 인지 계획」
  지난 9번은 탐지기 3클래스 · 섞인 분할이었다. 이번엔 **탐지 뒤 크롭 분류 + 장소 분리 평가**.

규칙
  · 1080p 환산 해상도로 자른 뒤 128 px 정사각 (운용 입력과 같은 정보량) · 여백 = 긴 변 × 1.5
  · 자세 3종: lying · sitting · standing (걷기·달리기는 standing)
  · 장소 분리 — AI-Hub 학습 산악2·3·4·6·평지(수풀)1 / val 산악8 / 평가 산악5 (= test_kr 장소)
  · Okutama 는 평가 전용 (비스듬 25편 · 수직 15편 · 1초 1장 · 1280×720 추출 프레임)
  · SARD 는 학습 보강용 (비스듬 · 사람이 붙인 자세 · not_defined 제외)
  · NOMAD 는 학습 보강용 (미국 농장 · 비스듬~수직 · 고도 10~90 m · 10-01 추가) — 자세는 **활동 구간에서 유도**
    Walking → standing · Laying · Hiding (Laying) → lying · Hiding 은 자세 불명이라 뺀다 · 가시도 50 % 미만 뺀다
    서기는 상한 3,000 (걷기 프레임이 압도적으로 많다) · 앉기 라벨 없음

출력
  data/pose_cls/<묶음>/{train,val}/{lying,sitting,standing}/*.jpg  — ultralytics 분류 폴더 구조
  data/pose_eval/<평가셋>/{lying,sitting,standing}/*.jpg
  각 폴더의 crops.csv — 파일 · 출처 · 장소 · 시점 · 고도 · 박스 w/h (1080p 환산 px) · 자세
  ⚠ 원본 파생물 — git 에 올리지 않는다 (data/ 는 gitignore)

실행
  python make_posture_crops.py aihub      # AI-Hub 학습·val·평가(산악5)
  python make_posture_crops.py okutama    # Okutama 평가 (비스듬 · 수직)
  python make_posture_crops.py sard       # SARD 학습 보강
  python make_posture_crops.py nomad      # NOMAD 학습 보강 (10-01)
  python make_posture_crops.py all
"""
import csv
import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

BASE = Path(__file__).resolve().parent
RAW = Path("/home/se/JupyterLAB/Capstone/data/raw")
AIHUB = RAW / "AIHub182"
AIHUB_LAB = AIHUB / "064.드론_이동체_인지_영상(도로_고정)" / "01.데이터"
OKU = RAW / "okutama"
SARD = RAW / "sard2" / "search-and-rescue-2"
OUT_CLS = BASE / "data" / "pose_cls"
OUT_EVAL = BASE / "data" / "pose_eval"
SIZE, MARGIN = 128, 1.5
CLASSES = ("lying", "sitting", "standing")

AIHUB_SPLIT = {"산악2": "train", "산악3": "train", "산악4": "train", "산악6": "train", "평지(수풀)1": "train",
               "산악8": "val", "산악5": "test"}
OKU_NADIR = set("2.2.2 1.1.8 1.1.4 2.2.10 2.2.5 2.2.1 1.1.11 1.1.7 2.2.7 1.1.5 2.2.4 2.2.3 1.1.9 2.2.6 2.2.8".split())
OKU_POSE = {"Lying": "lying", "Sitting": "sitting", "Standing": "standing", "Walking": "standing", "Running": "standing"}
SARD_POSE = {0: "standing", 1: "standing", 2: "lying", 4: "sitting", 5: "standing"}   # 3 = not_defined 제외
NOMAD = RAW / "NOMAD"
NOMAD_POSE = {"Walking": "standing", "Laying": "lying", "Hiding (Laying)": "lying"}          # Hiding 은 뺀다
NOMAD_STAND_CAP, NOMAD_MIN_VIS = 3000, 50


def crop(img, box, to1080):
    """box = (x1, y1, x2, y2) 원본 px · to1080 = 원본 → 1080p 환산 배율. 정사각 여백 크롭 → SIZE."""
    x1, y1, x2, y2 = box
    cx, cy, side = (x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1) * MARGIN
    side = max(side, 8 / to1080)
    H, W = img.shape[:2]
    a, b = int(round(cx - side / 2)), int(round(cy - side / 2))
    c, d = a + int(round(side)), b + int(round(side))
    pad = [max(0, -b), max(0, d - H), max(0, -a), max(0, c - W)]
    patch = img[max(0, b):min(H, d), max(0, a):min(W, c)]
    if any(pad):
        patch = cv2.copyMakeBorder(patch, *pad, cv2.BORDER_REFLECT)
    if to1080 < 1:                               # 먼저 1080p 환산으로 줄여 정보량을 운용과 맞춘다
        s = max(4, int(round(patch.shape[0] * to1080)))
        patch = cv2.resize(patch, (s, s), interpolation=cv2.INTER_AREA)
    interp = cv2.INTER_AREA if patch.shape[0] > SIZE else cv2.INTER_LINEAR
    return cv2.resize(patch, (SIZE, SIZE), interpolation=interp)


def write_rows(root, rows):
    root.mkdir(parents=True, exist_ok=True)
    new = not (root / "crops.csv").exists()
    with open(root / "crops.csv", "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["file", "source", "place", "view", "alt", "w1080", "h1080", "pose"])
        w.writerows(rows)


# ── AI-Hub 182 ─────────────────────────────────────────────────────────────
def _aihub_job(job):
    img_path, lab_path, out_dir, tag = job
    d = json.loads(Path(lab_path).read_text(encoding="utf-8", errors="replace"))
    img = cv2.imread(img_path)
    if img is None:
        return []
    to1080 = 1080 / img.shape[0]
    alt = (d.get("Metadata") or {}).get("altitude")
    rows = []
    for k, a in enumerate(d.get("annotations", [])):
        pose = (a.get("attributes") or {}).get("person_pose")
        pts = a.get("points") or []
        if pose not in CLASSES or len(pts) < 3:
            continue
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        box = (min(xs), min(ys), max(xs), max(ys))
        if box[2] - box[0] < 4 or box[3] - box[1] < 4:
            continue
        name = f"{tag}_{Path(img_path).stem.split('_')[-1]}_{abs(hash(img_path)) % 10**8:08d}_{k}.jpg"
        cv2.imwrite(str(Path(out_dir) / pose / name), crop(img, box, to1080), [cv2.IMWRITE_JPEG_QUALITY, 95])
        rows.append([f"{pose}/{name}", "aihub", tag, "nadir", alt,
                     round((box[2] - box[0]) * to1080, 1), round((box[3] - box[1]) * to1080, 1), pose])
    return rows


def aihub():
    cache = BASE / "data" / "pose_cls" / "_aihub_label_index.json"
    if cache.exists():
        index = json.loads(cache.read_text())
    else:
        index = {p.stem: str(p) for p in AIHUB_LAB.rglob("03_survivor/**/*.json")}
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(index))
    print(f"AI-Hub 라벨 색인 {len(index):,}개")
    jobs, cnt = [], Counter()
    for img in sorted(AIHUB.glob("조난자_*/**/*.jpg")):
        place = next((p for p in img.parts if p in AIHUB_SPLIT), None)
        if place is None or img.stem not in index:
            cnt["건너뜀"] += 1
            continue
        split = AIHUB_SPLIT[place]
        out = (OUT_EVAL / "aihub_test") if split == "test" else (OUT_CLS / "aihub" / split)
        for c in CLASSES:
            (out / c).mkdir(parents=True, exist_ok=True)
        jobs.append((str(img), index[img.stem], str(out), place))
        cnt[split] += 1
    print("AI-Hub 이미지", dict(cnt))
    with Pool(14, initializer=cv2.setNumThreads, initargs=(1,)) as pool:
        per_out = defaultdict(list)
        for job, rows in zip(jobs, pool.imap(_aihub_job, jobs, chunksize=16)):
            per_out[job[2]] += rows
    for out, rows in per_out.items():
        write_rows(Path(out), rows)
        print(f"  {Path(out).relative_to(BASE)}: {dict(Counter(r[-1] for r in rows))}")


# ── Okutama (평가 전용) ─────────────────────────────────────────────────────
def _okutama_job(job):
    vid, frame_path, boxes, out_dir, view = job
    img = cv2.imread(frame_path)
    if img is None:
        return []
    rows = []
    for tid, (x1, y1, x2, y2, pose) in boxes:
        name = f"oku_{vid}_{Path(frame_path).stem}_{tid}.jpg"
        cv2.imwrite(str(Path(out_dir) / pose / name), crop(img, (x1, y1, x2, y2), 1.0), [cv2.IMWRITE_JPEG_QUALITY, 95])
        rows.append([f"{pose}/{name}", "okutama", vid, view, "", round(x2 - x1, 1), round(y2 - y1, 1), pose])
    return rows


def okutama():
    quoted = re.compile(r'"([^"]*)"')
    frame_dir = {os.path.basename(d): d for d in glob.glob(str(OKU / "Drone*/*/Extracted-Frames-1280x720/*"))}
    jobs = []
    for lp in sorted(glob.glob(str(OKU / "Labels/MultiActionLabels/3840x2160/*.txt"))):
        vid = Path(lp).stem
        if vid not in frame_dir:
            continue
        view = "nadir" if vid in OKU_NADIR else "oblique"
        out = OUT_EVAL / f"okutama_{'nadir' if view == 'nadir' else 'obl'}"
        for c in CLASSES:
            (out / c).mkdir(parents=True, exist_ok=True)
        per = defaultdict(list)
        for ln in open(lp):
            t = ln.split()
            if len(t) < 10 or int(t[6]):
                continue
            tid, x1, y1, x2, y2, fr = map(int, t[:6])
            if fr % 30:
                continue
            act = next((a for a in quoted.findall(ln)[1:] if a in OKU_POSE), None)
            if act and (x2 - x1) > 9 and (y2 - y1) > 9:          # 4K 기준 · 1280 에서 3 px 초과
                per[fr].append((tid, (x1 / 3, y1 / 3, x2 / 3, y2 / 3, OKU_POSE[act])))
        for fr, boxes in per.items():
            fp = f"{frame_dir[vid]}/{fr}.jpg"
            if os.path.exists(fp):
                jobs.append((vid, fp, boxes, str(out), view))
    with Pool(14, initializer=cv2.setNumThreads, initargs=(1,)) as pool:
        per_out = defaultdict(list)
        for job, rows in zip(jobs, pool.imap(_okutama_job, jobs, chunksize=8)):
            per_out[job[3]] += rows
    for out, rows in per_out.items():
        write_rows(Path(out), rows)
        print(f"  {Path(out).relative_to(BASE)}: {dict(Counter(r[-1] for r in rows))}")


# ── SARD (학습 보강 · 비스듬) ────────────────────────────────────────────────
def _sard_job(job):
    img_path, lab_path, out_dir = job
    img = cv2.imread(img_path)
    if img is None:
        return []
    H, W = img.shape[:2]
    to1080 = min(1.0, 1080 / H)
    rows = []
    for k, ln in enumerate(open(lab_path)):
        t = ln.split()
        if len(t) != 5 or int(t[0]) not in SARD_POSE:
            continue
        pose = SARD_POSE[int(t[0])]
        cx, cy, bw, bh = (float(v) for v in t[1:])
        box = ((cx - bw / 2) * W, (cy - bh / 2) * H, (cx + bw / 2) * W, (cy + bh / 2) * H)
        name = f"sard_{Path(img_path).stem[:40]}_{k}.jpg"
        cv2.imwrite(str(Path(out_dir) / pose / name), crop(img, box, to1080), [cv2.IMWRITE_JPEG_QUALITY, 95])
        rows.append([f"{pose}/{name}", "sard", "sard", "oblique", "", round(bw * W * to1080, 1),
                     round(bh * H * to1080, 1), pose])
    return rows


def sard():
    out = OUT_CLS / "sard" / "train"
    for c in CLASSES:
        (out / c).mkdir(parents=True, exist_ok=True)
    jobs = []
    for split in ("train", "valid", "test"):                  # 평가에 안 쓰므로 전부 학습 보강
        for img in sorted((SARD / split / "images").glob("*")):
            lab = SARD / split / "labels" / (img.stem + ".txt")
            if lab.exists():
                jobs.append((str(img), str(lab), str(out)))
    with Pool(14, initializer=cv2.setNumThreads, initargs=(1,)) as pool:
        rows = [r for rs in pool.imap(_sard_job, jobs, chunksize=8) for r in rs]
    write_rows(out, rows)
    print(f"  {out.relative_to(BASE)}: {dict(Counter(r[-1] for r in rows))}")


# ── NOMAD (학습 보강 · 활동 구간 유도 자세) ─────────────────────────────────
def _num(x):
    m = re.search(r"\d+(\.\d+)?", str(x))                    # 원본 오타 '1320}' · '1140.1598'
    return float(m.group()) if m else None


def _nomad_act(acts, actor, dist, frame):
    for name, spans in acts.get(actor, {}).get(dist, {}).items():
        for sp in spans:
            if sp:
                a, b = _num(sp[0]), _num(sp[-1])
                if a is not None and b is not None and a <= frame <= b:
                    return name
    return None


def _nomad_job(job):
    img_path, boxes, out_dir, tag = job
    img = cv2.imread(img_path)
    if img is None:
        return []
    to1080 = 1080 / img.shape[0]
    rows = []
    for k, (x, y, w, h, pose) in enumerate(boxes):
        if max(w, h) * to1080 < 12:                           # 1080p 에서 12 px 미만은 자세 단서가 없다
            continue
        name = f"nomad_{Path(img_path).stem}_{k}.jpg"
        cv2.imwrite(str(Path(out_dir) / pose / name), crop(img, (x, y, x + w, y + h), to1080), [cv2.IMWRITE_JPEG_QUALITY, 95])
        rows.append([f"{pose}/{name}", "nomad", tag, "oblique", tag.split("_")[-1], round(w * to1080, 1), round(h * to1080, 1), pose])
    return rows


def nomad():
    import random
    out = OUT_CLS / "nomad" / "train"
    for c in CLASSES:
        (out / c).mkdir(parents=True, exist_ok=True)
    ann = json.load(open(NOMAD / "annotations.json"))
    acts = {a["id"]: a["labels"] for a in json.load(open(NOMAD / "activityLabels.json"))}
    by_pose = defaultdict(list)
    for r in ann:
        fn = r["file_name"]; actor_s, dist_s, f_s = fn[:-4].split("_")
        img = NOMAD / "images" / actor_s / f"{actor_s}_{dist_s}" / fn
        if not img.exists():
            continue
        pose = NOMAD_POSE.get(_nomad_act(acts, int(actor_s[5:]), dist_s[1:], int(f_s[1:])))
        if pose is None:
            continue
        boxes = [(*b["bbox"], pose) for b in r["annotations"] if int(b.get("visibility", 100)) >= NOMAD_MIN_VIS]
        if boxes:
            by_pose[pose].append((str(img), boxes, str(out), f"{actor_s}_{dist_s}"))
    rng = random.Random(7)
    rng.shuffle(by_pose["standing"])
    jobs = by_pose["lying"] + by_pose["standing"][:NOMAD_STAND_CAP]
    print("NOMAD 프레임", {k: len(v) for k, v in by_pose.items()}, "→ 서기 상한", NOMAD_STAND_CAP)
    with Pool(14, initializer=cv2.setNumThreads, initargs=(1,)) as pool:
        rows = [r for rs in pool.imap(_nomad_job, jobs, chunksize=8) for r in rs]
    write_rows(out, rows)
    print(f"  {out.relative_to(BASE)}: {dict(Counter(r[-1] for r in rows))}")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name, fn in (("aihub", aihub), ("okutama", okutama), ("sard", sard), ("nomad", nomad)):
        if what in (name, "all"):
            print(f"== {name}")
            fn()
