"""
posture_v2_data.py — 자세 판별 학습 계획 P1 · P2 데이터 (2026-10-03)

계획 → obsidian 「11 서버 학습 계획/08 자세 판별 학습 계획」

  c2a    C2A 합성 (재난 배경 · 5자세) 크롭 → data/pose_cls/c2a/train/
         긴 변 ≥ 24 px 만 · 누움(2) → lying · 앉음(3) · 무릎(1) → sitting · 서기(4) → standing · 굽힘(0) 은 뺀다
         자세별 상한 12,000 · train + val 분할만 (test 분할은 안 씀) · 영상 ~430 px 그대로 (1080p 환산 안 함)
  relh   상대 키 표 — 크롭마다 「같은 장면 다른 사람 박스」 의 높이 · 긴 변 중앙값 (**자세 라벨은 안 쓴다**)
         Okutama  같은 영상 · 화면 높이(y) ±10 % 띠 · 1초 1장 · 다른 추적 ID
         SARD · C2A  같은 영상(이미지) 의 다른 사람
         NOMAD   같은 배우 · 같은 고도 시퀀스 전체 (한 장에 한 명)
         AI-Hub  수직이라 뺀다 (NaN)
         운용에선 GeoResolver 추정 키로 바꾼다 — 여기선 "같은 장면 사람들 대부분은 서 있다" 는 가정
출력: <크롭 폴더>/relh.csv  (file · rel_h · rel_long · n_ref) — rel = 내 박스 ÷ 다른 사람 중앙값 (n_ref 0 = 없음)
⚠ 원본 파생물 — data/ 는 gitignore
실행: python posture_v2_data.py c2a | relh | all
"""
import csv
import glob
import json
import random
import re
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

from make_posture_crops import CLASSES, NOMAD, NOMAD_MIN_VIS, OKU, OUT_CLS, OUT_EVAL, RAW, SARD, crop, write_rows

C2A = RAW / "C2A" / "new_dataset3"
C2A_POSE = {2: "lying", 3: "sitting", 1: "sitting", 4: "standing"}
C2A_MIN, C2A_CAP = 24, 12000


# ── P2 C2A ────────────────────────────────────────────────────────────────
def _c2a_job(job):
    img_path, items, out_dir = job
    img = cv2.imread(img_path)
    if img is None:
        return []
    rows = []
    for k, box, pose in items:
        name = f"c2a_{Path(img_path).stem}_{k}.jpg"
        cv2.imwrite(str(Path(out_dir) / pose / name), crop(img, box, 1.0), [cv2.IMWRITE_JPEG_QUALITY, 95])
        place = re.sub(r"_image\d+.*$", "", Path(img_path).stem)
        rows.append([f"{pose}/{name}", "c2a", place, "c2a", "", round(box[2] - box[0], 1), round(box[3] - box[1], 1), pose])
    return rows


def c2a():
    out = OUT_CLS / "c2a" / "train"
    if (out / "crops.csv").exists():
        print("  이미 있음 — 건너뜀"); return
    for c in CLASSES:
        (out / c).mkdir(parents=True, exist_ok=True)
    labdir = C2A / "All labels with Pose information" / "labels"
    imgs = {p.stem: p for s in ("train", "val") for p in (C2A / s / "images").glob("*")}
    cand = []                                                   # (이미지, k, box, pose)
    for lp in sorted(labdir.glob("*.txt")):
        if lp.stem not in imgs:
            continue
        ip = imgs[lp.stem]
        W, H = _size(ip)
        for k, ln in enumerate(open(lp)):
            t = ln.split()
            if len(t) < 6 or int(float(t[5])) not in C2A_POSE:
                continue
            cx, cy, bw, bh = (float(v) for v in t[1:5])
            if max(bw * W, bh * H) < C2A_MIN:
                continue
            cand.append((str(ip), k, ((cx - bw / 2) * W, (cy - bh / 2) * H, (cx + bw / 2) * W, (cy + bh / 2) * H),
                         C2A_POSE[int(float(t[5]))]))
    print("C2A 후보", dict(Counter(c[3] for c in cand)))
    rng = random.Random(7); rng.shuffle(cand)
    keep, n = [], Counter()
    for c in cand:
        if n[c[3]] < C2A_CAP:
            keep.append(c); n[c[3]] += 1
    per = defaultdict(list)
    for ip, k, box, pose in keep:
        per[ip].append((k, box, pose))
    jobs = [(ip, items, str(out)) for ip, items in per.items()]
    with Pool(14, initializer=cv2.setNumThreads, initargs=(1,)) as pool:
        rows = [r for rs in pool.imap(_c2a_job, jobs, chunksize=16) for r in rs]
    write_rows(out, rows)
    print(f"  {out}: {dict(Counter(r[-1] for r in rows))}")


def _size(p):
    from PIL import Image
    with Image.open(p) as im:
        return im.size


# ── P1 상대 키 ─────────────────────────────────────────────────────────────
def _ref(h, lng, others_h, others_l):
    """내 박스 높이 · 긴 변 ÷ 다른 사람들의 중앙값 (같은 단위면 된다)."""
    if len(others_h) == 0:
        return "", "", 0
    return round(h / max(float(np.median(others_h)), 1e-6), 4), round(lng / max(float(np.median(others_l)), 1e-6), 4), len(others_h)


def write_relh(root, rows):
    with open(root / "relh.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["file", "rel_h", "rel_long", "n_ref"]); w.writerows(rows)
    n = sum(1 for r in rows if r[3] and int(r[3]) > 0)
    print(f"  {root.relative_to(OUT_CLS.parent)}: relh {n:,}/{len(rows):,} 있음")


def relh_okutama():
    vids = defaultdict(list)                                     # vid → [(fr, tid, cy, h, long)] (1280×720 px)
    for lp in glob.glob(str(OKU / "Labels/MultiActionLabels/3840x2160/*.txt")):
        vid = Path(lp).stem
        seen = set()
        for ln in open(lp):
            t = ln.split()
            if len(t) < 10 or int(t[6]):
                continue
            tid, x1, y1, x2, y2, fr = map(int, t[:6])
            if fr % 30 or (x2 - x1) <= 9 or (y2 - y1) <= 9 or (tid, fr) in seen:
                continue
            seen.add((tid, fr))
            vids[vid].append((fr, tid, (y1 + y2) / 6, (y2 - y1) / 3, max(x2 - x1, y2 - y1) / 3))
    arr = {v: np.array(b) for v, b in vids.items()}
    for s in ("okutama_obl", "okutama_nadir"):
        root = OUT_EVAL / s
        out = []
        for r in csv.DictReader(open(root / "crops.csv")):
            _, vid, fr, tid = Path(r["file"]).stem.split("_")       # oku_<영상>_<프레임>_<추적 ID>
            a = arr.get(vid)
            me = a[(a[:, 0] == int(fr)) & (a[:, 1] == int(tid))] if a is not None else []
            if a is None or not len(me):
                out.append([r["file"], "", "", 0]); continue
            cy = me[0, 2]
            m = (np.abs(a[:, 2] - cy) < 72) & (a[:, 1] != int(tid))   # 720 px 의 10 %
            out.append([r["file"], *_ref(me[0, 3], me[0, 4], a[m, 3], a[m, 4])])
        write_relh(root, out)


def relh_sard():
    root = OUT_CLS / "sard" / "train"
    labs = {}
    for split in ("train", "valid", "test"):
        for img in sorted((SARD / split / "images").glob("*")):
            lab = SARD / split / "labels" / (img.stem + ".txt")
            if lab.exists():
                labs[img.stem[:40]] = [list(map(float, l.split()[1:5])) for l in open(lab) if len(l.split()) == 5]
    out = []
    for r in csv.DictReader(open(root / "crops.csv")):
        stem, k = Path(r["file"]).stem[5:].rsplit("_", 1)
        b = labs.get(stem)
        if not b or int(k) >= len(b):
            out.append([r["file"], "", "", 0]); continue
        me, o = b[int(k)], [x for i, x in enumerate(b) if i != int(k)]   # 정규화 좌표 — 같은 이미지 안 비율이라 단위 무관
        out.append([r["file"], *_ref(me[3], max(me[2], me[3]), [x[3] for x in o], [max(x[2], x[3]) for x in o])])
    write_relh(root, out)
    return labs


def relh_nomad():
    root = OUT_CLS / "nomad" / "train"
    seq = defaultdict(list)
    for r in json.load(open(NOMAD / "annotations.json")):
        a, d, _ = r["file_name"][:-4].split("_")
        for b in r["annotations"]:                              # 영상 높이로 나눠 둔다 (crops.csv 는 1080p 환산)
            if int(b.get("visibility", 100)) >= NOMAD_MIN_VIS:
                seq[f"{a}_{d}"].append((b["bbox"][3] / r["height"], max(b["bbox"][2], b["bbox"][3]) / r["height"]))
    out = []
    for r in csv.DictReader(open(root / "crops.csv")):
        s = np.array(seq.get(r["place"], []))
        if not len(s):
            out.append([r["file"], "", "", 0]); continue
        h, w = float(r["h1080"]) / 1080, float(r["w1080"]) / 1080
        out.append([r["file"], *_ref(h, max(h, w), s[:, 0], s[:, 1])])
    write_relh(root, out)


def relh_c2a():
    root = OUT_CLS / "c2a" / "train"
    if not (root / "crops.csv").exists():
        return
    labdir = C2A / "All labels with Pose information" / "labels"
    cache = {}
    out = []
    for r in csv.DictReader(open(root / "crops.csv")):
        stem, k = Path(r["file"]).stem[4:].rsplit("_", 1)
        if stem not in cache:
            ip = next((p for s in ("train", "val") for p in (C2A / s / "images").glob(stem + ".*")), None)
            W, H = _size(ip)
            cache[stem] = [(float(l.split()[4]) * H, max(float(l.split()[3]) * W, float(l.split()[4]) * H))
                           for l in open(labdir / f"{stem}.txt") if len(l.split()) >= 6]
        o = [x for i, x in enumerate(cache[stem]) if i != int(k)]
        h, w = float(r["h1080"]), float(r["w1080"])
        out.append([r["file"], *_ref(h, max(h, w), [x[0] for x in o], [x[1] for x in o])])
    write_relh(root, out)


def relh():
    relh_okutama(); relh_sard(); relh_nomad(); relh_c2a()


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("c2a", "all"):
        print("== C2A 크롭"); c2a()
    if what in ("relh", "all"):
        print("== 상대 키"); relh()
