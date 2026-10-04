"""
make_v10_data.py — v10 데이터 = v9 + LADD (R3 · 2026-10-04)

  LADD (Lacmus Drone Dataset · 러시아 숲 수색 · GPL-3.0 · Dataset Ninja 사본 · 4000×3000 등 · 'pedestrian')
  → HERIDAL 과 같은 방식: 원본에서 사람 31~94 px 로 맞춘 1280×720 크롭 1장 (make_v6_data.crop_one)
  장소 · 계절 태그가 없다 → **이미지 번호순 앞 70 % 학습 · 경계 50장 버림 · 뒤쪽은 참고 평가 (test_ladd_h)**
    (번호가 촬영 순서라는 가정 · 장소 분리는 아니다 — 판정은 test_obl · test_v2 · test_kr 로만)
출력  data/raw/ladd/yolo_labels/ · data/det_v6/ladd/images|labels/train/ · configs/lists/ladd_v10.txt
      data/test_ladd_h/ (test_ladd 의 뒤쪽 번호만 · 심볼릭 링크) · metrics/v10_data_stats.json
실행: python make_v10_data.py
"""
import json
import os
import zlib
from multiprocessing import Pool
from pathlib import Path

from make_v6_data import crop_one

BASE = Path(__file__).resolve().parent
D = BASE / "data"
RAW = (D / "raw" / "ladd").resolve()
YL = RAW / "yolo_labels"
OUT = D / "det_v6" / "ladd" / "images" / "train"
LISTS = BASE / "configs" / "lists"
TRAIN_FRAC, GAP = 0.70, 50


def ann_to_yolo(ann):
    a = json.loads(ann.read_text())
    W, H = a["size"]["width"], a["size"]["height"]
    rows = []
    for o in a["objects"]:
        if o.get("geometryType") != "rectangle":
            continue
        (x1, y1), (x2, y2) = o["points"]["exterior"][:2]
        x1, x2 = sorted((x1, x2)); y1, y2 = sorted((y1, y2))
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        rows.append(f"0 {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
    return rows


def main():
    anns = sorted(RAW.rglob("ann/*.json"), key=lambda p: int(p.name.split(".")[0]) if p.name.split(".")[0].isdigit() else 10 ** 9)
    n = len(anns); cut = int(n * TRAIN_FRAC)
    train_anns, hold_anns = anns[:cut], anns[cut + GAP:]
    YL.mkdir(parents=True, exist_ok=True)
    jobs, n_box = [], 0
    for ann in train_anns:
        img = ann.parent.parent / "img" / ann.name[:-len(".json")]
        rows = ann_to_yolo(ann)
        if not rows or not img.exists():
            continue
        n_box += len(rows)
        lab = YL / f"{img.stem}.txt"; lab.write_text("\n".join(rows) + "\n")
        jobs.append((str(img), str(lab), None, str(OUT / f"ladd_{img.stem}.jpg"), zlib.crc32(img.name.encode())))
    with Pool(6) as pool:
        crops = [r[0] for r in pool.imap_unordered(crop_one, jobs, chunksize=8) if r]
    crops.sort()
    (LISTS / "ladd_v10.txt").write_text("\n".join(crops) + "\n")
    # 참고 평가: test_ladd 중 뒤쪽 번호만
    hold = {a.name.split(".")[0] for a in hold_anns}
    src, dst = D / "test_ladd", D / "test_ladd_h"
    (dst / "images").mkdir(parents=True, exist_ok=True); (dst / "labels").mkdir(exist_ok=True)
    k = 0
    for im in sorted((src / "images").glob("*.jpg")):
        if im.stem in hold:
            for sub, ext in (("images", ".jpg"), ("labels", ".txt")):
                s, t = (src / sub / im.stem).with_suffix(ext), (dst / sub / im.stem).with_suffix(ext)
                if s.exists() and not t.exists():
                    os.symlink(os.path.realpath(s), t)
            k += 1
    stats = {"LADD_전체": n, "학습_이미지": len(train_anns), "학습_사람": n_box, "학습_크롭": len(crops),
             "경계_버림": GAP, "참고평가_이미지(test_ladd_h)": k}
    (BASE / "metrics" / "v10_data_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1))
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
