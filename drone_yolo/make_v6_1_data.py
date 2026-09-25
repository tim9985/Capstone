"""
make_v6_1_data.py — v6_1 데이터 (2026-09-25 · 진단 근거: diag_misses_v6_obl.json)

  v6 에서 바꾸는 것 (나머지는 train_v6.txt 그대로)
  ① 누운 사람 보강  SARD 1,980장 전부 → 크롭 (사람 31~94 px) — test_obl 누운 사람 재현율 0.32 · 앉은 0.50
  ② 비스듬 음성    Unicamp 사람 없는 프레임 500장 → 1280×720 창 — 음성 5 % 의 절반을 비스듬으로
                   (라벨이 빠짐없는 원천만 쓴다 · VisDrone 은 무시 영역에 사람이 숨어 있어 음성으로 안 쓴다)
  ③ WiSARD 1월(2022) 측면 비행 595장은 비스듬으로 센다 (분류만 · 파일은 그대로)
  val: val_v6b (장소가 겹치지 않는 val)

출력  data/det_v6/{sard,uc_neg}/ · configs/lists/train_v6_1.txt · configs/data_v6_1.yaml · metrics/v6_1_data_stats.json
"""
import json, math, random, zlib
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

from make_v6_data import crop_one, CW, CH, T_LO, T_HI

BASE = Path(__file__).resolve().parent
D = BASE / "data"
OUT = D / "det_v6"
LISTS = BASE / "configs" / "lists"
rng = random.Random(61)


def neg_one(job):
    """사람 없는 4K 프레임에서 크롭과 같은 배율 분포로 창 하나"""
    src, dst, seed = job
    r = random.Random(seed)
    img = cv2.imread(src)
    if img is None:
        return None
    H0, W0 = img.shape[:2]
    # 비스듬 크롭의 배율 분포와 맞춘다: 사람 중앙값 69 px(4K) → 목표 31~94 px
    s = min(math.exp(r.uniform(math.log(T_LO), math.log(T_HI))) / 69.0, 2.0)
    ww, wh = min(CW / s, W0), min(CH / s, H0)
    x0 = int(r.uniform(0, W0 - ww)); y0 = int(r.uniform(0, H0 - wh))
    crop = cv2.resize(img[y0:y0 + int(wh), x0:x0 + int(ww)], (min(CW, round(ww * s)), min(CH, round(wh * s))),
                      interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    canvas = np.full((CH, CW, 3), 114, np.uint8); canvas[:crop.shape[0], :crop.shape[1]] = crop
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(dst, canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
    l = Path(dst.replace("/images/", "/labels/")).with_suffix(".txt"); l.parent.mkdir(parents=True, exist_ok=True); l.write_text("")
    return dst


def main():
    tr = [l.strip() for l in open(LISTS / "train_v6.txt") if l.strip()]
    # ① SARD
    sroot = D / "raw" / "sard2" / "search-and-rescue-2"
    sj = []
    for split in ("train", "valid", "test"):
        for p in sorted((sroot / split / "images").glob("*")):
            if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                sj.append((str(p), str(sroot / split / "labels" / f"{p.stem}.txt"), None,
                           str(OUT / "sard/images/train" / f"{split}_{p.stem}.jpg"), zlib.crc32(p.name.encode())))
    # ② Unicamp 빈 프레임
    uroot = D / "raw" / "unicamp_uav"
    nj = []
    for split in ("train", "val", "test"):
        for p in sorted((uroot / split / "images").glob("*.jpg")):
            l = uroot / split / "labels" / f"{p.stem}.txt"
            if l.exists() and not l.read_text().strip():
                nj.append((str(p), str(OUT / "uc_neg/images/train" / f"{split}_{p.stem}.jpg"), zlib.crc32(p.name.encode())))
    with Pool(6) as pool:
        sard = [r[0] for r in pool.imap_unordered(crop_one, sj, chunksize=8) if r]
        uneg = [r for r in pool.imap_unordered(neg_one, nj, chunksize=8) if r]
    # 음성: 기존 수직 음성 절반 + 비스듬 음성 → 합계는 v6 와 같은 약 5 %
    old_neg = [p for p in tr if "/det_neg/" in p]
    keep_neg = rng.sample(old_neg, max(0, len(old_neg) - len(uneg))) if len(uneg) < len(old_neg) else []
    base = [p for p in tr if "/det_neg/" not in p]
    train = base + sard + keep_neg + uneg
    rng.shuffle(train)
    (LISTS / "train_v6_1.txt").write_text("\n".join(train) + "\n")
    (BASE / "configs" / "data_v6_1.yaml").write_text(
        "# v6_1 (2026-09-25) — v6 + SARD(누운 사람) + 비스듬 음성(Unicamp 빈 프레임) · val = val_v6b\n"
        f"train: {LISTS / 'train_v6_1.txt'}\nval: {LISTS / 'val_v6b.txt'}\nnc: 1\nnames: ['person']\n")
    jan = sum(1 for p in base if "(1-" in p and "2022" in p)
    obl = sum(1 for p in base if "/unicamp/" in p or "/visdrone/" in p) + jan
    pos = len(base) + len(sard)
    stats = {"train": len(train), "SARD": len(sard), "비스듬_음성": len(uneg), "수직_음성": len(keep_neg),
             "음성_비율": round((len(uneg) + len(keep_neg)) / len(train), 3),
             "비스듬_양성(1월 측면 포함)": obl, "양성": pos, "비스듬_비율(양성)": round(obl / pos, 3)}
    (BASE / "metrics" / "v6_1_data_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1))
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
