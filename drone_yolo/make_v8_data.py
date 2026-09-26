"""
make_v8_data.py — v8 추가 데이터: NII-CU 45° (2026-09-26)

  NII-CU Multispectral Aerial Person Detection (Speth et al. 2022 · CC BY-NC-SA 3.0)
  지바 야구장과 주변 · 2020-01 · 카메라 45° 아래 · 고도 20~50 m · 4K · 사람 긴 변 중앙 86 px (1920 기준 43 px)
  라벨 열: x1 y1 x2 y2 type occluded bad · type 0 = 둘 다 보임 · 1 = 열화상에만 · 2 = RGB 에만
  · RGB 에 안 보이는 사람(type 1)과 흐린 사람(bad 1)은 **회색(114)으로 가린다** — '사람 없음'으로 배우지 않게
  · 프레임은 10 간격(3 fps) → frame % 20 == 1 만 (0.67 s 간격) · train 만 학습에 쓴다
  · val(flight3 뒤쪽) 은 1920×1080 으로 줄여 data/test_nii — 학습과 **같은 장소**라 판정용이 아니다 (박스 습관 · 무학습 성능 확인용)
출력  data/raw/NII-CU/yolo/ · data/det_v6/nii/ · configs/lists/nii_train.txt · data/test_nii/ · metrics/v8_data_stats.json
"""
import json, zlib
from multiprocessing import Pool
from pathlib import Path
import cv2

from make_v6_data import crop_one

BASE = Path(__file__).resolve().parent
D = BASE / "data"
SRC = D / "raw" / "NII-CU" / "rgb-t"
STAGE = D / "raw" / "NII-CU" / "yolo"
OUT = D / "det_v6" / "nii" / "images" / "train"
TEST = D / "test_nii"
LISTS = BASE / "configs" / "lists"


def load(stem, split):
    keep, mask = [], []
    for ln in (SRC / "labels" / split / f"{stem}.txt").read_text().splitlines():
        t = ln.split()
        if len(t) < 7:
            continue
        x1, y1, x2, y2 = map(float, t[:4]); ty, occ, bad = int(t[4]), int(t[5]), int(t[6])
        (mask if ty == 1 or bad == 1 else keep).append((x1, y1, x2, y2, occ))
    return keep, mask


def stage_one(a):
    stem, split, out_img, out_lbl, scale = a
    img = cv2.imread(str(SRC / "images" / "rgb" / split / f"{stem}.jpg"))
    if img is None:
        return None
    H, W = img.shape[:2]
    keep, mask = load(stem, split)
    for x1, y1, x2, y2, _ in mask:
        dx, dy = 0.1 * (x2 - x1), 0.1 * (y2 - y1)
        img[max(0, int(y1 - dy)):min(H, int(y2 + dy)), max(0, int(x1 - dx)):min(W, int(x2 + dx))] = 114
    if scale != 1:
        img = cv2.resize(img, (int(W * scale), int(H * scale)), interpolation=cv2.INTER_AREA)
    Path(out_img).parent.mkdir(parents=True, exist_ok=True); Path(out_lbl).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_img), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    Path(out_lbl).write_text("".join(f"0 {(x1+x2)/2/W:.6f} {(y1+y2)/2/H:.6f} {(x2-x1)/W:.6f} {(y2-y1)/H:.6f}\n"
                                     for x1, y1, x2, y2, _ in keep))
    return stem, len(keep), sum(o for *_, o in keep), len(mask)


def main():
    tr = sorted(p.stem for p in (SRC / "labels" / "train").glob("*.txt")
                if int(p.stem.split("frame")[1]) % 20 == 1 and (SRC / "images" / "rgb" / "train" / f"{p.stem}.jpg").exists())
    va = sorted(p.stem for p in (SRC / "labels" / "val").glob("*.txt"))[::3]
    jobs = [(s, "train", STAGE / "images" / f"{s}.jpg", STAGE / "labels" / f"{s}.txt", 1) for s in tr]
    jobs += [(s, "val", TEST / "images" / f"{s}.jpg", TEST / "labels" / f"{s}.txt", 0.5) for s in va]
    with Pool(6) as pool:
        st = [r for r in pool.imap_unordered(stage_one, jobs, chunksize=8) if r]
    cj = [(str(STAGE / "images" / f"{s}.jpg"), str(STAGE / "labels" / f"{s}.txt"), None,
           str(OUT / f"nii_{s}.jpg"), zlib.crc32(s.encode())) for s in tr]
    with Pool(6) as pool:
        crops = [r for r in pool.imap_unordered(crop_one, cj, chunksize=8) if r]
    crops.sort()
    (LISTS / "nii_train.txt").write_text("\n".join(c[0] for c in crops) + "\n")
    trs = [x for x in st if x[0] in set(tr)]
    stats = {"train_프레임": len(tr), "train_사람": sum(x[1] for x in trs), "train_가림": sum(x[2] for x in trs),
             "가린_사람(RGB 안 보임·흐림)": sum(x[3] for x in trs), "크롭": len(crops),
             "크롭_사람px_중앙": round(sorted(c[1] for c in crops)[len(crops) // 2], 1) if crops else None,
             "test_nii(같은 장소 · 판정용 아님)": len(va)}
    (BASE / "metrics" / "v8_data_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1))
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
