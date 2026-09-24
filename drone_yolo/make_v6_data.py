"""
make_v6_data.py — 학습 계획 v6 (기본 마운트각 60°) 데이터 (2026-09-24)

  비스듬 ≥50 %  Unicamp-UAV train+val (영상당 상한 3,000) · VisDrone 비스듬 시퀀스(r ≥ 0.3 · 사람만)
               → 새로 크롭: 사람 긴 변 중앙값을 로그균등 31~94 px 에 맞춰 1280×720 (확대 최대 2배)
  수직 ≤50 %   기존 크롭 재사용 — AI-Hub 학습 3곳 · WiSARD(평가 2곳 제외) · NOMAD 상한 3,000
               · 사람 중앙값 25~120 px 만 · 장소당 상한 3,000 · 절반은 180° 회전을 미리 적용
               (60° 증강은 flipud 0 · degrees 10 이라 수직 크롭의 방향 다양성을 여기서 채운다)
  음성 5 %     det_neg (하드 네거티브)
  val(모니터링) 수직 val_v3_place 표집 3,000 + 비스듬(Unicamp test · VisDrone val) — 판정은 test_obl · test_v2

출력  data/det_v6/<원천>/images|labels/<split>/ · configs/lists/{train,val}_v6.txt · configs/data_v6.yaml
      metrics/v6_data_stats.json
"""
import csv, json, math, random, re, zlib
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

BASE = Path(__file__).resolve().parent
D = BASE / "data"
OUT = D / "det_v6"
LISTS = BASE / "configs" / "lists"
CW, CH = 1280, 720
T_LO, T_HI, MAX_UP = 31.0, 94.0, 2.0
PLACE_CAP = 3000
rng = random.Random(42)


# ── 비스듬: 새로 크롭 ─────────────────────────────────────────────
def read_yolo(lbl, w, h, keep=None):
    out = []
    if not lbl.exists():
        return out
    for ln in lbl.read_text().splitlines():
        t = ln.split()
        if len(t) < 5 or (keep is not None and int(t[0]) not in keep):
            continue
        cx, cy, bw, bh = (float(v) for v in t[1:5])
        out.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
    return out


def crop_one(job):
    src, lbl, keep, out_img, seed = job
    r = random.Random(seed)
    img = cv2.imread(str(src))
    if img is None:
        return None
    H0, W0 = img.shape[:2]
    boxes = read_yolo(Path(lbl), W0, H0, keep)
    if not boxes:
        return None
    b = np.array(boxes, np.float32)
    L = float(np.median(np.maximum(b[:, 2] - b[:, 0], b[:, 3] - b[:, 1])))
    t = math.exp(r.uniform(math.log(T_LO), math.log(T_HI)))
    s = min(t / max(L, 1.0), MAX_UP)
    ww, wh = CW / s, CH / s
    a = b[r.randrange(len(b))]
    cx, cy = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
    x0 = cx - r.uniform(0.15, 0.85) * ww if ww < W0 else 0.0
    y0 = cy - r.uniform(0.15, 0.85) * wh if wh < H0 else 0.0
    x0 = int(min(max(x0, 0), max(W0 - ww, 0))); y0 = int(min(max(y0, 0), max(H0 - wh, 0)))
    x1, y1 = int(min(x0 + ww, W0)), int(min(y0 + wh, H0))
    crop = img[y0:y1, x0:x1]
    nw, nh = min(CW, round((x1 - x0) * s)), min(CH, round((y1 - y0) * s))
    crop = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    canvas = np.full((CH, CW, 3), 114, np.uint8); canvas[:nh, :nw] = crop
    rows, px = [], []
    for bx in b:
        ox1, oy1 = max(bx[0], x0), max(bx[1], y0); ox2, oy2 = min(bx[2], x1), min(bx[3], y1)
        if ox2 <= ox1 or oy2 <= oy1:
            continue
        if (ox2 - ox1) * (oy2 - oy1) < 0.5 * (bx[2] - bx[0]) * (bx[3] - bx[1]):
            continue
        X1, Y1, X2, Y2 = (ox1 - x0) * s, (oy1 - y0) * s, (ox2 - x0) * s, (oy2 - y0) * s
        if X2 - X1 < 3 or Y2 - Y1 < 3:
            continue
        rows.append(f"0 {(X1+X2)/2/CW:.6f} {(Y1+Y2)/2/CH:.6f} {(X2-X1)/CW:.6f} {(Y2-Y1)/CH:.6f}")
        px.append(max(X2 - X1, Y2 - Y1))
    if not rows:
        return None
    out_img = Path(out_img)
    out_img.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_img), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
    ol = Path(str(out_img).replace("/images/", "/labels/")).with_suffix(".txt")
    ol.parent.mkdir(parents=True, exist_ok=True)
    ol.write_text("\n".join(rows) + "\n")
    return str(out_img), float(np.median(px))


def unicamp_jobs():
    root = D / "raw" / "unicamp_uav"
    jobs = {"train": [], "val": []}
    for split in ("train", "val"):
        per = defaultdict(list)
        for p in sorted((root / split / "images").glob("*.jpg")):
            per[p.stem.rsplit("_", 1)[0]].append(p)
        for vid, ps in per.items():
            rng.shuffle(ps)
            for p in ps:
                jobs["train"].append((vid, p, root / split / "labels" / f"{p.stem}.txt"))
    # 영상당 상한 3,000 (학습) · test 분할 → 모니터링 val (영상이 같아 판정용 아님)
    cnt, train = Counter(), []
    for vid, p, l in jobs["train"]:
        if cnt[vid] < PLACE_CAP:
            cnt[vid] += 1; train.append((str(p), str(l), None, str(OUT / "unicamp/images/train" / f"{p.stem}.jpg"), zlib.crc32(p.name.encode())))
    tp = sorted((root / "test" / "images").glob("*.jpg")); rng.shuffle(tp)
    val = [(str(p), str(root / "test/labels" / f"{p.stem}.txt"), None,
            str(OUT / "unicamp/images/val" / f"{p.stem}.jpg"), zlib.crc32(p.name.encode())) for p in tp[:600]]
    return train, val


def visdrone_jobs():
    rows = list(csv.DictReader(open(BASE / "metrics/survey_tilt2.csv", encoding="utf-8")))
    obl = {r["group"].split("seq")[1] for r in rows
           if r["group"].startswith("VisDrone") and float(r["corr_cy_px"]) >= 0.3}
    root = D / "raw" / "VisDrone"
    out = {}
    for split, dst in (("train", "train"), ("val", "val"), ("test", "train")):
        for p in sorted((root / "images" / split).glob("*.jpg")):
            if p.stem.split("_")[0] in obl:
                out.setdefault(dst, []).append((str(p), str(root / "labels" / split / f"{p.stem}.txt"), {0, 1},
                                                str(OUT / f"visdrone/images/{dst}" / f"{p.stem}.jpg"), zlib.crc32(p.name.encode())))
    return out.get("train", []), out.get("val", []), len(obl)


# ── 수직: 기존 크롭 재사용 ─────────────────────────────────────────
def med_px(img_path):
    l = Path(img_path.replace("/images/", "/labels/")).with_suffix(".txt")
    if not l.exists():
        return None
    L = [max(float(t[3]) * CW, float(t[4]) * CH) for t in (x.split() for x in l.read_text().splitlines()) if len(t) >= 5]
    return float(np.median(L)) if L else None


def place_of(p):
    n = Path(p).name
    if "/det_aihub/" in p:
        return "aihub_" + n.split("_")[0]
    if "/wisard" in p:
        m = re.match(r"\d+_([A-Za-z]+)", n)
        return "wisard_" + (m.group(1) if m else "?")
    return "nomad"


def nadir_pool():
    v3 = [l.strip() for l in open(LISTS / "train_v3_place_neg.txt") if l.strip()]
    pos = [p for p in v3 if "/det_neg/" not in p]
    neg = [p for p in v3 if "/det_neg/" in p]
    aihub = sorted(str(p) for p in (D / "det_aihub/images/train").glob("*.jpg"))
    return pos + aihub, neg


def rot180(p):
    img = cv2.imread(p)
    dst = OUT / "nadir_rot/images/train" / Path(p).name
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), np.ascontiguousarray(img[::-1, ::-1]), [cv2.IMWRITE_JPEG_QUALITY, 92])
    src_l = Path(p.replace("/images/", "/labels/")).with_suffix(".txt")
    rows = []
    for t in (x.split() for x in src_l.read_text().splitlines()):
        if len(t) >= 5:
            rows.append(f"{t[0]} {1-float(t[1]):.6f} {1-float(t[2]):.6f} {t[3]} {t[4]}")
    dl = Path(str(dst).replace("/images/", "/labels/")).with_suffix(".txt")
    dl.parent.mkdir(parents=True, exist_ok=True)
    dl.write_text("\n".join(rows) + "\n")
    return str(dst)


def main():
    stats = {}
    uc_tr, uc_va = unicamp_jobs()
    vd_tr, vd_va, n_seq = visdrone_jobs()
    print(f"작업: Unicamp {len(uc_tr)}+{len(uc_va)} · VisDrone(시퀀스 {n_seq}) {len(vd_tr)}+{len(vd_va)}", flush=True)
    with Pool(6) as pool:
        res = {k: [r for r in pool.imap_unordered(crop_one, jobs, chunksize=8) if r]
               for k, jobs in (("uc_tr", uc_tr), ("uc_va", uc_va), ("vd_tr", vd_tr), ("vd_va", vd_va))}
    obl_tr = [r[0] for r in res["uc_tr"] + res["vd_tr"]]
    obl_va = [r[0] for r in res["uc_va"] + res["vd_va"]]
    obl_px = [r[1] for r in res["uc_tr"] + res["vd_tr"]]
    stats["비스듬"] = {k: len(v) for k, v in res.items()}
    print("비스듬 크롭", stats["비스듬"], flush=True)

    # 수직: 크기 25~120 px · 장소 상한 3,000 · NOMAD 3,000 · 총량 = 비스듬 수 이하
    pos, neg = nadir_pool()
    by_place = defaultdict(list)
    for p in pos:
        m = med_px(p)
        if m is not None and 25 <= m <= 120:
            by_place[place_of(p)].append(p)
    for v in by_place.values():
        rng.shuffle(v)
    capped = {k: v[:PLACE_CAP] for k, v in by_place.items()}
    budget = len(obl_tr)
    total = sum(len(v) for v in capped.values())
    frac = min(1.0, budget / max(total, 1))
    nadir = []
    for k, v in capped.items():
        nadir += v[:int(round(len(v) * frac))]
    rng.shuffle(nadir)
    half = len(nadir) // 2
    print(f"수직 {len(nadir)}장 (180° 회전 {half}장 생성 중)", flush=True)
    with Pool(6) as pool:
        rotated = list(pool.imap_unordered(rot180, nadir[:half], chunksize=16))
    nadir_final = rotated + nadir[half:]
    stats["수직_장소별"] = {k: int(round(len(v) * frac)) for k, v in capped.items()}

    n_neg = int(round(0.05 * (len(obl_tr) + len(nadir_final)) / 0.95))
    rng.shuffle(neg)
    negs = neg[:n_neg]
    train = obl_tr + nadir_final + negs
    rng.shuffle(train)

    v3v = [l.strip() for l in open(LISTS / "val_v3_place.txt") if l.strip()]
    rng.shuffle(v3v)
    val = obl_va + v3v[:3000]

    (LISTS / "train_v6.txt").write_text("\n".join(train) + "\n")
    (LISTS / "val_v6.txt").write_text("\n".join(val) + "\n")
    (BASE / "configs" / "data_v6.yaml").write_text(
        "# 학습 계획 v6 (60°) — make_v6_data.py 생성 · 판정은 test_obl · test_v2 (학습에 안 쓴 장소)\n"
        f"train: {LISTS / 'train_v6.txt'}\nval: {LISTS / 'val_v6.txt'}\nnc: 1\nnames: ['person']\n")
    stats.update({"train": len(train), "비스듬_train": len(obl_tr), "수직_train": len(nadir_final),
                  "음성_train": len(negs), "비스듬_비율": round(len(obl_tr) / len(train), 3),
                  "val": len(val), "val_비스듬": len(obl_va),
                  "비스듬_사람px_분위수_5_25_50_75_95": np.percentile(obl_px, [5, 25, 50, 75, 95]).round(1).tolist()})
    (BASE / "metrics" / "v6_data_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1))
    print(json.dumps(stats, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
