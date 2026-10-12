"""
make_v12_data.py — v12 데이터: 원본부터 다시 짠 학습셋 (10-12 · 사용자 「데이터셋 구성부터 다시」)

  v7 파이프라인 점검 (볼트 「17 v12 데이터 재구성」) 에서 나온 문제 → v12 규칙
  ① 확대된 흐린 사람  VisDrone 크롭 90 % 확대 (58 % 는 2배 상한) · Unicamp 39 · SARD 38 · HERIDAL 28 %
                     → **확대 금지** — 배율 s ≤ 1 (원래 크기 또는 축소 · INTER_AREA = 더 높이서 찍은 화질)
  ② 장면 중복        Unicamp 영상 3개가 학습셋의 22 % · AI-Hub 3곳 · WiSARD 비행당 77장
                     → **시퀀스 (영상 · 비행 · 고도×시각) 마다 상한 · 시간 간격 고르게** · 출처 예산은 장면 다양성으로
  ③ 라벨 위생        VisDrone 무시 영역 (그림의 46 %) 안 군중이 배경으로 · 크롭 가장자리에 잘린 사람 (보이는 넓이 < 40~50 %) 이 라벨 없이 남음
                     → 무시 영역 · 잘린 조각 (보이는 넓이 < 25 %) 은 **가린다** (창 평균색) · 25 % 이상은 잘린 박스로 라벨
  ④ 사람 중심 크롭    사람 하나를 가운데 두고 자름 → **타일처럼** 창 위치를 그림 전체에서 고르게 (사람이 가장자리에도)
  ⑤ 안 쓴 데이터      비스듬 자연 지형 (NOMAD 2만 · Archangel) · 한국 수직 (AI-Hub 8곳 4.6만) · VisDrone 수직 · LADD · NII-CU
  ⑥ 음성            4.4 % → 약 8 % · 같은 그림의 사람 없는 창 (실제 물체 · 차 · 장비가 섞인 배경)

  영역 (양성 창 예산)  비스듬 도시 VisDrone 3,000 · Unicamp 450 · NII-CU 500 | 비스듬 자연 NOMAD 3,000 · SARD 1,500 · Archangel 1,500
                      수직 도시 VisDrone 1,200 | 수직 자연 WiSARD 2,800 · HERIDAL 1,600 · LADD 900 | 수직 한국 AI-Hub 2,500
  평가셋 제외        WiSARD Karen · Carnation (test_v2) · AI-Hub 산악5 (test_kr) · LADD 뒤쪽 30 % (test_ladd_h) · NII-CU flight3 (test_nii) · Okutama 안 씀
  val_v12 (모니터링) VisDrone val · Unicamp DJI_0085 · NOMAD 배우 096~100 · WiSARD 비행 3개 · AI-Hub 저수지1 — 학습과 겹치지 않음
  ⚠ 원본 파생물 — git 에 올리지 않는다 (data/ 는 git 밖)
출력  data/det_v12/{images,labels}/{train,val} · configs/lists/{train,val}_v12.txt · configs/data_v12.yaml · metrics/v12_data_stats.json
실행: /home/se/miniconda3/envs/drone/bin/python make_v12_data.py [--dry] [--only 출처,…]
"""
import argparse
import csv
import json
import math
import random
import re
import zlib
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

BASE = Path(__file__).resolve().parent
RAW = BASE / "data" / "raw"
OUT = BASE / "data" / "det_v12"
LISTS = BASE / "configs" / "lists"
CW, CH = 1280, 720
VIS_MIN, NEG_EVERY = 0.25, 12            # 잘린 사람 라벨 기준 · 선택 프레임 12장마다 음성 창 1장 (≈ 8 %)
TEST_WISARD = ("Karen", "Carnation")
WISARD_VAL = ("200426_SkookumCreek_Mavic_Mini_VIS_0006", "210924_FHL_Enterprise_VIS_0403", "210924_FHL_Enterprise_VIS_0409")


def yolo(lbl, W, H, keep=None):
    out = []
    if not Path(lbl).exists():
        return np.zeros((0, 4), np.float32)
    for ln in open(lbl, encoding="utf-8", errors="replace"):
        t = ln.split()
        if len(t) != 5:
            continue
        try:
            c, cx, cy, bw, bh = int(float(t[0])), *map(float, t[1:])
        except ValueError:
            continue
        if keep is not None and c not in keep:
            continue
        out.append(((cx - bw / 2) * W, (cy - bh / 2) * H, (cx + bw / 2) * W, (cy + bh / 2) * H))
    return np.array(out, np.float32).reshape(-1, 4)


def imsize(p):
    im = cv2.imread(str(p), cv2.IMREAD_REDUCED_GRAYSCALE_8)
    return (im.shape[1] * 8, im.shape[0] * 8) if im is not None else (0, 0)


# ── 출처 어댑터: 프레임 기록 = dict(src, dom, img, lab, kind, seq, order, split, scale) ───────────────
#    kind = 라벨 읽는 법 · scale = (규칙, 인자) — native(lo) · target(t_lo, t_hi) · p1080(u_lo)  (모두 s ≤ 1)
def src_visdrone():
    rows = list(csv.DictReader(open(BASE / "metrics/survey_tilt2.csv", encoding="utf-8")))
    obl = {r["group"].split("seq")[1] for r in rows if r["group"].startswith("VisDrone") and float(r["corr_cy_px"]) >= 0.3}
    out = []
    for split, raw in (("train", "train"), ("val", "val"), ("test", "test-dev")):
        for p in sorted((RAW / "VisDrone/images" / split).glob("*.jpg")):
            seq = p.stem.split("_")[0]; o = seq in obl
            out.append(dict(src="visdrone_obl" if o else "visdrone_nadir", dom="obl_urban" if o else "nad_urban", img=str(p),
                            lab=str(RAW / f"VisDrone/VisDrone2019-DET-{raw}/annotations/{p.stem}.txt"), kind="visdrone",
                            seq=seq, order=int(p.stem.split("_")[1]), split="val" if split == "val" else "train", scale=("native", 0.8)))
    return out


def src_unicamp():
    out = []
    for split in ("train", "val", "test"):
        for p in sorted((RAW / "unicamp_uav" / split / "images").glob("*.jpg")):
            vid, fr = p.stem.rsplit("_", 1)
            out.append(dict(src="unicamp", dom="obl_urban", img=str(p), lab=str(p.parent.parent / "labels" / f"{p.stem}.txt"),
                            kind="yolo", seq=vid, order=int(fr), split="val" if vid == "DJI_0085" else "train", scale=("target", 24, 80)))
    return out


def src_niicu():
    out = []
    for p in sorted((RAW / "NII-CU/yolo/images").glob("*")):
        fl, fr = p.stem.split("_frame")
        if fl == "flight3":                                       # test_nii
            continue
        out.append(dict(src="niicu", dom="obl_urban", img=str(p), lab=str(RAW / "NII-CU/yolo/labels" / f"{p.stem}.txt"), kind="yolo",
                        seq=fl, order=int(fr), split="train", scale=("target", 20, 80)))
    return out


def src_nomad():
    out = []
    for p in sorted((RAW / "NOMAD/images").glob("Actor*/Actor*_a*/*.jpg")):
        actor, alt = p.parent.name.split("_")
        lab = RAW / "NOMAD/labels" / actor / p.parent.name / f"{p.stem}.txt"
        fr = int(re.search(r"_f(\d+)", p.stem).group(1)) if re.search(r"_f(\d+)", p.stem) else 0
        out.append(dict(src="nomad", dom="obl_nat", img=str(p), lab=str(lab), kind="yolo", seq=p.parent.name, order=fr,
                        split="val" if int(actor[5:]) >= 96 else "train", scale=("target", 18, 100)))
    return out


def src_sard():
    root = RAW / "sard2/search-and-rescue-2"; out = []
    for sp in ("train", "valid", "test"):
        for k, p in enumerate(sorted((root / sp / "images").glob("*"))):
            out.append(dict(src="sard", dom="obl_nat", img=str(p), lab=str(root / sp / "labels" / f"{p.stem}.txt"), kind="yolo",
                            seq=f"{sp}_{k // 20}", order=k, split="train", scale=("target", 20, 90)))
    return out


def src_archangel():
    out = []
    for l in open(LISTS / "arch_v11.txt"):
        p = l.strip()
        if p:
            vid = Path(p).stem.rsplit("_", 1)[0]
            out.append(dict(src="archangel", dom="obl_nat", img=p, lab=p.replace("/images/", "/labels/")[:-4] + ".txt", kind="yolo_crop",
                            seq=vid, order=int(Path(p).stem.rsplit("_", 1)[1]), split="train", scale=("native", 1.0)))
    return out


def src_wisard():
    out = []
    for fd in sorted((RAW / "WiSARD").iterdir()):
        if not fd.is_dir() or "_VIS" not in fd.name or any(t in fd.name for t in TEST_WISARD):
            continue
        val = any(fd.name.startswith(v) or v in fd.name for v in WISARD_VAL)
        for k, p in enumerate(sorted(fd.glob("*.jpg"))):
            out.append(dict(src="wisard", dom="nad_nat", img=str(p), lab=str(p.with_suffix(".txt")), kind="yolo", seq=fd.name, order=k,
                            split="val" if val else "train", scale=("p1080", 0.667)))
    return out


def src_heridal():
    root = RAW / "HERIDAL/heridal_keras_retinanet_voc"; out = []
    for k, p in enumerate(sorted((root / "JPEGImages").glob("*.jpg"))):
        out.append(dict(src="heridal", dom="nad_nat", img=str(p), lab=str(RAW / "HERIDAL/yolo_labels" / f"{p.stem}.txt"), kind="yolo",
                        seq=f"h{k // 10}", order=k, split="train", scale=("target", 16, 72)))
    return out


def src_ladd():
    anns = sorted((RAW / "ladd/ds0/ann").glob("*.json"), key=lambda p: int(p.name.split(".")[0]) if p.name.split(".")[0].isdigit() else 10 ** 9)
    cut = int(len(anns) * 0.70); out = []
    for a in anns[:cut]:                                          # 뒤쪽 30 % = test_ladd_h (make_v10_data 와 같은 자르기)
        n = a.name.split(".")[0]
        img = next((RAW / "ladd/ds0/img").glob(f"{n}.*"), None)
        if img is None:
            continue
        out.append(dict(src="ladd", dom="nad_nat", img=str(img), lab=str(RAW / "ladd/yolo_labels" / f"{n}.txt"), kind="yolo",
                        seq=f"l{int(n) // 20 if n.isdigit() else 0}", order=int(n) if n.isdigit() else 0, split="train", scale=("target", 16, 64)))
    return out


def src_aihub():
    root = RAW / "AIHub182"
    lab = {p.stem: p for p in root.rglob("라벨링데이터/**/*.json")}
    places = {"산악2", "산악3", "산악4", "산악6", "산악8", "저수지2", "평지(수풀)1", "평지(흙)3", "저수지1"}
    out = []
    for p in sorted(root.rglob("원천데이터/**/*.jpg")):
        place = next((q for q in p.parts if q in places), None)
        if place is None or p.stem not in lab:
            continue
        m = re.search(r"-(\d+)m?-(\d+h\d+m)_(\d+)$", p.stem)
        seq = f"{place}_{m.group(1)}_{m.group(2)}" if m else place
        out.append(dict(src="aihub", dom="nad_kr", img=str(p), lab=str(lab[p.stem]), kind="aihub", seq=seq,
                        order=int(m.group(3)) if m else 0, split="val" if place == "저수지1" else "train", scale=("p1080", 0.667)))
    return out


# 출처 → (어댑터, 학습 양성 창 예산, 시퀀스당 상한, val 창 상한)
SOURCES = {
    "visdrone_obl": (None, 3000, 40, 300), "visdrone_nadir": (None, 1200, 40, 0), "unicamp": (src_unicamp, 450, 150, 60),
    "niicu": (src_niicu, 500, 250, 0), "nomad": (src_nomad, 3000, 15, 300), "sard": (src_sard, 1500, 20, 0),
    "archangel": (src_archangel, 1500, 20, 0), "wisard": (src_wisard, 2800, 110, 300), "heridal": (src_heridal, 1600, 10, 0),
    "ladd": (src_ladd, 900, 20, 0), "aihub": (src_aihub, 2500, 10, 300),
}


def read_boxes(rec):
    """(사람 박스 N×4, 가릴 박스 M×4, W, H) — 원본 px"""
    k = rec["kind"]
    if k == "visdrone":
        W, H = imsize(rec["img"]); P, I = [], []
        if Path(rec["lab"]).exists():
            for ln in open(rec["lab"]):
                t = ln.strip().split(",")
                if len(t) < 6:
                    continue
                x, y, w, h, cat = int(t[0]), int(t[1]), int(t[2]), int(t[3]), int(t[5])
                if cat in (1, 2):
                    P.append((x, y, x + w, y + h))
                elif cat == 0:
                    I.append((x, y, x + w, y + h))
        return np.array(P, np.float32).reshape(-1, 4), np.array(I, np.float32).reshape(-1, 4), W, H
    if k == "aihub":
        from aihub_prep import boxes_from_json
        W, H = imsize(rec["img"]); bx, _ = boxes_from_json(Path(rec["lab"]))
        P = [(b["bbox"][0], b["bbox"][1], b["bbox"][0] + b["bbox"][2], b["bbox"][1] + b["bbox"][3]) for b in bx]
        return np.array(P, np.float32).reshape(-1, 4), np.zeros((0, 4), np.float32), W, H
    W, H = imsize(rec["img"])
    return yolo(rec["lab"], W, H), np.zeros((0, 4), np.float32), W, H


def pick_scale(rec, b, W, H, r):
    fill = max(CW / W, CH / H)                                    # 창을 채우는 최소 배율 (> 1 이면 원본이 창보다 작다 → 1 · 여백)
    rule = rec["scale"]
    if rule[0] == "native":
        s = r.uniform(rule[1], 1.0)
    elif rule[0] == "p1080":                                      # 1080p 로 맞춘 뒤 u 배 (u ∈ [하한, 1])
        s = (1080.0 / H) * r.uniform(rule[1], 1.0)
    else:
        L = float(np.median(np.maximum(b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]))) if len(b) else 60.0
        t = math.exp(r.uniform(math.log(rule[1]), math.log(rule[2]))); s = t / max(L, 1.0)
    return float(min(1.0, max(s, min(fill, 1.0))))


def window(job):
    rec, k, want_neg = job
    r = random.Random(zlib.crc32(f"{rec['img']}|{k}".encode()))
    try:
        b, ign, W, H = read_boxes(rec)
    except Exception:
        return None
    if W == 0 or (not len(b) and not want_neg):
        return None
    s = pick_scale(rec, b, W, H, r)
    img = cv2.imread(rec["img"])
    if img is None:
        return None
    nw, nh = max(1, round(W * s)), max(1, round(H * s))
    if s < 1:
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    b = b * s; ign = ign * s
    # 타일처럼 창 위치를 고르게 — 양성은 사람이 하나 이상 (보이는 넓이 ≥ 50 %) · 음성은 사람과 안 겹치게
    best = None
    for _ in range(30):
        x0 = r.randint(0, max(nw - CW, 0)); y0 = r.randint(0, max(nh - CH, 0))
        if len(b):
            ix = np.clip(np.minimum(b[:, 2], x0 + CW) - np.maximum(b[:, 0], x0), 0, None)
            iy = np.clip(np.minimum(b[:, 3], y0 + CH) - np.maximum(b[:, 1], y0), 0, None)
            vis = ix * iy / np.maximum((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]), 1e-6)
        else:
            vis = np.zeros(0)
        if want_neg and (not len(vis) or vis.max() == 0):
            best = (x0, y0, vis); break
        if not want_neg and len(vis) and (vis >= 0.5).any():
            best = (x0, y0, vis); break
    if best is None:
        return None
    x0, y0, vis = best
    crop = np.full((CH, CW, 3), 114, np.uint8)
    part = img[y0:y0 + CH, x0:x0 + CW]; crop[:part.shape[0], :part.shape[1]] = part
    fillc = tuple(int(v) for v in part.reshape(-1, 3).mean(0))
    lines, px = [], []
    for (x1, y1, x2, y2), v in zip(b, vis):
        if v <= 0:
            continue
        cx1, cy1, cx2, cy2 = max(x1 - x0, 0), max(y1 - y0, 0), min(x2 - x0, CW), min(y2 - y0, CH)
        if v >= VIS_MIN and cx2 - cx1 >= 2 and cy2 - cy1 >= 2:
            lines.append(f"0 {(cx1 + cx2) / 2 / CW:.6f} {(cy1 + cy2) / 2 / CH:.6f} {(cx2 - cx1) / CW:.6f} {(cy2 - cy1) / CH:.6f}")
            px.append(max(cx2 - cx1, cy2 - cy1))
        else:                                                      # 잘린 조각 · 너무 작은 사람 → 가림 (라벨 없는 사람을 배경으로 배우지 않게)
            cv2.rectangle(crop, (int(cx1), int(cy1)), (int(math.ceil(cx2)), int(math.ceil(cy2))), fillc, -1)
    nmask = 0
    for x1, y1, x2, y2 in ign:                                     # VisDrone 무시 영역
        a1, b1, a2, b2 = int(max(x1 - x0, 0)), int(max(y1 - y0, 0)), int(min(x2 - x0, CW)), int(min(y2 - y0, CH))
        if a2 > a1 and b2 > b1:
            cv2.rectangle(crop, (a1, b1), (a2, b2), fillc, -1); nmask += 1
    if not want_neg and not lines:
        return None
    sp = rec["split"]
    stem = f"{rec['src']}_{zlib.crc32(rec['img'].encode()):08x}_{k}{'_n' if want_neg else ''}"
    oi = OUT / "images" / sp / f"{stem}.jpg"
    cv2.imwrite(str(oi), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
    (OUT / "labels" / sp / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
    return dict(path=str(oi), src=rec["src"], dom=rec["dom"], split=sp, neg=want_neg, px=px, s=s, masked=nmask)


def select(recs, budget, cap, rng):
    """시퀀스마다 시간 순으로 고르게 ≤ cap 장 → 전체 예산에 맞춰 시퀀스 사이 고르게"""
    by = defaultdict(list)
    for x in recs:
        by[x["seq"]].append(x)
    pick = []
    for s, xs in by.items():
        xs.sort(key=lambda x: x["order"])
        n = min(cap, len(xs)); idx = np.linspace(0, len(xs) - 1, n).round().astype(int) if n else []
        pick.append([xs[i] for i in sorted(set(idx))])
    out = []
    while len(out) < budget and any(pick):                        # 시퀀스 돌아가며 한 장씩 → 큰 시퀀스가 독식하지 않게
        rng.shuffle(pick)
        for xs in pick:
            if xs and len(out) < budget:
                out.append(xs.pop(rng.randrange(len(xs))))
        pick = [xs for xs in pick if xs]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    rng = random.Random(12)
    vis = src_visdrone()
    pool_by_src = {"visdrone_obl": [x for x in vis if x["src"] == "visdrone_obl"], "visdrone_nadir": [x for x in vis if x["src"] == "visdrone_nadir"]}
    for name, (fn, *_r) in SOURCES.items():
        if fn is not None and (a.only is None or name in a.only.split(",")):
            pool_by_src[name] = fn()
    jobs, plan = [], {}
    for name, (_, budget, cap, vcap) in SOURCES.items():
        if a.only and name not in a.only.split(","):
            continue
        recs = pool_by_src.get(name, [])
        tr = [x for x in recs if x["split"] == "train"]; va = [x for x in recs if x["split"] == "val"]
        sel = select(tr, int(budget * 1.15), cap, rng)            # 양성 실패 (사람 없음 · 창 못 찾음) 여유 15 %
        vsel = select(va, vcap, max(1, math.ceil(vcap / max(len({x["seq"] for x in va}), 1))), rng) if vcap else []
        plan[name] = {"원본 학습 프레임": len(tr), "원본 val 프레임": len(va), "시퀀스": len({x["seq"] for x in tr}),
                      "고른 학습": len(sel), "고른 val": len(vsel)}
        for i, x in enumerate(sel):
            jobs.append((x, 0, False))
            if i % NEG_EVERY == 0:
                jobs.append((x, 1, True))
        if name.startswith("visdrone") and len(sel) < budget * 1.15:  # VisDrone DET 은 시퀀스당 그림이 적다 → 큰 그림 (창 2장 넘는 넓이) 에서 다른 위치 창 하나 더
            big = [x for x in sel if (lambda w, h: w * h >= 2 * CW * CH)(*imsize(x["img"]))]
            extra = big[: int(budget * 1.15) - len(sel)]
            jobs += [(x, 2, False) for x in extra]; plan[name]["큰 그림 둘째 창"] = len(extra)
        jobs += [(x, 0, False) for x in vsel]
    print(json.dumps(plan, ensure_ascii=False, indent=1), flush=True)
    if a.dry:
        return
    for sp in ("train", "val"):
        (OUT / "images" / sp).mkdir(parents=True, exist_ok=True); (OUT / "labels" / sp).mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    with Pool(12) as pool:
        res = [x for x in pool.imap_unordered(window, jobs, chunksize=4) if x]
    # 출처별 예산으로 자르기 (양성) — 여유분은 버림
    keep, cnt = [], Counter()
    for x in sorted(res, key=lambda x: x["path"]):
        if x["split"] == "train" and not x["neg"]:
            if cnt[x["src"]] >= SOURCES[x["src"]][1]:
                continue
            cnt[x["src"]] += 1
        keep.append(x)
    tr = [x["path"] for x in keep if x["split"] == "train"]; va = [x["path"] for x in keep if x["split"] == "val"]
    if not a.only:
        (LISTS / "train_v12.txt").write_text("\n".join(sorted(tr)) + "\n")
        (LISTS / "val_v12.txt").write_text("\n".join(sorted(va)) + "\n")
        (BASE / "configs" / "data_v12.yaml").write_text(
            "# v12 (2026-10-12) — 원본부터 다시: 확대 금지 · 시퀀스 상한 · 무시 영역 · 잘린 조각 가림 · 타일처럼 창 · 영역 예산 · val = val_v12\n"
            f"train: {LISTS / 'train_v12.txt'}\nval: {LISTS / 'val_v12.txt'}\nnc: 1\nnames: ['person']\n")
    bins = [0, 16, 24, 36, 50, 80, 1e9]
    def dist(xs):
        p = np.array([q for x in xs for q in x["px"]]); h = np.histogram(p, bins=bins)[0] if len(p) else np.zeros(len(bins) - 1)
        return {"사람": int(len(p)), "px 분포": {f"{bins[i]}-{bins[i + 1] if bins[i + 1] < 1e9 else '∞'}": round(v / max(len(p), 1), 3) for i, v in enumerate(h)}}
    T = [x for x in keep if x["split"] == "train"]
    st = {"계획": plan, "학습 창": len(T), "음성 창": sum(x["neg"] for x in T), "val 창": len(va),
          "출처별 학습 창 (양성 · 음성)": {s: [sum(1 for x in T if x["src"] == s and not x["neg"]), sum(1 for x in T if x["src"] == s and x["neg"])] for s in SOURCES},
          "영역별 양성": dict(Counter(x["dom"] for x in T if not x["neg"])),
          "배율 s 중앙 (출처별)": {s: round(float(np.median([x["s"] for x in T if x["src"] == s])), 3) for s in SOURCES if any(x["src"] == s for x in T)},
          "가림 창 수 (무시 영역)": sum(1 for x in T if x["masked"]), "전체": dist(T),
          "영역별": {d: dist([x for x in T if x["dom"] == d]) for d in sorted({x["dom"] for x in T})}}
    (BASE / "metrics" / ("v12_data_stats.json" if not a.only else "v12_data_stats_partial.json")).write_text(json.dumps(st, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in st.items() if k != "계획"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
