"""
aihub_prep.py — AI-Hub 182 조난자 수색 원천 → YOLO 크롭 데이터셋 (2026-09-18)

nomad_prep.py · wisard_prep.py 와 같은 규칙으로 만든다 (1280×720 창 · 로그 균등 16~160 px · MAX_UPSCALE 2.2).
다른 점 세 가지:
  1) 라벨이 **4점 폴리곤 JSON** 이다 → min/max 로 박스화
  2) 메타가 풍부하다 — `altitude` · `angle` · `person_pose`(standing · sitting · lying) · 지형 → 크롭마다 meta/*.json 에 남긴다
  3) **장소(지형 폴더) 단위로 나눈다.** NFR-V03 의 "학습/평가 장소 분리" 는 지금 어느 데이터셋도 만족하지 못한다
     (NOMAD 는 배우 단위 · WiSARD 는 비행 단위이고 val 장소 6곳이 train 에도 전부 있다 — 09-18 확인)

기본 분할 (원천 하나 = 장소 하나)
  train : 저수지2(9,053) · 평지(흙)3(3,861) · 산악6(6,380)
  val   : 저수지1(1,285)        ← **학습에 없는 장소**
  제외  : 산악5(화성26 3,614)   ← [[예산안 화각 비교]] 시험 원본이라 학습에 넣지 않는다

> 이 데이터는 쉽다 (마른 흙 위 가림 없는 단독 피사체 · 같은 30 px 에서 0.86 vs NOMAD·WiSARD 0.33).
> **학습에 섞는 용도**이고, 합격 판정은 가림 지형으로 한다.

실행: python aihub_prep.py            (기본값으로 data/det_aihub 생성)
출력: data/det_aihub/{images,labels,meta}/{train,val} + aihub_prep_stats.json
"""
import argparse
import json
import multiprocessing as mp
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
CROP_W, CROP_H = 1280, 720
MAX_UPSCALE = 2.2
TARGET_MIN, TARGET_MAX = 16.0, 160.0
MIN_BOX_PX = 8            # 창 가장자리에서 잘린 박스 하한 (nomad_prep 과 같다)
KEEP_FRAC = 0.4           # 잘려도 넓이 40 % 이상 남으면 유지
SEED = 42

# 지형 폴더 이름 → 분할. 여기 없는 장소는 쓰지 않는다
PLACE_SPLIT = {"저수지2": "train", "평지(흙)3": "train", "산악6": "train", "저수지1": "val"}
EXCLUDE_PLACES = {"산악5"}        # 화각 비교 시험 원본


def boxes_from_json(p):
    d = json.loads(p.read_text(encoding="utf-8", errors="replace"))
    out = []
    for a in d.get("annotations", []):
        pts = a.get("points") or []
        if len(pts) < 3:
            continue
        xs = [q[0] for q in pts]; ys = [q[1] for q in pts]
        out.append({"bbox": (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)),
                    "pose": (a.get("attributes") or {}).get("person_pose", "unknown")})
    meta = d.get("Metadata", {})
    return out, {"alt": meta.get("altitude"), "angle": meta.get("angle"),
                 "weather": meta.get("weather"), "time": meta.get("time")}


def _init_worker():
    cv2.setNumThreads(1)


def process_one(job):
    img_path_s, lab_path_s, split, place, per_image, neg_per, out_s, seed = job
    img_path, out = Path(img_path_s), Path(out_s)
    rnd = random.Random(seed)
    boxes, meta = boxes_from_json(Path(lab_path_s))
    if not boxes:
        return {"skipped_nolabel": 1}
    img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return {"skipped_read": 1}
    H, W = img.shape[:2]
    st = Counter(); px_list = []; pose_cnt = Counter()

    for k in range(per_image + neg_per):
        negative = k >= per_image
        ref = max(boxes, key=lambda b: b["bbox"][2] * b["bbox"][3])
        bx, by, bw, bh = ref["bbox"]
        long_side = max(bw, bh)
        if long_side <= 1:
            continue
        s_fill = max(CROP_W / W, CROP_H / H)
        lo, hi = max(TARGET_MIN, long_side * s_fill), min(TARGET_MAX, long_side * MAX_UPSCALE)
        if lo > hi:
            st["skipped_upscale"] += 1
            continue
        target = float(np.exp(rnd.uniform(np.log(lo), np.log(hi))))
        scale = target / long_side
        nw, nh = int(round(W * scale)), int(round(H * scale))
        if nw < CROP_W or nh < CROP_H:
            scale = s_fill
            nw, nh = int(round(W * scale)), int(round(H * scale))
        rs = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)

        if negative:
            # 사람에서 창 하나 이상 떨어진 자리 — 배경만 담는다 (오탐 억제용)
            pcx, pcy = (bx + bw / 2) * scale, (by + bh / 2) * scale
            for _ in range(20):
                x0 = rnd.randint(0, max(0, nw - CROP_W)); y0 = rnd.randint(0, max(0, nh - CROP_H))
                if abs(x0 + CROP_W / 2 - pcx) > CROP_W or abs(y0 + CROP_H / 2 - pcy) > CROP_H:
                    break
            else:
                st["skipped_neg"] += 1
                continue
        else:
            pcx, pcy = (bx + bw / 2) * scale, (by + bh / 2) * scale
            x0 = int(round(pcx - CROP_W / 2 + rnd.uniform(-0.30, 0.30) * CROP_W))
            y0 = int(round(pcy - CROP_H / 2 + rnd.uniform(-0.30, 0.30) * CROP_H))
            x0 = max(0, min(x0, nw - CROP_W)); y0 = max(0, min(y0, nh - CROP_H))
        crop = rs[y0:y0 + CROP_H, x0:x0 + CROP_W]
        if crop.shape[:2] != (CROP_H, CROP_W):
            continue

        lines, cmeta = [], []
        for b in boxes:
            sx, sy, sw, sh = [v * scale for v in b["bbox"]]
            x1, y1, x2, y2 = sx - x0, sy - y0, sx - x0 + sw, sy - y0 + sh
            cx1, cy1, cx2, cy2 = max(x1, 0), max(y1, 0), min(x2, CROP_W), min(y2, CROP_H)
            if cx2 - cx1 < MIN_BOX_PX or cy2 - cy1 < MIN_BOX_PX:
                continue
            if (cx2 - cx1) * (cy2 - cy1) < KEEP_FRAC * sw * sh:
                continue
            w_, h_ = cx2 - cx1, cy2 - cy1
            lines.append(f"0 {(cx1 + w_/2)/CROP_W:.6f} {(cy1 + h_/2)/CROP_H:.6f} {w_/CROP_W:.6f} {h_/CROP_H:.6f}")
            cmeta.append({"pose": b["pose"], "px": round(max(w_, h_), 1)})
            px_list.append(max(w_, h_)); pose_cnt[b["pose"]] += 1
        if negative:
            if lines:                       # 사람이 걸리면 음성이 아니다
                st["skipped_neg"] += 1
                continue
        elif not lines:
            st["skipped_nobox"] += 1
            continue

        stem = f"{place}_{img_path.stem}_c{k}" + ("_n" if negative else "")
        stem = re.sub(r"[^\w가-힣().-]", "_", stem)
        ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
        buf.tofile(str(out / "images" / split / f"{stem}.jpg"))
        (out / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        (out / "meta" / split / f"{stem}.json").write_text(json.dumps(
            {"place": place, "alt": meta["alt"], "angle": meta["angle"], "weather": meta["weather"],
             "time": meta["time"], "scale": round(scale, 4), "boxes": cmeta}, ensure_ascii=False))
        st["neg" if negative else "crops"] += 1

    return {**st, "person_px": px_list, "poses": pose_cnt, "place": place, "split": split}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(BASE_DIR / "data" / "raw" / "AIHub182"))
    ap.add_argument("--out", default=str(BASE_DIR / "data" / "det_aihub"))
    ap.add_argument("--per-image", type=int, default=1, help="프레임당 양성 크롭 수")
    ap.add_argument("--neg-every", type=int, default=12, help="이 프레임마다 음성 크롭 1장")
    ap.add_argument("--step", type=int, default=1, help="프레임 간격 (1 = 전부)")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    raw, out = Path(args.raw), Path(args.out)
    if out.exists() and any(out.rglob("*.jpg")):
        raise SystemExit(f"출력 폴더에 이미 크롭이 있다: {out}")
    for split in ("train", "val"):
        for sub in ("images", "labels", "meta"):
            (out / sub / split).mkdir(parents=True, exist_ok=True)

    print("라벨 색인 중…", flush=True)
    lab_by_stem = {p.stem: p for p in raw.rglob("라벨링데이터/**/*.json")}
    print(f"라벨 {len(lab_by_stem):,}개")

    jobs, by_place = [], Counter()
    for p in sorted(raw.rglob("원천데이터/**/*.jpg")):
        place = next((q for q in p.parts if q in PLACE_SPLIT or q in EXCLUDE_PLACES), None)
        if place is None or place in EXCLUDE_PLACES:
            by_place[f"건너뜀:{place}"] += 1
            continue
        if p.stem not in lab_by_stem:
            by_place["라벨없음"] += 1
            continue
        by_place[place] += 1
        jobs.append((place, p))
    print("장소별 원본:", dict(by_place))

    tasks = []
    for i, (place, p) in enumerate(jobs[:: args.step]):
        neg = 1 if i % args.neg_every == 0 else 0
        tasks.append((str(p), str(lab_by_stem[p.stem]), PLACE_SPLIT[place], place,
                      args.per_image, neg, str(out), SEED + i))
    print(f"작업 {len(tasks):,}개 (음성 {sum(1 for t in tasks if t[5]):,})", flush=True)

    agg = Counter(); px = []; poses = Counter(); per_place = defaultdict(Counter)
    with mp.Pool(args.workers, initializer=_init_worker) as pool:
        for n, r in enumerate(pool.imap_unordered(process_one, tasks, chunksize=8), 1):
            px += r.pop("person_px", [])
            poses += r.pop("poses", Counter())
            place = r.pop("place", None); split = r.pop("split", None)
            if place:
                per_place[place].update({k: v for k, v in r.items() if isinstance(v, int)})
            agg.update({k: v for k, v in r.items() if isinstance(v, int)})
            if n % 2000 == 0:
                print(f"  {n:,}/{len(tasks):,} · 크롭 {agg['crops']:,} · 음성 {agg['neg']:,}", flush=True)

    q = np.percentile(px, [5, 25, 50, 75, 95]).round(1).tolist() if px else []
    stats = {"counts": dict(agg), "person_px_quantiles_5_25_50_75_95": q,
             "poses": dict(poses), "per_place": {k: dict(v) for k, v in per_place.items()},
             "place_split": PLACE_SPLIT, "excluded_places": sorted(EXCLUDE_PLACES),
             "target_px": [TARGET_MIN, TARGET_MAX], "crop": [CROP_W, CROP_H]}
    (out / "aihub_prep_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1))
    for split in ("train", "val"):
        n = len(list((out / "images" / split).glob("*.jpg")))
        print(f"{split}: {n:,}장")
    print(f"사람 px 분위수(5·25·50·75·95): {q}")
    print(f"자세: {dict(poses)}")
    print(f"→ {out}")


if __name__ == "__main__":
    main()
