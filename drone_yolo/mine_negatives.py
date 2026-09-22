"""
mine_negatives.py — 하드 네거티브 마이닝 (2026-09-21)

왜
  학습 크롭 30,056장 중 음성(사람 없는 장)이 **808장 · 2.6 %** 뿐이고, 그마저 전부 WiSARD다.
  NOMAD 는 음성이 **0장** — 모델이 오탐을 배울 기회가 구조적으로 없었다.
  그런데 타일 추론은 프레임당 4회 추론이라 오탐 노출이 4배다 (현재 1.718 건/프레임).

왜 랜덤 배경이 아니라 마이닝인가
  NOMAD 원본은 5472×3078 에 배우 한 명이고 나머지는 대부분 밭·잔디다.
  거기서 무작위로 뽑으면 **모델이 이미 오탐을 안 내는 빈 땅**이 대부분이라 배울 게 없다.
  → 현재 가중치로 돌려 **실제로 오탐이 나는 자리**만 음성으로 만든다.

절차 (프레임마다)
  1. 정답 박스를 읽는다. 사람이 0명이면 건너뛴다 (배율 기준이 없다)
  2. det_fov 와 **같은 규칙**으로 목표 px 를 로그 균등에서 뽑아 리샘플 — 양성과 배율 분포를 맞춘다
  3. 리샘플 이미지를 1280×720 타일로 잘라 추론 (배율 1.0 유지 · 학습 크롭과 같은 형식)
  4. conf ≥ --conf 이면서 **어떤 정답과도 안 겹치는** 예측 = 오탐
  5. 가장 높은 conf 오탐을 중심으로 1280×720 창을 잡되, 창에 정답이 조금이라도 걸리면 버린다
  6. 프레임당 최대 1장 (같은 배경 중복 방지)

실행:
  python mine_negatives.py --weights runs_person/fov_11m_1280_all/weights/best.pt --target 4500
출력: data/det_neg/{images,labels,meta}/train
"""
import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

BASE = Path(__file__).resolve().parent
CROP_W, CROP_H = 1280, 720
MAX_UPSCALE = 2.2                      # nomad_prep.py 와 같은 값 — 넘으면 화질이 무너진다
TARGET_MIN, TARGET_MAX = 16.0, 160.0   # det_fov 생성에 쓴 로그 균등 범위
SEED = 42

# det_fov 의 val 배우 — 여기서 캔 음성은 **train 전용**이라 전부 빼야 한다
VAL_ACTORS_ALL = {"004", "008", "014", "018", "024", "028",
                  "048", "059", "071", "079", "088", "094"}
NOMAD_BATCHES = {                      # 원본 디렉터리 → 뺄 배우
    # 기본 NOMAD 디렉터리는 배우 60명이 다 들어 있다 → val 배우 **전체**를 뺀다
    # (09-21 스모크에서 Actor014 가 섞여 나와 발견)
    "NOMAD": VAL_ACTORS_ALL,
    "NOMAD_b1_10": VAL_ACTORS_ALL,
    "NOMAD_b11_20": VAL_ACTORS_ALL,
    "NOMAD_b21_30": VAL_ACTORS_ALL,
    "NOMAD_sel31_100": VAL_ACTORS_ALL,
}


def imread_u(p):
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)


def collect_nomad(limit_per_batch=None):
    """(경로, [xyxy 정답], 출처) 목록 — val 배우는 뺀다"""
    out = []
    seen = set()                       # 기본 NOMAD 디렉터리가 배치들을 포함한다 — 파일명으로 중복 제거
    for name, val_actors in NOMAD_BATCHES.items():
        root = BASE / "data" / "raw" / name
        ann = root / "annotations.json"
        if not ann.exists():
            continue
        idx = {p.name: p for p in (root / "images").rglob("*.jpg")}
        recs = json.load(open(ann, encoding="utf-8"))
        got = 0
        for r in recs:
            fn = r["file_name"]
            m = re.match(r"Actor(\d+)_", fn)
            if not m or m.group(1) in val_actors:
                continue
            p = idx.get(fn)
            if p is None or not r.get("annotations") or fn in seen:
                continue
            seen.add(fn)
            gt = np.array([[b["bbox"][0], b["bbox"][1], b["bbox"][0] + b["bbox"][2], b["bbox"][1] + b["bbox"][3]]
                           for b in r["annotations"]], dtype=np.float32)
            if not len(gt):
                continue
            out.append((p, gt, name))
            got += 1
            if limit_per_batch and got >= limit_per_batch:
                break
    return out


def collect_wisard(val_stems, limit=None):
    root = BASE / "data" / "raw" / "WiSARD"
    out = []
    for p in sorted(root.rglob("*.jpg")):
        if p.stem in val_stems:
            continue
        t = p.with_suffix(".txt")
        if not t.exists():
            continue
        rows = [l.split() for l in t.read_text().splitlines() if l.strip()]
        if not rows:
            continue                                   # 사람 없는 프레임은 배율 기준이 없다
        try:                                           # 헤더만 읽는다 — 2만장을 디코딩하면 수십 분이다
            with Image.open(p) as im:
                W, H = im.size
        except Exception:
            continue
        gt = []
        for _, cx, cy, w, h in rows:
            cx, cy, w, h = float(cx) * W, float(cy) * H, float(w) * W, float(h) * H
            gt.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
        out.append((p, np.array(gt, dtype=np.float32), "WiSARD"))
        if limit and len(out) >= limit:
            break
    return out


def val_stems_of(dataset):
    d = BASE / "data" / "det_fov" / dataset / "images" / "val"
    return {re.sub(r"_c\d+$", "", p.stem) for p in d.glob("*.jpg")} if d.exists() else set()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="runs_person/fov_11m_1280_all/weights/best.pt")
    ap.add_argument("--out", default=str(BASE / "data" / "det_neg"))
    ap.add_argument("--target", type=int, default=4500, help="만들 음성 장수")
    ap.add_argument("--conf", type=float, default=0.10, help="이 신뢰도 이상인 오탐만 쓴다")
    ap.add_argument("--max-frames", type=int, default=14000, help="훑을 원본 프레임 수 (균등 표집)")
    ap.add_argument("--batch", type=int, default=8, help="타일 배치 크기")
    args = ap.parse_args()

    from ultralytics import YOLO

    rnd = random.Random(SEED)
    print("원본 목록 작성 중… (첫 실행만 몇 분 · 이후 캐시)", flush=True)
    cache = BASE / "data" / "det_neg" / "_frames.json"
    if cache.exists():
        frames = [(Path(a), np.array(b, dtype=np.float32), c) for a, b, c in json.loads(cache.read_text())]
    else:
        frames = collect_nomad() + collect_wisard(val_stems_of("wisard"))
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps([[str(a), b.tolist(), c] for a, b, c in frames]))
    rnd.shuffle(frames)
    if len(frames) > args.max_frames:
        frames = frames[:args.max_frames]
    src_cnt = Counter(s for _, _, s in frames)
    print(f"대상 프레임 {len(frames):,}장 — {dict(src_cnt)}", flush=True)

    out = Path(args.out)
    for sub in ("images/train", "labels/train", "meta/train"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)
    st = Counter(); made = 0; scales = []
    for i, (path, gt, src) in enumerate(frames):
        if made >= args.target:
            break
        if i % 200 == 0:
            print(f"  {i:,}/{len(frames):,} 프레임 · 음성 {made:,}장", flush=True)
        img = imread_u(path)
        if img is None:
            st["읽기실패"] += 1
            continue
        H, W = img.shape[:2]
        r = random.Random(SEED + i)

        long_side = float(np.maximum(gt[:, 2] - gt[:, 0], gt[:, 3] - gt[:, 1]).max())
        if long_side <= 1:
            st["박스이상"] += 1
            continue
        s_fill = max(CROP_W / W, CROP_H / H)
        lo, hi = max(TARGET_MIN, long_side * s_fill), min(TARGET_MAX, long_side * MAX_UPSCALE)
        if lo > hi:
            st["배율불가"] += 1
            continue
        scale = float(np.exp(r.uniform(np.log(lo), np.log(hi)))) / long_side
        nw, nh = int(round(W * scale)), int(round(H * scale))
        if nw < CROP_W or nh < CROP_H:
            scale = s_fill
            nw, nh = int(round(W * scale)), int(round(H * scale))
        rs = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
        g = gt * scale

        # 겹침 없는 타일로 훑는다 — 배율 1.0 유지 (학습 크롭과 같은 형식)
        xs = list(range(0, max(1, nw - CROP_W + 1), CROP_W)) or [0]
        ys = list(range(0, max(1, nh - CROP_H + 1), CROP_H)) or [0]
        if xs[-1] + CROP_W < nw:
            xs.append(nw - CROP_W)
        if ys[-1] + CROP_H < nh:
            ys.append(nh - CROP_H)
        origins = [(x, y) for y in ys for x in xs]
        tiles = [rs[y:y + CROP_H, x:x + CROP_W] for x, y in origins]
        tiles = [t for t in tiles if t.shape[:2] == (CROP_H, CROP_W)]
        if not tiles:
            st["타일없음"] += 1
            continue

        best = None
        for s in range(0, len(tiles), args.batch):
            chunk = tiles[s:s + args.batch]
            res = model.predict(chunk, imgsz=1280, conf=args.conf, batch=len(chunk),
                                quantize="fp16", verbose=False)
            for (ox, oy), rr in zip(origins[s:s + args.batch], res):
                b = rr.boxes
                if not len(b):
                    continue
                xyxy = b.xyxy.cpu().numpy() + np.array([ox, oy, ox, oy])
                cf = b.conf.cpu().numpy()
                for k in range(len(xyxy)):
                    x1, y1, x2, y2 = xyxy[k]
                    ix1 = np.maximum(g[:, 0], x1); iy1 = np.maximum(g[:, 1], y1)
                    ix2 = np.minimum(g[:, 2], x2); iy2 = np.minimum(g[:, 3], y2)
                    if np.any((ix2 > ix1) & (iy2 > iy1)):
                        continue                       # 정답과 조금이라도 겹치면 오탐이 아니다
                    if best is None or cf[k] > best[0]:
                        best = (float(cf[k]), (x1 + x2) / 2, (y1 + y2) / 2)
        if best is None:
            st["오탐없음"] += 1
            continue

        conf, cx, cy = best
        x0 = int(round(cx - CROP_W / 2 + r.uniform(-0.25, 0.25) * CROP_W))
        y0 = int(round(cy - CROP_H / 2 + r.uniform(-0.25, 0.25) * CROP_H))
        x0 = max(0, min(x0, nw - CROP_W)); y0 = max(0, min(y0, nh - CROP_H))
        # 창에 정답이 조금이라도 걸리면 음성이 아니다
        if np.any((g[:, 2] > x0) & (g[:, 0] < x0 + CROP_W) & (g[:, 3] > y0) & (g[:, 1] < y0 + CROP_H)):
            st["사람걸림"] += 1
            continue
        crop = rs[y0:y0 + CROP_H, x0:x0 + CROP_W]
        if crop.shape[:2] != (CROP_H, CROP_W):
            st["창밖"] += 1
            continue

        stem = re.sub(r"[^\w.-]", "_", f"{src}_{path.stem}_neg")
        ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
        buf.tofile(str(out / "images" / "train" / f"{stem}.jpg"))
        (out / "labels" / "train" / f"{stem}.txt").write_text("")
        (out / "meta" / "train" / f"{stem}.json").write_text(json.dumps(
            {"src": src, "frame": path.name, "scale": round(scale, 4),
             "fp_conf": round(conf, 4), "origin": [x0, y0]}, ensure_ascii=False))
        st[f"만듦:{src}"] += 1; made += 1; scales.append(scale)

    print(f"\n=== 음성 {made:,}장 생성 → {out}/images/train")
    for k, v in sorted(st.items()):
        print(f"  {k:>12} {v:,}")
    if scales:
        print(f"  배율 분위수 5/50/95: {np.round(np.percentile(scales, [5, 50, 95]), 3)}")
    (out / "mine_stats.json").write_text(json.dumps(
        {"made": made, "weights": args.weights, "conf": args.conf,
         "stats": dict(st)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
