"""
make_budget_testset.py — 예산안 1안 vs 2안 렌즈 화각 비교 시험셋 (2026-09-17)

무엇이 다른가 (기존 make_fov_testsets.py 와 비교)
  · 원본이 **AI-Hub 182 화성26** 이다 — 한국 산악5 · 하향 90° · 겨울 · 4K · 고도 10~40 m 실촬
    → 크기만 흉내 내던 합성 시험셋과 달리 **고도마다 진짜 그 고도에서 찍은 프레임**을 쓴다
  · 자세가 **실제 라벨**이다 (`person_pose` standing · sitting · lying) — 0.5 m / 1.7 m 크기 가정이 아니다
  · 장소가 학습과 완전히 분리된다 (NOMAD · WiSARD · SARD 어디에도 없는 산악5) → NFR-V03 의 장소 분리 조건

어떻게 화각을 흉내 내나
  AI-Hub 카메라 초점거리를 라벨 박스로 역산했다 (09-17 · 표본 6,000장):
    누운 사람 긴 변 × 고도 = 4,725 / 4,760 / 5,040 (고도 10 / 20 / 40 m) → 거의 일정
    누운 사람 1.7 m 를 기준자로 f = 2,848 px(4K) · HFOV 68.0°
  우리 후보 카메라(1920 입력)의 f 로 **같은 고도 프레임을 축소**하면 사람 px 가 정확히 그 카메라 값이 된다:
    배율 s = f_후보 / 2848      (전부 1 미만 → 축소만 · 없는 디테일을 만들지 않는다)
  같은 고도 · 같은 프레임 · 같은 크롭 위치를 모든 후보가 공유한다 → 오직 배율만 다른 짝지은 비교.
  흉내 내지 못하는 것: **렌즈 왜곡 · 색수차 · 가장자리 옆모습** (68° 원본에 그 신호가 없다) → 광각은 실제보다 좋게 나온다.

출력 칸: <out>/<카메라>_<고도>m/{images,labels,meta}/ + manifest.json
  meta/<이름>.json = 라벨 줄 순서대로 {"pose", "orig", "seam"}
    orig = 원본 칸 사람 · seam = 거울 이음새에 붙어 매칭이 흔들리는 원본 (93° · 160° 칸에서만 생긴다)

실행: python make_budget_testset.py
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

from make_fov_testsets import mirror_index, tile_span

BASE_DIR = Path(__file__).resolve().parent
CROP_W, CROP_H = 1280, 720
F_AI_4K = 2848.0            # AI-Hub 182 초점거리 (4K 기준 · 09-17 라벨 역산 · HFOV 68.0°)
STANDING_M = 0.81           # 하향 90° 서 있는 사람 박스 긴 변 (같은 역산에서 나온 값)
SEED = 42
MANIFEST_VERSION = 1

# 이름 → 1920 입력 기준 f(px). 예산안 후보 + 원본 해상도 기준점
CAMERAS = (
    ("68ref", 1424, "AI-Hub 원본을 1920 으로 (기준점)"),
    ("75", 1251, "1안 75°"),
    ("90", 960, "1안 90°"),
    ("93", 911, "2안 93°"),
    ("160eq", 688, "1안 160° 어안 (등거리 투영 가정)"),
)
ALTS = (10, 20, 25, 30, 40)


def boxes_from_json(p):
    """AI-Hub 라벨 → [(x, y, w, h, pose)] · 4점 폴리곤을 min/max 로 박스화"""
    d = json.loads(p.read_text(encoding="utf-8", errors="replace"))
    out = []
    for a in d.get("annotations", []):
        pts = a.get("points") or []
        if len(pts) < 3:
            continue
        xs = [q[0] for q in pts]
        ys = [q[1] for q in pts]
        pose = (a.get("attributes") or {}).get("person_pose", "unknown")
        out.append((min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys), pose))
    return out


def work(job):
    img_path, lab_path, alt, seed, out = job
    img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    boxes = boxes_from_json(lab_path)
    if not boxes:
        return None
    H, W = img.shape[:2]
    rnd = random.Random(seed)
    bx, by, bw, bh, _ = boxes[0]                       # 기준 인물 — 모든 카메라 칸이 같은 사람을 가운데 둔다
    jx, jy = rnd.uniform(-0.3, 0.3), rnd.uniform(-0.3, 0.3)
    stem = img_path.stem
    stats = {}

    for name, f_px, _ in CAMERAS:
        s = f_px / F_AI_4K
        nw, nh = max(1, round(W * s)), max(1, round(H * s))
        rs = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        cx, cy = (bx + bw / 2) * s, (by + bh / 2) * s
        x0 = int(round(cx - CROP_W / 2 + jx * CROP_W))
        y0 = int(round(cy - CROP_H / 2 + jy * CROP_H))
        x0 = min(max(x0, min(0, nw - CROP_W)), max(0, nw - CROP_W))
        y0 = min(max(y0, min(0, nh - CROP_H)), max(0, nh - CROP_H))
        canvas = rs[mirror_index(y0, CROP_H, nh)[:, None], mirror_index(x0, CROP_W, nw)[None, :]]

        kx_rng = range(x0 // nw, (x0 + CROP_W - 1) // nw + 1)
        ky_rng = range(y0 // nh, (y0 + CROP_H - 1) // nh + 1)
        left, right, top, bottom = x0 < 0, x0 + CROP_W > nw, y0 < 0, y0 + CROP_H > nh
        lines, meta = [], []
        for b in boxes:
            b1x, b1y, b2x, b2y = b[0] * s, b[1] * s, (b[0] + b[2]) * s, (b[1] + b[3]) * s
            L = max(b2x - b1x, b2y - b1y)
            seam = ((left and b1x < L) or (right and nw - b2x < L) or (top and b1y < L) or (bottom and nh - b2y < L))
            for ky in ky_rng:
                Y1, Y2 = tile_span(b1y, b2y, ky, nh)
                for kx in kx_rng:
                    X1, X2 = tile_span(b1x, b2x, kx, nw)
                    x1, y1, x2, y2 = X1 - x0, Y1 - y0, X2 - x0, Y2 - y0
                    cx1, cy1, cx2, cy2 = max(x1, 0), max(y1, 0), min(x2, CROP_W), min(y2, CROP_H)
                    if cx2 - cx1 < 2 or cy2 - cy1 < 2 or (cx2 - cx1) * (cy2 - cy1) < 0.4 * (x2 - x1) * (y2 - y1):
                        continue
                    ww, hh = cx2 - cx1, cy2 - cy1
                    lines.append(f"0 {(cx1 + ww / 2) / CROP_W:.6f} {(cy1 + hh / 2) / CROP_H:.6f} {ww / CROP_W:.6f} {hh / CROP_H:.6f}")
                    orig = kx == 0 and ky == 0
                    meta.append({"pose": b[4], "orig": orig, "seam": bool(orig and seam),
                                 "px": round(max(b2x - b1x, b2y - b1y), 1)})
        if not any(m["orig"] for m in meta):
            continue
        d = Path(out) / f"{name}_{alt}m"
        ok, buf = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
        buf.tofile(str(d / "images" / f"{stem}.jpg"))
        (d / "labels" / f"{stem}.txt").write_text("\n".join(lines) + "\n")
        (d / "meta" / f"{stem}.json").write_text(json.dumps(meta, ensure_ascii=False))
        n_orig = sum(m["orig"] for m in meta)
        stats[name] = {"orig_frac": min(nw, CROP_W) * min(nh, CROP_H) / (CROP_W * CROP_H),
                       "orig": n_orig, "copies": len(meta) - n_orig,
                       "seam": sum(m["seam"] for m in meta),
                       "px": float(np.median([m["px"] for m in meta if m["orig"]])),
                       "px_by_pose": {m["pose"]: m["px"] for m in meta if m["orig"]},
                       "poses": Counter(m["pose"] for m in meta if m["orig"])}
    return (alt, stats) if stats else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(BASE_DIR / "data" / "raw" / "AIHub182" / "조난자_화성26"))
    ap.add_argument("--labels", default=str(BASE_DIR / "data" / "raw" / "AIHub182"))
    ap.add_argument("--out", default=str(BASE_DIR / "data" / "det_budget"))
    ap.add_argument("--step", type=int, default=1, help="고도별 프레임 간격 (1 = 전부)")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"출력 폴더가 비어 있지 않다: {out}")

    lab_by_stem = {p.stem: p for p in Path(args.labels).rglob("라벨링데이터/**/조난자_화성26/**/*.json")}
    imgs = sorted(Path(args.src).rglob("*.jpg"))
    print(f"원천 이미지 {len(imgs):,}장 · 라벨 {len(lab_by_stem):,}개")

    by_alt = defaultdict(list)
    for p in imgs:
        m = re.search(r"-(\d{2,3})-(\d{1,3})m-", p.name)
        if not m or m.group(1) != "90":
            continue
        alt = int(m.group(2))
        if alt in ALTS and p.stem in lab_by_stem:
            by_alt[alt].append(p)

    jobs = []
    for alt in ALTS:
        ps = sorted(by_alt[alt])[:: args.step]
        print(f"  고도 {alt} m: {len(ps)}장")
        jobs += [(p, lab_by_stem[p.stem], alt, SEED + i, str(out)) for i, p in enumerate(ps)]

    for name, _, _ in CAMERAS:
        for alt in ALTS:
            for sub in ("images", "labels", "meta"):
                (out / f"{name}_{alt}m" / sub).mkdir(parents=True, exist_ok=True)

    agg = defaultdict(lambda: {"n": 0, "orig": 0, "copies": 0, "seam": 0, "px": [], "poses": Counter(),
                               "orig_frac": [], "px_by_pose": defaultdict(list)})
    with mp.Pool(args.workers) as pool:
        for r in pool.imap_unordered(work, jobs, chunksize=8):
            if not r:
                continue
            alt, stats = r
            for name, st in stats.items():
                a = agg[(name, alt)]
                a["n"] += 1
                a["orig"] += st["orig"]; a["copies"] += st["copies"]; a["seam"] += st["seam"]
                a["px"].append(st["px"]); a["poses"] += st["poses"]; a["orig_frac"].append(st["orig_frac"])
                for k, v in st["px_by_pose"].items():
                    a["px_by_pose"][k].append(v)

    table = []
    for name, f_px, label in CAMERAS:
        for alt in ALTS:
            a = agg[(name, alt)]
            if not a["n"]:
                continue
            table.append({"cam": name, "f_px": f_px, "label": label, "alt": alt,
                          "dir": f"{name}_{alt}m", "images": a["n"],
                          "scale": round(f_px / F_AI_4K, 4),
                          "px_median": round(float(np.median(a["px"])), 1),
                          "px_standing_calc": round(f_px * STANDING_M / alt, 1),
                          "orig": a["orig"], "copies": a["copies"], "seam": a["seam"],
                          "orig_frac": round(float(np.mean(a["orig_frac"])), 3),
                          "px_by_pose": {k: round(float(np.median(v)), 1) for k, v in a["px_by_pose"].items()},
                          "poses": dict(a["poses"])})
    (out / "manifest.json").write_text(json.dumps(
        {"version": MANIFEST_VERSION, "source": "AI-Hub 182 · 조난자_화성26 (Validation · 산악5)",
         "f_ai_4k": F_AI_4K, "standing_m": STANDING_M, "cameras": [list(c) for c in CAMERAS],
         "alts": list(ALTS), "crop": [CROP_W, CROP_H], "table": table}, ensure_ascii=False, indent=1))

    print(f"\n{'칸':>12} {'장수':>5} {'배율':>6} {'사람px(중앙)':>12} {'원본칸비율':>9} {'이음새':>5}  자세")
    for t in table:
        po = " · ".join(f"{k} {v}장 {t['px_by_pose'].get(k, 0):.0f}px" for k, v in sorted(t["poses"].items(), key=lambda x: -x[1]))
        print(f"{t['dir']:>12} {t['images']:>5} {t['scale']:>6.3f} {t['px_median']:>12.1f} "
              f"{t['orig_frac']:>9.2f} {t['seam']:>5}  {po}")
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
