"""
make_fov_testsets.py — 화각 · 고도별 사람 크기 시험셋 (2026-09-15 화각 비교 실험 · v4)

무엇을 흉내 내나
  우리 카메라(IMX415 · 1080p 원본을 입력 1920 으로 1:1 추론)에 렌즈 화각 · 고도를 바꿨을 때
  하향 90° 화면 중앙의 사람이 몇 px 로 찍히는지만 흉내 낸다.
    사람 px = (1920/2) / tan(수평화각/2) × 사람 길이 / 고도
    서 있는 사람 = 위에서 본 어깨 폭 0.5 m · 누운 사람 = 키 1.7 m
  렌즈 왜곡 · 가장자리 옆모습 · 하향 시점 모습은 흉내 내지 않는다 → 광각 결과는 실제보다 좋게 나온다.
  고도 10 · 20 · 25 · 30 · 40 m (사용자 요청 09-15). 학습 크기 16~160 px(+scale 0.3 → 약 11~208 px) 밖 칸
  (80~90° · 40 m 서 있음 12~14 px, 54~65° · 10 m 누움 256~320 px)은 외삽이다.

공정한 비교 (로컬 세션 검토 반영)
  · 원본 세트 = 자세 × 고도 묶음 {10 m, 20~40 m}. 같은 세트 안의 모든 칸은 **같은 원본 · 같은 기준 인물 ·
    같은 크롭 위치 비율**에서 나온다. 원본 조건은 세트마다 기준 인물 ≥ (그 세트 최대 px)/2.2.
    한 조건으로 묶으면 10 m 칸(최대 320 px)이 조건을 끌어올려 누운 사람 원본이 NOMAD a10 + 쉬운 비행으로 쏠렸다 (v3).
    → 10 m 행은 원본 세트가 달라 20 m 이상 행과 직접 비교하지 않는다.
  · 줄인 원본이 1280×720 창보다 작으면 회색 대신 **거울 반사 바둑판**으로 채운다 (홀수 칸 좌우 · 상하 뒤집기).
    회색 여백은 작은 칸에서 화면의 85 % 까지 차지해 배경 방해물이 사라져 점수가 좋게 나왔다 (v1).
  · 복제 칸의 사람도 라벨에 넣는다 (넣지 않으면 그 탐지가 오탐으로 잡힌다).
    채점용 표시는 meta/<이름>.json 에 라벨 줄 순서대로:
      orig = 원본 칸 사람 · seam = 원본 칸 사람 중 거울 이음새에서 박스 긴 변 안 (대칭 쌍둥이로 붙어 매칭이 흔들린다)
      near_ref = 긴 변이 기준 인물의 ±30 % 안 (그 칸의 목표 px 로 찍힌 사람) · vis = NOMAD 가시도
    eval_fov.py 는 orig & !seam & near_ref 박스로 이미지 단위 재현율을 내고, 세트 안 모든 칸에서 채점된 원본(교집합)으로 평균한다.
  · 원본은 학습에 안 쓴 것만: NOMAD val 배우 · WiSARD val 비행 (시간 분할 비행은 val_flights 에 없어 자동 제외).

실행:
  python make_fov_testsets.py --det data/det_fov --out data/det_fov_test
출력: <out>/<세트>_<px>/{images,labels,meta}/ + data.yaml (세트 = s10 · s20 · l10 · l20)
      <out>/manifest.json (표 · 세트별 원본 구성 · 칸별 원본 칸 비율 · 복제 · 이음새 통계)
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
RAW = BASE_DIR / "data" / "raw"
CROP_W, CROP_H = 1280, 720
MAX_UPSCALE = 2.2          # nomad_prep · wisard_prep 와 같은 확대 상한
NEAR_REF = 0.3             # 기준 인물 긴 변 ±30 % 안이면 그 칸의 목표 크기로 본다
FOVS = (54, 65, 80, 90)    # 수평 화각 — 5.5 · 4.4 · 3.3 · 2.8 mm (IMX415)
ALT_GROUPS = (("10", (10,)), ("20", (20, 25, 30, 40)))
POSES = (("standing", 0.5), ("lying", 1.7))
NOMAD_DISTS = {"standing": ("10", "30", "50", "70"), "lying": ("10", "30")}
INPUT_W = 1920
SEED = 42
MANIFEST_VERSION = 4


def person_px(fov, alt, length):
    return INPUT_W / 2 / np.tan(np.radians(fov / 2)) * length / alt


def fov_table():
    rows = []
    for f in FOVS:
        for g, alts in ALT_GROUPS:
            for a in alts:
                for p, L in POSES:
                    px = int(round(person_px(f, a, L)))
                    s = f"{p[0]}{g}"
                    rows.append({"fov": f, "alt": a, "pose": p, "px": px, "set": s, "dir": f"{s}_{px}"})
    return sorted(rows, key=lambda r: (r["pose"], r["fov"], r["alt"]))


def read_yolo(txt, W, H):
    out = []
    if txt.exists():
        for ln in txt.read_text(errors="replace").splitlines():
            q = ln.split()
            if len(q) == 5:
                _, cx, cy, bw, bh = map(float, q)
                out.append(((cx - bw / 2) * W, (cy - bh / 2) * H, bw * W, bh * H))
    return out


def mirror_index(start, length, n):
    """거울 반사 평면 좌표 start..start+length-1 → 원본 인덱스 (홀수 칸은 뒤집힘)"""
    xs = np.arange(start, start + length)
    k = np.floor_divide(xs, n)
    m = xs - k * n
    return np.where(k % 2 == 1, n - 1 - m, m)


def tile_span(a1, a2, k, n):
    """원본 좌표 구간 [a1, a2] 가 k 번째 칸(홀수는 뒤집힘)에서 차지하는 평면 좌표"""
    return (k * n + a1, k * n + a2) if k % 2 == 0 else (k * n + (n - a2), k * n + (n - a1))


def work(job):
    set_name, kind, group, path, boxes, vis, seed, px_list, out = job
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    H, W = img.shape[:2]
    if boxes is None:
        boxes = read_yolo(path.with_suffix(".txt"), W, H)
        vis = [-1] * len(boxes)
    if not boxes:
        return None
    ref = float(np.median([max(b[2], b[3]) for b in boxes]))
    if ref * MAX_UPSCALE < max(px_list):      # 이 세트의 가장 큰 목표를 못 만들면 세트 전 칸에서 뺀다
        return None
    rnd = random.Random(seed)
    bx, by, bw, bh = min(boxes, key=lambda b: abs(max(b[2], b[3]) - ref))   # 기준 인물
    jx, jy = rnd.uniform(-0.3, 0.3), rnd.uniform(-0.3, 0.3)                  # 모든 칸에 같은 위치 비율
    near = [abs(max(b[2], b[3]) / ref - 1) <= NEAR_REF for b in boxes]
    stem = f"{kind}_{path.stem}"
    stats = {}
    for P in px_list:
        s = P / ref
        nw, nh = max(1, round(W * s)), max(1, round(H * s))
        rs = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
        cx, cy = (bx + bw / 2) * s, (by + bh / 2) * s
        x0 = int(round(cx - CROP_W / 2 + jx * CROP_W))
        y0 = int(round(cy - CROP_H / 2 + jy * CROP_H))
        # 원본이 창보다 크면 [0, nw-CROP_W] 안, 작으면 [nw-CROP_W, 0] 안 → 원본 칸은 항상 창 안에 온전히 들어온다
        x0 = min(max(x0, min(0, nw - CROP_W)), max(0, nw - CROP_W))
        y0 = min(max(y0, min(0, nh - CROP_H)), max(0, nh - CROP_H))
        canvas = rs[mirror_index(y0, CROP_H, nh)[:, None], mirror_index(x0, CROP_W, nw)[None, :]]

        kx_rng = range(x0 // nw, (x0 + CROP_W - 1) // nw + 1)
        ky_rng = range(y0 // nh, (y0 + CROP_H - 1) // nh + 1)
        left, right, top, bottom = x0 < 0, x0 + CROP_W > nw, y0 < 0, y0 + CROP_H > nh
        lines, meta = [], []
        for b, v, nr in zip(boxes, vis, near):
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
                    meta.append({"orig": orig, "seam": bool(orig and seam), "near_ref": bool(nr), "vis": int(v)})
        if not any(m["orig"] for m in meta):
            continue
        d = Path(out) / f"{set_name}_{P}"
        ok, buf = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
        buf.tofile(str(d / "images" / f"{stem}.jpg"))
        (d / "labels" / f"{stem}.txt").write_text("\n".join(lines) + "\n")
        (d / "meta" / f"{stem}.json").write_text(json.dumps(meta))
        n_orig = sum(m["orig"] for m in meta)
        stats[P] = {"orig_frac": min(nw, CROP_W) * min(nh, CROP_H) / (CROP_W * CROP_H),
                    "copies": len(meta) - n_orig, "orig": n_orig, "seam": sum(m["seam"] for m in meta),
                    "scorable": sum(m["orig"] and not m["seam"] and m["near_ref"] for m in meta)}
    return (set_name, kind, group, stats) if stats else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", default=str(BASE_DIR / "data" / "det_fov"), help="val 배우 · 비행을 읽을 학습 데이터 폴더")
    ap.add_argument("--out", default=str(BASE_DIR / "data" / "det_fov_test"))
    ap.add_argument("--nomad-step", type=int, default=4, help="NOMAD val 프레임 간격")
    ap.add_argument("--wisard-step", type=int, default=4, help="WiSARD val 프레임 간격")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--fovs", default="", help="쉼표로 화각 지정 (기본 54,65,80,90) — 예산안 비교는 68,75,90,93")
    ap.add_argument("--standing-m", type=float, default=None,
                    help="서 있는 사람 박스 긴 변(m). 기본 0.5 (옛 가정) · AI-Hub 라벨 역산값은 0.81")
    args = ap.parse_args()

    global FOVS, POSES
    if args.fovs:
        FOVS = tuple(int(x) for x in args.fovs.split(","))
    if args.standing_m:
        POSES = (("standing", args.standing_m), ("lying", 1.7))
    print(f"화각 {FOVS} · 크기 가정 {dict(POSES)}")

    det, out = Path(args.det), Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"출력 폴더가 비어 있지 않다: {out} — 섞이지 않게 지우거나 다른 경로를 쓴다")
    val_actors = sorted({a for f in det.glob("nomad_*/nomad_prep_stats.json") for a in json.load(open(f))["val_actors"]})
    val_flights = json.load(open(det / "wisard" / "wisard_prep_stats.json"))["val_flights"]
    print(f"NOMAD val 배우 {len(val_actors)}명: {val_actors}\nWiSARD val 비행 {len(val_flights)}개: {val_flights}")

    table = fov_table()
    sets = {}
    for r in table:
        sets.setdefault(r["set"], {"set": r["set"], "pose": r["pose"], "alts": set(), "px": set()})
        sets[r["set"]]["alts"].add(r["alt"]); sets[r["set"]]["px"].add(r["px"])
    for s in sets.values():
        s["alts"], s["px"] = sorted(s["alts"]), sorted(s["px"])
        s["min_ref"] = round(max(s["px"]) / MAX_UPSCALE, 1)
        for P in s["px"]:
            d = out / f"{s['set']}_{P}"
            for sub in ("images", "labels", "meta"):
                (d / sub).mkdir(parents=True, exist_ok=True)
            (d / "data.yaml").write_text(f"path: {d.resolve()}\ntrain: images\nval: images\nnc: 1\nnames: ['person']\n")

    nomad = []
    for r in json.load(open(RAW / "NOMAD" / "annotations.json")):
        m = re.match(r"Actor(\d+)_a(\d+)_", r["file_name"])
        if not m or m.group(1) not in val_actors or not r["annotations"]:
            continue
        p = RAW / "NOMAD" / "images" / f"Actor{m.group(1)}" / f"Actor{m.group(1)}_a{m.group(2)}" / r["file_name"]
        if p.exists():
            nomad.append((m.group(2), p, [tuple(b["bbox"]) for b in r["annotations"]],
                          [int(b.get("visibility", 100)) for b in r["annotations"]]))
    wisard = []
    for f in val_flights:
        d = RAW / "WiSARD" / f
        wisard += [(f, p) for p in sorted([*d.glob("*.jpg"), *d.glob("*.jpeg"), *d.glob("*.png")])[::args.wisard_step]]

    jobs, seed = [], SEED
    for s in sets.values():
        for dist, p, boxes, vis in [x for x in nomad if x[0] in NOMAD_DISTS[s["pose"]]][::args.nomad_step]:
            jobs.append((s["set"], "nomad", f"a{dist}", p, boxes, vis, seed, s["px"], str(out))); seed += 1
        for f, p in wisard:
            jobs.append((s["set"], "wisard", f, p, None, None, seed, s["px"], str(out))); seed += 1
    print(f"작업 {len(jobs)}개 (세트 × 원본) · " + " · ".join(f"{k}{v['alts']} {v['px']} ref≥{v['min_ref']}" for k, v in sets.items()))

    used = {k: Counter() for k in sets}
    agg = defaultdict(lambda: {"images": 0, "orig_frac": [], "copies": [], "orig": 0, "seam": 0, "scorable": 0})
    with mp.Pool(args.workers, initializer=cv2.setNumThreads, initargs=(1,)) as pool:
        for i, res in enumerate(pool.imap_unordered(work, jobs, chunksize=4), 1):
            if res:
                set_name, kind, group, stats = res
                used[set_name][f"{kind}:{group}"] += 1
                for P, st in stats.items():
                    q = agg[f"{set_name}_{P}"]
                    q["images"] += 1; q["orig_frac"].append(st["orig_frac"]); q["copies"].append(st["copies"])
                    for k in ("orig", "seam", "scorable"):
                        q[k] += st[k]
            if i % 1000 == 0:
                print(f"  {i}/{len(jobs)}", flush=True)

    summary = {}
    for s in sets.values():
        for P in s["px"]:
            q = agg[f"{s['set']}_{P}"]
            fr = np.array(q["orig_frac"]) if q["orig_frac"] else None
            summary[f"{s['set']}_{P}"] = {
                "images": q["images"],
                "orig_frac_mean": round(float(fr.mean()), 3) if fr is not None else None,
                "orig_frac_lt50pct": round(float((fr < 0.5).mean()), 3) if fr is not None else None,
                "copies_mean": round(float(np.mean(q["copies"])), 2) if q["copies"] else None,
                "orig_boxes": q["orig"], "seam_ratio": round(q["seam"] / q["orig"], 3) if q["orig"] else None,
                "scorable_boxes": q["scorable"]}
    manifest = {"version": MANIFEST_VERSION, "table": table, "sets": list(sets.values()),
                "sources_used": {k: dict(sorted(c.items())) for k, c in used.items()}, "per_cell": summary,
                "val_actors": val_actors, "val_flights": val_flights, "input_w": INPUT_W,
                "alts": sorted({a for _, al in ALT_GROUPS for a in al}), "fovs": list(FOVS),
                "poses": dict(POSES),
                "note": "화면 중앙 · 하향 90° 사람 크기만 흉내 (왜곡 · 옆모습 없음) · 거울 바둑판 채움 · "
                        "채점은 meta orig & !seam & near_ref · 세트(자세 × 10 m / 20~40 m) 안 교집합 원본으로 평균"}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    for k, c in used.items():
        print(f"[{k}] 원본 {sum(c.values())}장: {dict(sorted(c.items()))}")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"→ {out}")


if __name__ == "__main__":
    main()
