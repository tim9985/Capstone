"""
make_fullframe_testset.py — 1920×1080 전체 프레임 시험셋 (2026-09-16)

왜
  학습은 1920 프레임에서 잘라낸 1280×720 창으로 했다. 창 밖 배경이 적어서
  **프레임당 오탐**이 실제보다 적게 보일 수 있다 → 전체 프레임으로 따로 재야 한다.
  덤으로 추론 입력(1280 / 1600 / 1920)에 따라 재현율 · 오탐이 어떻게 변하는지도 같은 프레임으로 본다.

무엇을 만드나 — 칸은 **사람 px** 로 정의된다
  s38  38 px  ·  l128 128 px  ·  neg 사람 없는 프레임
  ⚠ 원래 표기 '54° · 25 m · 하향 90°' 는 **옛 가정(서 있는 사람 0.5 m)** 으로 계산한 것이다 (09-17 실측 0.81 m 로 정정).
    0.81 m 로 다시 계산하면 같은 조건의 서 있는 사람은 61 px 이고, 38 px 은 54°·40 m 쯤에 해당한다.
    누운 사람 128 px 은 1.7 m 기준이라 그대로 유효하다. **마운트각은 모형화하지 않는다** (크기만 재현).
  · val 원본만 (NOMAD val 배우 · WiSARD val 비행) — make_fov_testsets.py v4 와 같은 원본 규칙
  · 원본을 목표 크기로 줄이고 1920×1080 창을 뜬다. 창보다 작아지면 거울 반사 바둑판으로 채운다
  · 음성 프레임은 그 비행 양성 프레임의 중앙값 배율을 그대로 써서 같은 GSD 로 맞춘다
  · meta/<이름>.json: 라벨 줄 순서대로 orig · seam · near_ref · vis, 그리고 선명도(라플라시안 분산)

실행: python make_fullframe_testset.py --det data/det_fov --out data/det_fullframe
"""
import argparse, json, multiprocessing as mp, random, re, statistics
from collections import defaultdict
from pathlib import Path
import cv2, numpy as np
from make_fov_testsets import mirror_index, tile_span, read_yolo, MAX_UPSCALE, NEAR_REF

BASE_DIR = Path(__file__).resolve().parent
RAW = BASE_DIR / "data" / "raw"
W_OUT, H_OUT = 1920, 1080
SETS = {"s38": ("standing", 38, ("10", "30", "50", "70")), "l128": ("lying", 128, ("10", "30"))}
SEED = 42


def sharpness(img):
    return float(cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


def canvas_from(rs, x0, y0):
    nh, nw = rs.shape[:2]
    return rs[mirror_index(y0, H_OUT, nh)[:, None], mirror_index(x0, W_OUT, nw)[None, :]]


def clamp(v, n, out):
    return min(max(v, min(0, n - out)), max(0, n - out))


def work(job):
    kind, group, path, boxes, vis, seed, target, out, scale_fixed = job
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    H, W = img.shape[:2]
    if boxes is None:
        boxes = read_yolo(path.with_suffix(".txt"), W, H)
        vis = [-1] * len(boxes)
    rnd = random.Random(seed)
    if scale_fixed is not None:          # 음성 프레임 — 그 비행 양성 배율을 그대로
        s, ref = scale_fixed, None
        bx = by = bw = bh = None
    else:
        if not boxes:
            return None
        ref = float(np.median([max(b[2], b[3]) for b in boxes]))
        if ref * MAX_UPSCALE < target:
            return None
        s = target / ref
        bx, by, bw, bh = min(boxes, key=lambda b: abs(max(b[2], b[3]) - ref))
    nw, nh = max(1, round(W * s)), max(1, round(H * s))
    rs = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    if scale_fixed is not None:
        x0, y0 = clamp(rnd.randint(0, max(nw - W_OUT, 0)), nw, W_OUT), clamp(rnd.randint(0, max(nh - H_OUT, 0)), nh, H_OUT)
    else:
        x0 = clamp(int(round((bx + bw / 2) * s - W_OUT / 2 + rnd.uniform(-0.3, 0.3) * W_OUT)), nw, W_OUT)
        y0 = clamp(int(round((by + bh / 2) * s - H_OUT / 2 + rnd.uniform(-0.3, 0.3) * H_OUT)), nh, H_OUT)
    canvas = canvas_from(rs, x0, y0)

    lines, meta = [], []
    if scale_fixed is None:
        kx_rng = range(x0 // nw, (x0 + W_OUT - 1) // nw + 1)
        ky_rng = range(y0 // nh, (y0 + H_OUT - 1) // nh + 1)
        left, right, top, bottom = x0 < 0, x0 + W_OUT > nw, y0 < 0, y0 + H_OUT > nh
        for b, v in zip(boxes, vis):
            b1x, b1y, b2x, b2y = b[0] * s, b[1] * s, (b[0] + b[2]) * s, (b[1] + b[3]) * s
            L = max(b2x - b1x, b2y - b1y)
            seam = (left and b1x < L) or (right and nw - b2x < L) or (top and b1y < L) or (bottom and nh - b2y < L)
            near = abs(max(b[2], b[3]) / ref - 1) <= NEAR_REF
            for ky in ky_rng:
                Y1, Y2 = tile_span(b1y, b2y, ky, nh)
                for kx in kx_rng:
                    X1, X2 = tile_span(b1x, b2x, kx, nw)
                    x1, y1, x2, y2 = X1 - x0, Y1 - y0, X2 - x0, Y2 - y0
                    cx1, cy1, cx2, cy2 = max(x1, 0), max(y1, 0), min(x2, W_OUT), min(y2, H_OUT)
                    if cx2 - cx1 < 2 or cy2 - cy1 < 2 or (cx2 - cx1) * (cy2 - cy1) < 0.4 * (x2 - x1) * (y2 - y1):
                        continue
                    ww, hh = cx2 - cx1, cy2 - cy1
                    lines.append(f"0 {(cx1+ww/2)/W_OUT:.6f} {(cy1+hh/2)/H_OUT:.6f} {ww/W_OUT:.6f} {hh/H_OUT:.6f}")
                    meta.append({"orig": kx == 0 and ky == 0, "seam": bool((kx == 0 and ky == 0) and seam),
                                 "near_ref": bool(near), "vis": int(v)})
        if not any(m["orig"] for m in meta):
            return None
    d = Path(out)
    stem = f"{kind}_{path.stem}"
    ok, buf = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
    buf.tofile(str(d / "images" / f"{stem}.jpg"))
    (d / "labels" / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
    (d / "meta" / f"{stem}.json").write_text(json.dumps({"boxes": meta, "sharpness": sharpness(canvas),
                                                         "scale": s, "kind": kind, "group": group}))
    return (kind, group, s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", default=str(BASE_DIR / "data" / "det_fov"))
    ap.add_argument("--out", default=str(BASE_DIR / "data" / "det_fullframe"))
    ap.add_argument("--nomad-step", type=int, default=4)
    ap.add_argument("--wisard-step", type=int, default=4)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    det, out = Path(args.det), Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"출력 폴더가 비어 있지 않다: {out}")
    val_actors = sorted({a for f in det.glob("nomad_*/nomad_prep_stats.json") for a in json.load(open(f))["val_actors"]})
    val_flights = json.load(open(det / "wisard" / "wisard_prep_stats.json"))["val_flights"]
    for name in list(SETS) + ["neg"]:
        for sub in ("images", "labels", "meta"):
            (out / name / sub).mkdir(parents=True, exist_ok=True)
        (out / name / "data.yaml").write_text(f"path: {(out/name).resolve()}\ntrain: images\nval: images\nnc: 1\nnames: ['person']\n")

    ann = json.load(open(RAW / "NOMAD" / "annotations.json"))
    nomad = []
    for r in ann:
        m = re.match(r"Actor(\d+)_a(\d+)_", r["file_name"])
        if not m or m.group(1) not in val_actors or not r["annotations"]:
            continue
        p = RAW / "NOMAD" / "images" / f"Actor{m.group(1)}" / f"Actor{m.group(1)}_a{m.group(2)}" / r["file_name"]
        if p.exists():
            nomad.append((m.group(2), p, [tuple(b["bbox"]) for b in r["annotations"]],
                          [int(b.get("visibility", 100)) for b in r["annotations"]]))
    wis_pos, wis_neg = [], []
    for f in val_flights:
        for p in sorted([*(RAW / "WiSARD" / f).glob("*.jpg"), *(RAW / "WiSARD" / f).glob("*.jpeg"), *(RAW / "WiSARD" / f).glob("*.png")])[::args.wisard_step]:
            (wis_pos if p.with_suffix(".txt").exists() and p.with_suffix(".txt").read_text().strip() else wis_neg).append((f, p))

    jobs = []
    seed = SEED
    for name, (pose, target, dists) in SETS.items():
        d = out / name
        for dist, p, boxes, vis in [x for x in nomad if x[0] in dists][::args.nomad_step]:
            jobs.append(("nomad", f"a{dist}", p, boxes, vis, seed, target, str(d), None)); seed += 1
        for f, p in wis_pos:
            jobs.append(("wisard", f, p, None, None, seed, target, str(d), None)); seed += 1
    print(f"양성 작업 {len(jobs)}개 · 음성 후보 {len(wis_neg)}장")

    scales = defaultdict(list)
    used = defaultdict(int)
    with mp.Pool(args.workers, initializer=cv2.setNumThreads, initargs=(1,)) as pool:
        for res in pool.imap_unordered(work, jobs, chunksize=4):
            if res:
                kind, group, s = res
                used[f"{kind}:{group}"] += 1
                scales[group].append(s)
    # 음성: 그 비행 양성 배율 중앙값 (s38 기준 GSD)
    njobs = []
    for f, p in wis_neg[::1]:
        if f in scales:
            njobs.append(("wisard", f, p, [], [], seed, 0, str(out / "neg"), statistics.median(scales[f]))); seed += 1
    with mp.Pool(args.workers, initializer=cv2.setNumThreads, initargs=(1,)) as pool:
        nneg = sum(1 for r in pool.imap_unordered(work, njobs, chunksize=4) if r)
    counts = {n: len(list((out / n / "images").glob("*.jpg"))) for n in list(SETS) + ["neg"]}
    (out / "manifest.json").write_text(json.dumps({"sets": {k: {"pose": v[0], "px": v[1]} for k, v in SETS.items()},
        "frame": [W_OUT, H_OUT], "operating_point": "칸은 사람 px 로 정의 (s38=38px · l128=128px) · 옛 0.5 m 가정 표기였음 · 마운트각 미모형화", "counts": counts,
        "sources_used": dict(used), "negatives": nneg, "val_actors": val_actors, "val_flights": val_flights}, indent=2, ensure_ascii=False))
    print(f"장수 {counts} · 음성 {nneg}\n→ {out}")


if __name__ == "__main__":
    main()
