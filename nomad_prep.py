"""
nomad_prep.py — NOMAD 원본 → 운용 조건에 맞춘 YOLO 학습셋 변환

왜 그냥 못 쓰는가
  NOMAD 원본은 5472x3078 이고 사람 크기가 거리별로 27~233px 로 제각각이다.
  운용 조건(수평 화각 54°, 1280px 입력, 고도 16~30m)에서 사람 1.7m 는 약 71~133px 이다
  (렌즈를 4.8mm · 수평 60.5° 로 바꾸면 62~117px). 목표 94px 에 지터를 줘 두 경우를 덮는다.
  원본을 그대로 넣으면 추론 해상도로 축소될 때 사람이 10px 이하로 뭉개져 학습이 안 된다.
  → 사람 크기를 목표 픽셀로 맞춰 리샘플링한 뒤 1280x720 창을 크롭한다.

거리별 필요 배율 (실측 중앙값 기준)
  a10 233px → 0.40배(축소, 최상)   a30 78px → 1.2배(최적)
  a50  47px → 2.0배(한계)          a70/a90 → 2.8~3.5배 업스케일이라 기본 제외

과적합 대응
  · train/val 을 **배우(Actor) 단위**로 분할한다. 같은 배우의 연속 프레임이 양쪽에
    섞이면 성능이 부풀려진다(프레임 간 상관이 매우 높음).
  · 목표 사람 크기에 지터를 줘 스케일 다양성을 만든다.
  · 크롭 위치도 지터를 줘 사람이 항상 중앙에 오지 않게 한다.

병렬화 (2026-09-11 추가)
  이미지 단위 크롭 생성은 서로 독립이라 프로세스 풀로 병렬 처리한다. 재현성을 지키려고
  전역 random 대신 이미지마다 결정적 시드(SEED + usable 리스트에서의 순번)를 쓴다 —
  usable 순서는 항상 annotations.json 순서로 고정되므로 워커가 몇 개든, 처리 순서가
  뒤섞여도 실행마다 같은 크롭이 나온다. --workers 로 조절(기본 CPU 코어 수).

실행:
  python nomad_prep.py                        # 기본: a10,a30 사용
  python nomad_prep.py --distances 10,30,50 --per-image 2
  python nomad_prep.py --min-visibility 30    # 가림 심한 표본 제외
  python nomad_prep.py --workers 12           # 병렬 워커 수 직접 지정
출력: data/det/nomad_actor01_10/{images,labels}/{train,val} + data.yaml + nomad_prep_stats.json
"""
import argparse
import json
import multiprocessing as mp
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
NOMAD_DIR = BASE_DIR / "data" / "raw" / "NOMAD"
OUT_ROOT = BASE_DIR / "data" / "det" / "nomad_actor01_10"

CROP_W, CROP_H = 1280, 720
# 목표 크기 근거 (2026-09-12 수정)
#   이전 주석 "고도 20m, FOV 60°" 의 60° 는 예산안 v2.0 의 **대각** 화각이었다. 수평은 54.0°.
#   1280px 입력 · 사람 1.7m 기준, 고도 16~30m 에서
#     5.5mm(수평 54.0°)  → 71~133px
#     4.8mm(수평 60.5°)  → 62~117px   (화각 실측 후 렌즈 교체 가능성 — 기준서 §8)
#   화각이 확정 전이라 목표 94 는 유지하고 지터를 넓혀 61~136px 로 두 경우를 모두 덮는다.
TARGET_PERSON_PX = 94
SCALE_JITTER = (0.65, 1.45)     # 목표 크기에 곱하는 지터 → 스케일 다양성 (이전 0.75~1.35)
MAX_UPSCALE = 2.2               # 이 배율을 넘는 확대는 화질이 무너져 버린다
VAL_ACTOR_RATIO = 0.2
SEED = 42


def imread_u(path):
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def imwrite_u(path, img, quality=92):
    ext = Path(path).suffix or ".jpg"
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if ext.lower() in (".jpg", ".jpeg") else []
    ok, buf = cv2.imencode(ext, img, params)
    if not ok:
        raise IOError(f"인코딩 실패: {path}")
    buf.tofile(str(path))


def index_images(root):
    """파일명 → 실제 경로"""
    idx = {}
    for p in root.rglob("*.jpg"):
        idx[p.name] = p
    return idx


def _init_worker():
    # OpenCV 는 프로세스당 내부 스레드풀을 또 만든다. 멀티프로세싱과 같이 쓰면
    # 코어 수를 초과해 경합만 생기므로 워커 프로세스당 1스레드로 고정한다.
    cv2.setNumThreads(1)


def process_one(args_tuple):
    """이미지 1장 → 크롭 args_tuple['per_image']개. (crops_written, stats_dict) 반환."""
    u, split, target_px, per_image, out_root_s, seed = args_tuple
    out_root = Path(out_root_s)
    rnd = random.Random(seed)   # 워커·순서에 무관하게 항상 같은 값이 나오는 로컬 RNG

    img = imread_u(u["path"])
    if img is None:
        return 0, {"skipped_read": 1}
    H, W = img.shape[:2]

    stats = {"crops": 0, "skipped_upscale": 0, "by_dist": Counter(),
              "by_split": Counter(), "person_px": [], "visibility": Counter()}

    for k in range(per_image):
        bx, by, bw, bh = max(u["boxes"], key=lambda b: b["bbox"][2] * b["bbox"][3])["bbox"]
        long_side = max(bw, bh)
        if long_side <= 1:
            continue
        target = target_px * rnd.uniform(*SCALE_JITTER)
        scale = target / long_side
        if scale > MAX_UPSCALE:
            stats["skipped_upscale"] += 1
            continue

        new_w, new_h = int(round(W * scale)), int(round(H * scale))
        if new_w < CROP_W or new_h < CROP_H:
            scale = max(CROP_W / W, CROP_H / H)
            new_w, new_h = int(round(W * scale)), int(round(H * scale))
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
        resized = cv2.resize(img, (new_w, new_h), interpolation=interp)

        pcx, pcy = (bx + bw / 2) * scale, (by + bh / 2) * scale
        jx = rnd.uniform(-0.30, 0.30) * CROP_W
        jy = rnd.uniform(-0.30, 0.30) * CROP_H
        x0 = int(round(pcx - CROP_W / 2 + jx))
        y0 = int(round(pcy - CROP_H / 2 + jy))
        x0 = max(0, min(x0, new_w - CROP_W))
        y0 = max(0, min(y0, new_h - CROP_H))
        crop = resized[y0:y0 + CROP_H, x0:x0 + CROP_W]
        if crop.shape[0] != CROP_H or crop.shape[1] != CROP_W:
            continue

        lines = []
        for b in u["boxes"]:
            sx, sy, sw, sh = [v * scale for v in b["bbox"]]
            x1, y1 = sx - x0, sy - y0
            x2, y2 = x1 + sw, y1 + sh
            cx1, cy1 = max(x1, 0), max(y1, 0)
            cx2, cy2 = min(x2, CROP_W), min(y2, CROP_H)
            if cx2 - cx1 < 8 or cy2 - cy1 < 8:
                continue
            if (cx2 - cx1) * (cy2 - cy1) < 0.4 * sw * sh:
                continue
            w_, h_ = cx2 - cx1, cy2 - cy1
            lines.append(f"0 {(cx1 + w_/2)/CROP_W:.6f} {(cy1 + h_/2)/CROP_H:.6f} "
                         f"{w_/CROP_W:.6f} {h_/CROP_H:.6f}")
            stats["person_px"].append(max(w_, h_))
            stats["visibility"][b.get("visibility", "?")] += 1
        if not lines:
            continue

        stem = f"{u['rec']['image_id'].replace('.jpg','')}_c{k}"
        imwrite_u(out_root / "images" / split / f"{stem}.jpg", crop)
        (out_root / "labels" / split / f"{stem}.txt").write_text(
            "\n".join(lines) + "\n", encoding="utf-8")
        stats["crops"] += 1
        stats["by_dist"][u["dist"]] += 1
        stats["by_split"][split] += 1

    return stats["crops"], stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nomad", default=str(NOMAD_DIR))
    ap.add_argument("--out", default=str(OUT_ROOT))
    ap.add_argument("--distances", default="10,30",
                    help="사용할 거리(m) 목록. 기본 10,30 (업스케일 과한 50/70/90 제외)")
    ap.add_argument("--per-image", type=int, default=1, help="원본 1장당 생성할 크롭 수")
    ap.add_argument("--min-visibility", type=int, default=0,
                    help="이 값 미만 가시성 표본 제외 (0=전부 사용)")
    ap.add_argument("--target-px", type=int, default=TARGET_PERSON_PX)
    ap.add_argument("--limit", type=int, default=0, help="디버그용 최대 처리 수")
    ap.add_argument("--all-train", action="store_true",
                    help="이번 배치를 전부 train 으로 (검증셋을 기존 것으로 고정할 때)")
    ap.add_argument("--workers", type=int, default=0,
                    help="병렬 워커 수. 0이면 CPU 코어 수 그대로 (서버는 16코어)")
    args = ap.parse_args()

    random.seed(SEED)
    nomad = Path(args.nomad)
    out_root = Path(args.out)
    dists = {int(d) for d in args.distances.split(",") if d.strip()}

    ann_path = nomad / "annotations.json"
    if not ann_path.exists():
        raise SystemExit(f"주석 없음: {ann_path}")
    records = json.load(open(ann_path, encoding="utf-8"))
    img_index = index_images(nomad)
    print(f"주석 {len(records)}개 / 보유 이미지 {len(img_index)}장")

    # 사용할 레코드 선별 (annotations.json 순서 그대로 — 재현성의 기준)
    usable = []
    for r in records:
        p = img_index.get(r["file_name"])
        if p is None or not r["annotations"]:
            continue
        m = re.match(r"Actor(\d+)_a(\d+)_", r["file_name"])
        if not m:
            continue
        actor, dist = m.group(1), int(m.group(2))
        if dist not in dists:
            continue
        boxes = [b for b in r["annotations"]
                 if int(b.get("visibility", 100)) >= args.min_visibility]
        if not boxes:
            continue
        usable.append({"rec": r, "path": p, "actor": actor, "dist": dist, "boxes": boxes})
    if not usable:
        raise SystemExit("조건에 맞는 표본이 없음 — --distances / --min-visibility 확인")
    if args.limit:
        usable = usable[:args.limit]

    # ── 배우 단위 train/val 분할 (프레임 단위로 나누면 누수) ──
    actors = sorted({u["actor"] for u in usable})
    if args.all_train:
        val_actors = set()
    else:
        random.shuffle(actors)
        n_val = max(1, round(len(actors) * VAL_ACTOR_RATIO))
        val_actors = set(actors[:n_val])
    print(f"배우 {len(actors)}명 → val {sorted(val_actors) or '없음(전부 train)'} / "
          f"train {len(actors)-len(val_actors)}명")

    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (out_root / sub).mkdir(parents=True, exist_ok=True)

    n_workers = args.workers or os.cpu_count() or 1
    print(f"워커 {n_workers}개로 병렬 처리 ({len(usable)}장)")

    jobs = []
    for i, u in enumerate(usable):
        split = "val" if u["actor"] in val_actors else "train"
        jobs.append((u, split, args.target_px, args.per_image, str(out_root), SEED + i))

    stats = {"crops": 0, "skipped_upscale": 0, "skipped_read": 0,
             "by_dist": Counter(), "by_split": Counter(),
             "person_px": [], "visibility": Counter()}

    done = 0
    with mp.Pool(processes=n_workers, initializer=_init_worker) as pool:
        for _, s in pool.imap_unordered(process_one, jobs, chunksize=8):
            stats["crops"] += s.get("crops", 0)
            stats["skipped_read"] += s.get("skipped_read", 0)
            stats["skipped_upscale"] += s.get("skipped_upscale", 0)
            stats["by_dist"].update(s.get("by_dist", {}))
            stats["by_split"].update(s.get("by_split", {}))
            stats["person_px"].extend(s.get("person_px", []))
            stats["visibility"].update(s.get("visibility", {}))
            done += 1
            if done % 300 == 0:
                print(f"  {done}/{len(jobs)} 처리 (크롭 {stats['crops']}장)")

    (out_root / "data.yaml").write_text(
        f"path: {out_root.resolve().as_posix()}\n"
        "train: images/train\nval: images/val\nnc: 1\nnames: ['person']\n",
        encoding="utf-8")

    px = np.array(stats["person_px"]) if stats["person_px"] else np.array([0])
    summary = {
        "crops": stats["crops"],
        "train": stats["by_split"]["train"], "val": stats["by_split"]["val"],
        "val_actors": sorted(val_actors),
        "by_distance": dict(stats["by_dist"]),
        "skipped_upscale": stats["skipped_upscale"],
        "person_px_quartiles": np.percentile(px, [0, 25, 50, 75, 100]).round(1).tolist(),
        "target_person_px": args.target_px,
        "crop_size": [CROP_W, CROP_H],
        "visibility": dict(stats["visibility"]),
        "split_policy": "actor-level (프레임 단위 분할 시 누수)",
    }
    (out_root / "nomad_prep_stats.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n=== 변환 완료 ===")
    print(f"크롭 {summary['crops']}장 (train {summary['train']} / val {summary['val']})")
    print(f"거리별: {summary['by_distance']}  | 업스케일 초과로 제외 {summary['skipped_upscale']}")
    print(f"사람 크기(px) 사분위: {summary['person_px_quartiles']} (목표 {args.target_px})")
    print(f"저장: {out_root}")


if __name__ == "__main__":
    main()
