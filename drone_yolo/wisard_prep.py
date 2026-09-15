"""
wisard_prep.py — WiSARD 원본 → 운용 조건에 맞춘 YOLO 학습셋 변환

NOMAD 대비 차이
  · 라벨이 **이미 YOLO txt** 형식이라 파싱이 단순하다(annotations.json 불필요)
  · 해상도가 4종으로 섞여 있다(1920x1080 / 2720x1530 / 3840x2160 / 4096x2160)
    → 해상도마다 사람 크기가 59~118px 로 달라서 **프레임별로 배율을 따로 계산**한다
  · 프레임당 사람이 평균 1.8명이라 기준 인물을 **중앙값 크기**로 잡는다
    (최대 인물 기준으로 맞추면 나머지가 너무 작아진다)
  · 분할 단위는 배우가 아니라 **비행(폴더)** 이다. 한 비행의 연속 프레임은 거의 같은
    장면이라 프레임 단위로 나누면 검증 점수가 부풀려진다.

계절 정보
  폴더명 YYMMDD_장소_기체_TYPE 에서 월을 추출해 메타로 남긴다. 나중에 겨울(1월) 표본만
  뽑아 시현 환경 대비 성능을 따로 평가하기 위함이다.

병렬화 (2026-09-11 추가)
  프레임 단위 처리는 서로 독립이라 프로세스 풀로 병렬 처리한다. split(train/val)은
  미리(메인 프로세스에서) 정해서 각 작업에 실어 보내고, 재현성을 위해 프레임마다
  결정적 시드(SEED + 전역 프레임 순번)를 쓴다 — 워커 수·완료 순서가 달라져도
  실행마다 같은 크롭이 나온다. --workers 로 조절(기본 CPU 코어 수).

실행:
  python wisard_prep.py                          # 기본: VIS 전체
  python wisard_prep.py --months 09,01           # 특정 월만
  python wisard_prep.py --per-image 1 --negatives 0.1
  python wisard_prep.py --workers 12             # 병렬 워커 수 직접 지정
출력: data/det/wisard/{images,labels}/{train,val} + data.yaml + wisard_prep_stats.json
"""
import argparse
import json
import multiprocessing as mp
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
WISARD_DIR = BASE_DIR / "data" / "raw" / "WiSARD"
OUT_ROOT = BASE_DIR / "data" / "det" / "wisard"

CROP_W, CROP_H = 1280, 720
# 목표 크기 근거는 nomad_prep.py 와 같다 (2026-09-12 수정): 수평 화각 54°(5.5mm) · 1280px ·
# 고도 16~30m 에서 사람 1.7m = 71~133px, 4.8mm(60.5°)면 62~117px. 지터로 61~136px 를 덮는다.
TARGET_PERSON_PX = 94
SCALE_JITTER = (0.65, 1.45)    # 이전 0.75~1.35
MAX_UPSCALE = 2.2              # 이 배율 초과 확대는 화질이 무너진다
VAL_FLIGHT_RATIO = 0.2
NEGATIVE_RATIO = 0.10          # 라벨 없는 프레임(배경)을 이 비율만큼 섞는다
SEED = 42
IMG_EXT = ("*.jpeg", "*.jpg", "*.png")


def imread_u(path):
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def imwrite_u(path, img, quality=92):
    ext = Path(path).suffix or ".jpg"
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if ext.lower() in (".jpg", ".jpeg") else []
    ok, buf = cv2.imencode(ext, img, params)
    if not ok:
        raise IOError(f"인코딩 실패: {path}")
    buf.tofile(str(path))


def read_yolo(txt_path, W, H):
    """YOLO txt → [(x, y, w, h)] 픽셀 절대좌표"""
    out = []
    if not txt_path.exists():
        return out
    for ln in txt_path.read_text(encoding="utf-8", errors="replace").splitlines():
        q = ln.split()
        if len(q) != 5:
            continue
        try:
            _, cx, cy, bw, bh = [float(v) for v in q]
        except ValueError:
            continue
        out.append(((cx - bw / 2) * W, (cy - bh / 2) * H, bw * W, bh * H))
    return out


def list_flights(root, months=None, kind="VIS"):
    """비행 폴더 목록. (폴더명, 월) 반환."""
    flights = []
    for d in sorted(os.listdir(root)):
        p = root / d
        if not p.is_dir() or f"_{kind}" not in d:
            continue
        m = re.match(r"\d{2}(\d{2})\d{2}_", d)
        mon = m.group(1) if m else "??"
        if months and mon not in months:
            continue
        flights.append((d, mon))
    return flights


# 목표 크기 분포 (--target-dist): jitter = TARGET_PERSON_PX × SCALE_JITTER (기존) · loguniform = [min, max] 로그 균등
TARGET_DIST, TARGET_MIN, TARGET_MAX = "jitter", 22.0, 120.0


def _init_worker(target_dist="jitter", target_min=22.0, target_max=120.0):
    cv2.setNumThreads(1)  # 멀티프로세싱과 OpenCV 내부 스레드풀이 겹치지 않게
    # Windows(spawn)는 전역을 물려받지 않으므로 크기 설정은 initargs 로 넘긴다
    global TARGET_DIST, TARGET_MIN, TARGET_MAX
    TARGET_DIST, TARGET_MIN, TARGET_MAX = target_dist, float(target_min), float(target_max)


def process_frame(job):
    """프레임 1장 → 크롭(들) 또는 배경 1장. (crops_written, stats_dict) 반환."""
    img_p, split, mon, target_px, per_image, negatives, out_root_s, seed = job
    out_root = Path(out_root_s)
    rnd = random.Random(seed)

    stats = {"crops": 0, "negatives": 0, "boxes": 0, "skip_upscale": 0, "skip_read": 0,
              "by_split": Counter(), "by_month": Counter(), "by_res": Counter(),
              "person_px": []}

    txt_p = img_p.with_suffix(".txt")
    img = imread_u(img_p)
    if img is None:
        stats["skip_read"] += 1
        return 0, stats
    H, W = img.shape[:2]
    boxes = read_yolo(txt_p, W, H)

    if not boxes:
        if negatives <= 0 or rnd.random() > negatives:
            return 0, stats
        scale = max(CROP_W / W, CROP_H / H)
        resized = cv2.resize(img, (int(W * scale), int(H * scale)),
                             interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
        nh, nw = resized.shape[:2]
        x0 = rnd.randint(0, max(nw - CROP_W, 0))
        y0 = rnd.randint(0, max(nh - CROP_H, 0))
        crop = resized[y0:y0 + CROP_H, x0:x0 + CROP_W]
        if crop.shape[:2] != (CROP_H, CROP_W):
            return 0, stats
        stem = f"{img_p.stem}_n"
        imwrite_u(out_root / "images" / split / f"{stem}.jpg", crop)
        (out_root / "labels" / split / f"{stem}.txt").write_text("", encoding="utf-8")
        stats["crops"] += 1; stats["negatives"] += 1
        stats["by_split"][split] += 1; stats["by_month"][mon] += 1
        stats["by_res"][f"{W}x{H}"] += 1
        return 1, stats

    for k in range(per_image):
        longs = [max(b[2], b[3]) for b in boxes]
        ref = float(np.median(longs))
        if ref <= 1:
            continue
        if TARGET_DIST == "loguniform":
            # 원본마다 만들 수 있는 크기 범위와 목표 범위의 교집합에서 로그 균등 (nomad_prep.py 와 같은 규칙)
            s_fill = max(CROP_W / W, CROP_H / H)
            lo, hi = max(TARGET_MIN, ref * s_fill), min(TARGET_MAX, ref * MAX_UPSCALE)
            if lo > hi:
                stats["skip_upscale"] += 1   # 도달 불가 (기존 통계 이름 유지)
                continue
            target = float(np.exp(rnd.uniform(np.log(lo), np.log(hi))))
        else:
            target = target_px * rnd.uniform(*SCALE_JITTER)
        scale = target / ref
        if scale > MAX_UPSCALE:
            stats["skip_upscale"] += 1
            continue
        nw, nh = int(round(W * scale)), int(round(H * scale))
        if nw < CROP_W or nh < CROP_H:
            scale = max(CROP_W / W, CROP_H / H)
            nw, nh = int(round(W * scale)), int(round(H * scale))
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
        resized = cv2.resize(img, (nw, nh), interpolation=interp)

        bx, by, bw, bh = rnd.choice(boxes)
        pcx, pcy = (bx + bw / 2) * scale, (by + bh / 2) * scale
        x0 = int(round(pcx - CROP_W / 2 + rnd.uniform(-0.3, 0.3) * CROP_W))
        y0 = int(round(pcy - CROP_H / 2 + rnd.uniform(-0.3, 0.3) * CROP_H))
        x0 = max(0, min(x0, nw - CROP_W))
        y0 = max(0, min(y0, nh - CROP_H))
        crop = resized[y0:y0 + CROP_H, x0:x0 + CROP_W]
        if crop.shape[:2] != (CROP_H, CROP_W):
            continue

        lines = []
        for sbx, sby, sbw, sbh in boxes:
            x1, y1 = sbx * scale - x0, sby * scale - y0
            w_, h_ = sbw * scale, sbh * scale
            cx1, cy1 = max(x1, 0), max(y1, 0)
            cx2, cy2 = min(x1 + w_, CROP_W), min(y1 + h_, CROP_H)
            if cx2 - cx1 < 8 or cy2 - cy1 < 8:
                continue
            if (cx2 - cx1) * (cy2 - cy1) < 0.4 * w_ * h_:
                continue
            ww, hh = cx2 - cx1, cy2 - cy1
            lines.append(f"0 {(cx1+ww/2)/CROP_W:.6f} {(cy1+hh/2)/CROP_H:.6f} "
                         f"{ww/CROP_W:.6f} {hh/CROP_H:.6f}")
            stats["person_px"].append(max(ww, hh))
        if not lines:
            continue

        stem = f"{img_p.stem}_c{k}"
        imwrite_u(out_root / "images" / split / f"{stem}.jpg", crop)
        (out_root / "labels" / split / f"{stem}.txt").write_text(
            "\n".join(lines) + "\n", encoding="utf-8")
        stats["crops"] += 1; stats["boxes"] += len(lines)
        stats["by_split"][split] += 1; stats["by_month"][mon] += 1
        stats["by_res"][f"{W}x{H}"] += 1

    return stats["crops"], stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(WISARD_DIR))
    ap.add_argument("--out", default=str(OUT_ROOT))
    ap.add_argument("--kind", default="VIS", choices=("VIS", "IR"),
                    help="VIS=가시광(기본). IR은 열화상 카메라 도입 시에만 의미 있음")
    ap.add_argument("--months", default="", help="쉼표 구분 월 필터. 예: 09,01 (비우면 전체)")
    ap.add_argument("--per-image", type=int, default=1, help="원본 1장당 생성할 크롭 수")
    ap.add_argument("--negatives", type=float, default=NEGATIVE_RATIO,
                    help="라벨 없는 배경 프레임 비율 (0이면 제외)")
    ap.add_argument("--target-px", type=int, default=TARGET_PERSON_PX)
    ap.add_argument("--limit", type=int, default=0, help="디버그용 최대 처리 프레임 수")
    ap.add_argument("--all-train", action="store_true",
                    help="전부 train 으로 (검증셋을 기존 것으로 고정할 때)")
    ap.add_argument("--time-split-single", action="store_true", default=True,
                    help="비행이 1개뿐인 월은 비행 내부를 시간순 분할 (겨울 평가용, 기본 켜짐)")
    ap.add_argument("--no-time-split-single", dest="time_split_single",
                    action="store_false", help="시간분할 끄기")
    ap.add_argument("--time-split", type=float, default=0.3,
                    help="시간분할 시 뒤쪽 몇 비율을 val 로 쓸지 (기본 0.3)")
    ap.add_argument("--workers", type=int, default=0,
                    help="병렬 워커 수. 0이면 CPU 코어 수 그대로 (서버는 16코어)")
    ap.add_argument("--target-dist", default="jitter", choices=("jitter", "loguniform"),
                    help="jitter = --target-px × SCALE_JITTER (기존) · loguniform = [--target-min, --target-max] 로그 균등")
    ap.add_argument("--target-min", type=float, default=22.0)
    ap.add_argument("--target-max", type=float, default=120.0)
    args = ap.parse_args()

    random.seed(SEED)
    src, out_root = Path(args.src), Path(args.out)
    months = {m.strip() for m in args.months.split(",") if m.strip()} or None

    flights = list_flights(src, months, args.kind)
    if not flights:
        raise SystemExit(f"조건에 맞는 비행 폴더 없음: {src} (kind={args.kind}, months={months})")
    print(f"비행 폴더 {len(flights)}개 (kind={args.kind}"
          f"{', months=' + ','.join(sorted(months)) if months else ''})")

    val_flights, time_split = set(), {}
    if not args.all_train:
        by_mon = defaultdict(list)
        for d, mon in flights:
            by_mon[mon].append(d)
        for mon, ds in by_mon.items():
            if len(ds) > 1:
                random.shuffle(ds)
                k = max(1, round(len(ds) * VAL_FLIGHT_RATIO))
                val_flights.update(ds[:k])
            elif args.time_split_single:
                time_split[ds[0]] = args.time_split
    print(f"val 비행 {len(val_flights)}개: {sorted(val_flights)[:5]}"
          f"{' ...' if len(val_flights) > 5 else ''}")
    if time_split:
        for d, r in time_split.items():
            print(f"시간분할: {d} → 뒤 {int(r*100)}% 를 val (비행이 1개뿐인 월)")

    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (out_root / sub).mkdir(parents=True, exist_ok=True)

    # ── 작업 목록 구성 (split 은 여기서 미리 확정) ──
    jobs = []
    seed_i = 0
    for flight, mon in flights:
        fdir = src / flight
        flight_split = "val" if flight in val_flights else "train"
        imgs = []
        for pat in IMG_EXT:
            imgs.extend(sorted(fdir.glob(pat)))
        cut = int(len(imgs) * (1 - time_split[flight])) if flight in time_split else None
        for fi, img_p in enumerate(imgs):
            if args.limit and len(jobs) >= args.limit:
                break
            split = ("val" if cut is not None and fi >= cut else
                     ("train" if cut is not None else flight_split))
            jobs.append((img_p, split, mon, args.target_px, args.per_image,
                         args.negatives, str(out_root), SEED + seed_i))
            seed_i += 1
        if args.limit and len(jobs) >= args.limit:
            break

    n_workers = args.workers or os.cpu_count() or 1
    print(f"워커 {n_workers}개로 병렬 처리 ({len(jobs)}프레임)")

    stats = {"crops": 0, "skip_upscale": 0, "skip_read": 0, "negatives": 0,
             "by_split": Counter(), "by_month": Counter(), "by_res": Counter(),
             "person_px": [], "boxes": 0}

    done = 0
    print(f"목표 크기: {args.target_dist}" + (f" {args.target_min:g}~{args.target_max:g} px" if args.target_dist == "loguniform" else f" {args.target_px} × {SCALE_JITTER}"))
    with mp.Pool(processes=n_workers, initializer=_init_worker,
                 initargs=(args.target_dist, args.target_min, args.target_max)) as pool:
        for _, s in pool.imap_unordered(process_frame, jobs, chunksize=16):
            stats["crops"] += s.get("crops", 0)
            stats["negatives"] += s.get("negatives", 0)
            stats["boxes"] += s.get("boxes", 0)
            stats["skip_upscale"] += s.get("skip_upscale", 0)
            stats["skip_read"] += s.get("skip_read", 0)
            stats["by_split"].update(s.get("by_split", {}))
            stats["by_month"].update(s.get("by_month", {}))
            stats["by_res"].update(s.get("by_res", {}))
            stats["person_px"].extend(s.get("person_px", []))
            done += 1
            if done % 1000 == 0:
                print(f"  {done}/{len(jobs)} 처리 (크롭 {stats['crops']}장)")

    (out_root / "data.yaml").write_text(
        f"path: {out_root.resolve().as_posix()}\n"
        "train: images/train\nval: images/val\nnc: 1\nnames: ['person']\n",
        encoding="utf-8")

    px = np.array(stats["person_px"]) if stats["person_px"] else np.array([0])
    summary = {
        "crops": stats["crops"], "boxes": stats["boxes"],
        "train": stats["by_split"]["train"], "val": stats["by_split"]["val"],
        "negatives": stats["negatives"],
        "val_flights": sorted(val_flights),
        "by_month": dict(sorted(stats["by_month"].items())),
        "by_source_resolution": dict(stats["by_res"]),
        "skip_upscale": stats["skip_upscale"], "skip_read": stats["skip_read"],
        "person_px_quartiles": np.percentile(px, [0, 25, 50, 75, 100]).round(1).tolist(),
        "target_person_px": args.target_px, "crop_size": [CROP_W, CROP_H],
        "split_policy": "flight-level, 월별 층화 (프레임 단위 분할 시 누수)",
    }
    (out_root / "wisard_prep_stats.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n=== 변환 완료 ===")
    print(f"크롭 {summary['crops']}장 (train {summary['train']} / val {summary['val']}) "
          f"| 박스 {summary['boxes']} | 배경 {summary['negatives']}")
    print(f"월별: {summary['by_month']}")
    print(f"원본 해상도별: {summary['by_source_resolution']}")
    print(f"사람 크기(px) 사분위: {summary['person_px_quartiles']} (목표 {args.target_px})")
    print(f"업스케일 초과 제외 {summary['skip_upscale']} | 읽기 실패 {summary['skip_read']}")
    print(f"저장: {out_root}")


if __name__ == "__main__":
    main()
