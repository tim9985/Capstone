"""
train_person.py — 사람 1클래스 탐지 학습 (서버 · 2026-09-24 보완)

현행 (v6 · 60° 비스듬)
  python train_person.py --stage 1 --hyp configs/hyp/v6_oblique.yaml \\
      --weights yolo11m.pt --data configs/data_v6.yaml --name v6_obl
  … --smoke      연기 실행 — 학습 64장 · 평가 20장 · 1에폭. 결과 파일이 안 생기면 exit 2
  … --dry-run    최종 설정만 찍고 끝낸다 (GPU 안 씀)

설정 우선순위 (뒤가 이긴다)
  옛 기본값·단계 기본값 < --hyp 파일 < 명시한 인자(--imgsz --scale --degrees …) < --set 키=값

단계 기본값 — 옛 명령과 체인이 그대로 돌도록 바꾸지 않는다
  stage 1 : 하향 90° 전제 — degrees 180 · flipud 0.5 · hsv 0.02/0.8/0.5 · scale 0.5 · translate 0.15
            ⚠ 60° 비스듬 데이터는 configs/hyp/v6_oblique.yaml (degrees 10 · flipud 0) 로 돌린다
  stage 2 : 합성 혼합 재파인튜닝 — 합성 데이터 보류 중 (옛 노트북 경로)

지키는 것
  · --model-yaml 파일명에 크기 글자(n/s/m/l/x)가 없으면 멈춘다 — 없으면 nano 로 학습된다 (09-23 사고)
  · 끝나면 best.pt → weights/<이름>.pt · results.csv·args.yaml·command.txt → metrics/train_runs/<이름>/
    (metrics 는 git 에 올라가므로 학습 곡선이 서버 밖에 남는다)
  · 시각은 KST 로 찍는다 (서버 시계는 UTC)
"""
import argparse
import csv
import json
import random
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE_DIR = Path(__file__).resolve().parent
NOMAD_YAML = BASE_DIR / "data" / "det" / "nomad_actor01_10" / "data.yaml"
SYNTH_IMG = BASE_DIR / "data" / "det" / "synth" / "yolo" / "images" / "train"
MIXED_YAML = BASE_DIR / "configs" / "data_stage2.yaml"
BASE_WEIGHTS = BASE_DIR / "weights" / "yolov8s_visdrone.pt"
RUNS = BASE_DIR / "runs_person"
BACKUP = BASE_DIR / "metrics" / "train_runs"
KST = ZoneInfo("Asia/Seoul")
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")

# --hyp 도 명시 인자도 없을 때 쓰는 옛 기본값 (노트북 시절 값 — 옛 명령 호환용)
LEGACY = dict(imgsz=960, patience=15, batch=-1)
# 스크립트가 직접 정하는 키 — --hyp · --set 으로 바꾸지 못한다
RESERVED = {"data", "model", "project", "name", "exist_ok", "device", "resume", "mode", "task"}
# 명시했을 때만 덮어쓰는 인자 (argparse 기본값 None)
CLI_KEYS = ("epochs", "imgsz", "patience", "lr0", "freeze", "time", "close_mosaic", "scale",
            "translate", "degrees", "flipud", "optimizer", "warmup_epochs")


def now():
    return datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S KST")


def stage_defaults(stage):
    if stage == 1:
        # 단일 도메인(여름·정오)이라 색상 증강을 기본보다 강하게.
        # degrees=180/flipud=0.5 — 하향 90° 시점엔 화면에 "위쪽"이 없다 (SERVER.md 5절 실험 A).
        # 60° 비스듬 화면은 위(먼 쪽)·아래(가까운 쪽)가 있어 맞지 않는다 → --hyp v6_oblique.yaml
        return dict(epochs=60, hsv_h=0.02, hsv_s=0.8, hsv_v=0.5, degrees=180.0,
                    flipud=0.5, translate=0.15, scale=0.5, fliplr=0.5, mosaic=1.0)
    # 이어 학습이므로 증강을 약하게, 학습률도 낮게
    return dict(epochs=8, hsv_h=0.015, hsv_s=0.6, hsv_v=0.4, degrees=8.0,
                translate=0.1, scale=0.4, fliplr=0.5, mosaic=0.5, lr0=0.002)


def parse_batch(v):
    """-1 → 자동(VRAM 60 %) · 0~1 소수 → 그 비율 목표 · 정수 → 고정 배치"""
    s = str(v)
    if s == "-1":
        return -1
    return float(s) if "." in s else int(s)


def resolve(p):
    """상대 경로는 현재 폴더 → drone_yolo 폴더 순으로 찾는다 (체인을 다른 곳에서 불러도 되게)"""
    p = Path(p)
    if p.exists() or p.is_absolute():
        return p
    q = BASE_DIR / p
    return q if q.exists() else p


def valid_keys():
    from ultralytics.cfg import DEFAULT_CFG_DICT
    return set(DEFAULT_CFG_DICT)


def check_keys(d, where):
    bad = sorted(set(d) - valid_keys())
    if bad:
        raise SystemExit(f"{where}: ultralytics 에 없는 키 {bad} — 오타인지 확인")
    fixed = sorted(set(d) & RESERVED)
    if fixed:
        raise SystemExit(f"{where}: {fixed} 는 스크립트 인자(--data · --name · --device …)로 준다")


def load_hyp(path):
    p = resolve(path)
    if not p.exists():
        raise SystemExit(f"--hyp 파일 없음: {path}")
    d = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(d, dict):
        raise SystemExit(f"--hyp 파일은 '키: 값' 모음이어야 한다: {p}")
    check_keys(d, f"--hyp {p.name}")
    if "batch" in d:
        d["batch"] = parse_batch(d["batch"])
    return d, p


def parse_sets(items):
    out = {}
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"--set 은 키=값 형식이다: {it}")
        k, v = it.split("=", 1)
        out[k.strip()] = yaml.safe_load(v)       # 10 → int · 0.5 → float · true → bool
    check_keys(out, "--set")
    if "batch" in out:
        out["batch"] = parse_batch(out["batch"])
    return out


def check_model_yaml(path):
    """구조 파일명에 크기 글자가 없으면 ultralytics 가 nano 로 만든다 (09-23 · 21시간 헛학습)"""
    p = resolve(path)
    if not p.exists():
        raise SystemExit(f"--model-yaml 파일 없음: {path}")
    y = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if "scales" in y and not re.search(r"yolo(e-)?[v]?\d+([nslmx])", p.stem):
        raise SystemExit(
            f"모델 yaml 파일명에 크기 글자가 없다: {p.name}\n"
            f"  → 'yolo11m-p2.yaml' 처럼 이름에 n/s/m/l/x 를 넣는다. 없으면 nano 로 학습된다")
    return p


# ── 연기 실행용 작은 데이터 ─────────────────────────────────────────
def _expand(entry, root):
    p = Path(entry)
    if not p.is_absolute():
        p = root / p
    if p.is_dir():
        return sorted(str(q) for q in p.rglob("*") if q.suffix.lower() in IMG_EXT)
    if p.suffix == ".txt":
        parent = str(p.parent) + "/"
        lines = [x.strip() for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
        return [x.replace("./", parent, 1) if x.startswith("./") else x for x in lines]
    if p.suffix.lower() in IMG_EXT:
        return [str(p)]
    raise SystemExit(f"데이터 항목을 읽을 수 없다: {p}")


def split_images(data_yaml, key):
    d = yaml.safe_load(Path(data_yaml).read_text(encoding="utf-8"))
    root = Path(d["path"]) if d.get("path") else Path(data_yaml).parent
    entries = d[key] if isinstance(d[key], list) else [d[key]]
    return [x for e in entries for x in _expand(e, root)], d


def make_smoke_data(data_yaml, out_dir, n_train, n_val, write=True):
    train, d = split_images(data_yaml, "train")
    val, _ = split_images(data_yaml, "val")
    rng = random.Random(42)
    tr = rng.sample(train, min(n_train, len(train)))
    va = rng.sample(val, min(n_val, len(val)))
    y = out_dir / "smoke_data.yaml"
    if write:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "train.txt").write_text("\n".join(tr) + "\n", encoding="utf-8")
        (out_dir / "val.txt").write_text("\n".join(va) + "\n", encoding="utf-8")
        y.write_text(yaml.safe_dump({"train": str(out_dir / "train.txt"), "val": str(out_dir / "val.txt"),
                                     "nc": d["nc"], "names": d["names"]}, allow_unicode=True),
                     encoding="utf-8")
    return y, len(train), len(tr), len(va)


# ── 결과 정리 ───────────────────────────────────────────────────────
def read_results(run):
    f = run / "results.csv"
    if not f.exists():
        return []
    with f.open(encoding="utf-8") as fh:
        return [{k.strip(): v.strip() for k, v in r.items()} for r in csv.DictReader(fh)]


def summarize(run):
    rows = read_results(run)
    k95, k50 = "metrics/mAP50-95(B)", "metrics/mAP50(B)"
    if not rows or k95 not in rows[0]:
        return None
    b = max(rows, key=lambda r: float(r[k95]))
    return dict(epochs=len(rows), best_epoch=int(float(b["epoch"])),
                mAP50=round(float(b[k50]), 4), mAP50_95=round(float(b[k95]), 4))


def backup(name):
    """best.pt → weights/ · 학습 곡선·설정 → metrics/train_runs/ (git 으로 서버 밖에 남는다)"""
    run = RUNS / name
    best = run / "weights" / "best.pt"
    if best.exists():
        dest = BASE_DIR / "weights" / f"{name}.pt"   # 모델 정보는 name 에 (yolov8s_ 접두사 없음 · 09-11)
        dest.parent.mkdir(exist_ok=True)
        shutil.copy(best, dest)
        print(f"  가중치 백업 : {dest}")
    dst = BACKUP / name
    copied = []
    for f in ("results.csv", "args.yaml", "command.txt"):
        if (run / f).exists():
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copy(run / f, dst / f)
            copied.append(f)
    if copied:
        print(f"  기록 백업   : {dst}/ ({' · '.join(copied)}) — git 에 올리면 서버 밖에 남는다")


def log_command(run, cfg, extra=""):
    run.mkdir(parents=True, exist_ok=True)
    with (run / "command.txt").open("a", encoding="utf-8") as fh:
        fh.write(f"# {now()} {extra}\n{sys.executable} {' '.join(sys.argv)}\n"
                 f"{json.dumps(cfg, ensure_ascii=False, sort_keys=True, default=str)}\n\n")


def schedule_shutdown(sec):
    """노트북(Windows) 전용 — 학습이 정상 완료된 뒤에만 호출된다. 취소: shutdown /a"""
    if sys.platform != "win32":
        print("※ --shutdown 은 Windows 노트북 전용이다 — 무시")
        return
    import subprocess
    print(f"\n※ {sec}초 뒤 컴퓨터를 종료합니다. 취소하려면 다른 터미널에서:  shutdown /a")
    try:
        subprocess.run(["shutdown", "/s", "/t", str(sec),
                        "/c", "Claude Code: 학습 완료로 자동 종료"], check=True)
    except Exception as e:
        print(f"종료 예약 실패(무시): {e}")


def write_mixed_yaml(repeat=3):
    """stage2용 data.yaml — 실데이터 전체 train + 합성 train, 검증은 **실데이터만**.

    합성을 검증에 넣으면 "자기가 만든 그림체를 자기가 알아보는가"를 재게 되어 수치가 부풀려진다.
    repeat: 합성 411장은 실사 14,342장의 2.8 % 뿐이라 3배(약 8 %)로 넣는다.
    """
    D = BASE_DIR / "data"
    real_train = [D / n / "images" / "train" for n in
                  ("det/nomad_actor01_10", "det/nomad_actor11_20", "det/nomad_actor21_30")]
    real_train.append(D / "det" / "wisard" / "images" / "train")
    real_val = [D / "det" / "nomad_actor01_10" / "images" / "val",
                D / "det" / "nomad_actor11_20" / "images" / "val",
                D / "det" / "wisard" / "images" / "val"]

    if repeat <= 1:
        train_block = "train:\n" + "".join(
            f"  - {d.as_posix()}\n" for d in real_train + [SYNTH_IMG])
    else:
        # 반복 투입은 폴더 지정으로 표현할 수 없어 이미지 목록 파일을 만든다.
        # 실사 크롭은 .jpg, 합성은 .png 다. 확장자를 고정하면 조용히 0장이 섞인다.
        def imgs(d):
            return sorted(q for q in d.iterdir()
                          if q.suffix.lower() in (".jpg", ".jpeg", ".png"))

        lst = BASE_DIR / "configs" / "stage2_train_list.txt"
        paths = []
        for d in real_train:
            paths += [str(q) for q in imgs(d)]
        n_real = len(paths)
        synth = [str(q) for q in imgs(SYNTH_IMG)]
        if not synth:
            raise SystemExit(f"합성 이미지를 찾지 못했습니다: {SYNTH_IMG}")
        paths += synth * repeat
        lst.write_text("\n".join(paths), encoding="utf-8")
        print(f"  학습 구성: 실사 {n_real}장 + 합성 {len(synth)}장×{repeat} "
              f"= {len(paths)}장 (합성 비중 {len(synth)*repeat/len(paths)*100:.1f}%)")
        train_block = f"train: {lst.as_posix()}\n"

    MIXED_YAML.write_text(
        f"# stage2: 실데이터 + 합성 혼합 (합성 {repeat}배). val 은 실데이터만\n"
        + train_block
        + "val:\n" + "".join(f"  - {d.as_posix()}\n" for d in real_val)
        + "nc: 1\nnames: ['person']\n",
        encoding="utf-8")
    return MIXED_YAML


def build_parser():
    ap = argparse.ArgumentParser(description="사람 1클래스 탐지 학습 — 설정 우선순위: "
                                             "기본값 < --hyp < 명시 인자 < --set")
    g = ap.add_argument_group("무엇을")
    g.add_argument("--stage", type=int, choices=(1, 2), required=True)
    g.add_argument("--data", default=None, help="data.yaml (예: configs/data_v3_place_neg.yaml)")
    g.add_argument("--weights", default=None, help="시작 가중치 (서버 실험은 yolo11m.pt)")
    g.add_argument("--model-yaml", default=None,
                   help="구조 파일(예: configs/models/yolo11m-p2.yaml). --weights 는 호환 층만 이식. "
                        "파일명에 크기 글자 필수")
    g.add_argument("--name", default=None, help="결과 폴더명 (runs_person/<이름>)")
    g.add_argument("--synth-repeat", type=int, default=3,
                   help="stage2 에서 합성 데이터를 몇 배로 넣을지 (1이면 원본 그대로)")

    g = ap.add_argument_group("설정 묶음")
    g.add_argument("--hyp", default=None,
                   help="설정 파일 (예: configs/hyp/v6_oblique.yaml) — ultralytics 학습 인자 모음")
    g.add_argument("--set", nargs="+", default=None, metavar="키=값",
                   help="그 밖의 ultralytics 인자를 직접 (예: --set box=7.5 fraction=0.5). 가장 우선")

    g = ap.add_argument_group("학습 (명시하면 --hyp 보다 우선)")
    g.add_argument("--epochs", type=int, default=None)
    g.add_argument("--imgsz", type=int, default=None, help="기본 960 (옛 값) · 서버 실험은 1280")
    g.add_argument("--batch", default=None,
                   help="-1 이면 VRAM 60%% 목표로 자동. 0<x<1 소수면 그 비율을 목표로 자동"
                        "(예: 0.85 → 3090 24GB의 85%%). 정수면 고정 배치 (기본 -1)")
    g.add_argument("--patience", type=int, default=None, help="기본 15 (옛 값)")
    # 파인튜닝에서 학습률은 결정적이다. 기본 0.01 은 처음부터 학습할 때의 값이라
    # 좋은 모델을 이어받을 때는 사전학습 특징을 초기 몇 에폭 만에 망가뜨린다(파국적 망각).
    g.add_argument("--lr0", type=float, default=None,
                   help="시작 학습률. 잘 되는 모델을 이어받을 때는 0.001 이하를 쓴다")
    # optimizer=auto(기본)면 ultralytics 가 --lr0 를 무시하고 스스로 고른다 (09-12 로그로 확인).
    g.add_argument("--optimizer", default=None,
                   help="옵티마이저 이름 (예: SGD, MuSGD, AdamW). 지정해야 --lr0 가 적용된다")
    g.add_argument("--warmup-epochs", type=float, default=None,
                   help="워밍업 에폭 (ultralytics 기본 3). 중간 가중치에서 이어갈 때는 줄인다")
    g.add_argument("--freeze", type=int, default=None,
                   help="앞쪽 N개 층을 고정한다. 특징 추출부를 보존하고 헤드만 학습시킬 때")
    # 마감이 정해진 학습용. ultralytics 가 첫 에폭 속도를 보고 에폭 수와 학습률 스케줄을 다시 맞춘다.
    g.add_argument("--time", type=float, default=None,
                   help="최대 학습 시간(시간 단위). 지정하면 --epochs 를 덮어쓴다")
    g.add_argument("--close-mosaic", type=int, default=None,
                   help="마지막 N 에폭은 모자이크를 끈다 (ultralytics 기본 10 · 우리 조건에선 0 — 09-20)")
    g.add_argument("--rect", action="store_true",
                   help="직사각 배치 — 추론 엔진 [736,1280] 과 모양을 맞춘다. ⚠ mosaic·shuffle 이 꺼진다")

    g = ap.add_argument_group("증강 (명시하면 --hyp 보다 우선)")
    g.add_argument("--scale", type=float, default=None,
                   help="크기 증강 (stage1 기본 0.5). 전처리에서 크기 분포를 이미 넓혔으면 0.3")
    g.add_argument("--translate", type=float, default=None,
                   help="위치 증강 (stage1 기본 0.15). 0.30 은 09-21 실패")
    g.add_argument("--degrees", type=float, default=None,
                   help="회전 증강 ±도 (stage1 기본 180 = 하향 90° 전제). 60° 비스듬은 10")
    g.add_argument("--flipud", type=float, default=None,
                   help="상하 반전 확률 (stage1 기본 0.5 = 하향 90° 전제). 60° 비스듬은 0")

    g = ap.add_argument_group("실행")
    g.add_argument("--smoke", action="store_true",
                   help="연기 실행 — 학습 N장 · 평가 20장 · 1에폭으로 끝까지 돌려 보고 결과 파일을 확인")
    g.add_argument("--smoke-train", type=int, default=64, help="연기 실행 학습 장수 (기본 64)")
    g.add_argument("--smoke-val", type=int, default=20, help="연기 실행 평가 장수 (기본 20)")
    g.add_argument("--dry-run", action="store_true", help="최종 설정만 찍고 끝낸다")
    g.add_argument("--resume", action="store_true", help="중단된 학습을 last.pt 에서 이어서 진행")
    g.add_argument("--device", default=0)
    # 노트북(RTX 3050 · 4코어) 기준 값이 박혀 있던 것을 뺐다. 결과에는 영향이 없고 에폭 시간만 바뀐다.
    g.add_argument("--workers", type=int, default=8, help="데이터 로더 워커 수 (노트북 4 · 서버 8)")
    g.add_argument("--cache", default="disk", choices=("False", "disk", "ram"),
                   help="이미지 캐시. 'ram' 은 9,008장에 ~24GB 필요하니 여유를 확인할 것")
    g.add_argument("--shutdown", type=int, default=0, metavar="SEC",
                   help="(Windows 노트북 전용) 학습 정상 완료 후 지정 초 뒤 종료")
    return ap


def resolve_config(args):
    """(cfg, 출처 표기) — 기본값 < --hyp < 명시 인자 < --set"""
    cfg = {**LEGACY, **stage_defaults(args.stage)}
    src = {k: "기본" for k in cfg}
    if args.hyp:
        hyp, hp = load_hyp(args.hyp)
        cfg.update(hyp)
        src.update({k: f"hyp:{hp.name}" for k in hyp})
    cli = {k: getattr(args, k) for k in CLI_KEYS if getattr(args, k) is not None}
    if args.batch is not None:
        cli["batch"] = parse_batch(args.batch)
    if args.rect:
        cli["rect"] = True
    cfg.update(cli)
    src.update({k: "인자" for k in cli})
    sets = parse_sets(args.set)
    cfg.update(sets)
    src.update({k: "--set" for k in sets})
    return cfg, src


def print_config(title, data, weights, model_yaml, name, cfg, src):
    print(f"=== {title} · {now()} ===")
    print(f"  데이터      : {data}")
    print(f"  시작 가중치 : {weights}" + (f"  (구조 {model_yaml})" if model_yaml else ""))
    print(f"  결과        : {RUNS / name}")
    w = max(len(k) for k in cfg)
    for k in sorted(cfg):
        print(f"    {k:<{w}} = {cfg[k]!s:<10} ← {src.get(k, '')}")
    if cfg.get("rect"):
        print("  ⚠ rect=True — ultralytics 가 mosaic · mixup 을 끄고 shuffle 도 끈다")
    if cfg.get("degrees", 0) >= 90 or cfg.get("flipud", 0) > 0:
        print("  ※ 회전 180°·상하 반전은 하향 90° 전제다 — 60° 비스듬 데이터면 --hyp configs/hyp/v6_oblique.yaml")


def main():
    args = build_parser().parse_args()

    # ── 이어 학습 ──
    if args.resume:
        from ultralytics import YOLO
        # --name 을 준 실행(예: stage1_nomad20)을 이어받을 수 있어야 한다
        name = args.name or ("stage1_nomad" if args.stage == 1 else "stage2_synth")
        last = RUNS / name / "weights" / "last.pt"
        if not last.exists():
            raise SystemExit(f"이어받을 체크포인트가 없습니다: {last}")
        print(f"=== stage{args.stage} 이어 학습: {last} · {now()} ===")
        resume_kw = {}
        if args.time is not None:
            # 체크포인트에 저장된 time(시간 제한)을 덮어쓴다. resume 하면 시간 측정이 0부터 다시
            # 시작돼 에폭 수가 재계산되므로, 0 을 주면 시간 제한 없이 저장된 epochs 까지 정확히 간다.
            resume_kw["time"] = args.time
            print(f"  시간 제한 덮어쓰기: {args.time}")
        if args.dry_run:
            return
        log_command(RUNS / name, resume_kw, "resume")
        YOLO(str(last)).train(resume=True, **resume_kw)
        print(f"\n=== 끝 · {now()} · {summarize(RUNS / name)} ===")
        backup(name)
        if args.shutdown:
            schedule_shutdown(args.shutdown)
        return

    # ── 무엇을 ──
    if args.stage == 1:
        data = NOMAD_YAML
        weights = resolve(args.weights) if args.weights else BASE_WEIGHTS
        name = "stage1_nomad"
    else:
        default_w = RUNS / "stage1_all" / "weights" / "best.pt"
        weights = resolve(args.weights) if args.weights else default_w
        if not weights.exists():
            raise SystemExit(f"stage1 가중치가 없습니다: {weights}\n먼저 --stage 1 을 실행하세요.")
        data = None if args.data else write_mixed_yaml(args.synth_repeat)
        name = "stage2_synth"
    if args.data:
        data = resolve(args.data)
    if args.name:
        name = args.name
    if not Path(data).exists():
        raise SystemExit(f"데이터 설정 없음: {data}")
    if not weights.exists():
        raise SystemExit(f"가중치 없음: {weights}")
    model_yaml = check_model_yaml(args.model_yaml) if args.model_yaml else None

    cfg, src = resolve_config(args)

    smoke = None
    if args.smoke:
        name = f"{name}_smoke"
        y, n_full, n_tr, n_va = make_smoke_data(data, RUNS / name / "smoke_data",
                                                args.smoke_train, args.smoke_val,
                                                write=not args.dry_run)
        smoke = dict(n_full=n_full, n_train=n_tr, n_val=n_va, data=str(data))
        data = y
        cfg["epochs"] = 1
        cfg.pop("time", None)                    # time 이 있으면 에폭 수가 다시 계산된다
        src["epochs"] = "연기 실행"

    fixed = dict(amp=True, cache=False if (args.cache == "False" or smoke) else args.cache,
                 workers=args.workers, seed=42, val=True, plots=not smoke)
    final = {**fixed, **cfg}

    fsrc = {k: "고정" for k in fixed}
    fsrc.update(workers="인자" if args.workers != 8 else "기본",
                cache="연기 실행" if smoke else ("인자" if args.cache != "disk" else "기본"),
                plots="연기 실행" if smoke else "고정")
    print_config(f"stage{args.stage} 학습" + (" · 연기 실행" if smoke else ""),
                 data, weights, model_yaml, name, final, {**fsrc, **src})
    if smoke:
        print(f"  연기 실행   : 학습 {smoke['n_train']}/{smoke['n_full']:,}장 · 평가 {smoke['n_val']}장 · 1에폭")
    if args.dry_run:
        print("  (--dry-run — 여기서 끝)")
        return

    from ultralytics import YOLO
    if model_yaml:
        model = YOLO(str(model_yaml)).load(str(weights))
    else:
        model = YOLO(str(weights))
    n_params = sum(p.numel() for p in model.model.parameters()) / 1e6
    print(f"  파라미터    : {n_params:.2f} M")

    run = RUNS / name
    log_command(run, {**final, "data": str(data), "weights": str(weights),
                      "model_yaml": str(model_yaml) if model_yaml else None,
                      "params_M": round(n_params, 2)})
    t0 = time.time()
    model.train(data=str(data), device=args.device, project=str(RUNS), name=name,
                exist_ok=True, **final)
    took = time.time() - t0

    if smoke:
        rows = read_results(run)
        ok = bool(rows) and (run / "weights" / "last.pt").exists()
        per_img = None
        if rows and "time" in rows[-1]:
            per_img = float(rows[-1]["time"]) / max(smoke["n_train"], 1)
        rep = dict(ok=ok, params_M=round(n_params, 2), took_s=round(took, 1),
                   results_rows=len(rows), last_pt=(run / "weights" / "last.pt").exists(),
                   est_full_epoch_min=round(per_img * smoke["n_full"] / 60, 1) if per_img else None)
        (run / "smoke_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n=== 연기 실행 {'통과 ✅' if ok else '실패 ❌'} · {now()} ===")
        print(f"  파라미터 {rep['params_M']} M · 걸린 시간 {took/60:.1f}분 · results.csv {len(rows)}줄 · "
              f"last.pt {'있음' if rep['last_pt'] else '없음'}")
        if rep["est_full_epoch_min"]:
            print(f"  전체 1에폭 추정 (대략) : {rep['est_full_epoch_min']}분 — 학습 {smoke['n_full']:,}장 기준")
        sys.exit(0 if ok else 2)

    print(f"\n=== 끝 · {now()} · {took/3600:.1f}시간 · {summarize(run)} ===")
    backup(name)
    print(f"결과: {run}")
    if args.shutdown:
        schedule_shutdown(args.shutdown)


if __name__ == "__main__":
    main()
