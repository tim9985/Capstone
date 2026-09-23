"""
train_person.py — 사람 탐지 모델 학습 (2단계)

  stage1 : VisDrone 가중치 → NOMAD 실데이터 파인튜닝
  stage2 : stage1 결과 → NOMAD + 합성데이터 혼합 재파인튜닝 (쓰러진 자세 보강)

핵심 설계
  · 검증셋은 **항상 NOMAD 실데이터만** 쓴다. 합성 데이터로 검증하면 마네킹 질감에
    과적합된 모델이 좋아 보이는 착시가 생긴다.
  · stage2 의 합성 비율은 13% 수준(합성 411 / 전체 3068)으로 제한한다. 합성이 많으면
    금속 마네킹 질감을 학습해 실제 성능이 떨어진다.
  · 학습 해상도 = 추론 해상도(운용 조건). 크롭이 1280x720, 사람 94px 기준이므로
    imgsz 960 학습 → 추론도 960 으로 맞춘다.
  · NOMAD 는 여름·정오·미국 시골로 조명/계절이 단일하다. 이를 보완하려 색상·밝기
    증강(hsv_*)을 기본값보다 강하게 준다.

실행:
  python train_person.py --stage 1
  python train_person.py --stage 2                # stage1 최종 가중치에서 이어서
  python train_person.py --stage 1 --epochs 40 --batch 4 --imgsz 960
"""
import argparse
import shutil
import sys
from pathlib import Path

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


def schedule_shutdown(sec):
    """학습이 정상 완료된 뒤에만 호출된다. 취소: 다른 터미널에서 shutdown /a"""
    import subprocess
    print(f"\n※ {sec}초 뒤 컴퓨터를 종료합니다. 취소하려면 다른 터미널에서:  shutdown /a")
    try:
        subprocess.run(["shutdown", "/s", "/t", str(sec),
                        "/c", "Claude Code: 학습 완료로 자동 종료"], check=True)
    except Exception as e:
        print(f"종료 예약 실패(무시): {e}")


def write_mixed_yaml(repeat=3):
    """stage2용 data.yaml — 실데이터 전체 train + 합성 train, 검증은 **실데이터만**.

    합성을 검증에 넣으면 "자기가 만든 그림체를 자기가 알아보는가"를 재게 되어
    수치가 부풀려진다. 실사에서 좋아져야 의미가 있다.

    repeat: 합성 데이터를 몇 배로 넣을지. 합성 411장은 실사 14,342장의 2.8%뿐이라
      그대로 넣으면 효과가 있어도 묻힌다. 3배(약 8%)면 드러날 만하면서도
      합성 편향이 실사를 덮지 않는다.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, choices=(1, 2), required=True)
    ap.add_argument("--synth-repeat", type=int, default=3,
                    help="stage2 에서 합성 데이터를 몇 배로 넣을지 (1이면 원본 그대로)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--batch", default=-1,
                    help="-1 이면 VRAM 60%% 목표로 자동. 0<x<1 소수면 그 비율을 목표로 자동"
                         "(예: 0.85 → 3090 24GB의 85%%). 정수면 고정 배치")
    ap.add_argument("--weights", default=None, help="시작 가중치 직접 지정")
    ap.add_argument("--rect", action="store_true",
                    help="직사각 배치(패딩 최소) — 추론 엔진 [736,1280] 과 모양을 맞춘다 (09-23)")
    ap.add_argument("--model-yaml", default=None,
                    help="구조 파일(예: configs/models/yolo11m-p2.yaml). --weights 는 호환 층만 이식")
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--device", default=0)
    # 아래 두 값은 노트북(RTX 3050 4GB · 4코어) 기준으로 코드에 박혀 있었다.
    # 서버(i9-11900 8코어16스레드 · RAM 64GB)에서는 크게 낭비되므로 인자로 뺀다.
    # 결과에는 영향이 없고 에폭당 시간만 줄인다 — 실험 비교의 공정성은 유지된다.
    ap.add_argument("--workers", type=int, default=8,
                    help="데이터 로더 워커 수 (노트북 4 · 서버 8)")
    ap.add_argument("--cache", default="disk",
                    choices=("False", "disk", "ram"),
                    help="이미지 캐시. 'ram' 은 9,008장에 ~24GB 필요하니 여유를 확인할 것")
    ap.add_argument("--resume", action="store_true",
                    help="중단된 학습을 last.pt 에서 이어서 진행")
    ap.add_argument("--data", default=None,
                    help="data.yaml 직접 지정 (예: data_nomad20.yaml — 배우 20명 재학습)")
    ap.add_argument("--name", default=None, help="결과 폴더명 직접 지정")
    # 파인튜닝에서 학습률은 결정적이다. 기본 0.01 은 처음부터 학습할 때의 값이라
    # 사전학습 특징을 초기 몇 에폭 만에 망가뜨린다(파국적 망각).
    # 자세 2클래스 시도(pose2)가 그래서 실패했다 — 박스 손실이 1.2 → 1.9 로 악화.
    ap.add_argument("--lr0", type=float, default=None,
                    help="시작 학습률. 잘 되는 모델을 이어받을 때는 0.001 이하를 쓴다")
    ap.add_argument("--freeze", type=int, default=None,
                    help="앞쪽 N개 층을 고정한다. 특징 추출부를 보존하고 헤드만 학습시킬 때")
    # 마감이 정해진 학습용. ultralytics 가 첫 에폭 속도를 보고 에폭 수와 학습률 스케줄을
    # 다시 맞추므로, 중간에 잘리는 게 아니라 줄어든 스케줄로 끝까지 마무리된다.
    ap.add_argument("--time", type=float, default=None,
                    help="최대 학습 시간(시간 단위). 지정하면 --epochs 를 덮어쓴다")
    ap.add_argument("--scale", type=float, default=None,
                    help="학습 중 크기 증강 (stage1 기본 0.5). 전처리에서 크기 분포를 이미 넓혔으면 0.3 (계획 4절)")
    ap.add_argument("--close-mosaic", type=int, default=None,
                    help="마지막 N 에폭은 모자이크를 끈다 (ultralytics 기본 10). 에폭이 적을 때 줄인다")
    ap.add_argument("--translate", type=float, default=None,
                    help="학습 중 위치 증강 (stage1 기본 0.15). 타일 추론을 쓰면 사람이 타일 가장자리에도 "
                         "오므로 키워 본다 — 저장된 크롭은 가장자리 100 px 안에 1.8 %% 뿐이다 (09-20)")
    # optimizer=auto(기본)면 ultralytics 가 --lr0 를 무시하고 스스로 고른다 (09-12 로그로 확인).
    # 학습률을 직접 정해야 하는 이어 학습에서는 옵티마이저를 명시한다.
    ap.add_argument("--optimizer", default=None,
                    help="옵티마이저 이름 (예: SGD, MuSGD, AdamW). 지정해야 --lr0 가 적용된다")
    ap.add_argument("--warmup-epochs", type=float, default=None,
                    help="워밍업 에폭 (ultralytics 기본 3). 중간 가중치에서 이어갈 때는 줄인다")
    ap.add_argument("--shutdown", type=int, default=0, metavar="SEC",
                    help="학습 정상 완료 후 지정 초 뒤 컴퓨터 종료 (예: --shutdown 120). "
                         "취소는 다른 터미널에서 'shutdown /a'")
    args = ap.parse_args()

    from ultralytics import YOLO

    # ── 이어 학습 ──
    if args.resume:
        # --name 을 준 실행(예: stage1_nomad20)을 이어받을 수 있어야 한다
        name = args.name or ("stage1_nomad" if args.stage == 1 else "stage2_synth")
        last = RUNS / name / "weights" / "last.pt"
        if not last.exists():
            raise SystemExit(f"이어받을 체크포인트가 없습니다: {last}")
        print(f"=== stage{args.stage} 이어 학습: {last} ===")
        resume_kw = {}
        if args.time is not None:
            # 체크포인트에 저장된 time(시간 제한)을 덮어쓴다. resume 하면 시간 측정이 0부터 다시
            # 시작돼 에폭 수가 재계산되므로, 0 을 주면 시간 제한 없이 저장된 epochs 까지 정확히 간다.
            resume_kw["time"] = args.time
            print(f"  시간 제한 덮어쓰기: {args.time}")
        YOLO(str(last)).train(resume=True, **resume_kw)
        best = RUNS / name / "weights" / "best.pt"
        if best.exists():
            # 실험 이름(name)에 이미 모델 정보가 들어있다(예: e7_11m_1280).
            # "yolov8s_" 를 고정으로 붙이면 yolo11m/l 로 학습해도 파일명이
            # yolov8s 라고 나와 오해를 준다(2026-09-11 발견) — 접두사 제거.
            dest = BASE_DIR / "weights" / f"{name}.pt"
            shutil.copy(best, dest)
            print(f"\n최종 가중치 복사: {dest}")
        if args.shutdown:
            schedule_shutdown(args.shutdown)
        return

    if args.stage == 1:
        data = NOMAD_YAML
        weights = Path(args.weights) if args.weights else BASE_WEIGHTS
        epochs = args.epochs or 60
        name = "stage1_nomad"
        # 단일 도메인(여름·정오)이라 색상 증강을 기본보다 강하게.
        # degrees=180/flipud=0.5 — 하향 90° 시점엔 화면에 "위쪽"이 없다.
        # 드론이 요잉하면 장면 전체가 돌고, 쓰러진 사람은 어느 방향으로든
        # 누울 수 있다. 기존 ±10°·flipud 0 은 이 물리적 조건과 안 맞았다
        # (SERVER.md 5절 실험 A). 과적합 완화 효과도 겸한다 — train/val
        # loss가 15에폭 근처부터 벌어지는 패턴이 E1·E2에서 보였는데,
        # 방향 다양성 부족이 원인 중 하나로 보인다.
        extra = dict(hsv_h=0.02, hsv_s=0.8, hsv_v=0.5, degrees=180.0,
                     flipud=0.5, translate=0.15, scale=0.5, fliplr=0.5, mosaic=1.0)
    else:
        data = write_mixed_yaml(args.synth_repeat)
        default_w = RUNS / "stage1_all" / "weights" / "best.pt"
        weights = Path(args.weights) if args.weights else default_w
        if not weights.exists():
            raise SystemExit(f"stage1 가중치가 없습니다: {weights}\n먼저 --stage 1 을 실행하세요.")
        epochs = args.epochs or 8
        name = "stage2_synth"
        # 이어 학습이므로 증강을 약하게, 학습률도 낮게
        extra = dict(hsv_h=0.015, hsv_s=0.6, hsv_v=0.4, degrees=8.0,
                     translate=0.1, scale=0.4, fliplr=0.5, mosaic=0.5,
                     lr0=0.002)

    if args.data:
        data = Path(args.data)
    if args.name:
        name = args.name

    if not Path(data).exists():
        raise SystemExit(f"데이터 설정 없음: {data}")
    if not weights.exists():
        raise SystemExit(f"가중치 없음: {weights}")

    print(f"=== stage{args.stage} 학습 ===")
    print(f"  시작 가중치 : {weights}")
    print(f"  데이터      : {data}")
    print(f"  imgsz {args.imgsz} / epochs {epochs} / batch {args.batch} / patience {args.patience}")

    if args.lr0 is not None:
        extra["lr0"] = args.lr0
        print(f"  학습률     : lr0 {args.lr0} (기본 0.01 대신 — 사전학습 보존)")
    if args.freeze is not None:
        extra["freeze"] = args.freeze
        print(f"  고정        : 앞 {args.freeze}개 층")
    if args.time is not None:
        extra["time"] = args.time
        print(f"  시간 제한   : {args.time}시간 (epochs 무시, 스케줄 자동 조정)")
    if args.close_mosaic is not None:
        extra["close_mosaic"] = args.close_mosaic
        print(f"  모자이크 끔 : 마지막 {args.close_mosaic}에폭")
    if args.scale is not None:
        extra["scale"] = args.scale
        print(f"  크기 증강   : scale {args.scale}")
    if args.translate is not None:
        extra["translate"] = args.translate
        print(f"  위치 증강   : translate {args.translate}")
    if args.optimizer is not None:
        extra["optimizer"] = args.optimizer
        print(f"  옵티마이저  : {args.optimizer}")
    if args.warmup_epochs is not None:
        extra["warmup_epochs"] = args.warmup_epochs
        print(f"  워밍업      : {args.warmup_epochs}에폭")

    # -1(문자열/정수 모두 허용) → 그대로, 소수(0~1) → VRAM 비율 목표, 그 외 → 정수 배치
    batch_str = str(args.batch)
    if batch_str == "-1":
        batch_val = -1
    elif "." in batch_str:
        batch_val = float(batch_str)
    else:
        batch_val = int(batch_str)

    if args.model_yaml:
        # 파일명에 크기(m)가 있어야 한다 — 없으면 nano 로 만들어진다 (09-23 사고)
        model = YOLO(args.model_yaml).load(str(weights))
    else:
        model = YOLO(str(weights))
    if args.rect:
        extra["rect"] = True
    model.train(
        data=str(data),
        epochs=epochs,
        imgsz=args.imgsz,
        batch=batch_val,
        device=args.device,
        project=str(RUNS),
        name=name,
        exist_ok=True,
        patience=args.patience,
        amp=True,
        cache=False if args.cache == "False" else args.cache,
        workers=args.workers,
        seed=42,
        val=True,
        plots=True,
        **extra,
    )

    best = RUNS / name / "weights" / "best.pt"
    if best.exists():
        dest = BASE_DIR / "weights" / f"{name}.pt"   # 이어 학습 분기와 같은 규칙 (모델 정보는 name 에)
        shutil.copy(best, dest)
        print(f"\n최종 가중치 복사: {dest}")
    print(f"결과: {RUNS / name}")
    if args.shutdown:
        schedule_shutdown(args.shutdown)


if __name__ == "__main__":
    main()
