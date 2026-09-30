"""
train_posture_cls.py — 자세 분류기 학습 (상태 인지 A2 · 2026-09-30)

  크롭(make_posture_crops.py) 묶음을 합쳐 YOLO11-cls 를 학습한다. 탐지기는 건드리지 않는다.
  val = AI-Hub 산악8 (학습 장소와 다름) · 판정은 eval_posture.py 로 **장소 분리 평가셋**에서만

묶음 (한 번에 하나만 바꾼다)
  s1_aihub       AI-Hub 5곳 (수직 90° · 한국)
  s1_aihub_sard  + SARD 전량 (비스듬 · 사람이 붙인 자세)

실행
  python train_posture_cls.py --set s1_aihub [--model yolo11s-cls.pt] [--epochs 30] [--smoke]
  --smoke : 학습 1 % · 1에폭 — 체인 걸기 전 연기 실행 (실패하면 exit 2)
출력
  runs_posture/<이름>/weights/best.pt → weights/<이름>.pt 로 복사 · results.csv 등 → metrics/train_runs/<이름>/
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
CLS = BASE / "data" / "pose_cls"
SETS = {"s1_aihub": ["aihub"], "s1_aihub_sard": ["aihub", "sard"]}
CLASSES = ("lying", "sitting", "standing")


def build(name, parts):
    """묶음 폴더 = 원천 크롭의 하드링크 (복사 없음) · val 은 AI-Hub 산악8."""
    root = CLS / name
    if root.exists():
        shutil.rmtree(root)
    for split in ("train", "val"):
        for c in CLASSES:
            (root / split / c).mkdir(parents=True, exist_ok=True)
    n = 0
    for p in parts:
        for c in CLASSES:
            for f in (CLS / p / "train" / c).glob("*.jpg"):
                os.link(f, root / "train" / c / f"{p}_{f.name}")
                n += 1
    for c in CLASSES:
        for f in (CLS / "aihub" / "val" / c).glob("*.jpg"):
            os.link(f, root / "val" / c / f.name)
    return root, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, choices=list(SETS))
    ap.add_argument("--model", default="yolo11s-cls.pt")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=128)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--name", default=None)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    from ultralytics import YOLO
    name = a.name or a.set + ("_smoke" if a.smoke else "")
    root, n = build(a.set + ("_smoke" if a.smoke else ""), SETS[a.set])
    print(f"묶음 {root.name}: 학습 크롭 {n:,}")
    if n == 0:
        print("학습 크롭 0 — 중단")
        sys.exit(2)
    model = YOLO(a.model)
    try:
        model.train(data=str(root), epochs=1 if a.smoke else a.epochs, imgsz=a.imgsz, batch=a.batch,
                    fraction=0.01 if a.smoke else 1.0, workers=8, project=str(BASE / "runs_posture"), name=name,
                    exist_ok=True, fliplr=0.5, flipud=0.0, scale=0.25, erasing=0.2, seed=0, deterministic=False,
                    plots=False, verbose=False)
    except Exception as e:                                   # 연기 실행에서 체인이 알 수 있게
        print(f"학습 실패: {e}")
        sys.exit(2)
    run = BASE / "runs_posture" / name
    best = run / "weights" / "best.pt"
    if not best.exists():
        print("best.pt 없음 — 중단")
        sys.exit(2)
    if not a.smoke:                                          # 대여 서버 — 바로 백업
        (BASE / "weights").mkdir(exist_ok=True)
        shutil.copy2(best, BASE / "weights" / f"{name}.pt")
        dst = BASE / "metrics" / "train_runs" / name
        dst.mkdir(parents=True, exist_ok=True)
        for f in ("results.csv", "args.yaml"):
            if (run / f).exists():
                shutil.copy2(run / f, dst / f)
        (dst / "command.txt").write_text(" ".join(sys.argv) + "\n")
    print(f"완료 {best}")


if __name__ == "__main__":
    main()
