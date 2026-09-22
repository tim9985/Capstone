"""
make_place_split.py — 장소 분리 학습 목록 (2026-09-22)

왜
  NFR-V03 은 "학습과 **장소 분리**" 를 요구하는데 지금 학습셋은 WiSARD 를 **비행 단위**로만 갈랐다.
  같은 장소의 다른 비행이 train/val 양쪽에 있어서 val 점수가 낙관적이다.
  평가셋 `test_v2`(Carnation · Karen)를 만들었으므로, 그 두 장소를 **학습에서 뺀다.**

  크롭을 다시 만들 필요는 없다 — 크롭 파일명에 비행명이 들어 있어
  **목록(.txt)에서 걸러내면** 된다. ultralytics 는 train/val 에 .txt 목록을 받는다.

실행: python make_place_split.py
출력: configs/lists/*.txt · configs/data_v3_place.yaml
"""
import argparse, re
from pathlib import Path

BASE = Path(__file__).resolve().parent
EVAL_PLACES = ("Carnation", "Karen")          # make_testset_v2.py 와 같아야 한다
SETS = ["nomad_actor01_10", "nomad_actor11_20", "nomad_actor21_30",
        "nomad_actor_sel31_100", "wisard"]


def collect(root: Path, split: str):
    d = root / "images" / split
    return sorted(p for p in d.glob("*.jpg")) if d.exists() else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(BASE / "data" / "det_fov"))
    ap.add_argument("--neg", default=str(BASE / "data" / "det_neg"))
    ap.add_argument("--with-neg", action="store_true", help="하드 네거티브도 포함 (v3a 설정)")
    ap.add_argument("--out", default=str(BASE / "configs"))
    args = ap.parse_args()

    pat = re.compile("|".join(EVAL_PLACES))
    out = Path(args.out); (out / "lists").mkdir(parents=True, exist_ok=True)

    kept, dropped = [], 0
    for s in SETS:
        for p in collect(Path(args.data) / s, "train"):
            if pat.search(p.name):
                dropped += 1
            else:
                kept.append(p)
    if args.with_neg:
        for p in collect(Path(args.neg), "train"):
            if pat.search(p.name):
                dropped += 1
            else:
                kept.append(p)

    val = []
    for s in SETS:
        val += collect(Path(args.data) / s, "val")
    val_bad = sum(1 for p in val if pat.search(p.name))

    tag = "neg" if args.with_neg else "base"
    tr = out / "lists" / f"train_v3_place_{tag}.txt"
    va = out / "lists" / "val_v3_place.txt"
    tr.write_text("\n".join(str(p) for p in kept) + "\n")
    va.write_text("\n".join(str(p) for p in val) + "\n")

    yml = out / f"data_v3_place_{tag}.yaml"
    yml.write_text(
        f"# 장소 분리 학습 (2026-09-22) — 평가 전용 장소 {list(EVAL_PLACES)} 를 train 에서 뺐다\n"
        f"# 평가셋: data/test_v2 (같은 두 장소 · 1920x1080 전체 프레임)\n"
        f"# 크롭은 재생성하지 않았다 — 목록에서 걸러냈을 뿐이다\n"
        f"train: {tr}\nval: {va}\nnc: 1\nnames: ['person']\n")

    print(f"학습 {len(kept):,}장 (평가 장소 {dropped:,}장 제외)")
    print(f"검증 {len(val):,}장 · 그중 평가 장소 {val_bad}장 {'⚠ 누수!' if val_bad else '(없음 ✓)'}")
    print(f"→ {tr}\n→ {va}\n→ {yml}")


if __name__ == "__main__":
    main()
