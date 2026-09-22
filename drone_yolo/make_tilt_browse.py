"""
make_tilt_browse.py — 마운트각 태그를 눈으로 확인할 폴더 (2026-09-23)

왜
  태그는 **추정**이다 (박스 크기 vs y 상관). 지형 경사·배우 이동이 섞인다.
  사람이 원본을 보고 "이게 정말 비스듬한가" 를 확인할 수 있어야 한다.

왜 원본인가
  학습 크롭은 1280×720 조각이라 **원근이 안 보인다** — 비스듬 여부를 눈으로 못 가린다.
  원본 프레임을 봐야 판단이 된다.

비용
  **심볼릭 링크**라 디스크를 거의 안 쓰고 원본도 안 건드린다. 지우려면 폴더만 지우면 된다.

실행: python make_tilt_browse.py [--per-group 4] [--crops]
출력: data/by_tilt/<태그>/<r값>_<그룹>/…
"""
import argparse, csv, random, re, shutil
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent


def tag_of(r):
    return "1_비스듬_아래큼" if r >= 0.30 else ("2_비스듬_위큼" if r <= -0.30 else "3_수직하향")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-group", type=int, default=4, help="그룹당 원본 프레임 수")
    ap.add_argument("--out", default=str(BASE / "data" / "by_tilt"))
    ap.add_argument("--crops", action="store_true", help="학습 크롭도 전부 링크 (42,448개)")
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)

    sur = {}
    for f in ("survey_tilt.csv", "survey_tilt2.csv"):
        p = BASE / "metrics" / f
        if p.exists():
            for r in csv.DictReader(open(p, encoding="utf-8")):
                sur[r["group"]] = float(r["corr_cy_px"])

    rnd = random.Random(42)
    n = 0
    for g, r in sorted(sur.items(), key=lambda x: -abs(x[1])):
        src, key = g.split(" ", 1)
        files = []
        if src == "NOMAD":
            files = sorted(BASE.glob(f"data/raw/NOMAD*/images/**/{key}_f*.jpg"))
        elif src == "WiSARD":
            d = BASE / "data/raw/WiSARD" / key
            files = sorted(list(d.rglob("*.jpg")) + list(d.rglob("*.jpeg"))) if d.exists() else []
        elif src == "Okutama":
            ds = list(BASE.glob(f"data/raw/okutama/**/Extracted-Frames-1280x720/{key}"))
            files = sorted(ds[0].glob("*.jpg")) if ds else []
        elif src == "VisDrone":
            files = sorted(BASE.glob(f"data/raw/VisDrone/images/*/{key[3:]}_*.jpg"))
        elif src == "SARD":
            files = sorted(BASE.glob(f"data/raw/sard2/search-and-rescue-2/{key}/images/*.jpg"))
        if not files:
            continue
        pick = rnd.sample(files, min(args.per_group, len(files)))
        safe = re.sub(r"[^-\w가-힣.]", "_", g)
        d = out / tag_of(r) / f"r{r:+.2f}_{safe}"
        d.mkdir(parents=True, exist_ok=True)
        for f in pick:
            try:
                (d / f.name).symlink_to(f.resolve())
                n += 1
            except FileExistsError:
                pass

    print(f"원본 프레임 {n:,}개 링크 → {out}")
    for t in sorted(p.name for p in out.iterdir() if p.is_dir()):
        gs = list((out / t).iterdir())
        print(f"  {t:<16} 그룹 {len(gs):>3} · 프레임 {sum(len(list(x.iterdir())) for x in gs):>5}")

    if args.crops:
        cp = out / "_학습크롭"
        c = 0
        for row in csv.DictReader(open(BASE / "metrics/tilt_tags.csv", encoding="utf-8")):
            if row["tilt"] in ("미상", "음성"):
                continue
            d = cp / row["tilt"]
            d.mkdir(parents=True, exist_ok=True)
            s = BASE / row["crop"]
            try:
                (d / s.name).symlink_to(s.resolve()); c += 1
            except FileExistsError:
                pass
        print(f"  학습 크롭 {c:,}개 링크 → {cp}")

    print("\n지우려면: rm -rf", out, "  (심볼릭 링크라 원본은 그대로)")


if __name__ == "__main__":
    main()
