"""
compare_ci.py — 두 모델 AP50 차이의 짝 부트스트랩 95 % 구간 (2026-09-25)

  같은 평가셋 · 같은 블록 재표집으로 두 모델을 동시에 재므로, 평가셋 자체의 난이도 흔들림이 상쇄된다.
  구간이 0 을 포함하지 않으면 "차이가 있다" 로 본다.
  입력: eval_test_v2.py 가 남긴 runs_person/<모델>/eval_<tag>.npz
실행: python compare_ci.py --tag test_obl v6_obl v6_p2m [v3_place:v6_obl …]
"""
import argparse, csv
from pathlib import Path
import numpy as np
from eval_test_v2 import ap101, blocks_of

BASE = Path(__file__).resolve().parent


def load(name, tag):
    if "@" in name:                      # 같은 모델의 다른 평가 방식 비교: 모델@태그
        name, tag = name.split("@")
    d = np.load(BASE / "runs_person" / name / f"eval_{tag}.npz", allow_pickle=False)
    return d["s"], d["t"].astype(bool), d["img"], d["ngt"], [str(x) for x in d["names"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("pairs", nargs="+", help="기준:비교 (예: v6_obl:v6_p2m)")
    args = ap.parse_args()
    rows = []
    for pr in args.pairs:
        a, b = pr.split(":")
        A, Bm = load(a, args.tag), load(b, args.tag)
        assert A[4] == Bm[4], "두 평가의 이미지 목록이 다르다"
        blk, nblk, ngrp = blocks_of(A[4])
        rng = np.random.default_rng(0)
        pa, pb = blk[A[2]], blk[Bm[2]]
        ia = [np.where(pa == k)[0] for k in range(nblk)]; ib = [np.where(pb == k)[0] for k in range(nblk)]
        gt = np.bincount(blk, weights=A[3], minlength=nblk)
        d = []
        for _ in range(args.B):
            pick = rng.integers(0, nblk, nblk)
            xa = np.concatenate([ia[k] for k in pick]); xb = np.concatenate([ib[k] for k in pick])
            g = gt[pick].sum()
            d.append(ap101(Bm[0][xb], Bm[1][xb], g) - ap101(A[0][xa], A[1][xa], g))
        d = np.array(d)
        full = ap101(Bm[0], Bm[1], A[3].sum()) - ap101(A[0], A[1], A[3].sum())
        lo, hi = np.percentile(d, [2.5, 97.5])
        verdict = "차이 있음" if lo > 0 or hi < 0 else "구분 안 됨"
        print(f"{args.tag}  {b} − {a} = {full*100:+.1f} %p  (95 % {lo*100:+.1f} ~ {hi*100:+.1f} %p · 블록 {nblk}) → {verdict}")
        rows.append([args.tag, a, b, round(full, 4), round(lo, 4), round(hi, 4), nblk, verdict])
    out = BASE / "metrics" / f"compare_ci_{args.tag}.csv"
    new = not out.exists()
    with open(out, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["tag", "base", "model", "diff", "lo", "hi", "blocks", "verdict"])
        w.writerows(rows)


if __name__ == "__main__":
    main()
