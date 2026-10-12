"""
compare_means.py — 반복 여러 개의 **평균 AP** 끼리 짝 부트스트랩 (10-12 · v12 판정)

  왜  수프 하나 (soup_v7r2) 를 기준으로 삼으면 기준이 「운 좋은 회차」 에 묶인다
      — v7 같은 설정 4회 test_obl 0.604 ~ 0.631 · soup_v7r2 에 든 1회차가 가장 높다 · 수프는 BN 이 어긋나면 못 만든다
  방법 같은 블록 재표집으로 그룹마다 모델별 AP 를 재고 그룹 평균의 차 (B − A) · 2,000회 → 95 % 구간
      출력 줄은 compare_ci 와 같은 꼴 → judge_pairs.py 가 그대로 읽는다 (이름 = 평균[모델+모델])
실행: python compare_means.py --tag test_obl v7,v7_r2,v7_r3,v7_r4 v12_f91,v12_f92
"""
import argparse
import csv
from pathlib import Path

import numpy as np

from compare_ci import load
from eval_test_v2 import ap101, blocks_of

BASE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("a"); ap.add_argument("b")
    a = ap.parse_args()
    A, Bm = [load(m, a.tag) for m in a.a.split(",")], [load(m, a.tag) for m in a.b.split(",")]
    names = A[0][4]
    assert all(x[4] == names for x in A + Bm), "이미지 목록이 다르다"
    blk, nblk, _ = blocks_of(names)
    gt = np.bincount(blk, weights=A[0][3], minlength=nblk)
    def idx(m):
        p = blk[m[2]]; return [np.where(p == k)[0] for k in range(nblk)]
    IA, IB = [idx(m) for m in A], [idx(m) for m in Bm]
    rng = np.random.default_rng(0); d = []
    for _ in range(a.B):
        pick = rng.integers(0, nblk, nblk); g = gt[pick].sum()
        ma = np.mean([ap101(m[0][np.concatenate([ii[k] for k in pick])], m[1][np.concatenate([ii[k] for k in pick])], g) for m, ii in zip(A, IA)])
        mb = np.mean([ap101(m[0][np.concatenate([ii[k] for k in pick])], m[1][np.concatenate([ii[k] for k in pick])], g) for m, ii in zip(Bm, IB)])
        d.append(mb - ma)
    n = A[0][3].sum()
    full = np.mean([ap101(m[0], m[1], n) for m in Bm]) - np.mean([ap101(m[0], m[1], n) for m in A])
    lo, hi = np.percentile(d, [2.5, 97.5])
    na, nb = f"평균[{a.a.replace(',', '+')}]", f"평균[{a.b.replace(',', '+')}]"
    verdict = "차이 있음" if lo > 0 or hi < 0 else "구분 안 됨"
    print(f"{a.tag}  {nb} − {na} = {full*100:+.1f} %p  (95 % {lo*100:+.1f} ~ {hi*100:+.1f} %p · 블록 {nblk}) → {verdict}")
    out = BASE / "metrics" / f"compare_means_{a.tag}.csv"; new = not out.exists()
    with open(out, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["tag", "base_group", "model_group", "diff", "lo", "hi", "blocks", "verdict"])
        w.writerow([a.tag, a.a, a.b, round(full, 4), round(lo, 4), round(hi, 4), nblk, verdict])


if __name__ == "__main__":
    main()
