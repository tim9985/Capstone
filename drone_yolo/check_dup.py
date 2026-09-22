"""
check_dup.py — 크롭 묶음의 실제 중복률 (2026-09-21)

왜
  하드 네거티브는 연속 영상 프레임에서 캐낸다 → 이웃 프레임끼리 같은 그림일 수 있다.
  프레임 번호 근접으로 세면 과대평가된다 (프레임마다 목표 px 를 새로 뽑아 배율이 갈리기 때문).
  → **픽셀로** 잰다: 8×8 평균 해시(aHash)의 해밍 거리.

실행: python check_dup.py data/det_neg/images/train
"""
import sys, glob, collections
import cv2, numpy as np

d = sys.argv[1] if len(sys.argv) > 1 else "data/det_neg/images/train"
fs = sorted(glob.glob(f"{d}/*.jpg"))
if not fs:
    raise SystemExit(f"이미지가 없다: {d}")
H = []
for f in fs:
    im = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    if im is None:
        continue
    s = cv2.resize(im, (8, 8), interpolation=cv2.INTER_AREA)
    H.append((s > s.mean()).flatten())
H = np.array(H, dtype=np.uint8); n = len(H)
print(f"{d} — {n:,}장")

dup_edges = []
CH = 2000                                   # 쌍 행렬을 통째로 만들면 메모리가 터진다
for i0 in range(0, n, CH):
    a = H[i0:i0 + CH]
    for j0 in range(i0, n, CH):
        b = H[j0:j0 + CH]
        dd = (a[:, None, :] != b[None, :, :]).sum(2)
        for x, y in zip(*np.where(dd <= 5)):
            gi, gj = i0 + x, j0 + y
            if gi < gj:
                dup_edges.append((gi, gj))

par = list(range(n))
def find(a):
    while par[a] != a:
        par[a] = par[par[a]]; a = par[a]
    return a
for i, j in dup_edges:
    a, b = find(i), find(j)
    if a != b:
        par[a] = b
sizes = collections.Counter(find(i) for i in range(n))
dup = sum(v - 1 for v in sizes.values() if v > 1)
print(f"  해밍 ≤5 쌍 {len(dup_edges):,} · 군집 {len(sizes):,}개")
print(f"  버려도 되는 장수 {dup:,} / {n:,} = **{dup / n:.1%}**")
