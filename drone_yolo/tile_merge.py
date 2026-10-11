"""
tile_merge.py — 타일 합치기: 안쪽 경계에서 잘린 박스 지우기 (10-11 · v7 넘기 검토 I1)

  문제  1920×1080 → 1280×720 타일 4장 (겹침 50 %) · 사람이 한 타일의 안쪽 경계에 걸리면 그 타일은 잘린 박스, 이웃 타일은 온전한 박스를 낸다
        잘린 박스는 넓이가 0.4 배쯤이라 온전한 박스와 IoU < 0.6 → NMS 0.6 이 못 지워 확신 높은 오탐으로 남는다
  규칙  NMS 0.6 뒤에, 자기 타일의 안쪽 경계 (이미지 테두리가 아닌 쪽) 에 3 px 안으로 닿은 박스가
        다른 타일의 더 큰 박스 안에 60 % 이상 들어가면 지운다 · 그 사람의 점수는 큰 박스로 넘긴다 (max)
        겹침이 50 % 라 640×360 px 보다 작은 사람은 늘 어느 한 타일에 온전히 들어간다
  고른 법  soup_v7r2 가 학습하지 않은 1920×1080 셋 (test_ladd · test_nii · test_cy) 에서 규칙 3개 비교 → 이것 하나를 평가셋에 한 번 (diag_tile_merge.py)
"""
import numpy as np

W, H, TW, TH = 1920, 1080, 1280, 720
TILES = [(x, y) for y in (0, H - TH) for x in (0, W - TW)]      # eval_test_v2.TILES 와 같은 순서


def ios(a, B):
    """a 가 B 각각에 들어간 비율 = 교집합 / a 넓이"""
    x1 = np.maximum(a[0], B[:, 0]); y1 = np.maximum(a[1], B[:, 1]); x2 = np.minimum(a[2], B[:, 2]); y2 = np.minimum(a[3], B[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    return inter / max(float((a[2] - a[0]) * (a[3] - a[1])), 1e-6)


def touches(b, t, iw=W, ih=H, e=3.0):
    """박스 b 가 타일 t 의 안쪽 경계에 e px 안으로 닿았나 (t < 0 = 타일 아님)"""
    if t < 0:
        return False
    x0, y0 = TILES[t]
    return bool((x0 > 0 and b[0] - x0 <= e) or (x0 + TW < iw and x0 + TW - b[2] <= e)
                or (y0 > 0 and b[1] - y0 <= e) or (y0 + TH < ih and y0 + TH - b[3] <= e))


def edge_merge(b, s, t, iw=W, ih=H, e=3.0, thr=0.6):
    """NMS 를 마친 박스 b (xyxy · 원본 좌표) · 점수 s · 타일 번호 t → 잘린 박스를 뺀 (b, s)"""
    if (iw, ih) != (W, H) or not len(b):
        return b, s
    s = s.copy(); area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]); drop = np.zeros(len(b), bool)
    for i in np.argsort(area):                                       # 작은 (잘린) 박스부터
        if not touches(b[i], t[i], iw, ih, e):
            continue
        c = np.flatnonzero((t != t[i]) & ~drop & (area > area[i]))
        if len(c):
            r = ios(b[i], b[c])
            if r.max() >= thr:
                j = c[np.argmax(r)]; s[j] = max(s[j], s[i]); drop[i] = True
    return b[~drop], s[~drop]
