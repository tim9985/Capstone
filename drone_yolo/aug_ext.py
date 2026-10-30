"""
aug_ext.py — 학습 증강 고치기 (10-30 · 누운 사람 강건성 · 파이프라인 점검)

  타원 박스 (Kalra et al., "Towards Rotation Invariance in Object Detection", ICCV 2021)
    ultralytics RandomPerspective.apply_bboxes 는 박스 네 꼭짓점을 돌린 뒤 감싸는 「가장 큰 박스」 를 라벨로 씀
    → ±10° 만 돌려도 누운 사람 (가로/세로 3) 박스 세로 ×1.51 · 서 있는 사람 (0.4) 가로 ×1.42 로 부풀어 같은 사람 라벨이 흔들림
    → 박스에 내접하는 타원 위 점 32개를 돌려 감쌈 (같은 10° 에서 누움 세로 ×1.11) · 회전 없을 땐 원래와 같음 (축 정렬 타원의 감싸는 박스 = 원래 박스)
  train_person.py --ellipse-box 로 켬 (데이터 로더 프로세스는 fork 라 같이 적용)
"""
import numpy as np


def enable_ellipse_box(k=32):
    from ultralytics.data.augment import RandomPerspective
    t = np.linspace(0, 2 * np.pi, k, endpoint=False)
    ct, st = np.cos(t), np.sin(t)

    def apply_bboxes(self, bboxes, M):
        n = len(bboxes)
        if n == 0:
            return bboxes
        cx, cy = (bboxes[:, 0] + bboxes[:, 2]) / 2, (bboxes[:, 1] + bboxes[:, 3]) / 2
        a, b = (bboxes[:, 2] - bboxes[:, 0]) / 2, (bboxes[:, 3] - bboxes[:, 1]) / 2
        xs = cx[:, None] + a[:, None] * ct[None]; ys = cy[:, None] + b[:, None] * st[None]
        xy = np.ones((n * k, 3), dtype=bboxes.dtype)
        xy[:, 0], xy[:, 1] = xs.ravel(), ys.ravel()
        xy = xy @ M.T
        xy = (xy[:, :2] / xy[:, 2:3] if self.perspective else xy[:, :2]).reshape(n, k, 2)
        return np.stack([xy[..., 0].min(1), xy[..., 1].min(1), xy[..., 0].max(1), xy[..., 1].max(1)], 1).astype(bboxes.dtype)

    RandomPerspective.apply_bboxes = apply_bboxes
    print("  증강        : 회전 박스 = 타원 방식 (aug_ext · Kalra ICCV 2021)")
