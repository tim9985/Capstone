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


def enable_degrade(p_resize=0.2, s_resize=(0.5, 0.9), p_blur=0.3, s_blur=(0.3, 1.2), p_noise=0.2, s_noise=(2, 8), p_jpeg=0.5, q_jpeg=(30, 95)):
    """화질 열화 증강 (10-31 「C」) — 실기체 실시간 전송 (RunCam WiFiLink 2-G · H.264 4~8 Mbps) 흉내

    근거: pose_bench (10-31) — 배포 v12 를 JPEG 30 으로 다시 저장하면 Okutama 누움 뜸 −9.8 · 앉음 −7.6 · 서기 −7.2 %p
          (v12 학습 창은 「확대 금지 · 면적 평균 축소」 로 매끈 → 블록 · 잡음 · 흐림을 본 적 없음)
    학습 증강 RandomHSV.apply_image 뒤에 차례대로: 축소 뒤 확대 → 흐림 → 노이즈 → JPEG 재압축 (마지막 = 카메라 → 인코더 순서)
    검증 변환엔 RandomHSV 가 없어 안 걸림 · 판정은 학습에 안 쓴 WebP 로 (pose_bench w30)
    train_person.py --degrade 로 켬 (데이터 로더 프로세스는 fork 라 같이 적용)
    """
    import random
    import cv2
    from ultralytics.data.augment import RandomHSV
    orig = RandomHSV.apply_image

    def apply_image(self, labels, params=None):
        labels = orig(self, labels, params)
        img = labels["img"]
        if img.ndim != 3 or img.shape[-1] != 3:
            return labels
        h, w = img.shape[:2]
        if random.random() < p_resize:
            s = random.uniform(*s_resize)
            img = cv2.resize(cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA), (w, h), interpolation=cv2.INTER_LINEAR)
        if random.random() < p_blur:
            img = cv2.GaussianBlur(img, (0, 0), random.uniform(*s_blur))
        if random.random() < p_noise:
            img = np.clip(img.astype(np.float32) + np.random.normal(0.0, random.uniform(*s_noise), img.shape), 0, 255).astype(np.uint8)
        if random.random() < p_jpeg:
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(random.uniform(*q_jpeg))])
            if ok:
                img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        labels["img"] = np.ascontiguousarray(img)
        return labels

    RandomHSV.apply_image = apply_image
    print(f"  증강        : 화질 열화 (aug_ext · 축소 확대 p {p_resize} · 흐림 p {p_blur} · 노이즈 p {p_noise} · JPEG {q_jpeg[0]}~{q_jpeg[1]} p {p_jpeg})")
