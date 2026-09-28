"""
geo_resolver.py — 탐지 박스 → 실좌표 (설계명세서 4.4 GeoResolver · 2026-09-28)

  깊이 센서 없이 **박스 아래 변 가운데(발끝) 광선과 지면의 교차점**을 좌표로 낸다.
    ① 발끝 화소 → 카메라 광선        f = (W/2)/tan(HFOV/2)  (88° 렌즈: 수평 80.2° · f 1,141 px)
    ② 카메라 → 기체 → NED 회전        짐벌 안정화면 기체 roll·pitch 를 빼고 짐벌 절대각만 쓴다
    ③ 지면과 교차                     평지(이륙점 높이) 또는 지형 함수 terrain(n, e) → 반복 교차
    ④ NED 오프셋 → 위경도             coord_transform.ned_to_gps (소규모 근사)
  좌표보류(PENDING) — 버리지 않고 사유를 남긴다 (UC-1204)
    · 광선이 지평선 위로 감 · 교차 거리가 너무 멂 · 발끝이 화면 아래 끝에 걸림(잘렸을 수 있음)
    · 기체 기울기 > 6° 또는 자세 변화 > 1.5°/프레임 (NFR-V04 안정 프레임 조건)
  오차 타원 — GPS · 고도 · 짐벌 피치 · 방위 오차를 편미분으로 전파 (coord_error.py 와 같은 식)

좌표계: NED (x 북 · y 동 · z 아래) · 기체 x 앞 · y 오른쪽 · z 아래 · 카메라 x 광축 · y 오른쪽 · z 아래
       카메라 pitch 는 수평에서 아래가 음수 (마운트 60° → pitch −60°) — coord_transform 과 같은 규약
실행 (자체 점검 · 합성 장면 몬테카를로): python geo_resolver.py
"""
import math
from dataclasses import dataclass, field
from coord_transform import rotation_body_to_ned, _matvec, ned_to_gps

D = math.radians


@dataclass
class CameraModel:
    width: int = 1920
    height: int = 1080
    hfov_deg: float = 80.2          # LN012 대각 88° → 16:9 수평 80.2°

    @property
    def f(self):
        return (self.width / 2) / math.tan(D(self.hfov_deg) / 2)

    def ray(self, u, v):
        """화소 → 카메라 좌표 광선 (x = 광축)"""
        return (1.0, (u - self.width / 2) / self.f, (v - self.height / 2) / self.f)

    def project(self, p_cam):
        """카메라 좌표 점 → 화소 (시험용 · 광축 뒤면 None)"""
        if p_cam[0] <= 0:
            return None
        return (self.width / 2 + self.f * p_cam[1] / p_cam[0], self.height / 2 + self.f * p_cam[2] / p_cam[0])


@dataclass
class Telemetry:
    """한 프레임의 기체 상태 — MAVLink 에서 영상 시각에 맞춰 받는다"""
    lat: float
    lon: float
    alt_agl: float                   # 이륙점(지면) 기준 고도 m
    roll: float = 0.0                # rad
    pitch: float = 0.0
    yaw: float = 0.0                 # rad · 북 0 · 시계 방향 +
    gimbal_pitch_deg: float = -60.0  # 수평 기준 · 아래가 음수 (마운트 60°)
    gimbal_yaw_deg: float = 0.0      # 기수 기준
    stabilized: bool = True          # True: 짐벌이 roll·pitch 를 상쇄 (GM3 V2 3축)
    attitude_rate_deg: float = 0.0   # 직전 프레임 대비 자세 변화 (안정 프레임 판정)


@dataclass
class Result:
    status: str                      # "OK" | "PENDING"
    lat: float = None
    lon: float = None
    north: float = None              # 기체 기준 NED 오프셋 m
    east: float = None
    range_m: float = None            # 기체 → 지점 수평 거리
    depression_deg: float = None     # 광선 하향각
    ellipse: tuple = None            # (거리 방향 σ m, 옆 방향 σ m, 거리 방향 방위 deg)
    reasons: list = field(default_factory=list)


@dataclass
class ErrorModel:
    """오차 타원용 1σ (초안 · 실측 후 교체)"""
    gps_m: float = 1.5               # 수평 축당
    alt_m: float = 1.0               # 기압·GPS 고도
    pitch_deg: float = 1.0           # 짐벌 피치 + 기체 잔여 기울기
    yaw_deg: float = 2.0             # 나침반 방위
    terrain_m: float = 0.0           # 평지 가정일 때 지형 기복 (산지면 키운다)


class GeoResolver:
    MAX_TILT_DEG = 6.0
    MAX_RATE_DEG = 1.5
    MAX_RANGE_M = 200.0
    EDGE_PX = 4                      # 아래 끝에서 이만큼 안이면 발끝이 잘렸을 수 있다

    def __init__(self, cam=None, err=None, terrain=None):
        self.cam = cam or CameraModel()
        self.err = err or ErrorModel()
        self.terrain = terrain       # f(north, east) → 이륙점 기준 지면 높이 m (없으면 평지)

    def cam_to_ned(self, tel: Telemetry):
        """카메라 → NED 회전행렬"""
        gp, gy = D(tel.gimbal_pitch_deg), D(tel.gimbal_yaw_deg)
        if tel.stabilized:          # 짐벌 절대각: 기체 방위 + 짐벌 방위 · 짐벌 피치 · roll 0
            return rotation_body_to_ned(0.0, gp, tel.yaw + gy)
        from coord_transform import _matmul
        return _matmul(rotation_body_to_ned(tel.roll, tel.pitch, tel.yaw), rotation_body_to_ned(0.0, gp, gy))

    def to_world(self, bbox, tel: Telemetry):
        x1, y1, x2, y2 = bbox
        res = Result("OK")
        tilt = math.degrees(math.acos(max(-1, min(1, math.cos(tel.roll) * math.cos(tel.pitch)))))
        if tilt > self.MAX_TILT_DEG:
            res.reasons.append(f"기체 기울기 {tilt:.1f}° > {self.MAX_TILT_DEG}°")
        if tel.attitude_rate_deg > self.MAX_RATE_DEG:
            res.reasons.append(f"자세 변화 {tel.attitude_rate_deg:.1f}° > {self.MAX_RATE_DEG}°")
        if y2 >= self.cam.height - self.EDGE_PX:
            res.reasons.append("발끝이 화면 아래 끝 — 잘렸을 수 있음")
        u, v = (x1 + x2) / 2, y2
        d = _matvec(self.cam_to_ned(tel), self.cam.ray(u, v))
        if d[2] <= 1e-6:
            res.status = "PENDING"; res.reasons.append("광선이 지평선 위"); return res
        z0 = -tel.alt_agl                                  # 기체 z (NED · 지면 0)
        ground = 0.0
        for _ in range(5 if self.terrain else 1):          # 지형: 교차 → 그 지점 높이로 다시
            t = (-ground - z0) / d[2]
            n, e = t * d[0], t * d[1]
            if self.terrain:
                ground = self.terrain(n, e)
        rng = math.hypot(n, e)
        if rng > self.MAX_RANGE_M:
            res.status = "PENDING"; res.reasons.append(f"거리 {rng:.0f} m > {self.MAX_RANGE_M:.0f} m"); return res
        dep = math.degrees(math.atan2(d[2], math.hypot(d[0], d[1])))
        res.lat, res.lon = ned_to_gps((tel.lat, tel.lon), n, e)
        res.north, res.east, res.range_m, res.depression_deg = n, e, rng, dep
        res.ellipse = self.ellipse(tel.alt_agl - ground, dep, rng, math.degrees(math.atan2(e, n)))
        if res.reasons:
            res.status = "PENDING"
        return res

    def ellipse(self, h, dep_deg, rng, bearing_deg):
        """거리 방향: ∂d/∂φ = −h/sin²φ · ∂d/∂h = 1/tanφ · 기복 Δz/tanφ · 옆 방향: 거리 × 방위 오차"""
        phi = D(max(dep_deg, 1.0))
        e = self.err
        s_along = math.sqrt((h / math.sin(phi) ** 2 * D(e.pitch_deg)) ** 2 + (e.alt_m / math.tan(phi)) ** 2
                            + (e.terrain_m / math.tan(phi)) ** 2 + e.gps_m ** 2)
        s_cross = math.sqrt((rng * D(e.yaw_deg)) ** 2 + e.gps_m ** 2)
        return (round(s_along, 2), round(s_cross, 2), round(bearing_deg, 1))


# ── 자체 점검 · 합성 장면 ─────────────────────────────────────────────
def _synthetic(n=4000, seed=0):
    """사람을 땅에 놓고 → 화소로 투영 → 되돌려 좌표 오차 (① 오차 0 확인 ② 텔레메트리 잡음 영향)"""
    import random
    rnd = random.Random(seed)
    cam = CameraModel(); gr = GeoResolver(cam)
    em = ErrorModel()
    rows = []
    for mount in (45, 60, 75):
        for h in (16, 20, 25, 30):
            exact, noisy = [], []
            for _ in range(n // 12):
                tel = Telemetry(35.0, 128.0, h, roll=D(rnd.uniform(-3, 3)), pitch=D(rnd.uniform(-3, 3)),
                                yaw=D(rnd.uniform(0, 360)), gimbal_pitch_deg=-mount, stabilized=True)
                # 화면 안 임의 화소 (아래 끝 5 % 제외) → 지면 점 = 정답
                u, v = rnd.uniform(0, cam.width), rnd.uniform(0.05 * cam.height, 0.95 * cam.height)
                d = _matvec(gr.cam_to_ned(tel), cam.ray(u, v))
                if d[2] <= 0:
                    continue
                t = tel.alt_agl / d[2]; gt = (t * d[0], t * d[1])
                r = gr.to_world((u - 5, v - 40, u + 5, v), tel)
                exact.append(math.hypot(r.north - gt[0], r.east - gt[1]))
                # 텔레메트리 잡음 (ErrorModel 1σ)
                tn = Telemetry(35.0, 128.0, h + rnd.gauss(0, em.alt_m), yaw=tel.yaw + D(rnd.gauss(0, em.yaw_deg)),
                               gimbal_pitch_deg=-mount + rnd.gauss(0, em.pitch_deg), stabilized=True)
                rn = gr.to_world((u - 5, v - 40, u + 5, v), tn)
                if rn.status == "OK" or rn.north is not None:
                    gx, gy = rnd.gauss(0, em.gps_m), rnd.gauss(0, em.gps_m)
                    noisy.append(math.hypot(rn.north + gx - gt[0], rn.east + gy - gt[1]))
            noisy.sort()
            rows.append((mount, h, max(exact), sum(noisy) / len(noisy), noisy[len(noisy) // 2], noisy[int(0.95 * len(noisy))]))
    return rows


if __name__ == "__main__":
    import sys
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    gr = GeoResolver()
    tel = Telemetry(35.0, 128.0, 30.0, gimbal_pitch_deg=-60.0)
    r = gr.to_world((955, 520, 965, 540), tel)           # 화면 가운데 발끝
    print(f"중앙 발끝 · 60° · 30 m → 앞 {r.north:.2f} m (기대 {30 / math.tan(D(60)):.2f}) · 타원 {r.ellipse} · {r.status}")
    r = gr.to_world((900, 1000, 940, 1078), tel)
    print(f"아래 끝 박스 → {r.status} · {r.reasons}")
    print("\n합성 장면 — 마운트 · 고도 · 오차 없음 최대 · 잡음(GPS 1.5 · 고도 1 · 피치 1° · 방위 2°) 평균/중앙/P95 m")
    for m, h, ex, mean, med, p95 in _synthetic():
        print(f"  {m:>2}° {h:>2} m  exact≤{ex:.1e}  잡음 {mean:5.2f} / {med:5.2f} / {p95:5.2f}")
