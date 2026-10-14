"""
pose.py — 백엔드 프레임 pose (`frame-pose/1.0` · 통합 v1.1 · 10-14) → 발끝 광선 좌표 「worker 주장」

  규칙 (통합 v1.1): 빠진 자세를 0° · 짐벌 −45° · 평지로 **채우지 않는다** — 없으면 PENDING + 사유
    · 카메라 방향: camera_attitude (CAMERA_FRD_TO_NED) 그대로 · 없으면 body_attitude × mount (CAMERA_FRD_TO_BODY_FRD) · 둘 다 없으면 보류
      카메라 FRD = x 광축 · y 오른쪽 · z 아래 = geo.CameraModel.ray 와 같은 축
    · 지면 높이: terrain(north, east) (NED · 아래 +) 가 주어지면 그것 · 없으면 보류 (GROUND_UNKNOWN)
      AGL 고도만 있으면 「기체 바로 아래와 같은 높이」 를 **가정으로 명시**하고 계산 (assumptions 에 LOCAL_FLAT_AGL) — allow_agl_flat=True 일 때만
    · 동기 오차 sync_error_ms > MAX_SYNC_MS · 시계 (REAL 은 UTC · DEVICE_MONOTONIC 만) · missing_reasons 는 그대로 사유로
  결과는 공식 위치가 아니라 worker 주장 — status "CLAIM" (계산됨) | "PENDING" · 공식 승격은 백엔드 (물리 검증 뒤)
"""
import math

from .coord_transform import ned_to_gps
from .geo import ErrorModel, GeoResolver

SCHEMA = "frame-pose/1.0"
MAX_SYNC_MS = 50.0
MAX_RANGE_M = 200.0
EDGE_PX = 4


def quat_to_matrix(q):
    """단위 쿼터니언 {w,x,y,z} → 회전행렬 (출발 축 → 도착 축)"""
    w, x, y, z = (float(q[k]) for k in ("w", "x", "y", "z"))
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]]


def matrix_to_quat(R):
    """회전행렬 → {w,x,y,z} (시험 · UE 기록 변환용)"""
    t = R[0][0] + R[1][1] + R[2][2]
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = (0.25 * s, (R[2][1] - R[1][2]) / s, (R[0][2] - R[2][0]) / s, (R[1][0] - R[0][1]) / s)
    elif R[0][0] > R[1][1] and R[0][0] > R[2][2]:
        s = math.sqrt(1.0 + R[0][0] - R[1][1] - R[2][2]) * 2
        q = ((R[2][1] - R[1][2]) / s, 0.25 * s, (R[0][1] + R[1][0]) / s, (R[0][2] + R[2][0]) / s)
    elif R[1][1] > R[2][2]:
        s = math.sqrt(1.0 + R[1][1] - R[0][0] - R[2][2]) * 2
        q = ((R[0][2] - R[2][0]) / s, (R[0][1] + R[1][0]) / s, 0.25 * s, (R[1][2] + R[2][1]) / s)
    else:
        s = math.sqrt(1.0 + R[2][2] - R[0][0] - R[1][1]) * 2
        q = ((R[1][0] - R[0][1]) / s, (R[0][2] + R[2][0]) / s, (R[1][2] + R[2][1]) / s, 0.25 * s)
    return dict(zip("wxyz", q))


def _mm(A, B):
    return [[sum(A[i][k] * B[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def _mv(A, v):
    return [sum(A[i][k] * v[k] for k in range(3)) for i in range(3)]


def camera_rotation(pose):
    """CAMERA_FRD → NED 회전 · (R, 출처, 사유)"""
    ca = pose.get("camera_attitude")
    if ca and ca.get("reference") == "CAMERA_FRD_TO_NED":
        return quat_to_matrix(ca["quaternion"]), "camera_attitude", None
    ba, mt = pose.get("body_attitude"), pose.get("mount")
    if ba and mt and (mt.get("orientation") or {}).get("reference") == "CAMERA_FRD_TO_BODY_FRD":
        return _mm(quat_to_matrix(ba["quaternion"]), quat_to_matrix(mt["orientation"]["quaternion"])), "body_attitude×mount", None
    return None, None, "CAMERA_ATTITUDE_MISSING"


class _Claim:
    """registry.CandidateRegistry 가 읽는 꼴 (geo.Result 와 같은 이름) — 좌표 주장이 있을 때만"""
    def __init__(self, c):
        self.status, self.lat, self.lon, self.ellipse = "OK", c["lat"], c["lon"], c["ellipse"]


def as_registry_geo(c):
    return _Claim(c) if c and c["status"] == "CLAIM" else None


def geo_json(c, claims=True):
    """vision-ingest/1.0 Geo — claims=False 면 통합 v1.1 그대로 (항상 PENDING · 사유) · True 면 계산된 주장을 VALID 로 (공식 승격은 백엔드)"""
    if claims and c and c["status"] == "CLAIM":
        a, b, az = c["ellipse"]
        major, minor = max(a, b), min(a, b)
        return {"status": "VALID", "lat": round(c["lat"], 7), "lon": round(c["lon"], 7), "reasons": [],
                "error_ellipse": {"major_m": round(major, 2), "minor_m": round(minor, 2), "azimuth_deg": round((az if a >= b else az + 90) % 360, 1), "k": 1},
                "method": ("FOOT_RAY_POSE+" + "+".join(c["assumptions"]) if c["assumptions"] else "FOOT_RAY_POSE")[:80]}
    reasons = (c or {}).get("reasons") or []
    base = ["POSE_MISSING"] if "POSE_MISSING" in reasons else ["PHYSICAL_VALIDATION_REQUIRED"]
    return {"status": "PENDING", "lat": None, "lon": None, "reasons": (base + [r for r in reasons if r not in base])[:16],
            "error_ellipse": None, "method": "FOOT_RAY_POSE" if c and c.get("source") else None}


class PoseGeo:
    """frame-pose/1.0 → 발끝 광선 × 지면 (worker 주장)"""

    def __init__(self, cam, terrain=None, allow_agl_flat=False, environment="SIM", err=None):
        self.cam, self.terrain, self.allow_agl_flat, self.env = cam, terrain, allow_agl_flat, environment
        self._ell = GeoResolver(cam, err or ErrorModel())                 # 오차 타원 식만 빌림 (geo.GeoResolver.ellipse)

    def check(self, pose):
        """프레임 단위 사유 (박스와 상관없음)"""
        if not pose:
            return ["POSE_MISSING"]
        r = []
        if pose.get("schema_version") != SCHEMA:
            r.append("POSE_SCHEMA")
        if self.env == "REAL" and pose.get("clock_domain") not in ("UTC", "DEVICE_MONOTONIC"):
            r.append("POSE_CLOCK")
        s = pose.get("sync_error_ms")
        if s is not None and s > MAX_SYNC_MS:
            r.append(f"SYNC_ERROR {s:.0f} ms > {MAX_SYNC_MS:.0f}")
        if not pose.get("geographic"):
            r.append("GEOGRAPHIC_MISSING")
        r += [f"SOURCE:{m}" for m in pose.get("missing_reasons") or []]
        return r

    @staticmethod
    def _unused(reason, source):
        """카메라 자세를 직접 받았으면 마운트 · 짐벌 · 기체 자세가 빠진 것은 계산에 안 쓰므로 막지 않는다 (사유는 남김)"""
        return source == "camera_attitude" and reason.startswith("SOURCE:") and any(k in reason.upper() for k in ("MOUNT", "GIMBAL", "BODY_ATTITUDE"))

    def locate(self, bbox, pose):
        """→ {"status": "CLAIM" | "PENDING", "lat", "lon", "north_m", "east_m", "range_m", "ellipse", "source", "assumptions", "reasons"}
        ellipse = (거리 방향 σ m, 옆 방향 σ m, 거리 방향 방위 deg) — geo.GeoResolver.ellipse 와 같은 식"""
        out = {"status": "PENDING", "lat": None, "lon": None, "north_m": None, "east_m": None, "range_m": None, "ellipse": None,
               "source": None, "assumptions": [], "reasons": self.check(pose)}
        if "POSE_MISSING" in out["reasons"]:
            return out
        R, src, why = camera_rotation(pose)
        out["source"] = src
        if why:
            out["reasons"].append(why); return out
        x1, y1, x2, y2 = bbox
        if y2 >= self.cam.height - EDGE_PX:
            out["reasons"].append("FOOT_AT_EDGE")
        d = _mv(R, self.cam.ray((x1 + x2) / 2, y2))
        if d[2] <= 1e-6:
            out["reasons"].append("RAY_ABOVE_HORIZON"); return out
        cp, geo = pose.get("camera_position_ned"), pose.get("geographic") or {}
        if self.terrain and cp:                                        # 지형 함수 (NED down · 지면 높이 = −down)
            hit = self._terrain_hit(d, cp)
            if hit is None:
                out["reasons"].append("NO_TERRAIN_HIT"); return out
            n, e = hit
            h = self.terrain(float(cp["north_m"]) + n, float(cp["east_m"]) + e) - float(cp["down_m"])
        elif self.allow_agl_flat and geo.get("altitude_datum") == "AGL":
            h = float(geo["altitude_m"])
            t = h / d[2]; n, e = t * d[0], t * d[1]
            out["assumptions"].append("LOCAL_FLAT_AGL")
        else:
            out["reasons"].append("GROUND_UNKNOWN"); return out
        rng = math.hypot(n, e)
        if rng > MAX_RANGE_M:
            out["reasons"].append(f"RANGE {rng:.0f} m > {MAX_RANGE_M:.0f}"); return out
        dep = math.degrees(math.atan2(d[2], math.hypot(d[0], d[1])))
        out.update(north_m=round(n, 2), east_m=round(e, 2), range_m=round(rng, 2),
                   ellipse=self._ell.ellipse(h, dep, rng, math.degrees(math.atan2(e, n))))
        if geo:
            out["lat"], out["lon"] = ned_to_gps((float(geo["latitude"]), float(geo["longitude"])), n, e)
        blocking = [r for r in out["reasons"] if not self._unused(r, src)]
        out["notes"] = [r for r in out["reasons"] if r not in blocking]
        out["reasons"] = blocking
        if not out["reasons"] and out["lat"] is not None:
            out["status"] = "CLAIM"
        return out

    def _terrain_hit(self, d, cp, step_m=1.0):
        """카메라 위치에서 광선을 수평 step_m 씩 → 처음 지면 아래 → 이분법 · 반환 = 카메라 기준 (north, east) 오프셋"""
        n0, e0, z0 = float(cp["north_m"]), float(cp["east_m"]), float(cp["down_m"])
        hz = math.hypot(d[0], d[1])
        below = lambda s: (z0 + d[2] * s / max(hz, 1e-9)) - self.terrain(n0 + d[0] * s / max(hz, 1e-9), e0 + d[1] * s / max(hz, 1e-9))
        if hz < 1e-9:
            return (0.0, 0.0) if z0 < self.terrain(n0, e0) else None
        if below(0.0) >= 0:                                            # 카메라가 이미 지면 아래
            return None
        s0, s = 0.0, step_m
        while s <= MAX_RANGE_M * 1.5:
            if below(s) >= 0:
                lo, hi = s0, s
                for _ in range(30):
                    mid = (lo + hi) / 2
                    lo, hi = (mid, hi) if below(mid) < 0 else (lo, mid)
                return hi * d[0] / hz, hi * d[1] / hz
            s0, s = s, s + step_m
        return None
