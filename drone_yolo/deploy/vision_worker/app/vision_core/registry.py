"""
registry.py — 후보 기억 v0 (설계명세서 CandidateRegistry · 2026-10-05)

  같은 사람을 하나의 후보로 묶고, 화면에서 사라져도 후보는 남긴다 (세계 좌표 기억)
    · 좌표 OK  → 기존 후보와 거리 ≤ 문턱 (두 오차 타원 1σ 큰 축 합 × GATE_SIGMA · 최소 MIN_GATE_M) 이면 같은 후보
                 위치는 분산 역가중 평균
    · 좌표 보류 → 최근 IMG_WINDOW_S 안 같은 후보 박스와 IoU ≥ IMG_IOU 면 같은 후보 (화면 안 연결만)
  같은 프레임의 탐지끼리는 묶지 않는다 (NMS 뒤라 서로 다른 사람)
  확정 = CONFIRM_WINDOW_S 안에 서로 다른 프레임 CONFIRM_HITS 장 이상에서 탐지 (관측 시간 규칙 · 9.28 — 팀 합의 전 기본값)
  후보를 지우지 않는다 — 운영자 판단 전까지 남는다 (UC · 거르지 않고 순위만)

  추적기 연결 (V4 · 10-07): worker 가 BoT-SORT 추적 키 → 후보를 기억해 같은 추적이면 attach 로 바로 붙인다 (ID 바뀜 끊기 뒤엔 새 키)
  남은 한계: 추적 기반 재점수 (R1) · 문턱 시뮬레이션 (W3-2) 전 기본값
"""
import math
from dataclasses import dataclass, field

GATE_SIGMA = 3.0
MIN_GATE_M = 3.0
IMG_IOU = 0.3
IMG_WINDOW_S = 1.0
CONFIRM_HITS = 2
CONFIRM_WINDOW_S = 3.0


def _iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    it = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - it
    return it / ua if ua > 0 else 0.0


def _metres(lat0, lon0, lat1, lon1):
    k = 111_320.0
    return math.hypot((lat1 - lat0) * k, (lon1 - lon0) * k * math.cos(math.radians(lat0)))


@dataclass
class Candidate:
    cid: str
    first_t: float
    last_t: float
    hits: list = field(default_factory=list)     # 탐지 시각 (s)
    best_conf: float = 0.0
    lat: float = None
    lon: float = None
    sigma_m: float = None                        # 1σ 큰 축 (평균 뒤)
    geo_status: str = "PENDING"
    last_box: tuple = None
    confirmed: bool = False
    color: object = None                         # ColorAccumulator (worker 가 붙인다)
    color_t: float = -1e9                        # 마지막 색 판정 시각 (s)

    def to_json(self):
        return {"candidate_id": self.cid, "first_seen_s": round(self.first_t, 3), "last_seen_s": round(self.last_t, 3),
                "n_hits": len(self.hits), "best_conf": round(self.best_conf, 3), "confirmed": self.confirmed,
                "geo_status": self.geo_status,
                "lat": None if self.lat is None else round(self.lat, 7),
                "lon": None if self.lon is None else round(self.lon, 7),
                "sigma_m": None if self.sigma_m is None else round(self.sigma_m, 2),
                "upper_color": None if self.color is None else self.color.result().get("top")}


class CandidateRegistry:
    def __init__(self, mission_id):
        self.mission_id, self.items, self._n = mission_id, [], 0

    def _new(self, t):
        self._n += 1
        c = Candidate(cid=f"{self.mission_id}-C{self._n:04d}", first_t=t, last_t=t)
        self.items.append(c)
        return c

    def update(self, t, box, conf, geo):
        """한 탐지를 넣는다 · 반환 (후보, 사건 이름 new · update · confirmed · best)"""
        geo_ok = geo is not None and geo.status == "OK"
        sig = max(geo.ellipse[0], geo.ellipse[1]) if geo_ok else None
        match = None
        if geo_ok:
            best = None
            for c in self.items:
                if c.lat is None or c.last_t == t:      # 같은 프레임의 다른 탐지는 다른 사람
                    continue
                d = _metres(c.lat, c.lon, geo.lat, geo.lon)
                gate = max(MIN_GATE_M, GATE_SIGMA * ((c.sigma_m or sig) + sig))
                if d <= gate and (best is None or d < best[0]):
                    best = (d, c)
            match = best[1] if best else None
        if match is None:                        # 좌표가 없거나 못 찾았으면 화면 안 연결
            best = None
            for c in self.items:
                if c.last_box is not None and 0 < t - c.last_t <= IMG_WINDOW_S:
                    o = _iou(box, c.last_box)
                    if o >= IMG_IOU and (best is None or o > best[0]):
                        best = (o, c)
            match = best[1] if best else None
        if match is None:
            return self.attach(self._new(t), t, box, conf, geo, "new")
        return self.attach(match, t, box, conf, geo)

    def attach(self, c, t, box, conf, geo, event="update"):
        """정해진 후보에 탐지를 붙인다 (추적기가 같은 사람이라고 한 경우도 여기로) · 반환 (후보, 사건)"""
        geo_ok = geo is not None and geo.status == "OK"
        sig = max(geo.ellipse[0], geo.ellipse[1]) if geo_ok else None
        c.last_t, c.last_box = t, tuple(float(v) for v in box)
        c.hits.append(t)
        if geo_ok:
            if c.lat is None:
                c.lat, c.lon, c.sigma_m, c.geo_status = geo.lat, geo.lon, sig, "OK"
            else:                                # 분산 역가중 평균
                wa, wb = 1 / c.sigma_m ** 2, 1 / sig ** 2
                c.lat = (c.lat * wa + geo.lat * wb) / (wa + wb)
                c.lon = (c.lon * wa + geo.lon * wb) / (wa + wb)
                c.sigma_m = (wa + wb) ** -0.5
        if conf > c.best_conf:
            c.best_conf = conf
            if event == "update":
                event = "best"
        recent = {h for h in c.hits if t - h <= CONFIRM_WINDOW_S}     # 서로 다른 프레임 수
        if not c.confirmed and len(recent) >= CONFIRM_HITS:
            c.confirmed, event = True, "confirmed"
        return c, event
