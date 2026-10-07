"""
telemetry.py — 촬영 시각 · 자세 sidecar 읽기 (형식은 **제안** · 동료 · 서성훈 합의 전)

  missions/<id>/metadata/telemetry.jsonl   기체 · 짐벌 상태 (초당 10회 이상 · 시각순)
    {"t_us": 1791117033123456, "lat": 36.1, "lon": 127.4, "alt_agl_m": 20.0,
     "roll_deg": 0.5, "pitch_deg": -1.2, "yaw_deg": 87.0,
     "gimbal_pitch_deg": -45.0, "gimbal_yaw_deg": 0.0, "gimbal_stabilized": true}
  missions/<id>/metadata/frames.jsonl      영상 프레임 ↔ 촬영 시각 (프레임마다 또는 키프레임마다)
    {"pts_ms": 123456.7, "t_us": 1791117033100000}

  프레임 시각 = frames.jsonl 로 PTS → 촬영 시각 (사이는 선형) · 없으면 받은 시각 (time_source="receive")
  자세 = 프레임 시각 앞뒤 telemetry 선형 보간 · 앞뒤 간격이 MAX_GAP_S 넘으면 없음 → 좌표 보류 (PENDING)
  t_us = UTC 마이크로초 (Pi 와 서버 시계를 같은 기준으로 — NTP/PTP)
"""
import bisect
import json
import math
from pathlib import Path

from .geo import Telemetry

MAX_GAP_S = 0.5


def _read_jsonl(path, offset):
    """새로 붙은 줄만 읽는다 (파일이 계속 자란다)"""
    rows = []
    try:
        with open(path, "rb") as f:
            f.seek(offset)
            data = f.read()
    except OSError:
        return rows, offset
    end = data.rfind(b"\n") + 1                  # 덜 쓴 마지막 줄은 다음에
    for line in data[:end].splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows, offset + end


class TelemetryStore:
    def __init__(self, mission_dir):
        m = Path(mission_dir) / "metadata"
        self.tel_path, self.frm_path = m / "telemetry.jsonl", m / "frames.jsonl"
        self._to = self._fo = 0
        self.t, self.rows = [], []
        self.pts, self.pts_t = [], []

    def refresh(self):
        rows, self._to = _read_jsonl(self.tel_path, self._to)
        for r in rows:
            if "t_us" in r:
                self.t.append(r["t_us"]); self.rows.append(r)
        rows, self._fo = _read_jsonl(self.frm_path, self._fo)
        for r in rows:
            if "pts_ms" in r and "t_us" in r:
                self.pts.append(r["pts_ms"]); self.pts_t.append(r["t_us"])

    def frame_time(self, pts_ms):
        """PTS → 촬영 시각 (UTC µs) · 모르면 None"""
        if pts_ms is None or len(self.pts) < 1:
            return None
        i = bisect.bisect_left(self.pts, pts_ms)
        if i <= 0:
            return self.pts_t[0] + (pts_ms - self.pts[0]) * 1000
        if i >= len(self.pts):
            return self.pts_t[-1] + (pts_ms - self.pts[-1]) * 1000
        p0, p1, t0, t1 = self.pts[i - 1], self.pts[i], self.pts_t[i - 1], self.pts_t[i]
        return t0 + (t1 - t0) * (pts_ms - p0) / max(p1 - p0, 1e-9)

    def at(self, t_us):
        """촬영 시각의 자세 (geo.Telemetry) · 모르면 None"""
        if t_us is None or len(self.t) < 2:
            return None
        i = bisect.bisect_left(self.t, t_us)
        if i <= 0 or i >= len(self.t):
            return None
        a, b = self.rows[i - 1], self.rows[i]
        if (self.t[i] - self.t[i - 1]) / 1e6 > MAX_GAP_S:
            return None
        w = (t_us - self.t[i - 1]) / max(self.t[i] - self.t[i - 1], 1)
        lin = lambda k: a[k] + (b[k] - a[k]) * w
        ang = lambda k: a[k] + ((b[k] - a[k] + 180) % 360 - 180) * w     # 방위는 0/360 을 넘나든다
        rate = max(abs(b["roll_deg"] - a["roll_deg"]), abs(b["pitch_deg"] - a["pitch_deg"]))
        return Telemetry(lat=lin("lat"), lon=lin("lon"), alt_agl=lin("alt_agl_m"),
                         roll=math.radians(lin("roll_deg")), pitch=math.radians(lin("pitch_deg")),
                         yaw=math.radians(ang("yaw_deg")),
                         gimbal_pitch_deg=lin("gimbal_pitch_deg"), gimbal_yaw_deg=ang("gimbal_yaw_deg"),
                         stabilized=bool(a.get("gimbal_stabilized", True)), attitude_rate_deg=rate)
