"""
coord_dem_sim.py — G1: 좌표 산정 평지 가정 vs DEM — 금오공대 캠퍼스 지형 모의 (10-11 · 판정 기준은 _학습 큐 「10-11 G1」)

  지형 「참」  Copernicus GLO-30 (data/raw/dem/glo30_N36E128.tif · 30 m DSM · 쌍선형) — 압축 (deflate + 실수 예측자) 을 직접 푼 .npy
  추정        flat   이륙점 높이 평면 (지금 worker)
              team   팀 지형 타일 (drone-dev spatial campus · Terrarium PNG z16 · 원본 90 m DEM 을 10 m 로 다시 뽑음)
              coarse GLO-30 을 90 m 로 뭉갠 것 (해상도 몫)
              각 추정의 기체 높이 = 그 DEM 의 이륙점 높이 + 고도 (기압 고도 = 이륙점 기준) → 높이 기준 차는 상쇄
  기하        이륙점 20 (핵심 구역 무작위 · 시드 0) · 기체 반경 200 m 안 25 m 간격 · 고도 16 · 20 m · 마운트 45° (짐벌 안정) · 방위 8 · 화면 3×3
              광선 = worker GeoResolver.cam_to_ned · cam.ray (같은 카메라 모델) · 「참」 땅 위 8 m 미만 기체는 뺌
  지표        「참」 교차점과 수평 오차 중앙 · P95 · 이륙점 블록 부트스트랩
실행: /home/se/miniconda3/envs/drone/bin/python coord_dem_sim.py   → metrics/coord_dem_sim.json
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / "deploy" / "vision_worker"))
from app.vision_core.geo import GeoResolver, Telemetry   # noqa: E402

DEM = BASE.parent / "data" / "raw" / "dem" / "glo30_N36E128.npy"
TEAM = Path("/home/se/sandboxes/drone-dev/data/spatial/campus-04e1b46514574d45/terrain/16")
CORE = (128.38058, 36.13870, 128.39922, 36.15080)               # world.json coreBounds (lon0, lat0, lon1, lat1)
LAT0, LON0 = 36.1452, 128.3916
R = 6378137.0
MN = math.radians(1) * R; ME = MN * math.cos(math.radians(LAT0))   # m / 도


def ll(n, e):
    return LAT0 + n / MN, LON0 + e / ME


class Glo:
    def __init__(self, coarse=False):
        a = np.load(DEM)                                         # 3600×3600 · 왼쪽 위 (37 N, 128 E) · 1"
        if coarse:                                               # 90 m (3×3 칸) 평균 → 다시 1" 로 (해상도 몫)
            h, w = a.shape; b = a[:h // 3 * 3, :w // 3 * 3].reshape(h // 3, 3, w // 3, 3).mean((1, 3)); a = np.kron(b, np.ones((3, 3), np.float32))
        self.a = a

    def __call__(self, n, e):
        lat, lon = ll(n, e); y = (37.0 - lat) * 3600 - 0.5; x = (lon - 128.0) * 3600 - 0.5
        y0, x0 = int(math.floor(y)), int(math.floor(x)); fy, fx = y - y0, x - x0; a = self.a
        return float((a[y0, x0] * (1 - fx) + a[y0, x0 + 1] * fx) * (1 - fy) + (a[y0 + 1, x0] * (1 - fx) + a[y0 + 1, x0 + 1] * fx) * fy)


class Team:
    def __init__(self):
        self.cache = {}

    def _px(self, tx, ty):
        if (tx, ty) not in self.cache:
            im = np.array(Image.open(TEAM / str(tx) / f"{ty}.png").convert("RGB")).astype(np.float64)
            self.cache[(tx, ty)] = im[..., 0] * 256 + im[..., 1] + im[..., 2] / 256 - 32768
        return self.cache[(tx, ty)]

    def __call__(self, n, e):
        lat, lon = ll(n, e); z = 2 ** 16
        X = (lon + 180) / 360 * z * 256 - 0.5; Y = (1 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi) / 2 * z * 256 - 0.5
        x0, y0 = int(math.floor(X)), int(math.floor(Y)); fx, fy = X - x0, Y - y0
        v = lambda xx, yy: self._px(xx // 256, yy // 256)[yy % 256, xx % 256]
        return float((v(x0, y0) * (1 - fx) + v(x0 + 1, y0) * fx) * (1 - fy) + (v(x0, y0 + 1) * (1 - fx) + v(x0 + 1, y0 + 1) * fx) * fy)


def march(ground, n0, e0, z0, d, step=0.5, max_h=400.0):
    """광선 (n0, e0, 높이 z0 · 방향 d = (북, 동, 아래)) × 지형 → 첫 교차 (n, e) · 없으면 None"""
    hz = math.hypot(d[0], d[1]); k = step / max(hz, 1e-6); t = 0.0; prev = (z0 - ground(n0, e0))
    while t * hz < max_h:
        t += k; n, e, z = n0 + t * d[0], e0 + t * d[1], z0 - t * d[2]
        cur = z - ground(n, e)
        if cur <= 0:
            lo, hi = t - k, t
            for _ in range(25):
                m = (lo + hi) / 2
                if z0 - m * d[2] - ground(n0 + m * d[0], e0 + m * d[1]) > 0:
                    lo = m
                else:
                    hi = m
            return n0 + hi * d[0], e0 + hi * d[1]
        prev = cur
    return None


def main():
    truth, team, coarse = Glo(), Team(), Glo(coarse=True)
    geo = GeoResolver(); rng = np.random.default_rng(0)
    nmin, emin = (CORE[1] - LAT0) * MN, (CORE[0] - LON0) * ME; nmax, emax = (CORE[3] - LAT0) * MN, (CORE[2] - LON0) * ME
    pix = [(u, v) for v in (0.30 * 1080, 0.55 * 1080, 0.90 * 1080) for u in (0.15 * 1920, 0.5 * 1920, 0.85 * 1920)]
    rows = []                                                    # (이륙점, 고도, 기체 위 땅높이, flat, team, coarse)
    for ti in range(20):
        tn, te = rng.uniform(nmin, nmax), rng.uniform(emin, emax)
        gT = {"truth": truth(tn, te), "team": team(tn, te), "coarse": coarse(tn, te)}
        for alt in (16.0, 20.0):
            for dn in np.arange(-200, 201, 25):
                for de in np.arange(-200, 201, 25):
                    if math.hypot(dn, de) > 200:
                        continue
                    pn, pe = tn + dn, te + de
                    zt = gT["truth"] + alt
                    if zt - truth(pn, pe) < 8:
                        continue
                    for yaw in range(0, 360, 45):
                        tel = Telemetry(LAT0, LON0, alt, yaw=math.radians(yaw), gimbal_pitch_deg=-45.0, stabilized=True)
                        M = geo.cam_to_ned(tel)
                        for u, v in pix:
                            r = geo.cam.ray(u, v); d = [sum(M[i][j] * r[j] for j in range(3)) for i in range(3)]
                            if d[2] <= 1e-3:
                                continue
                            hit = march(truth, pn, pe, zt, d)
                            if hit is None:
                                continue
                            tf = alt / d[2]; flat = (pn + tf * d[0], pe + tf * d[1])                 # 이륙점 높이 평면
                            est = {"flat": flat}
                            for name, g in (("team", team), ("coarse", coarse)):
                                est[name] = march(g, pn, pe, gT[name] + alt, d)
                            err = {k: (math.dist(hit, p) if p is not None else np.nan) for k, p in est.items()}
                            rows.append((ti, alt, zt - truth(pn, pe), err["flat"], err["team"], err["coarse"]))
        print(f"이륙점 {ti + 1}/20 · 광선 {len(rows):,}", flush=True)
    A = np.array(rows, float)
    res = {"rays": len(A), "takeoffs": 20}
    for j, name in ((3, "평지"), (4, "팀 DEM (90 m 원본)"), (5, "GLO-30 을 90 m 로")):
        v = A[:, j]; ok = ~np.isnan(v)
        res[name] = {"중앙 m": round(float(np.median(v[ok])), 2), "P95 m": round(float(np.percentile(v[ok], 95)), 2), "교차 실패 비율": round(float(1 - ok.mean()), 4)}
        for alt in (16.0, 20.0):
            m = ok & (A[:, 1] == alt); res[name][f"고도 {int(alt)} m 중앙 · P95"] = [round(float(np.median(v[m])), 2), round(float(np.percentile(v[m], 95)), 2)]
    rng = np.random.default_rng(1); blocks = [np.where(A[:, 0] == t)[0] for t in range(20)]; bs = []
    for _ in range(1000):
        s = np.concatenate([blocks[t] for t in rng.integers(0, 20, 20)]); f, tm = A[s, 3], A[s, 4]; ok = ~np.isnan(tm)
        bs.append((np.percentile(f, 95) - np.percentile(tm[ok], 95)) / np.percentile(f, 95))
    res["팀 DEM 의 P95 줄임 비율 95%"] = [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]
    fp = res["평지"]["P95 m"]; red = (fp - res["팀 DEM (90 m 원본)"]["P95 m"]) / fp
    res["G1 판정 (DEM 도입 권고)"] = bool(fp > 3 and red >= 0.30)
    (BASE / "metrics" / "coord_dem_sim.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
