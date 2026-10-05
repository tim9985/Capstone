"""
ue_geo_flight_eval.py — ue_geo_flight.py 기록으로 GeoResolver 검증 (노트북 · 2026-10-05)

두 단계 (가상환경이 달라서 나눈다 — 새 패키지 설치 없음)
  detect : soup_v7r2 탐지 (vision_worker PersonDetector · 타일 4장 · conf 0.15 · NMS 0.6) → dets.json
           conda drone 환경 (ultralytics)
  eval   : 세 가지 박스로 GeoResolver → 정답 배우 위치와 수평 오차
           팀 simulation_world 가상환경 (rasterio · pyproj — terrain.tif 지형)

검증 단계 (서버 10-05 요청과 같은 이름)
  a  oracle : 배우 정답 발 위치 → 카메라 참값 자세로 화소 투영 → GeoResolver(참값 자세) → 0 이어야 함 (좌표계 · 높이 기준)
  a2 tel    : 같은 화소 + SITL 텔레메트리(영상 시각에 보간) — 텔레메트리 추정 · 시각 어긋남 · 카메라 오프셋 효과
  c  det    : soup_v7r2 탐지 박스 아래 변 가운데 + SITL 텔레메트리 — 실제 운용 오차 (판정은 안정 프레임)
  b (분할 박스)는 못 함 — 팀 맵은 인스턴스 분할이 꺼져 있다 (InitialInstanceSegmentation false · 256×144)

좌표 틀 (시뮬 전용 약속)
  UE · AirSim NED 축 = UTM 52N 격자축 (팀 world frame). SITL 위경도는 home(spawnGeographic) + NED 소규모 근사라
  정답 · 결과를 모두 home 기준 NED 로 바꿔 비교한다. DEM 은 같은 NED → UTM 격자로 읽는다 (회전 없음).
  UE 지면 = DEM + 0.08 m (시나리오 5곳 같음) → oracle 은 이 0.08 을 빼고, tel · det 은 운용처럼 DEM 그대로.

실행
  conda drone : python ue_geo_flight_eval.py detect ue_geo/run01 --weights <soup_v7r2 best.pt>
  sim venv    : python ue_geo_flight_eval.py eval ue_geo/run01 [--lag 0.05]
출력 metrics/ue_geo_flight_<run>.csv · .md
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

BASE = Path(__file__).resolve().parent
D = math.radians


def load_csv(p):
    with open(p, encoding="utf-8") as f:
        return [{k: (float(v) if k not in ("frame", "phase", "armed") else v) for k, v in r.items()} for r in csv.DictReader(f)]


# ── detect ───────────────────────────────────────────────────────────
def detect(run, weights):
    import cv2
    sys.path.insert(0, str(BASE / "deploy" / "vision_worker" / "app"))
    from vision.detector import PersonDetector
    det = PersonDetector(weights)
    out, frames = {}, sorted((run / "frames").glob("*.jpg"))
    for i, p in enumerate(frames):
        boxes, conf, ms = det.detect(cv2.imread(str(p)))
        out[p.stem] = [[*map(float, b), float(c)] for b, c in zip(boxes, conf)]
        if i % 100 == 0:
            print(f"  {i}/{len(frames)} · {ms:.0f} ms · 사람 {len(out[p.stem])}", flush=True)
    (run / "dets.json").write_text(json.dumps(out), encoding="utf-8")
    print(f"탐지 {len(out)} 장 → {run / 'dets.json'}")


# ── 기하 ─────────────────────────────────────────────────────────────
def quat_to_R(w, x, y, z):
    """쿼터니언(body→NED) → 회전행렬"""
    return ((1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)),
            (2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
            (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)))


def R_to_euler(R):
    """Rz(yaw)·Ry(pitch)·Rx(roll) 분해 (coord_transform 규약)"""
    return math.atan2(R[2][1], R[2][2]), -math.asin(max(-1, min(1, R[2][0]))), math.atan2(R[1][0], R[0][0])


def project(R, cam, p, W, H, f):
    d = (p[0] - cam[0], p[1] - cam[1], p[2] - cam[2])
    c = tuple(sum(R[k][i] * d[k] for k in range(3)) for i in range(3))     # Rᵀ·d
    if c[0] < 0.5:
        return None
    u, v = W / 2 + f * c[1] / c[0], H / 2 + f * c[2] / c[0]
    return (u, v) if 0 <= u < W and 0 <= v < H else None


def interp(rows, t, key):
    lo, hi = 0, len(rows) - 1
    if t <= rows[0]["t"] or t >= rows[-1]["t"]:
        return None
    while hi - lo > 1:
        m = (lo + hi) // 2
        lo, hi = (m, hi) if rows[m]["t"] <= t else (lo, m)
    a, b = rows[lo], rows[hi]
    if b["t"] - a["t"] > 0.5:                 # 텔레메트리 끊김
        return None
    f = (t - a["t"]) / (b["t"] - a["t"])
    va, vb = a[key], b[key]
    if key == "yaw":
        vb = va + (vb - va + math.pi) % (2 * math.pi) - math.pi
    return va + (vb - va) * f


def pct(v, q):
    v = sorted(v)
    if not v:
        return float("nan")
    k = (len(v) - 1) * q
    i = int(k)
    return v[i] if i + 1 >= len(v) else v[i] + (v[i + 1] - v[i]) * (k - i)


def summary(rows, label):
    e = [r["err"] for r in rows]
    if not e:
        return f"| {label} | 0 | | | | | | | | |"
    al = [abs(r["along"]) for r in rows]; cr = [abs(r["cross"]) for r in rows]
    inside = [r for r in rows if r.get("in2s") is not None]
    cov = f"{100 * sum(r['in2s'] for r in inside) / len(inside):.0f}" if inside else ""
    return (f"| {label} | {len(e)} | {pct(e, .5):.2f} | {sum(e) / len(e):.2f} | {pct(e, .9):.2f} | {pct(e, .95):.2f} | "
            f"{max(e):.2f} | {pct(al, .5):.2f} | {pct(cr, .5):.2f} | {cov} |")


def evaluate(run, lag_list):
    import rasterio
    import pyproj
    from geo_resolver import CameraModel, GeoResolver, Telemetry
    from coord_transform import gps_to_ned

    meta = json.loads((run / "run.json").read_text(encoding="utf-8"))
    actors = json.loads((run / "actors.json").read_text(encoding="utf-8"))
    frames, tel = load_csv(run / "frames.csv"), load_csv(run / "telemetry.csv")
    tel.sort(key=lambda r: r["t"])
    # 영상 time_stamp 는 시뮬 시계 — 부하가 걸리면 벽시계보다 느리게 간다 (run01: 7분 동안 차이 29 → 53 s).
    # 텔레메트리는 벽시계라 상수 하나로는 못 맞춘다 → 앞뒤 31장의 (요청 가운데 벽시계 − 영상 시각) 이동 중앙값으로 잇는다
    raw = [(fr["t_req"] + fr["t_recv"]) / 2 - fr["t_img"] for fr in frames]
    smooth = [sorted(raw[max(0, i - 15):i + 16])[len(raw[max(0, i - 15):i + 16]) // 2] for i in range(len(raw))]
    resid = sorted(abs(r - s) for r, s in zip(raw, smooth))
    jitter = resid[int(len(resid) * .9)]
    clock = (smooth[0], smooth[-1])
    for fr, s in zip(frames, smooth):
        fr["t_img"] += s
    print(f"시계 맞춤: 영상 시각 + 이동 중앙값 ({clock[0]:.1f} → {clock[1]:.1f} s) · 남는 흔들림 P90 {jitter * 1000:.0f} ms")
    dets = json.loads((run / "dets.json").read_text()) if (run / "dets.json").exists() else {}
    W, H, hfov, mount = meta["w"], meta["h"], meta["hfov"], meta["mount_pitch_deg"]
    cam_model = CameraModel(W, H, hfov)
    f = cam_model.f
    sn, (hlon, hlat, halt) = meta["spawnNeu"], meta["spawnGeographic"]
    lon0, lat0, z0 = meta["origin"]
    off = meta["ground_offset_m"]

    ds = rasterio.open(Path(meta["release"]) / "terrain.tif"); Z = ds.read(1).astype(float)
    tf = pyproj.Transformer.from_crs(4326, 32652, always_xy=True)
    x0, y0 = tf.transform(lon0, lat0)
    inv = ~ds.transform

    def dem_ned(n, e):
        """home 기준 NED (UTM 격자축) → DEM 해발 (쌍선형)"""
        c, r = inv * (x0 + sn[1] + e, y0 + sn[0] + n); c -= .5; r -= .5
        i, j = int(r), int(c); fr, fc = r - i, c - j
        return Z[i, j] * (1 - fr) * (1 - fc) + Z[i, j + 1] * (1 - fr) * fc + Z[i + 1, j] * fr * (1 - fc) + Z[i + 1, j + 1] * fr * fc

    ref = dem_ned(0.0, 0.0)                    # 기준면 = 이륙점 지면 해발 (서버 권장)
    msl_of_d = lambda d: z0 + sn[2] - d        # UE NED z → 해발 (SITL home 고도와 같은 식)

    # 시각 어긋남 추정 — SITL NED 와 카메라 참값 위치가 가장 잘 맞는 지연
    def lag_err(lag):
        s, k = 0.0, 0
        for fr in frames[::5]:
            n_ = interp(tel, fr["t_img"] + lag, "ned_n"); e_ = interp(tel, fr["t_img"] + lag, "ned_e")
            if n_ is not None:
                s += math.hypot(n_ - fr["cam_n"], e_ - fr["cam_e"]); k += 1
        return s / max(k, 1)
    lags = [x / 100 for x in range(-50, 51, 2)]
    best_lag = min(lags, key=lag_err)
    print(f"카메라 참값 vs SITL 위치: 지연 0 s 평균 {lag_err(0):.2f} m · 최소 {lag_err(best_lag):.2f} m @ {best_lag:+.2f} s")

    def resolver_for(n_ref, e_ref):
        return GeoResolver(cam_model, terrain=lambda dn, de: dem_ned(n_ref + dn, e_ref + de) - ref)

    def score(res, gt, n_ref, e_ref, kind, extra):
        if res.north is None:
            return dict(kind=kind, status=res.status, reasons=";".join(res.reasons), **extra)
        rn, re_ = n_ref + res.north, e_ref + res.east
        en, ee = rn - gt[0], re_ - gt[1]
        bn, be = gt[0] - n_ref, gt[1] - e_ref
        b = math.hypot(bn, be) or 1.0
        along, cross = (en * bn + ee * be) / b, (-en * be + ee * bn) / b
        row = dict(kind=kind, status=res.status, reasons=";".join(res.reasons), err=math.hypot(en, ee),
                   along=along, cross=cross, range=res.range_m, dep=res.depression_deg, **extra)
        if res.ellipse:
            sa, sc, _ = res.ellipse
            row["in2s"] = int((along / sa) ** 2 + (cross / sc) ** 2 <= 4.0)
        return row

    import bisect
    tts = [r["t"] for r in tel]
    def phase_at(t):
        i = min(max(bisect.bisect_left(tts, t), 0), len(tel) - 1)
        return tel[i]["phase"]

    rows = []
    for fr in frames:
        paused = str(phase_at(fr["t_img"])).startswith("PAUSED")   # 게이트웨이 안전 정지 중 제자리 정지 (같은 시점 반복)
        R = quat_to_R(fr["qw"], fr["qx"], fr["qy"], fr["qz"])
        cam = (fr["cam_n"], fr["cam_e"], fr["cam_d"])
        cr, cp, cy = R_to_euler(R)
        boxes = dets.get(fr["frame"], [])
        for a in actors:
            foot = (a["ned"][0], a["ned"][1], a["ground_ned_d"])
            px = project(R, cam, foot, W, H, f)
            if px is None:
                continue
            gt = foot[:2]
            base = dict(frame=fr["frame"], actor=a["name"], mode=a["mode"], u=round(px[0], 1), v=round(px[1], 1),
                        paused=int(paused))
            # a oracle — 참값 카메라 자세 (gimbal 0 · 카메라 자세를 기체 자리에)
            t_or = Telemetry(0, 0, msl_of_d(cam[2]) - off - ref, cr, cp, cy, gimbal_pitch_deg=0.0, stabilized=False)
            r_or = resolver_for(cam[0], cam[1]).to_world((px[0], px[1], px[0], px[1]), t_or)
            rows.append(score(r_or, gt, cam[0], cam[1], "oracle", {**base, "lag": 0.0}))
            for lag in lag_list:
                t = fr["t_img"] + lag
                vals = {k: interp(tel, t, k) for k in ("lat", "lon", "alt_msl", "roll", "pitch", "yaw")}
                prev = {k: interp(tel, t - 0.2, k) for k in ("roll", "pitch", "yaw")}
                if None in vals.values() or None in prev.values():
                    continue
                rate = math.degrees(math.sqrt(sum((vals[k] - prev[k] if k != "yaw" else
                                                   (vals[k] - prev[k] + math.pi) % (2 * math.pi) - math.pi) ** 2
                                                  for k in prev)))
                n_ref, e_ref = gps_to_ned((hlat, hlon), vals["lat"], vals["lon"])
                tt = Telemetry(vals["lat"], vals["lon"], vals["alt_msl"] - ref, vals["roll"], vals["pitch"], vals["yaw"],
                               gimbal_pitch_deg=mount, stabilized=False, attitude_rate_deg=rate)
                tilt = math.degrees(math.acos(math.cos(vals["roll"]) * math.cos(vals["pitch"])))
                extra = {**base, "lag": lag, "roll": round(math.degrees(vals["roll"]), 2),
                         "pitch": round(math.degrees(vals["pitch"]), 2), "tilt": round(tilt, 2), "rate": round(rate, 2)}
                gr = resolver_for(n_ref, e_ref)
                rows.append(score(gr.to_world((px[0], px[1], px[0], px[1]), tt), gt, n_ref, e_ref, "tel", extra))
                # c det — 정답 발 화소가 (20 % 넓힌) 박스 안에 드는 탐지 중 아래 변 가운데가 가장 가까운 것
                cand = []
                for x1, y1, x2, y2, conf in boxes:
                    mx, my = 0.2 * (x2 - x1), 0.2 * (y2 - y1)
                    if x1 - mx <= px[0] <= x2 + mx and y1 - my <= px[1] <= y2 + my:
                        cand.append((math.hypot((x1 + x2) / 2 - px[0], y2 - px[1]), (x1, y1, x2, y2), conf))
                if cand:
                    _, bb, conf = min(cand)
                    rows.append(score(gr.to_world(bb, tt), gt, n_ref, e_ref, "det",
                                      {**extra, "conf": round(conf, 3), "box_h": round(bb[3] - bb[1], 1),
                                       "foot_dv": round(bb[3] - px[1], 1)}))
                elif lag == lag_list[0]:
                    rows.append(dict(kind="det", status="MISS", reasons="탐지 없음", **extra))

    tag = run.name
    keys = sorted({k for r in rows for k in r}, key=lambda k: ["kind", "frame", "actor", "mode", "lag", "status", "err"].index(k)
                  if k in ("kind", "frame", "actor", "mode", "lag", "status", "err") else 99)
    out_csv = BASE / "metrics" / f"ue_geo_flight_{tag}.csv"
    out_csv.parent.mkdir(exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as fo:
        w = csv.DictWriter(fo, fieldnames=keys); w.writeheader(); w.writerows(rows)

    hdr = ("| 구분 | n | 중앙(CEP50) | 평균 | P90 | P95 | 최대 | 거리방향 중앙 | 옆방향 중앙 | 2σ 안 % |\n"
           "|---|---|---|---|---|---|---|---|---|---|")
    ok = lambda r: "err" in r
    stable = lambda r: ok(r) and r["status"] == "OK"
    L = [f"# GeoResolver 비행 검증 — {tag}", "",
         f"- 프레임 {len(frames)} · 텔레메트리 {len(tel)} · 배우 {len(actors)} (" + ", ".join(f"{a['name']} {a['mode']}" for a in actors) + ")",
         f"- 카메라 {W}×{H} · HFOV {hfov}° · 기체 고정 {mount}° (짐벌 없음) · 고도 15 m · 지형 terrain.tif (10 m 격자)",
         f"- 시계: 영상 time_stamp(시뮬 시계)가 벽시계보다 느림 (차 {clock[0]:.1f} → {clock[1]:.1f} s) → 31장 이동 중앙값으로 맞춤 · 남는 흔들림 P90 {jitter * 1000:.0f} ms",
         f"- 카메라 참값 vs SITL 위치 평균 차: 지연 0 s {lag_err(0):.2f} m · 최소 {lag_err(best_lag):.2f} m @ {best_lag:+.2f} s "
         "(카메라는 기체 중심 앞 0.3 m · 그 차 포함)", "",
         "## 단계별 수평 오차 (m)", "", hdr]
    L.append(summary([r for r in rows if r["kind"] == "oracle" and ok(r)], "a oracle (참값 자세)"))
    for lag in lag_list:
        sel = [r for r in rows if r["kind"] == "tel" and r["lag"] == lag]
        L.append(summary([r for r in sel if ok(r)], f"a2 tel 전체 · 지연 {lag:+.2f} s"))
        L.append(summary([r for r in sel if stable(r)], f"a2 tel 안정 프레임 · 지연 {lag:+.2f} s"))
    for lag in lag_list:
        sel = [r for r in rows if r["kind"] == "det" and r.get("lag") == lag]
        L.append(summary([r for r in sel if ok(r)], f"c det 전체 · 지연 {lag:+.2f} s"))
        L.append(summary([r for r in sel if stable(r)], f"**c det 안정 프레임 · 지연 {lag:+.2f} s**"))
    main_det = [r for r in rows if r["kind"] == "det" and r.get("lag") == lag_list[0]]
    L += ["", "## c det 층별 (지연 " + f"{lag_list[0]:+.2f} s · 안정 프레임)", "", hdr]
    st = [r for r in main_det if stable(r)]
    for m in ("standing", "lying"):
        L.append(summary([r for r in st if r["mode"] == m], f"자세 {m}"))
    for lo, hi in ((0, 30), (30, 45), (45, 60), (60, 91)):
        L.append(summary([r for r in st if lo <= r["dep"] < hi], f"하향각 {lo}~{hi}°"))
    L += ["", "## c det 층별 (지연 " + f"{lag_list[0]:+.2f} s · 전체 · 기울기 · 자세 변화)", "", hdr]
    al = [r for r in main_det if ok(r)]
    for lo, hi in ((0, 3), (3, 6), (6, 90)):
        L.append(summary([r for r in al if lo <= r["tilt"] < hi], f"기울기 {lo}~{hi}°"))
    for lo, hi in ((0, 0.5), (0.5, 1.5), (1.5, 99)):
        L.append(summary([r for r in al if lo <= r["rate"] < hi], f"자세 변화 {lo}~{hi}°/0.2 s"))
    n_vis = sum(1 for r in main_det)
    n_miss = sum(1 for r in main_det if r["status"] == "MISS")
    mv = [r for r in main_det if not r["paused"]]
    n_miss_mv = sum(1 for r in mv if r["status"] == "MISS")
    n_pause = len({r["frame"] for r in rows if r.get("paused")})
    reasons = {}
    for r in main_det:
        if r["status"] == "MISS":
            continue
        for x in filter(None, (r.get("reasons") or "").split(";")):
            k = x.split(" ")[0] + (" " + x.split(" ")[1] if x.startswith("자세") or x.startswith("기체") else "")
            reasons[k] = reasons.get(k, 0) + 1
    L += ["", f"- 화면 안 배우(가림 무시 투영) {n_vis} · 탐지 짝 없음 {n_miss} ({100 * n_miss / max(n_vis, 1):.0f} %)"
          f" · 정지 구간(PAUSED · {n_pause} 장) 빼면 {n_miss_mv}/{len(mv)} ({100 * n_miss_mv / max(len(mv), 1):.0f} %)",
          f"- 좌표보류 사유: " + " · ".join(f"{k} {v}" for k, v in sorted(reasons.items(), key=lambda x: -x[1]))]
    dh = [r["foot_dv"] for r in main_det if "foot_dv" in r and r["mode"] == "standing"]
    if dh:
        L.append(f"- 탐지 박스 아래 변 − 정답 발 화소 (서 있음): 중앙 {pct(dh, .5):+.1f} px · P10 {pct(dh, .1):+.1f} · P90 {pct(dh, .9):+.1f}")
    md = "\n".join(L)
    (BASE / "metrics" / f"ue_geo_flight_{tag}.md").write_text(md + "\n", encoding="utf-8")
    print(md)
    print(f"\n저장: {out_csv}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("detect", "eval"))
    ap.add_argument("run")
    ap.add_argument("--weights", default=str(BASE.parent.parent / "drone_dev/weights/soup_v7r2/weights/best.pt"))
    ap.add_argument("--lag", type=float, nargs="*", default=[0.0, 0.05, 0.10], help="영상 시각에 일부러 더하는 어긋남 s")
    args = ap.parse_args()
    run = Path(args.run)
    if args.step == "detect":
        detect(run, args.weights)
    else:
        evaluate(run, args.lag)


if __name__ == "__main__":
    main()
