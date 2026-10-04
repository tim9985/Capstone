"""
ue_flight_sequence.py — UE(Cosys-AirSim, SimpleFlight) 에서 실제로 기체를 날리며 비행 영상 시퀀스를 기록한다 (노트북 · 2026-10-05)

왜
  사진 한 장씩 순간이동(ComputerVision)으로 찍은 자세 데이터와 달리, 실제 비행 경로를 따라
  프레임마다 기체 위치 · 자세 · 짐벌각 · 시각을 남겨야 좌표 산정 · 후보 기억(재관측) · 전체 흐름을 검증할 수 있다.

하는 일
  · 이륙 → 고도 --alt m → 잔디밭 위 지그재그(--legs 줄 · 줄 간격 --spacing m) → 같은 구역을 반대 방향으로 한 번 더 → 착륙
  · 비행 중 --fps 로 Scene + Segmentation 을 받아
      frames/<n>.jpg · labels/<n>.txt (사람 박스, YOLO 1클래스 · 분할 마스크 외곽)
      flight.csv — 프레임마다 시각 · NED 위치 · 위경도 · 고도 · roll/pitch/yaw · 짐벌 피치 · 화면에 잡힌 사람(배우 이름)
  · 짐벌 안정화 가정: 카메라 피치 = −마운트각 (기체 기울기는 따로 기록)

실행 (UE 를 settings_flight.json 으로 -game 실행 중일 때)
  python ue_flight_sequence.py --out ue_flight/rural01 --alt 20 --mount 45 --speed 4 --fps 5
"""
import argparse
import csv
import math
import re
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import cosysairsim as airsim
from cosysairsim.utils import euler_to_quaternion, quaternion_to_euler_angles

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

ACTOR = re.compile(r"Person_(Lying|Sitting|Kneeling|Crouching|Standing|Walking)_\d+", re.I)
W, H = 1920, 1080


def person_colors(client):
    """배우 이름 → 분할 색 (Cosys 인스턴스 분할 · 이름에 Person_ 이 들어간 것만)."""
    names = [n for n in client.simListInstanceSegmentationObjects() if ACTOR.fullmatch(n.split(".")[-1]) or ACTOR.match(n)]
    cmap = {}
    try:
        cols = client.simGetSegmentationColorMap()
        objs = client.simListInstanceSegmentationObjects()
        for i, n in enumerate(objs):
            if n in names and i < len(cols):
                c = cols[i]
                cmap[n] = (int(c[0]), int(c[1]), int(c[2]))
    except Exception:
        pass
    return names, cmap


def boxes_from_seg(seg, cmap):
    out = {}
    for name, (r, g, b) in cmap.items():
        m = (seg[:, :, 0] == r) & (seg[:, :, 1] == g) & (seg[:, :, 2] == b)
        n = int(m.sum())
        if n < 20:
            continue
        ys, xs = np.nonzero(m)
        x1, y1, x2, y2 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
        cut = x1 <= 0 or y1 <= 0 or x2 >= W or y2 >= H
        out[name] = ((float(x1), float(y1), float(x2), float(y2)), n, cut)
    return out


def lawnmower(x0, y0, legs, spacing, length, alt):
    pts = []
    for i in range(legs):
        y = y0 + i * spacing
        xs = (x0, x0 + length) if i % 2 == 0 else (x0 + length, x0)
        pts += [airsim.Vector3r(xs[0], y, -alt), airsim.Vector3r(xs[1], y, -alt)]
    return pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--alt", type=float, default=20.0)
    ap.add_argument("--mount", type=float, default=45.0, help="짐벌 아래 각 (°) · 카메라 피치 = −mount")
    ap.add_argument("--speed", type=float, default=4.0)
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--legs", type=int, default=4)
    ap.add_argument("--spacing", type=float, default=12.0)
    ap.add_argument("--length", type=float, default=50.0)
    ap.add_argument("--x0", type=float, default=-25.0)
    ap.add_argument("--y0", type=float, default=-22.0)
    ap.add_argument("--max-seconds", type=float, default=600.0)
    args = ap.parse_args()

    out = Path(args.out)
    for d in ("frames", "labels"):
        (out / d).mkdir(parents=True, exist_ok=True)
    c = airsim.MultirotorClient(); c.confirmConnection()
    names, cmap = person_colors(c)
    print(f"배우 {len(names)} 명 · 분할 색 {len(cmap)} 개")

    c.enableApiControl(True); c.armDisarm(True)
    c.simSetCameraPose("0", airsim.Pose(airsim.Vector3r(0, 0, 0), euler_to_quaternion(0, -math.radians(args.mount), 0)))
    c.takeoffAsync().join()
    c.moveToZAsync(-args.alt, 3).join()
    start = lawnmower(args.x0, args.y0, args.legs, args.spacing, args.length, args.alt)
    c.moveToPositionAsync(start[0].x_val, start[0].y_val, -args.alt, 5).join()
    # 같은 구역을 반대 방향으로 한 번 더 — 같은 사람을 다른 방위에서 다시 본다 (재관측)
    path = start + list(reversed(start))

    rows, stop = [], threading.Event()
    rec_client = airsim.MultirotorClient(); rec_client.confirmConnection()

    def record():
        n, t0, dt = 0, time.time(), 1.0 / args.fps
        while not stop.is_set() and time.time() - t0 < args.max_seconds:
            t_req = time.time()
            r = rec_client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False),
                                         airsim.ImageRequest("0", airsim.ImageType.Segmentation, False, False)])
            st = rec_client.getMultirotorState()
            k = st.kinematics_estimated
            gps = st.gps_location
            roll, pitch, yaw = quaternion_to_euler_angles(k.orientation)
            rgb = np.frombuffer(r[0].image_data_uint8, np.uint8).reshape(r[0].height, r[0].width, 3)
            seg = np.frombuffer(r[1].image_data_uint8, np.uint8).reshape(r[1].height, r[1].width, 3)
            bx = boxes_from_seg(seg, cmap)
            n += 1
            stem = f"{n:05d}"
            cv2.imwrite(str(out / "frames" / f"{stem}.jpg"), rgb[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])
            with open(out / "labels" / f"{stem}.txt", "w") as f:
                for an, (bb, px, cut) in bx.items():
                    f.write(f"0 {(bb[0] + bb[2]) / 2 / W:.6f} {(bb[1] + bb[3]) / 2 / H:.6f} {(bb[2] - bb[0]) / W:.6f} {(bb[3] - bb[1]) / H:.6f}\n")
            rows.append({"frame": stem, "t_s": round(t_req - t0, 3), "sim_ts_ns": r[0].time_stamp,
                         "ned_x": round(k.position.x_val, 3), "ned_y": round(k.position.y_val, 3), "ned_z": round(k.position.z_val, 3),
                         "lat": gps.latitude, "lon": gps.longitude, "alt_msl": round(gps.altitude, 2),
                         "roll_deg": round(math.degrees(roll), 2), "pitch_deg": round(math.degrees(pitch), 2), "yaw_deg": round(math.degrees(yaw), 2),
                         "gimbal_pitch_deg": -args.mount, "fov_h_deg": 80.2, "w": W, "h": H,
                         "people": len(bx), "actors": ";".join(sorted(bx))})
            if n % 25 == 0:
                print(f"  프레임 {n} · t {t_req - t0:.0f}s · 위치 ({k.position.x_val:.0f},{k.position.y_val:.0f},{-k.position.z_val:.0f}) · 사람 {len(bx)}", flush=True)
            time.sleep(max(0.0, dt - (time.time() - t_req)))

    th = threading.Thread(target=record, daemon=True); th.start()
    c.moveOnPathAsync(path, args.speed, args.max_seconds, airsim.DrivetrainType.ForwardOnly,
                      airsim.YawMode(False, 0), -1, 1).join()
    stop.set(); th.join()
    c.landAsync().join(); c.armDisarm(False); c.enableApiControl(False)

    if rows:
        with open(out / "flight.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    seen = {}
    for r in rows:
        for a in filter(None, r["actors"].split(";")):
            seen[a] = seen.get(a, 0) + 1
    dur = rows[-1]["t_s"] if rows else 0
    print(f"\n프레임 {len(rows)} · {dur:.0f}초 · 실제 {len(rows) / max(dur, 1e-6):.1f} fps · 한 번 이상 잡힌 배우 {len(seen)}/{len(names)} → {out}")


if __name__ == "__main__":
    main()
