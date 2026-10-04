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
  · ⚠ 카메라는 기체에 고정 (짐벌 안정화 없음) — 실제 카메라 자세 = 기체 roll/pitch/yaw 에 피치 −mount 를 더한 것.
    선회 구간은 기체가 20~30° 기운다 (10-05 · |roll| 최대 25.8°) → NFR-V04 안정 프레임(≤6°)은 roll_deg · pitch_deg 로 골라 쓴다

실행 (UE 를 settings_flight.json 으로 -game 실행 중일 때)
  python ue_flight_sequence.py --out ue_flight/rural01 --alt 20 --mount 45 --speed 4 --fps 5
"""
import argparse
import csv
import math
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import cosysairsim as airsim
from cosysairsim.utils import euler_to_quaternion, quaternion_to_euler_angles

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from ue_posture_capture import actors, boxes_in, W, H   # 배우 → 분할 색 · 마스크 박스 (검증된 것 재사용)


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
    acts = actors(c)
    print(f"배우 {len(acts)} 명")

    c.enableApiControl(True); c.armDisarm(True)
    c.simSetCameraPose("0", airsim.Pose(airsim.Vector3r(0, 0, 0), euler_to_quaternion(0, -math.radians(args.mount), 0)))
    c.takeoffAsync().join()
    c.moveToZAsync(-args.alt, 3).join()
    start = lawnmower(args.x0, args.y0, args.legs, args.spacing, args.length, args.alt)
    c.moveToPositionAsync(start[0].x_val, start[0].y_val, -args.alt, 5).join()
    # 같은 구역을 반대 방향으로 한 번 더 — 같은 사람을 다른 방위에서 다시 본다 (재관측)
    path = start + list(reversed(start))[1:]
    end = path[-1]
    c.moveOnPathAsync(path, args.speed, args.max_seconds, airsim.DrivetrainType.ForwardOnly,
                      airsim.YawMode(False, 0), -1, 1)          # join 하지 않고 기록하며 기다린다

    rows, n, t0, dt = [], 0, time.time(), 1.0 / args.fps
    near_since = None
    while time.time() - t0 < args.max_seconds:
        t_req = time.time()
        r = c.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False),
                            airsim.ImageRequest("0", airsim.ImageType.Segmentation, False, False)])
        st = c.getMultirotorState()
        k, gps = st.kinematics_estimated, st.gps_location
        roll, pitch, yaw = quaternion_to_euler_angles(k.orientation)
        rgb = np.frombuffer(r[0].image_data_uint8, np.uint8).reshape(r[0].height, r[0].width, 3)
        seg = np.frombuffer(r[1].image_data_uint8, np.uint8).reshape(r[1].height, r[1].width, 3)
        bx = boxes_in(seg, acts)
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
                     "cam_pitch_rel_body_deg": -args.mount, "fov_h_deg": 80.2, "w": W, "h": H,
                     "people": len(bx), "actors": ";".join(sorted(bx))})
        if n % 25 == 0:
            print(f"  프레임 {n} · t {t_req - t0:.0f}s · 위치 ({k.position.x_val:.0f},{k.position.y_val:.0f},{-k.position.z_val:.0f}) · 사람 {len(bx)}", flush=True)
        d_end = math.dist((k.position.x_val, k.position.y_val), (end.x_val, end.y_val))
        v = math.hypot(k.linear_velocity.x_val, k.linear_velocity.y_val)
        if d_end < 2.0 and v < 0.6:
            near_since = near_since or time.time()
            if time.time() - near_since > 1.5:
                break
        else:
            near_since = None
        time.sleep(max(0.0, dt - (time.time() - t_req)))

    c.landAsync().join(); c.armDisarm(False); c.enableApiControl(False)
    if rows:
        with open(out / "flight.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    seen = {}
    for r in rows:
        for a in filter(None, r["actors"].split(";")):
            seen[a] = seen.get(a, 0) + 1
    dur = rows[-1]["t_s"] if rows else 0
    print(f"\n프레임 {len(rows)} · {dur:.0f}초 · 실제 {len(rows) / max(dur, 1e-6):.1f} fps · 한 번 이상 잡힌 배우 {len(seen)}/{len(acts)} → {out}")


if __name__ == "__main__":
    main()
