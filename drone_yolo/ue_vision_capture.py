"""
ue_vision_capture.py — 비전 전체 검증용 측정 촬영 (팀 금오공대 맵 · 우리 VisionEval 레벨 · 노트북 · 2026-10-14)

왜
  팀 시연 경로(SITL · 게이트웨이 15 m · HFOV 78)는 그대로 두고, 측정은 우리 설정으로 따로 찍는다.
  SITL 없이 ComputerVision 모드 · 정지 → 0.1 s 진행 → 캡처 (가상 10 fps) — 렌더가 느려도 시각이 어긋나지 않는다.

하는 일
  · 카메라 1920×1080 · 수평 80.2° · 짐벌 안정화 가정(기체 수평 · 카메라 −45° 고정) · 고도 --alt m (이륙점 지면 기준)
  · 경로 = 시나리오의 지그재그를 --speed m/s 로 · 기수 = 진행 방향 (모서리는 ±2 m 앞뒤 방향으로 부드럽게)
  · 걷는 배우는 경로를 따라 매 단계 옮긴다 (제자리 걸음 애니 + 이동) · 높이 = DEM 지면 + 뼈 보정
  · 프레임마다 Scene + Segmentation → 사람 id 별 보이는 마스크 박스

출력 (서버 ue_vision_eval.py 형식) ue_flight/<run>/
  frames/<n>.jpg · flight.csv · boxes.jsonl · actors.json · run.json

준비: UE 를 우리 설정으로 띄운다 (팀 Documents\\AirSim 안 건드림)
  UnrealEditor.exe <팀 uproject> /Game/VisionEval/Maps/Vision_S1 -game -windowed -ResX=640 -ResY=360
      -settings=<저장소>/drone_yolo/configs/settings_vision_eval.json
실행 (팀 drone_simulater 가상환경 — cosysairsim 3.3)
  python ue_vision_capture.py --spec configs/vision_eval/S1.json --out ue_flight/S1_a16_test --alt 16 --seconds 30
"""
import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import cosysairsim as airsim
from cosysairsim.utils import euler_to_quaternion

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from coord_transform import ned_to_gps
from ue_posture_capture import actors as seg_actors, boxes_in

W, H, HFOV, MOUNT = 1920, 1080, 80.2, 45.0


def polyline_at(pts, s):
    """꼭짓점 목록 위 거리 s 지점 (끝이면 None)"""
    for a, b in zip(pts, pts[1:]):
        L = math.dist(a, b)
        if s <= L:
            f = s / L if L else 0.0
            return tuple(a[i] + (b[i] - a[i]) * f for i in range(len(a)))
        s -= L
    return None


def pingpong(pts, s):
    """왕복 경로 위 거리 s 의 (위치, 진행 방위 rad)"""
    total = sum(math.dist(a, b) for a, b in zip(pts, pts[1:]))
    ph = s % (2 * total)
    d, step = (ph, 0.3) if ph <= total else (2 * total - ph, -0.3)
    d = min(max(d, 0.3), total - 0.3)                         # 끝점에서도 방향을 잴 수 있게
    p, q = polyline_at(pts, d), polyline_at(pts, d + step)
    return p, math.atan2(q[1] - p[1], q[0] - p[0])


def zigzag(spec, args):
    """시나리오 묶음 중심 둘레 북쪽 방향 줄 3개 (NED · 줄 간격 --spacing)"""
    sn = spec["spawnNeu"]
    cn = sum(a["neu"][0] for a in spec["actors"]) / len(spec["actors"]) - sn[0]
    ce = sum(a["neu"][1] for a in spec["actors"]) / len(spec["actors"]) - sn[1]
    n0, n1 = cn - args.before, cn + args.after
    pts = []
    for i, de in enumerate((-args.spacing, 0.0, args.spacing)):
        a, b = (n0, n1) if i % 2 == 0 else (n1, n0)
        pts += [(a, ce + de), (b, ce + de)]
    return pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--alt", type=float, default=16.0, help="이륙점 지면 기준 m")
    ap.add_argument("--speed", type=float, default=3.5)
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--seconds", type=float, default=0, help="0 이면 경로 끝까지")
    ap.add_argument("--spacing", type=float, default=18.0)
    ap.add_argument("--before", type=float, default=55.0, help="묶음 중심 남쪽 시작 거리 m")
    ap.add_argument("--after", type=float, default=20.0)
    args = ap.parse_args()

    spec_p = Path(args.spec)
    spec = json.loads(spec_p.read_text(encoding="utf-8"))
    build = json.loads(spec_p.with_name(spec_p.stem + "_build.json").read_text(encoding="utf-8"))
    sn = spec["spawnNeu"]
    hlon, hlat = spec["spawnGeographic"][:2]
    d_alt = sn[2] - (spec["home_ground_neu_z"] + args.alt)          # NED d (AirSim 원점 = PlayerStart)
    out = Path(args.out)
    (out / "frames").mkdir(parents=True, exist_ok=True)

    c = airsim.VehicleClient(); c.confirmConnection()
    acts = seg_actors(c)
    names = {a["name"] for a in spec["actors"]}
    missing = names - set(acts)
    if missing:
        raise SystemExit(f"분할 객체에 배우가 없음 {sorted(missing)} — 레벨 · 설정(InitialInstanceSegmentation) 확인")
    print(f"배우 {len(names)} 명 분할 색 확인")
    c.simSetCameraPose("0", airsim.Pose(airsim.Vector3r(0, 0, 0), euler_to_quaternion(0, -math.radians(MOUNT), 0)))
    c.simSetCameraFov("0", HFOV)

    path = zigzag(spec, args)
    total = sum(math.dist(a, b) for a, b in zip(path, path[1:]))
    dur = total / args.speed if not args.seconds else min(args.seconds, total / args.speed)
    walkers = {a["name"]: a for a in spec["actors"] if a.get("path_neu")}

    def walker_pose(a, t):
        speed = build["actors"][a["name"]]["natural_speed_mps"] or a["speed_mps"]
        p, hd = pingpong(a["path_neu"], t * speed)
        z = p[2] + build["actors"][a["name"]]["z_offset_cm"] / 100.0
        return (p[0] - sn[0], p[1] - sn[1], sn[2] - z), hd

    c.simPause(True)
    rows, boxes_f = [], open(out / "boxes.jsonl", "w", encoding="utf-8")
    n_frames, t0 = int(dur * args.fps) + 1, time.time()
    for k in range(n_frames):
        t = k / args.fps
        s = t * args.speed
        p = polyline_at(path, s) or path[-1]
        pa, pb = polyline_at(path, max(s - 2, 0)) or path[0], polyline_at(path, s + 2) or path[-1]
        yaw = math.atan2(pb[1] - pa[1], pb[0] - pa[0])
        c.simSetVehiclePose(airsim.Pose(airsim.Vector3r(p[0], p[1], d_alt), euler_to_quaternion(0, 0, yaw)), True)
        wpos = {}
        for name, a in walkers.items():
            ned, hd = walker_pose(a, t)
            c.simSetObjectPose(name, airsim.Pose(airsim.Vector3r(*ned), euler_to_quaternion(0, 0, hd)), True)
            wpos[name] = ned
        c.simContinueForTime(1.0 / args.fps)                       # 애니 0.1 s 진행 후 다시 정지
        for _ in range(200):
            if c.simIsPause():
                break
            time.sleep(0.01)
        r = c.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False),
                            airsim.ImageRequest("0", airsim.ImageType.Segmentation, False, False)])
        rgb = np.frombuffer(r[0].image_data_uint8, np.uint8).reshape(r[0].height, r[0].width, 3)
        seg = np.frombuffer(r[1].image_data_uint8, np.uint8).reshape(r[1].height, r[1].width, 3)
        stem = f"{k + 1:05d}"
        cv2.imwrite(str(out / "frames" / f"{stem}.jpg"), rgb[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])
        bx = {}
        for name, (bb, px, cut) in boxes_in(seg, {n: acts[n] for n in names}).items():
            ned = wpos.get(name) or acts[name]["pos"]
            bx[name] = {"bb": [round(v, 1) for v in bb], "px": px, "cut": cut, "ned": [round(v, 3) for v in ned]}
        boxes_f.write(json.dumps({"frame": stem, "boxes": bx}) + "\n")
        lat, lon = ned_to_gps((hlat, hlon), p[0], p[1])
        rows.append({"frame": stem, "t_s": round(t, 3), "lat": lat, "lon": lon, "ned_x": round(p[0], 3),
                     "ned_y": round(p[1], 3), "ned_z": round(d_alt, 3), "roll_deg": 0.0, "pitch_deg": 0.0,
                     "yaw_deg": round(math.degrees(yaw), 2), "cam_pitch_rel_body_deg": -MOUNT, "people": len(bx)})
        if (k + 1) % 50 == 0:
            el = time.time() - t0
            print(f"  {k + 1}/{n_frames} · 가상 {t:.1f} s · 사람 {len(bx)} · 실제 {el:.0f} s ({(k + 1) / el:.1f} 장/s)", flush=True)
    boxes_f.close()
    c.simPause(False)

    with open(out / "flight.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    first = {a["name"]: a for a in spec["actors"]}
    (out / "actors.json").write_text(json.dumps({
        n: {"ned": [round(v, 3) for v in (walker_pose(first[n], 0.0)[0] if n in walkers else acts[n]["pos"])],
            "pose_raw": first[n]["pose_raw"], "upper_color": first[n]["upper_color"], "moving": first[n]["moving"],
            "character": first[n]["character"], "anim": build["actors"][n]["anim"]} for n in sorted(names)},
        indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "run.json").write_text(json.dumps({
        "home": [hlat, hlon], "start_ned": [path[0][0], path[0][1], round(d_alt, 3)], "mount_deg": MOUNT,
        "gimbal_stabilized": True, "hfov_deg": HFOV, "fps": args.fps, "alt_m": args.alt, "speed_mps": args.speed,
        "w": W, "h": H, "scenario": spec["scenario"], "map": spec["map_dst"], "path_ned": path,
        "time": "가상 시각 (정지 → 1/fps 진행 → 캡처)", "ned_origin": "PlayerStart = 팀 spawnNeu",
        "latlon": "home + NED 소규모 근사 (coord_transform.ned_to_gps)",
        "actor_ned": "배우 원점 (발 · 누움은 회전 축) · 걷는 배우는 프레임마다 boxes.jsonl ned"},
        indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n프레임 {len(rows)} · 가상 {rows[-1]['t_s']:.1f} s · 실제 {time.time() - t0:.0f} s → {out}")


if __name__ == "__main__":
    main()
