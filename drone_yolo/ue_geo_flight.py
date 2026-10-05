"""
ue_geo_flight.py — 팀 금오공대 시뮬(UE + Cosys-AirSim + ArduPilot SITL)에서 실제 비행하며 좌표 산정 검증용 기록 (노트북 · 2026-10-05)

왜
  ue_geo_validate.py 는 순간이동(ComputerVision) · 평지라 텔레메트리 추정 · 시각 어긋남 · 지형을 못 본다.
  여기서는 SITL 이 낸 위치·자세(MAVLink)와 영상을 같이 남겨 GeoResolver 를 실제 운용 입력으로 돌릴 수 있게 한다.

하는 일 (팀 게이트웨이 http://127.0.0.1:8015 가 떠 있고 기체가 착륙 · 시동 꺼짐 상태일 때)
  ① 평가 배우 3명을 지면에 놓는다 (서 있음 2 · 누움 1) — 팀 run_scenario 와 같은 규칙: 기체가 땅에 있을 때만 옮긴다
  ② 게이트웨이 GOTO 로 배우 묶음 중심을 지나는 현(chord) 비행 (고도 15 m 고정 · 2 m/s · 기수는 진행 방향)
  ③ 비행 중 --fps 로 RGB + 그 순간 카메라 참값 자세(영상 응답의 camera_position/orientation) 저장,
     별도 스레드가 게이트웨이 텔레메트리(SITL 추정 위경도 · 해발 고도 · roll/pitch/yaw)를 10 Hz 로 기록
  ④ 끝나면 RETURN(복귀 착륙) → 착륙 확인 후 배우를 원래 숨김 위치로

출력 ue_geo/<run>/ (git 밖)
  frames/<n>.jpg · frames.csv (영상 시각 ns · 카메라 NED 위치 · 쿼터니언) · telemetry.csv · actors.json · run.json
평가: ue_geo_flight_eval.py (탐지 · GeoResolver · 지표)

실행 (팀 drone_simulater 가상환경 — cosysairsim 3.3)
  C:/work/capstone1_project/drone_simulater/.venv/Scripts/python.exe ue_geo_flight.py --out ue_geo/run01
"""
import argparse
import csv
import json
import math
import sys
import threading
import time
import urllib.request
import uuid
from pathlib import Path

import cv2
import numpy as np
import cosysairsim as airsim

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from coord_transform import ned_to_gps

API = "http://127.0.0.1:8015/api/simulation"
CAMERA = "front_center"
VEHICLE = "Copter"
HIDDEN = ("Eval_Person_A", "Eval_Person_B", "Eval_Person_A_Moving", "Eval_Person_B_Moving", "Eval_Occluder")
GROUND_OFFSET_M = 0.08          # UE 지면 = DEM + 0.08 m (시나리오 5곳 모두 같음 · 10-05)


class Gateway:
    def __init__(self, runtime):
        self.token = json.loads((runtime / "relay-secrets.json").read_text())["operator"]   # 출력하지 않는다
        self.latest, self.rows, self.stop = None, [], False
        self.lock = threading.Lock()

    def state(self):
        with urllib.request.urlopen(f"{API}/state", timeout=3) as r:
            return json.load(r)

    def send(self, kind, target=None):
        s, now = self.state(), time.time()
        data = {**s["binding"], "commandId": uuid.uuid4().hex, "type": kind, "sequence": s["nextSequence"],
                "issuedAt": now, "expiresAt": now + 10}
        if target is not None:
            data["target"] = target
        req = urllib.request.Request(f"{API}/commands", data=json.dumps(data).encode(),
                                     headers={"Content-Type": "application/json", "Authorization": "Bearer " + self.token})
        with urllib.request.urlopen(req, timeout=5) as r:
            return json.load(r)["commandId"]

    def poll(self):
        """10 Hz 텔레메트리 기록 — 같은 MAVLink 시각은 한 번만"""
        last = None
        while not self.stop:
            try:
                s = self.state()
                with self.lock:
                    self.latest = s
                t = s.get("telemetry") or {}
                if t.get("valid") and t.get("timestamp") != last and t.get("position") and t.get("attitude"):
                    last = t["timestamp"]
                    lon, lat, alt = t["position"]
                    self.rows.append({"t": t["timestamp"], "recv": time.time(), "lat": lat, "lon": lon, "alt_msl": alt,
                                      "rel_alt": t.get("relativeAltitudeM"), "roll": t["attitude"][0],
                                      "pitch": t["attitude"][1], "yaw": t["attitude"][2],
                                      "ned_n": t["ned"][0], "ned_e": t["ned"][1], "ned_d": t["ned"][2],
                                      "armed": t.get("armed"), "phase": t.get("phase")})
            except Exception:
                pass
            time.sleep(0.1)

    def snap(self):
        with self.lock:
            return self.latest


def command_status(state, cid):
    return next((c["status"] for c in state.get("commands", []) if c["commandId"] == cid), None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--team-root", default="C:/work/capstone1_project")
    ap.add_argument("--fps", type=float, default=4.0)
    ap.add_argument("--radius", type=float, default=45.0, help="현 끝점 반지름 m (배우 묶음 중심 기준)")
    ap.add_argument("--max-minutes", type=float, default=15.0)
    args = ap.parse_args()

    runtime = Path(args.team_root) / "simulation_world" / "runtime"
    cfg = json.loads((runtime / "config.json").read_text(encoding="utf-8"))
    sn = cfg["spawnNeu"]
    spawn_lon, spawn_lat = cfg["spawnGeographic"][:2]
    out = Path(args.out)
    (out / "frames").mkdir(parents=True, exist_ok=True)

    gw = Gateway(runtime)
    s = gw.state()
    t = s.get("telemetry") or {}
    if not (s.get("connected") and t.get("valid") and not t.get("armed") and s.get("controlEnabled")):
        raise SystemExit("게이트웨이 연결 · 텔레메트리 유효 · 시동 꺼짐 · 로컬 제어 켜짐 상태여야 함")
    client = airsim.MultirotorClient(ip="127.0.0.1", port=41451, timeout_value=30)
    client.confirmConnection()

    def neu_pose(neu, yaw=0.0, pitch=0.0):
        return airsim.Pose(airsim.Vector3r(neu[0] - sn[0], neu[1] - sn[1], sn[2] - neu[2]),
                           airsim.euler_to_quaternion(0, pitch, yaw))

    # ① 배우 배치 — 통합 시나리오 지점(검증된 지면) 주변. 지면 z(NEU) = terrain.tif 쌍선형 − 원점 고도 + 0.08
    #    (팀 drone 가상환경엔 rasterio 가 없어 미리 계산 · 첫 점은 시나리오 값 −5.93 과 일치)
    people = [("Eval_Person_A", "standing", (60.02, -15.0, -5.934), 0.6),
              ("Eval_Person_A_Moving", "standing", (67.02, -6.0, -7.426), 2.2),
              ("Eval_Person_B", "lying", (54.02, -7.0, -6.169), 1.1)]
    ground = {name: z for name, _, (_, _, z), _ in people}
    actors = []
    for name, mode, (n, e, gz), yaw in people:
        z = gz + (0.32 if mode == "lying" else 0.0)
        client.simSetObjectPose(name, neu_pose((n, e, z), yaw, math.pi / 2 if mode == "lying" else 0.0), True)
        p = client.simGetObjectPose(name).position
        actors.append({"name": name, "mode": mode, "ned": [p.x_val, p.y_val, p.z_val], "yaw": yaw,
                       "ground_ned_d": sn[2] - ground[name]})
        print(f"배우 {name} {mode} NED ({p.x_val:.2f}, {p.y_val:.2f}, {p.z_val:.2f})")
    (out / "actors.json").write_text(json.dumps(actors, indent=2), encoding="utf-8")

    # ② 현 비행 — 끝점 각도: 180° · 135° 를 번갈아 (중심을 지나는 현 + 중심에서 0.38R 떨어진 현)
    cn = sum(a["ned"][0] for a in actors) / len(actors)
    ce = sum(a["ned"][1] for a in actors) / len(actors)
    angles, a = [0.0], 0.0
    for k in range(7):
        a = (a + (180.0 if k % 2 == 0 else 135.0)) % 360
        angles.append(a)
    legs = [(cn + args.radius * math.cos(math.radians(g)), ce + args.radius * math.sin(math.radians(g))) for g in angles]
    targets = []
    for n, e in legs:
        lat, lon = ned_to_gps((spawn_lat, spawn_lon), n, e)
        targets.append({"longitude": lon, "latitude": lat})
    print(f"현 {len(legs) - 1}개 · 끝점 각도 {angles}")

    threading.Thread(target=gw.poll, daemon=True).start()
    rows, n, dt = [], 0, 1.0 / args.fps
    t0 = time.time()
    leg, cid, returning = 0, gw.send("GOTO", targets[0]), False
    print("이륙 + 첫 끝점으로 이동")
    try:
        while time.time() - t0 < args.max_minutes * 60:
            t_req = time.time()
            st = gw.snap() or {}
            tel = st.get("telemetry") or {}
            status = command_status(st, cid) if st else None
            if status in ("FAILED", "REJECTED", "UNKNOWN"):
                raise RuntimeError(f"명령 실패 {status} (구간 {leg})")
            if status == "COMPLETED":
                if returning:
                    print("복귀 · 착륙 완료"); break
                leg += 1
                if leg < len(targets):
                    time.sleep(0.5)
                    cid = gw.send("GOTO", targets[leg]); print(f"  구간 {leg}/{len(targets) - 1} 시작")
                else:
                    cid, returning = gw.send("RETURN"), True; print("복귀 시작")
            if (tel.get("relativeAltitudeM") or 0) > 5 and not returning:
                r = client.simGetImages([airsim.ImageRequest(CAMERA, airsim.ImageType.Scene, False, False)], VEHICLE)[0]
                if r.width == 1920:
                    n += 1
                    rgb = np.frombuffer(r.image_data_uint8, np.uint8).reshape(r.height, r.width, 3)
                    cv2.imwrite(str(out / "frames" / f"{n:05d}.jpg"), rgb[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])
                    cp, cq = r.camera_position, r.camera_orientation
                    rows.append({"frame": f"{n:05d}", "t_img": r.time_stamp / 1e9, "t_req": t_req, "t_recv": time.time(),
                                 "cam_n": cp.x_val, "cam_e": cp.y_val, "cam_d": cp.z_val,
                                 "qw": cq.w_val, "qx": cq.x_val, "qy": cq.y_val, "qz": cq.z_val, "leg": leg})
                    if n % 50 == 0:
                        print(f"  프레임 {n} · 구간 {leg} · 고도 {tel.get('relativeAltitudeM'):.1f} m", flush=True)
            time.sleep(max(0.0, dt - (time.time() - t_req)))
        else:
            print("시간 초과 — 복귀 명령"); gw.send("RETURN")
    finally:
        def write(name, data):
            if data:
                with open(out / name, "w", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
        write("frames.csv", rows)
        # 착륙 · 시동 꺼짐 확인 후에만 배우를 숨긴다 (비행 중인 기체 아래로 옮기지 않는다)
        for _ in range(600):
            t = (gw.snap() or {}).get("telemetry") or {}
            if t.get("valid") and not t.get("armed"):
                for i, name in enumerate(HIDDEN):
                    client.simSetObjectPose(name, neu_pose((0, 0, -500 - i * 5)), True)
                print("배우 숨김 복원"); break
            time.sleep(0.5)
        gw.stop = True; time.sleep(0.3)
        write("telemetry.csv", gw.rows)
        (out / "run.json").write_text(json.dumps({"spawnNeu": sn, "spawnGeographic": cfg["spawnGeographic"],
            "origin": cfg["frame"]["origin"], "release": cfg["release"], "camera": CAMERA, "hfov": 78.0,
            "mount_pitch_deg": -45.0, "w": 1920, "h": 1080, "ground_offset_m": GROUND_OFFSET_M,
            "legs_ned": legs, "frames": len(rows), "telemetry": len(gw.rows)}, indent=2, ensure_ascii=False),
            encoding="utf-8")
        print(f"프레임 {len(rows)} · 텔레메트리 {len(gw.rows)} → {out}")


if __name__ == "__main__":
    main()
