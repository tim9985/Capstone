"""
tests_pose.py — frame-pose/1.0 → 좌표 (app/vision_core/pose.py) 검사 (10-14)

  ① 단위: 쿼터니언 왕복 · 자세 없음 / 지면 모름 → PENDING · AGL 평지 가정 표시 · REAL 시계 · 동기 오차
  ② UE 기록 (drone_yolo/ue_flight/<run> · 있으면): flight.csv → frame-pose/1.0 (카메라 자세 쿼터니언 · 카메라 NED 위치) →
     **정답 박스** 발끝 → 좌표 vs 배우 정답 위치 · 같은 장면을 옛 경로 (geo.GeoResolver · 짐벌 각) 로 낸 값과도 비교 (같아야 함)
실행: python tests_pose.py [S1_a16 S3_a20 …]
"""
import csv, json, math, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from app.vision_core.coord_transform import rotation_body_to_ned, gps_to_ned  # noqa: E402
from app.vision_core.geo import CameraModel, GeoResolver, Telemetry  # noqa: E402
from app.vision_core.pose import PoseGeo, matrix_to_quat, quat_to_matrix  # noqa: E402

D = math.radians
ok = 0


def check(cond, msg):
    global ok
    assert cond, msg
    ok += 1


def unit():
    R = rotation_body_to_ned(D(3), D(-45), D(120))
    R2 = quat_to_matrix(matrix_to_quat(R))
    check(max(abs(R[i][j] - R2[i][j]) for i in range(3) for j in range(3)) < 1e-9, "쿼터니언 왕복")
    cam = CameraModel()
    base = {"schema_version": "frame-pose/1.0", "clock_domain": "SIM", "geographic": {"latitude": 36.14, "longitude": 128.39, "altitude_m": 16.0, "altitude_datum": "AGL"},
            "camera_attitude": {"reference": "CAMERA_FRD_TO_NED", "quaternion": matrix_to_quat(rotation_body_to_ned(0, D(-45), 0))},
            "camera_position_ned": {"north_m": 0.0, "east_m": 0.0, "down_m": -16.0}, "missing_reasons": []}
    box = (950, 520, 970, 560)
    g = PoseGeo(cam).locate(box, base)
    check(g["status"] == "PENDING" and "GROUND_UNKNOWN" in g["reasons"], "지면 모르면 보류 (평지로 채우지 않음)")
    g = PoseGeo(cam, allow_agl_flat=True).locate(box, base)
    check(g["status"] == "CLAIM" and g["assumptions"] == ["LOCAL_FLAT_AGL"] and abs(g["north_m"] - 16.0 / math.tan(D(45))) < 1.0, "AGL 평지 가정은 표시하고 계산")
    g = PoseGeo(cam, terrain=lambda n, e: 0.0).locate(box, base)
    check(g["status"] == "CLAIM" and not g["assumptions"], "지형 함수로 계산")
    nb = dict(base); nb.pop("camera_attitude")
    check("CAMERA_ATTITUDE_MISSING" in PoseGeo(cam, terrain=lambda n, e: 0.0).locate(box, nb)["reasons"], "카메라 자세 없음 → 보류 (짐벌 −45° 로 채우지 않음)")
    mb = dict(nb, body_attitude={"reference": "BODY_FRD_TO_NED", "quaternion": matrix_to_quat(rotation_body_to_ned(0, 0, 0))},
              mount={"kind": "FIXED", "calibration_id": "t", "orientation": {"reference": "CAMERA_FRD_TO_BODY_FRD", "quaternion": matrix_to_quat(rotation_body_to_ned(0, D(-45), 0))}})
    g2 = PoseGeo(cam, terrain=lambda n, e: 0.0).locate(box, mb)
    check(g2["source"] == "body_attitude×mount" and abs(g2["north_m"] - g["north_m"]) < 1e-6 if g["north_m"] else True, "기체 자세 × 마운트 = 카메라 자세")
    check("POSE_CLOCK" in PoseGeo(cam, terrain=lambda n, e: 0.0, environment="REAL").locate(box, base)["reasons"], "REAL 은 SIM 시계 거부")
    check(any(r.startswith("SYNC_ERROR") for r in PoseGeo(cam, terrain=lambda n, e: 0.0).locate(box, dict(base, sync_error_ms=80.0))["reasons"]), "동기 오차 보류")
    check(PoseGeo(cam).locate(box, None)["reasons"] == ["POSE_MISSING"], "pose 없음")


def ue(run):
    d = HERE.parents[1] / "ue_flight" / run
    if not (d / "flight.csv").exists():
        print(f"  {run}: 기록 없음 — 건너뜀"); return None
    meta = json.load(open(d / "run.json")); actors = json.load(open(d / "actors.json"))
    gt = {json.loads(l)["frame"]: json.loads(l)["boxes"] for l in open(d / "boxes.jsonl")}
    gz = float(meta["ground_ned_z"]); cam = CameraModel(hfov_deg=float(meta["hfov_deg"]))
    pg, old = PoseGeo(cam, terrain=lambda n, e: gz), GeoResolver(cam)
    err, diff, n_claim, n_all = [], [], 0, 0
    home = None
    for r in csv.DictReader(open(d / "flight.csv")):
        yaw, cp = D(float(r["yaw_deg"])), D(float(r["cam_pitch_rel_body_deg"] or -meta["mount_deg"]))
        R = rotation_body_to_ned(0.0, cp, yaw) if meta.get("gimbal_stabilized") else rotation_body_to_ned(D(float(r["roll_deg"])), D(float(r["pitch_deg"])) + cp, yaw)
        pose = {"schema_version": "frame-pose/1.0", "clock_domain": "SIM", "missing_reasons": [],
                "geographic": {"latitude": float(r["lat"]), "longitude": float(r["lon"]), "altitude_m": gz - float(r["ned_z"]), "altitude_datum": "AGL"},
                "camera_attitude": {"reference": "CAMERA_FRD_TO_NED", "quaternion": matrix_to_quat(R)},
                "camera_position_ned": {"north_m": float(r["ned_x"]), "east_m": float(r["ned_y"]), "down_m": float(r["ned_z"])}}
        tel = Telemetry(float(r["lat"]), float(r["lon"]), gz - float(r["ned_z"]), D(float(r["roll_deg"])), D(float(r["pitch_deg"])), yaw,
                        gimbal_pitch_deg=math.degrees(cp), stabilized=bool(meta.get("gimbal_stabilized")))
        for n, g in (gt.get(r["frame"]) or {}).items():
            if g["cut"] or g["px"] < 300 or n not in actors:
                continue
            n_all += 1
            a = pg.locate(g["bb"], pose)
            if a["status"] != "CLAIM":
                continue
            n_claim += 1
            truth = g.get("ned") or actors[n]["ned"]
            err.append(math.hypot(float(r["ned_x"]) + a["north_m"] - truth[0], float(r["ned_y"]) + a["east_m"] - truth[1]))
            b = old.to_world(g["bb"], tel)
            if b.north is not None:
                diff.append(math.hypot(a["north_m"] - b.north, a["east_m"] - b.east))
    err.sort(); diff.sort()
    q = lambda v, p: v[min(len(v) - 1, int(p * len(v)))] if v else None
    res = {"run": run, "boxes": n_all, "claim": n_claim, "err_p50_m": round(q(err, .5), 2), "err_p95_m": round(q(err, .95), 2),
           "old_path_diff_max_m": round(max(diff), 4) if diff else None}
    print(f"  {run}: 정답 박스 {n_all} · 주장 {n_claim} · 오차 중앙 {res['err_p50_m']} · P95 {res['err_p95_m']} m · 옛 경로와 차 최대 {res['old_path_diff_max_m']} m")
    return res


def ue_dem(run, tiles="/home/se/sandboxes/drone-dev/data/spatial/campus-04e1b46514574d45/terrain"):
    """UE 기록 → frame-pose/1.0 (고도 MSL = PlayerStart 71.429 − NED z · AirSim gps 와 같은 식 · 10-17 로컬 실측) → DEM 좌표 vs 정답"""
    from app.vision_core.pose import DemTiles
    d = HERE.parents[1] / "ue_flight" / run
    if not (d / "flight.csv").exists() or not Path(tiles).exists():
        return None
    meta = json.load(open(d / "run.json")); actors = json.load(open(d / "actors.json"))
    gt = {json.loads(l)["frame"]: json.loads(l)["boxes"] for l in open(d / "boxes.jsonl")}
    pg = PoseGeo(CameraModel(hfov_deg=float(meta["hfov_deg"])), dem=DemTiles(tiles))
    err = []
    for r in list(csv.DictReader(open(d / "flight.csv")))[::3]:
        R = rotation_body_to_ned(0.0, D(float(r["cam_pitch_rel_body_deg"])), D(float(r["yaw_deg"])))
        pose = {"schema_version": "frame-pose/1.0", "clock_domain": "SIM", "missing_reasons": [],
                "geographic": {"latitude": float(r["lat"]), "longitude": float(r["lon"]), "altitude_m": 71.429 - float(r["ned_z"]), "altitude_datum": "MSL"},
                "camera_attitude": {"reference": "CAMERA_FRD_TO_NED", "quaternion": matrix_to_quat(R)}}
        for n, g in (gt.get(r["frame"]) or {}).items():
            if g["cut"] or g["px"] < 300 or n not in actors:
                continue
            a = pg.locate(g["bb"], pose)
            if a["status"] != "CLAIM":
                continue
            tr = g.get("ned") or actors[n]["ned"]
            err.append(math.hypot(float(r["ned_x"]) + a["north_m"] - tr[0], float(r["ned_y"]) + a["east_m"] - tr[1]))
    err.sort()
    res = {"run": run, "claim": len(err), "err_p50_m": round(err[len(err) // 2], 2), "err_p95_m": round(err[int(.95 * len(err))], 2)}
    print(f"  DEM {run}: 주장 {res['claim']} · 오차 중앙 {res['err_p50_m']} · P95 {res['err_p95_m']} m (method +DEM)")
    return res


if __name__ == "__main__":
    unit(); print(f"단위 {ok} 통과")
    runs = sys.argv[1:] or ["S1_a16", "S3_a16", "S4_a16", "S1_a20", "S3_a20", "S4_a20", "S2_a16", "S2_a20"]
    out = [x for x in (ue(r) for r in runs) if x]
    out_dem = [x for x in (ue_dem(r) for r in runs) if x]
    if out:
        p = HERE.parents[1] / "metrics" / "pose_geo_ue.json"; p.write_text(json.dumps({"flat_scene": out, "dem_msl": out_dem}, ensure_ascii=False, indent=1)); print("→", p)
