"""
tests_v1_2.py — 통합 v1.1 판 위 비전 1.2 교체 파일 시험 (10-14 · 백엔드 계약으로 검사)

  UE 촬영 (drone_yolo/ue_flight/<run> · 10 fps) → 프레임마다 frame-pose/1.0 · JPEG → 
    ① REAL: IngestWorker.process → VisionResult 본문을 **백엔드 real_contracts.VisionResult** 로 검사
    ② SIM : vision_sim.rich_observations → **백엔드 contracts.Observation** 으로 검사 (SIM_RICH_FIELDS=0 · 지금 계약)
  둘 다 VISION_GEO_CLAIMS 0 (기본 · v1.1 규칙) / 1 (주장) · 사람당 후보 · 상태 · 질의 · 좌표 오차를 같이 낸다
  백엔드 계약 파일은 샌드박스에서 **읽기만** (--backend 경로 · 기본 /home/se/sandboxes/drone-real)
실행: python tests_v1_2.py [--run S1_a16] [--frames 200] [--engine]
"""
import argparse, csv, json, math, os, sys, tempfile, uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]


def frame_pose(r, meta, fid, cap, clock="SIM"):
    from app.vision_core.coord_transform import rotation_body_to_ned
    from app.vision_core.pose import matrix_to_quat
    D = math.radians
    cp = D(float(r["cam_pitch_rel_body_deg"] or -meta["mount_deg"]))
    R = rotation_body_to_ned(0.0, cp, D(float(r["yaw_deg"])))
    gz = float(meta["ground_ned_z"])
    return {"schema_version": "frame-pose/1.0", "frame_id": fid, "capture_at": cap, "clock_domain": clock,
            "source_timestamp_ns": str(int(float(r["t_s"]) * 1e9)), "body_timestamp_ns": None, "sync_error_ms": None,
            "geographic": {"latitude": float(r["lat"]), "longitude": float(r["lon"]), "altitude_m": round(gz - float(r["ned_z"]), 3), "altitude_datum": "AGL"},
            "body_attitude": None, "camera_attitude": {"reference": "CAMERA_FRD_TO_NED", "quaternion": matrix_to_quat(R)},
            "camera_position_ned": {"north_m": float(r["ned_x"]), "east_m": float(r["ned_y"]), "down_m": float(r["ned_z"])},
            "mount": None, "missing_reasons": ["MOUNT_CALIBRATION_MISSING"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="S1_a16")
    ap.add_argument("--frames", type=int, default=0)
    ap.add_argument("--engine", action="store_true")
    ap.add_argument("--backend", default="/home/se/sandboxes/drone-real")
    a = ap.parse_args()
    # 배포 판 app 전체를 임시 폴더에 복사 → 교체 파일을 덮어씀 (= 통합 담당이 실제로 할 일) · 백엔드 계약은 읽기만
    import shutil
    stage = Path(tempfile.mkdtemp(prefix="v12stage_"))
    shutil.copytree(Path(a.backend) / "app", stage / "app", ignore=shutil.ignore_patterns("*.bak-*", "__pycache__"))
    for f in (HERE / "app").rglob("*.py"):
        dst = stage / "app" / f.relative_to(HERE / "app"); dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(f, dst)
    sys.path.insert(0, str(stage))
    sys.path.append(a.backend)
    import cv2
    from drone_backend.real_contracts import VisionResult
    from drone_backend.contracts import Observation as SimObservation
    from app.vision_core.detector import PersonDetector
    from app.vision_core.ingest import IngestWorker
    from app import vision_sim

    run = REPO / "drone_yolo" / "ue_flight" / a.run
    meta = json.load(open(run / "run.json")); actors = json.load(open(run / "actors.json"))
    gt = {json.loads(l)["frame"]: json.loads(l)["boxes"] for l in open(run / "boxes.jsonl")}
    rows = list(csv.DictReader(open(run / "flight.csv")))[: a.frames or None]
    W = REPO / "drone_yolo" / "runs_person" / "_pack_v12" / "v12_f91_swa"
    det = PersonDetector(W / "best.pt", engine=(W / "best_trt11.3.0.99_RTX3090.engine") if a.engine else None, conf=0.20)
    assert list(det.model.names.values()) == ["person"], det.model.names
    posture = REPO / "drone_yolo" / "runs_person" / "_pack_v12" / "posture.json"
    os.environ["VISION_GEO_AGL_FLAT"] = "1"                 # UE 장면은 배우 주변이 평지 — 가정은 method 에 표시됨
    t0 = datetime(2026, 10, 14, 6, 0, tzinfo=timezone.utc)
    out = {}
    for mode in ("REAL", "SIM"):
        for claims in (False, True):
            tmp = tempfile.mkdtemp(prefix="v12it_")
            q = Path(tmp) / "appearance_query.json"; q.write_text(json.dumps({"*": {"upper": ["green"]}}))
            w = IngestWorker(None, lambda: "x", det, str(uuid.uuid4()), scratch=tmp, upload_crops=False, posture=str(posture),
                             geo_claims=claims, environment=mode)
            mid = str(uuid.uuid4()); ok_n = 0; cand_act = {}; states = 0; matches = {}; err = []
            for r in rows:
                img = cv2.imread(str(run / "frames" / f"{r['frame']}.jpg"))
                fid = str(uuid.uuid4()); cap = (t0 + timedelta(seconds=float(r["t_s"]))).isoformat().replace("+00:00", "Z")
                pose = frame_pose(r, meta, fid, cap, "UTC" if mode == "REAL" else "SIM")
                fr = {"frame_id": fid, "mission_id": mid, "width": img.shape[1], "height": img.shape[0], "capture_at": cap, "pose": pose}
                data = cv2.imencode(".jpg", img)[1].tobytes()
                if mode == "REAL":
                    body = w.process(fr, data)
                    VisionResult.model_validate(body); ok_n += 1
                    obs = body["observations"]; bbs = [(o["bbox"]["x1"], o["bbox"]["y1"], o["bbox"]["x2"], o["bbox"]["y2"]) for o in obs]
                else:
                    job = {"analysis_run_id": str(uuid.uuid4())}
                    obs = vision_sim.observations(job, fr, data, det, worker=w)
                    for o in obs:
                        SimObservation.model_validate(o)
                    ok_n += 1
                    bbs = [(o["bbox"]["x1"], o["bbox"]["y1"], o["bbox"]["x2"], o["bbox"]["y2"]) for o in obs]
                for n, g in (gt.get(r["frame"]) or {}).items():
                    if n not in actors or g["px"] < 300:
                        continue
                    best, bi = 0.5, None
                    for k, b in enumerate(bbs):
                        ix = max(0, min(b[2], g["bb"][2]) - max(b[0], g["bb"][0])) * max(0, min(b[3], g["bb"][3]) - max(b[1], g["bb"][1]))
                        u = (b[2] - b[0]) * (b[3] - b[1]) + (g["bb"][2] - g["bb"][0]) * (g["bb"][3] - g["bb"][1]) - ix
                        if u > 0 and ix / u >= best:
                            best, bi = ix / u, k
                    if bi is None:
                        continue
                    o = obs[bi]; cand_act.setdefault(n, set()).add(o["candidate_id"])
                    if mode == "REAL":
                        states += o.get("state") is not None
                        m = (o["appearance"].get("match") or {}).get("verdict")
                        matches.setdefault(actors[n]["upper_color"] == "green", []).append(m == "MATCH")
                    geo = o["geo"] if mode == "REAL" else o["geo_result"]
                    if geo.get("status") == "VALID":
                        from app.vision_core.coord_transform import gps_to_ned
                        dn, de = gps_to_ned((float(r["lat"]), float(r["lon"])), geo["lat"], geo["lon"])
                        tr = g.get("ned") or actors[n]["ned"]
                        err.append(math.hypot(float(r["ned_x"]) + dn - tr[0], float(r["ned_y"]) + de - tr[1]))
            err.sort()
            key = f"{mode}_claims{int(claims)}"
            out[key] = {"frames_validated": ok_n, "cand_per_actor": round(sum(len(v) for v in cand_act.values()) / max(len(cand_act), 1), 2),
                        "obs_with_state": states, "geo_valid": len(err), "geo_err_p50_m": round(err[len(err) // 2], 2) if err else None,
                        "geo_err_p95_m": round(err[int(.95 * len(err))], 2) if err else None,
                        "query_green_match": round(sum(matches.get(True, [])) / max(len(matches.get(True, [])), 1), 3) if matches.get(True) else None,
                        "query_other_match": round(sum(matches.get(False, [])) / max(len(matches.get(False, [])), 1), 3) if matches.get(False) else None,
                        "backend": det.backend}
            print(key, out[key])
    (REPO / "drone_yolo" / "metrics" / f"integration_v1_2_{a.run}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
