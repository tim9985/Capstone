"""
tests_low_agl.py — 10-17 이착륙 프레임 후보 막기 (VISION_MIN_AGL_M) · 노트북 Kumoh 경로 녹화 재생 (ue_flight/route_132434 · 사람 없음)

  배포 판 app 복사 + 교체 파일 (tests_v1_2.py 와 같은 방식) → IngestWorker.observe 를 시각 순서로
  비교: VISION_MIN_AGL_M 0 (끔) vs 10 · DEM 있음 / 없음 (근거 없음 → 막지 않음) · REAL SKIPPED 본문이 백엔드 계약을 통과하는지
실행  python tests_low_agl.py
"""
import json, os, shutil, sys, tempfile, uuid
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
RUN = REPO / "drone_yolo" / "ue_flight" / "route_132434"
DEM = "/home/se/sandboxes/drone-dev/data/spatial/campus-04e1b46514574d45/terrain"
BACKEND = "/home/se/sandboxes/drone-real"


def main():
    stage = Path(tempfile.mkdtemp(prefix="lowagl_"))
    shutil.copytree(Path(BACKEND) / "app", stage / "app", ignore=shutil.ignore_patterns("*.bak-*", "__pycache__"))
    for f in (HERE / "app").rglob("*.py"):
        dst = stage / "app" / f.relative_to(HERE / "app"); dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(f, dst)
    sys.path.insert(0, str(stage)); sys.path.append(BACKEND)
    import cv2
    from drone_backend.real_contracts import VisionResult
    from app.vision_core.detector import PersonDetector
    from app.vision_core.ingest import IngestWorker

    W = REPO / "drone_yolo" / "runs_person" / "_pack_v12_sim" / "v12_f91_swa"
    det = PersonDetector(W / "best.pt", engine=W / "best_trt11.3.0.99_RTX3090.engine", conf=0.20)
    recs = [json.loads(l) for l in open(RUN / "poses.jsonl")]
    imgs = {r["frame"]: cv2.imread(str(RUN / "frames" / f"{r['frame']}.jpg")) for r in recs}
    out = {}
    for name, min_agl, dem in (("끔", "0", DEM), ("10 m · DEM", "10", DEM), ("10 m · DEM 없음", "10", "")):
        os.environ["VISION_MIN_AGL_M"] = min_agl; os.environ["VISION_DEM_DIR"] = dem
        w = IngestWorker(None, lambda: "x", det, str(uuid.uuid4()), scratch=tempfile.mkdtemp(prefix="lowagl_w_"),
                         upload_crops=False, posture=None, environment="SIM")
        mid = str(uuid.uuid4()); cands, n_obs, low_obs, cruise_obs = set(), 0, 0, 0
        for r in recs:
            pose = {"schema_version": "frame-pose/1.0", **r["pose"]}
            t = datetime.fromisoformat(pose["capture_at"]).timestamp()
            obs = w.observe(mid, imgs[r["frame"]], t, pose, upload_crops=False)
            n_obs += len(obs); cands |= {o["candidate_id"] for o in obs}
            if -pose["camera_position_ned"]["down_m"] < 9.5:      # 이륙점 기준 9.5 m 아래 = 그림자 구간 (route_check)
                low_obs += len(obs)
            else:
                cruise_obs += len(obs)
        row = {"관측": n_obs, "후보": len(cands), "낮은 구간 관측": low_obs, "높은 구간 관측": cruise_obs,
               "막은 장": w.stats.get("low_altitude_frames", 0), "막은 탐지": w.stats.get("low_altitude_dets", 0),
               "근거 없음 장": w.stats.get("agl_unknown_frames", 0)}
        out[name] = row; print(name, row)
    # REAL: 막은 프레임 → SKIPPED 본문 · 계약 통과
    os.environ["VISION_MIN_AGL_M"] = "10"; os.environ["VISION_DEM_DIR"] = DEM
    w = IngestWorker(None, lambda: "x", det, str(uuid.uuid4()), scratch=tempfile.mkdtemp(prefix="lowagl_r_"),
                     upload_crops=False, posture=None, environment="REAL")
    r = recs[34]; ok, enc = cv2.imencode(".jpg", imgs[r["frame"]])
    fr = {"mission_id": str(uuid.uuid4()), "frame_id": r["pose"]["frame_id"], "width": 1920, "height": 1080,
          "capture_at": r["pose"]["capture_at"], "pose": {"schema_version": "frame-pose/1.0", **r["pose"]}}
    body = w.process(fr, enc.tobytes())
    VisionResult.model_validate(body)
    out["REAL 막은 프레임"] = {"analysis_state": body["analysis_state"], "reason": body["reason"], "계약": "통과"}
    print(out["REAL 막은 프레임"])
    (REPO / "drone_yolo" / "metrics" / "low_agl_route_132434.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
