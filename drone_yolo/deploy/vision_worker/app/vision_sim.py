"""
vision_sim.py — 비전 worker SIM 실행 파일 (학교 SIM 업무 서비스 · 분석 작업 방식 · 2026-10-07)

  SIM 샌드박스 vision 컨테이너에서:  python -m app.vision_sim
    · 백엔드 · 모델 · /scratch 는 vision_ingest 와 같다 (app.common.client · /models/active.json · best*.engine)
    · active.json 의 model_config_id 가 uuid 면 그 모델 작업만 받고, 비어 있거나 "<…>" 면 작업이 고른 것을 그대로 받는다
  서버 밖 시험:  INGEST_BASE_URL=http://127.0.0.1:18081 INGEST_MODEL_DIR=... INGEST_SCRATCH=... python -m app.vision_sim
                 (가짜 SIM 백엔드: python tools/mock_sim_backend.py <JPEG 폴더>)
"""
import os
import re
import time

from app.vision_ingest import ROOT, load_detector, make_client, status

UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    status(state="starting", contract="sim-analysis-runs", model_loaded=False, flight_enabled=False)
    api, token = make_client()
    try:
        det, act = load_detector()
    except Exception as e:
        status(state="waiting_for_model", error=f"{type(e).__name__}: {e}"[:300]); raise
    from app.vision_core.ingest import IngestWorker
    from app.vision_core.sim import SimWorker
    mid = str(act.get("model_config_id") or "")
    core = IngestWorker(api, token, det, mid or "unset", scratch=ROOT, upload_crops=False)
    w = SimWorker(core, accept_model=mid if UUID.match(mid) else None, scratch=ROOT)
    status(state="running", model_loaded=True, model=act["name"], accept_model=w.accept_model, backend=det.backend, error=None)
    last, stop_after = 0.0, int(os.environ.get("INGEST_STOP_AFTER_IDLE", "0"))
    idle = 0
    while True:
        try:
            busy = w.run_once()
            if time.time() - last > 2:
                status(state="running", **w.stats, error=None); last = time.time()
            if not busy:
                idle += 1
                if stop_after and idle >= stop_after:             # 시험용 — 작업이 없으면 끝냄
                    status(state="stopped", **w.stats); return
                time.sleep(0.5)
            else:
                idle = 0
        except Exception as e:
            status(state="retrying", error=f"{type(e).__name__}: {str(e)[:200]}")
            idle += 1
            if stop_after and idle >= stop_after:                 # 시험용 — 백엔드가 끝났으면 끝냄
                status(state="stopped", **w.stats); return
            time.sleep(2)


if __name__ == "__main__":
    main()
