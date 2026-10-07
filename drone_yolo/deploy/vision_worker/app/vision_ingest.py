"""
vision_ingest.py — 비전 worker v1.0 실행 파일 (vision-ingest/1.0 · 2026-10-07)

  샌드박스 vision 컨테이너에서:  python -m app.vision_ingest
    · 백엔드 = unix 소켓 + mTLS + Bearer (app.common.client · /run/identity) — 비밀 파일은 읽기만, 출력 · 로그에 안 남김
    · 모델 = /models/active.json {"name": "soup_v7r2", "model_config_id": "<등록된 uuid>"} → /models/<name>/best.pt
             (같은 폴더에 best*.engine 이 있으면 TensorRT · 없으면 PyTorch FP16) · 후보 모델은 폴더만 갈아 끼우면 된다
    · 쓰기 = /scratch (outbox.jsonl · ack.json · ingest_state.json · worker-status.json)
    · 전송 = app.vision_outbox.send_pending (통합 담당 전송기 · 같은 컨테이너에 있으면) · 없으면 같은 규칙의 내장 전송기
  서버 밖 시험:  INGEST_BASE_URL=http://127.0.0.1:18080 INGEST_TOKEN=test INGEST_MODEL_DIR=... INGEST_SCRATCH=... python -m app.vision_ingest
"""
import hashlib
import json
import os
import time
from pathlib import Path

ROOT = Path(os.environ.get("INGEST_SCRATCH", "/scratch"))
MODELS = Path(os.environ.get("INGEST_MODEL_DIR", "/models"))


def status(**kw):
    p = ROOT / "worker-status.json"
    d = json.loads(p.read_text()) if p.exists() else {}
    d.update(kw, updated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    tmp = p.with_suffix(".tmp"); tmp.write_text(json.dumps(d, ensure_ascii=False)); os.replace(tmp, p)


def make_client():
    base = os.environ.get("INGEST_BASE_URL")
    if base:                                              # 서버 밖 시험 (가짜 백엔드)
        import httpx
        tok = os.environ.get("INGEST_TOKEN", "test")
        return httpx.Client(base_url=base, timeout=30, trust_env=False), (lambda: tok)
    from app.common import client, text
    return client("/run/backend-socket/service.sock", "backend.internal"), (lambda: text("token"))


def local_send_pending(api, token, root=ROOT):
    """app.vision_outbox.send_pending 과 같은 규칙 (체크포인트 · 경계 해시 · 성공한 줄까지만 기록)"""
    queue, ck = root / "outbox.jsonl", root / "ack.json"
    if not queue.exists():
        return 0
    st = json.loads(ck.read_text()) if ck.exists() else {"offset": 0}
    if queue.stat().st_size < st["offset"]:
        raise ValueError("outbox 가 줄었다 — 보낸 기록을 지키려고 멈춤")
    n = 0
    with queue.open("rb") as f:
        if st["offset"]:
            f.seek(st["offset"] - st["last_length"])
            if hashlib.sha256(f.read(st["last_length"])).hexdigest() != st["last_sha256"]:
                raise ValueError("보낸 경계가 바뀌었다 — 건너뛰지 않고 멈춤")
        f.seek(st["offset"])
        while True:
            line = f.readline(65538)
            if not line or not line.endswith(b"\n"):
                return n
            row = json.loads(line)
            path = {"result": "/internal/v1/vision/results", "reobservation": "/internal/v1/vision/reobservations"}[row["type"]]
            api.post(path, json=row["body"], headers={"Authorization": "Bearer " + token()}).raise_for_status()
            tmp = ck.with_suffix(".tmp")
            with tmp.open("w") as o:
                json.dump({"offset": f.tell(), "last_length": len(line), "last_sha256": hashlib.sha256(line).hexdigest()}, o); o.flush(); os.fsync(o.fileno())
            os.replace(tmp, ck); n += 1


def load_detector():
    from app.vision_core.detector import PersonDetector
    act = json.loads((MODELS / "active.json").read_text())
    d = MODELS / act["name"]
    eng = next(iter(sorted(d.glob("best*.engine"))), None)
    det = PersonDetector(d / "best.pt", engine=eng, conf=float(act.get("conf", 0.15)))
    return det, act


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    status(state="starting", model_loaded=False, flight_enabled=False)
    api, token = make_client()
    try:
        det, act = load_detector()
    except Exception as e:
        status(state="waiting_for_model", error=f"{type(e).__name__}: {e}"[:300]); raise
    from app.vision_core.ingest import IngestWorker
    w = IngestWorker(api, token, det, act["model_config_id"], scratch=ROOT, upload_crops=bool(act.get("upload_crops", True)))
    try:
        from app.vision_outbox import send_pending as sp
        send = lambda: sp(api, ROOT)
        sender = "app.vision_outbox"
    except Exception:
        send = lambda: local_send_pending(api, token)
        sender = "내장"
    status(state="running", model_loaded=True, model=act["name"], model_config_id=act["model_config_id"], backend=det.backend,
           sender=sender, producer_session_id=w.session, error=None)
    last = 0.0
    while True:
        try:
            n = w.poll_once(send=send)
            if time.time() - last > 2:
                status(state="running", **w.stats, sequence=w.state.sequence); last = time.time()
            if not n:
                time.sleep(0.5)
        except Exception as e:
            status(state="retrying", error=f"{type(e).__name__}: {str(e)[:200]}")
            time.sleep(2)


if __name__ == "__main__":
    main()
