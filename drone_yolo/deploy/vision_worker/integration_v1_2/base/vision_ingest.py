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


def load_detector(expected_sha256=None):
    import uuid
    import torch
    import numpy as np
    from app.vision_core.detector import PersonDetector
    act = json.loads((MODELS / "active.json").read_text())
    uuid.UUID(act['model_config_id'])
    d = (MODELS / act["name"]).resolve()
    if not d.is_relative_to(MODELS.resolve()):raise ValueError('Model path escapes model volume')
    manifest=json.loads((d/'model.json').read_text())
    weight=d/'best.pt'
    if not weight.resolve().is_relative_to(d):raise ValueError('Weight path escapes immutable model directory')
    if manifest['environment']!=os.environ.get('INGEST_ENVIRONMENT','REAL'):raise ValueError('Model environment mismatch')
    if hashlib.sha256(weight.read_bytes()).hexdigest()!=manifest['sha256']:raise ValueError('Model hash mismatch')
    if expected_sha256 is not None and manifest['sha256']!=expected_sha256:raise ValueError('Registered model hash mismatch')
    if not torch.cuda.is_available():raise RuntimeError('CUDA required; CPU fallback disabled')
    torch.set_num_threads(2)
    det = PersonDetector(weight, conf=float(act.get("conf", 0.15)))
    if list(det.model.names.values())!=['person']:raise ValueError('Expected a single person class')
    det.detect(np.zeros((1080,1920,3),np.uint8))
    act['artifact_sha256']=manifest['sha256']
    act['environment']=manifest['environment']
    return det, act


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    status(state="starting", model_loaded=False, flight_enabled=False)
    api, token = make_client()
    try:
        active=json.loads((MODELS/'active.json').read_text())
        registered=api.get('/internal/v1/vision/models/'+active['model_config_id'],headers={'Authorization':'Bearer '+token()})
        registered.raise_for_status()
        det, act = load_detector(registered.json()['payload']['artifact_sha256'])
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
    if act['environment']=='SIM':sender='leased-analysis-api'
    status(state="running", model_loaded=True, model=act["name"], model_config_id=act["model_config_id"], backend=det.backend,
           sender=sender, producer_session_id=w.session, error=None)
    if act['environment']=='SIM':
        from app.vision_sim import poll_once
        poll=lambda:poll_once(api,token,det,act,ROOT,w.stats)
    else:poll=lambda:w.poll_once(send=send)
    last = 0.0
    last_status = 0.0
    while True:
        try:
            if time.time()-last>=15:
                api.post('/internal/v1/vision/status',headers={'Authorization':'Bearer '+token()},json={
                    'model_config_id':act['model_config_id'],'artifact_sha256':act['artifact_sha256'],
                    'model_loaded':True,'cuda_verified':True,'state':'running'}).raise_for_status()
                last=time.time()
            n = poll()
            if time.time() - last_status > 2:
                status(state="running", **w.stats, sequence=w.state.sequence); last_status = time.time()
            if not n:
                time.sleep(0.5)
        except Exception as e:
            status(state="retrying", error=f"{type(e).__name__}: {str(e)[:200]}")
            time.sleep(2)


if __name__ == "__main__":
    main()
