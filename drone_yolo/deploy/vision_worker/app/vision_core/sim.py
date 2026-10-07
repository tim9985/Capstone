"""
sim.py — 비전 worker SIM 어댑터 (학교 SIM 업무 서비스 · 분석 작업 방식 · 2026-10-07)

  규격  schema/openapi-sim-vision.json (SIM 백엔드 drone_backend/analysis.py 로 동작 확인)
  흐름  POST /internal/v1/analysis-runs/claim → (작업 없음 {"job": null} 이면 쉼)
        → GET …/{rid}/input (X-Content-SHA256 = 작업의 input_hash 확인) → 디코드
        → IngestWorker.observe (탐지 · 좌표 · 후보 기억 · 색 — REAL 과 같은 코드)
        → SIM 관측 형식으로 줄임 → POST …/{rid}/completion {claim_id, attempt_no, observations}
  SIM 규칙 (백엔드 코드 기준)
        · 관측 = observation_id · candidate_id · detection_index (겹치면 안 됨) · bbox · confidence · geo_result · class_name
          → 색 · 상태 · 크롭 칸이 없다 (보낼 곳 없음 — 요청 사항)
        · geo_result VALID 는 503 REAL_COORDINATE_VALIDATION_NOT_READY → **늘 PENDING** + reason (문자열) · lat/lon 은 넣지 않는다
          계산된 추정 위치는 sim_estimate 에 따로 둔다 (DB 검사는 lat/lon 키만 본다)
        · 작업에 촬영 시각이 없다 → 후보 기억의 시각 = 받은 시각 (요청 사항)
        · 같은 작업 다시 받기 (임대 만료 · 재시도) → 이전에 만든 관측을 그대로 다시 보낸다 (후보 중복 방지)
        · completion 은 같은 본문이면 멱등 (result_hash) → 응답을 못 받으면 같은 본문으로 재전송
        · 못 읽는 입력 (디코드 · 해시) → final-failure · 그 밖 예외 → attempt-failure
"""
import hashlib
import json
import os
import time
import uuid
from pathlib import Path

import cv2
import numpy as np

SCOPE = "sim"                                             # 임무 id 를 모를 때 후보 기억 범위
KEEP = ("observation_id", "candidate_id", "detection_index", "bbox", "confidence")


def sim_geo(g):
    """vision-ingest 좌표 → SIM geo_result (VALID 못 보냄 → PENDING)"""
    out = {"status": "PENDING", "reason": None, "reasons": list(g.get("reasons") or []), "method": g.get("method")}
    if g.get("status") == "VALID":
        out["reason"] = "SIM_COORDINATE_NOT_ACCEPTED"
        out["sim_estimate"] = {"lat": g["lat"], "lon": g["lon"], "error_ellipse": g.get("error_ellipse")}
    else:
        out["reason"] = ",".join(out["reasons"]) or "NO_POSE"
    return out


def to_sim(o):
    d = {k: o[k] for k in KEEP}
    d["geo_result"] = sim_geo(o["geo"]); d["class_name"] = "person"
    return d


class Journal:
    """분석 작업별 만든 관측 (재시도 때 같은 것) · 보내지 못한 completion — /scratch/sim_journal.json"""
    def __init__(self, root, keep=5000):
        self.path, self.keep = Path(root) / "sim_journal.json", keep
        d = json.loads(self.path.read_text()) if self.path.exists() else {}
        self.obs, self.order = d.get("obs", {}), d.get("order", [])

    def put(self, rid, obs):
        if rid not in self.obs:
            self.order.append(rid)
        self.obs[rid] = obs
        while len(self.order) > self.keep:
            self.obs.pop(self.order.pop(0), None)
        self.save()

    def save(self):
        tmp = self.path.with_suffix(".tmp")
        with tmp.open("w") as f:
            json.dump({"obs": self.obs, "order": self.order}, f); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, self.path)


class SimWorker:
    def __init__(self, core, accept_model=None, scratch="/scratch", retries=3):
        self.core, self.api = core, core.api                 # core = IngestWorker (탐지 · 좌표 · 후보 · 색)
        self.accept_model = accept_model                      # None 이면 작업의 model_config_id 를 그대로 받음
        self.journal, self.retries = Journal(scratch), retries
        self.stats = {"jobs": 0, "observations": 0, "duplicates": 0, "final_failures": 0, "attempt_failures": 0, "lost_claims": 0}

    def _h(self):
        return self.core._h()

    def _post(self, path, body):
        """같은 본문으로 재시도 (completion · failure 모두 멱등)"""
        last = None
        for k in range(self.retries):
            try:
                return self.api.post(path, json=body, headers=self._h())
            except Exception as e:                            # 연결 끊김 — 서버가 반영했을 수도 있다
                last = e; time.sleep(0.5 * (k + 1))
        raise last

    def fail(self, job, final, reason):
        rid = job["analysis_run_id"]
        kind = "final-failure" if final else "attempt-failures"
        r = self._post(f"/internal/v1/analysis-runs/{rid}/{kind}",
                       {"claim_id": job["claim_id"], "attempt_no": int(job["attempt_no"]), "reason": reason[:512]})
        self.stats["final_failures" if final else "attempt_failures"] += 1
        return r.status_code

    def analyze(self, job):
        """작업 하나 → SIM 관측 목록 · 못 읽으면 (None, 이유, 최종 여부)"""
        rid = job["analysis_run_id"]
        if rid in self.journal.obs:                           # 이전 시도에서 이미 만든 것
            return self.journal.obs[rid], None, False
        ri = self.api.get(f"/internal/v1/analysis-runs/{rid}/input", headers=self._h())
        if ri.status_code != 200:
            return None, f"INPUT_HTTP_{ri.status_code}", False
        data = ri.content
        got = hashlib.sha256(data).hexdigest()
        if got != job["input_hash"] or ri.headers.get("X-Content-SHA256", got) != got:
            return None, "INPUT_HASH_MISMATCH", False
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return None, "DECODE_FAILED", True
        obs = [to_sim(o) for o in self.core.observe(job["mission_id"], img, time.time(), None, False)]
        for k, o in enumerate(obs):
            o["detection_index"] = k
        self.journal.put(rid, obs)
        return obs, None, False

    def run_once(self):
        """작업 하나를 받아 끝낸다 · 처리했으면 True"""
        r = self.api.post("/internal/v1/analysis-runs/claim", headers=self._h()); r.raise_for_status()
        job = r.json()
        if job.get("job", 1) is None:
            return False
        if self.accept_model and job["model_config_id"] != self.accept_model:
            self.fail(job, False, f"MODEL_CONFIG_NOT_LOADED {job['model_config_id']}"); return True
        job.setdefault("mission_id", SCOPE)                   # 작업에 임무 id 가 없다 (요청 사항) → 후보 기억 범위 하나
        try:
            obs, why, final = self.analyze(job)
        except Exception as e:
            self.fail(job, False, f"WORKER_ERROR {type(e).__name__}: {e}"); return True
        if obs is None:
            self.fail(job, final, why); return True
        body = {"claim_id": job["claim_id"], "attempt_no": int(job["attempt_no"]), "observations": obs}
        rc = self._post(f"/internal/v1/analysis-runs/{job['analysis_run_id']}/completion", body)
        if rc.status_code == 403 and obs:                     # 후보가 다른 임무 것 = 임무가 바뀜 → 후보 기억을 비우고 새 후보로 한 번 더
            self.new_scope(job["mission_id"], job["analysis_run_id"], obs)
            rc = self._post(f"/internal/v1/analysis-runs/{job['analysis_run_id']}/completion", body)
        if rc.status_code == 200:
            self.stats["jobs"] += 1; self.stats["observations"] += len(obs)
            self.stats["duplicates"] += bool(rc.json().get("duplicate"))
        elif rc.status_code == 409:                           # 임대 만료 · 지난 시도 → 다시 받으면 같은 관측을 보낸다
            self.stats["lost_claims"] += 1
        else:
            code = (rc.json().get("error") or {}).get("code", "") if rc.headers.get("content-type", "").startswith("application/json") else ""
            self.fail(job, False, f"COMPLETION_HTTP_{rc.status_code} {code}")
        return True

    def new_scope(self, mid, rid, obs):
        """임무가 바뀐 것으로 보고 후보 기억을 새로 시작 · 이번 관측의 후보 id 를 새로 매긴다"""
        self.core.reg.pop(mid, None)
        old = {k: v for k, v in self.core.uid.items() if k[0] == mid}
        for k in old:
            self.core.uid.pop(k)
        fresh = {}
        for o in obs:
            o["candidate_id"] = fresh.setdefault(o["candidate_id"], str(uuid.uuid4()))
        self.journal.put(rid, obs); self.stats["scope_resets"] = self.stats.get("scope_resets", 0) + 1
