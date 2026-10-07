"""
mock_sim_backend.py — SIM 업무 서비스 분석 작업 방식 가짜 백엔드 (표준 라이브러리 + jsonschema · 2026-10-07)

  실제 SIM 백엔드 (drone_backend/analysis.py) 규칙을 흉내 낸다:
    POST /internal/v1/analysis-runs/claim            가장 오래된 미완료 작업 · 임대 (--lease 초) · 시도 수 한도 3 → 넘으면 ANALYSIS_FAILED
    GET  /internal/v1/analysis-runs/{rid}             JobStatus (내 임대만)
    GET  /internal/v1/analysis-runs/{rid}/input        JPEG + X-Content-SHA256
    POST /internal/v1/analysis-runs/{rid}/completion   Result 스키마 · 임대 확인 (CLAIM_EXPIRED · OLD_ATTEMPT · ATTEMPT_FAILED)
                                                       · 끝난 작업에 같은 본문 → duplicate · 다르면 ID_CONFLICT
                                                       · detection_index 겹침 422 · 다른 임무 후보 403 · geo VALID 503 · geo 형식 (DB 검사) 422
    POST …/attempt-failures · …/final-failure          AttemptFailure 스키마
  작업 = 프레임마다 하나 (운영자가 POST /api/v1/analysis-runs 로 만드는 것을 미리 다 만들어 둠)

실행: python tools/mock_sim_backend.py <JPEG 폴더> [--port=18081] [--n=40] [--missions=1] [--lease=30]
        [--drop=2]     처음 2번의 completion 은 반영한 뒤 응답 없이 끊는다 (재전송 → duplicate 시험)
        [--expire=1]   처음 1번의 임대는 0초 (completion → CLAIM_EXPIRED → 다시 받기 시험)
        [--out=summary.json]
"""
import glob
import hashlib
import json
import re
import sys
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import jsonschema

HERE = Path(__file__).resolve().parent.parent
SPEC = json.load(open(HERE / "schema" / "openapi-sim-vision.json"))
ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
OPT = {a.split("=")[0][2:]: (a.split("=", 1)[1] if "=" in a else True) for a in sys.argv[1:] if a.startswith("--")}
LEASE, LIMIT = float(OPT.get("lease", 30)), 3
DROP, EXPIRE = [int(OPT.get("drop", 0))], [int(OPT.get("expire", 0))]


def schema(name):
    return {"$ref": f"#/components/schemas/{name}", "components": SPEC["components"]}


def sha(b):
    return hashlib.sha256(b).hexdigest()


def digest(v):
    return sha(json.dumps(v, sort_keys=True, separators=(",", ":")).encode())


def now():
    return datetime.now(timezone.utc)


FILES = sorted(glob.glob(str(Path(ARGS[0]) / "*.jpg")))[: int(OPT.get("n", 40))]
MISSIONS = [str(uuid.uuid4()) for _ in range(int(OPT.get("missions", 1)))]
MODEL = str(OPT.get("model", "11111111-2222-4333-8444-555555555555"))
JOBS, ATT, TERM, CAND, OBS = {}, {}, {}, {}, set()
for i, f in enumerate(FILES):
    data = open(f, "rb").read(); rid = str(uuid.uuid4())
    job = {"analysis_run_id": rid, "frame_id": str(uuid.uuid4()), "input_asset_id": str(uuid.uuid4()), "model_config_id": MODEL,
           "coordinate_config_id": None, "analysis_mode": "LIVE_DETECT", "input_hash": sha(data), "environment": "SIM",
           "pipeline_version": "SIM_V1", "analysis_revision": 1}
    job["job_hash"] = digest(job)
    JOBS[rid] = {"job": job, "data": data, "mission": MISSIONS[i * len(MISSIONS) // max(len(FILES), 1)]}
LOG = {"completed": 0, "observations": 0, "duplicates": 0, "errors": [], "geo": {}, "attempts": {}, "failures": [], "dropped_responses": 0,
       "expired_claims": 0}
LOCK = threading.Lock()


def valid_geo(g):
    """drone.valid_geo_result 와 같은 규칙 (coordinate_config 없음)"""
    if not isinstance(g, dict) or g.get("status") is None:
        return False
    if g["status"] in ("PENDING", "INVALID"):
        return isinstance(g.get("reason"), str) and g["reason"].strip() != "" and g.get("lon") is None and g.get("lat") is None
    return False                                           # VALID 은 coordinate_config 가 없으면 DB 가 막는다


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def reply(self, code, obj=None, raw=None, headers=None):
        body = raw if raw is not None else json.dumps(obj if obj is not None else {}).encode()
        self.send_response(code)
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.send_header("Content-Type", "application/octet-stream" if raw is not None else "application/json")
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

    def fail(self, code, err):
        LOG["errors"].append(f"{code} {err}")
        self.reply(code, {"error": {"code": err, "message": err}})

    def body(self):
        n = int(self.headers.get("Content-Length", 0)); raw = self.rfile.read(n)
        return json.loads(raw) if raw else {}

    def check_claim(self, rid, claim=None, attempt=None):
        a = ATT.get(rid, [])
        if not a:
            return "FORBIDDEN", 403
        a = a[-1]
        if claim is not None and a["claim_id"] != claim:
            return "CLAIM_EXPIRED", 409
        if attempt is not None and a["attempt_no"] != attempt:
            return "OLD_ATTEMPT", 409
        if datetime.fromisoformat(a["expires_at"]) <= now():
            return "CLAIM_EXPIRED", 409
        if a.get("failed"):
            return "ATTEMPT_FAILED", 409
        return None

    def do_GET(self):
        m = re.match(r"^/internal/v1/analysis-runs/([0-9a-f-]+)(/input)?$", self.path)
        if not m or m.group(1) not in JOBS:
            return self.fail(404, "RESOURCE_NOT_FOUND")
        rid = m.group(1)
        with LOCK:
            bad = self.check_claim(rid)
        if bad:
            return self.fail(bad[1], bad[0])
        j = JOBS[rid]
        if m.group(2):
            return self.reply(200, raw=j["data"], headers={"X-Content-SHA256": j["job"]["input_hash"]})
        out = {"job": j["job"], "claim": ATT[rid][-1], "terminal": None}
        jsonschema.validate(out, schema("JobStatus"))
        self.reply(200, out)

    def do_POST(self):
        try:
            b = self.body()
            if self.path == "/internal/v1/analysis-runs/claim":
                with LOCK:
                    for rid, j in JOBS.items():
                        if rid in TERM:
                            continue
                        a = ATT.get(rid, [])
                        if a and not a[-1].get("failed") and datetime.fromisoformat(a[-1]["expires_at"]) > now():
                            continue
                        n = len(a) + 1
                        if n > LIMIT:
                            TERM[rid] = {"type": "FAILED", "reason": "ATTEMPTS_EXHAUSTED"}; continue
                        lease = 0 if EXPIRE[0] > 0 else LEASE
                        if lease == 0:
                            EXPIRE[0] -= 1; LOG["expired_claims"] += 1
                        c = j["job"] | {"attempt_no": n, "worker_id": "vision-01", "claim_id": str(uuid.uuid4()),
                                        "expires_at": (now() + timedelta(seconds=lease)).isoformat()}
                        ATT.setdefault(rid, []).append(c); LOG["attempts"][str(n)] = LOG["attempts"].get(str(n), 0) + 1
                        jsonschema.validate(c, schema("Claim"))
                        return self.reply(200, c)
                return self.reply(200, {"job": None})
            m = re.match(r"^/internal/v1/analysis-runs/([0-9a-f-]+)/(completion|attempt-failures|final-failure)$", self.path)
            if not m or m.group(1) not in JOBS:
                return self.fail(404, "RESOURCE_NOT_FOUND")
            rid, kind = m.group(1), m.group(2)
            if kind != "completion":
                jsonschema.validate(b, schema("AttemptFailure"))
                with LOCK:
                    if rid in TERM:
                        return self.fail(409, "ANALYSIS_ALREADY_TERMINAL")
                    bad = self.check_claim(rid, b["claim_id"], b["attempt_no"])
                    if bad:
                        return self.fail(bad[1], bad[0])
                    LOG["failures"].append(f"{kind}: {b['reason'][:80]}")
                    if kind == "final-failure":
                        TERM[rid] = {"type": "FAILED", "reason": b["reason"]}
                    else:
                        ATT[rid][-1]["failed"] = True
                return self.reply(200, {"failed": True, "duplicate": False})
            jsonschema.validate(b, schema("Result"))
            rh = digest({"analysis_run_id": rid, "job_hash": JOBS[rid]["job"]["job_hash"], "observations": b["observations"]})
            with LOCK:
                if rid in TERM:
                    owner = ATT[rid][-1]
                    if owner["claim_id"] != b["claim_id"] or owner["attempt_no"] != b["attempt_no"] or TERM[rid].get("hash") != rh:
                        return self.fail(409, "ID_CONFLICT")
                    LOG["duplicates"] += 1
                    return self.reply(200, {"completed": True, "duplicate": True, "result_hash": rh})
                bad = self.check_claim(rid, b["claim_id"], b["attempt_no"])
                if bad:
                    return self.fail(bad[1], bad[0])
                obs = b["observations"]
                if len({o["detection_index"] for o in obs}) != len(obs):
                    return self.fail(422, "INVALID_DETECTION_INDEX")
                mission = JOBS[rid]["mission"]
                for o in obs:
                    if CAND.get(o["candidate_id"], mission) != mission:
                        return self.fail(403, "FORBIDDEN")
                    if o["geo_result"].get("status") == "VALID":
                        return self.fail(503, "REAL_COORDINATE_VALIDATION_NOT_READY")
                    if not valid_geo(o["geo_result"]):
                        return self.fail(422, "DB_CHECK valid_geo_contract")
                    if o["observation_id"] in OBS:
                        return self.fail(409, "OBSERVATION_ID_REUSED")
                for o in obs:
                    CAND[o["candidate_id"]] = mission; OBS.add(o["observation_id"])
                    s = o["geo_result"]["status"] + ("+estimate" if "sim_estimate" in o["geo_result"] else "")
                    LOG["geo"][s] = LOG["geo"].get(s, 0) + 1
                TERM[rid] = {"type": "COMPLETED", "hash": rh}
                LOG["completed"] += 1; LOG["observations"] += len(obs)
                if DROP[0] > 0:                              # 반영은 했지만 응답을 못 받은 것처럼 끊는다
                    DROP[0] -= 1; LOG["dropped_responses"] += 1
                    self.close_connection = True; self.connection.shutdown(2); return
            out = {"completed": True, "duplicate": False, "result_hash": rh, "detection_count": len(obs)}
            jsonschema.validate(out, schema("Completed"))
            self.reply(200, out)
        except jsonschema.ValidationError as e:
            self.fail(422, f"SCHEMA {e.message[:160]} @ {list(e.absolute_path)}")


def summary():
    done = [r for r, t in TERM.items() if t["type"] == "COMPLETED"]
    return {**LOG, "jobs": len(JOBS), "missions": len(MISSIONS), "terminal_completed": len(done),
            "terminal_failed": sum(t["type"] == "FAILED" for t in TERM.values()), "candidates": len(CAND),
            "candidates_per_mission": {m[:8]: sum(v == m for v in CAND.values()) for m in MISSIONS}}


if __name__ == "__main__":
    srv = ThreadingHTTPServer(("127.0.0.1", int(OPT.get("port", 18081))), H)
    print(f"가짜 SIM 백엔드 · 작업 {len(JOBS)} · 임무 {len(MISSIONS)} · 임대 {LEASE}s · 끊기 {DROP[0]} · 만료 {EXPIRE[0]}", flush=True)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        while len(TERM) < len(JOBS):
            time.sleep(0.5)
        time.sleep(1)
    finally:
        out = summary(); print(json.dumps(out, ensure_ascii=False, indent=1), flush=True)
        if OPT.get("out"):
            Path(OPT["out"]).write_text(json.dumps(out, ensure_ascii=False, indent=1))
        srv.shutdown()
