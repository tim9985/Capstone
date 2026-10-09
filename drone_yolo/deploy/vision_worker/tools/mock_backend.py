"""
mock_backend.py — 비전 worker v1.0 서버 밖 왕복 시험용 가짜 백엔드 (표준 라이브러리 · 2026-10-07)

  실제 백엔드 (drone_backend/real.py) 의 비전 쪽 규칙을 흉내 낸다:
    GET  /internal/v1/frames                    미분석 프레임 (최대 20 · 오래된 순)
    GET  /internal/v1/frames/{fid}/input         JPEG 바이트 + X-Content-SHA256
    POST /internal/v1/vision/media-bundles       예약 (Bundle 스키마)  ·  PUT …/media-assets/{aid}/content (크기 · SHA-256 확인)  ·  POST …/finalize
    POST /internal/v1/vision/results             VisionResult 스키마 검증 (openapi-vision.json) · 프레임 · 박스 범위 · worker_revision 증가 · 크롭 저장 완료
    POST /internal/v1/vision/reobservations      Reobserve 스키마
  인증: Authorization: Bearer <토큰> (기본 test)

실행: python tools/mock_backend.py <JPEG 폴더> [--port=18080] [--n=40] [--pose] [--dt=0.1] [--out=summary.json]
      --dt: 프레임 간격 초 (기본 0.1 · 재관측 규칙은 3초 넘게 봐야 나온다)
      --pose: 프레임에 가짜 자세 (위경도 · 고도 20 m · 짐벌 −45°) 를 넣어 좌표 VALID 경로를 시험
"""
import glob
import hashlib
import json
import re
import sys
import threading
import uuid
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import jsonschema

HERE = Path(__file__).resolve().parent.parent
SPEC = json.load(open(HERE / "schema" / "openapi-vision.json"))


def schema(name):
    return {"$ref": f"#/components/schemas/{name}", "components": SPEC["components"]}


ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
OPT = {a.split("=")[0][2:]: (a.split("=", 1)[1] if "=" in a else True) for a in sys.argv[1:] if a.startswith("--")}
FILES = sorted(glob.glob(str(Path(ARGS[0]) / "*.jpg")))[: int(OPT.get("n", 40))]
MISSION = str(uuid.uuid4()); T0 = datetime(2026, 10, 7, 3, 0, tzinfo=timezone.utc)
FRAMES = {}
for i, f in enumerate(FILES):
    fid = str(uuid.uuid4()); data = open(f, "rb").read()
    rec = {"frame_id": fid, "mission_id": MISSION, "capture_at": (T0 + timedelta(seconds=float(OPT.get("dt", 0.1)) * i)).isoformat().replace("+00:00", "Z"),
           "width": 1920, "height": 1080, "format": "JPEG", "time_evidence": {"time_source": "SIDECAR"}}
    if OPT.get("pose"):
        rec["pose"] = {"lat": 36.145, "lon": 128.393, "alt_agl_m": 20.0, "roll_deg": 0.5, "pitch_deg": -1.0, "yaw_deg": 90.0,
                       "gimbal_pitch_deg": -45.0, "gimbal_yaw_deg": 0.0, "gimbal_stabilized": True}
    FRAMES[fid] = {"rec": rec, "data": data, "sha": hashlib.sha256(data).hexdigest(), "state": "UNASSIGNED"}
ASSETS, BUNDLES, CAND, LOG = {}, {}, {}, {"results": 0, "observations": 0, "reobservations": 0, "errors": [], "geo": {}, "uploads": 0,
                                         "messages": set(), "duplicates": 0}
LOCK = threading.Lock()


class H(BaseHTTPRequestHandler):
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
        with LOCK:
            LOG["errors"].append(f"{code} {err}")
        self.reply(code, {"error": {"code": err}})

    def auth(self):
        if self.headers.get("Authorization") != "Bearer " + str(OPT.get("token", "test")):
            self.fail(401, "UNAUTHENTICATED"); return False
        return True

    def body(self):
        n = int(self.headers.get("Content-Length", 0)); return self.rfile.read(n)

    def do_GET(self):
        if not self.auth():
            return
        if self.path == "/internal/v1/frames":
            with LOCK:
                todo = [v["rec"] for v in FRAMES.values() if v["state"] == "UNASSIGNED"][:20]
            return self.reply(200, todo)
        m = re.match(r"^/internal/v1/frames/([0-9a-f-]+)/input$", self.path)
        if m and m.group(1) in FRAMES:
            f = FRAMES[m.group(1)]
            return self.reply(200, raw=f["data"], headers={"X-Content-SHA256": f["sha"]})
        self.fail(404, "RESOURCE_NOT_FOUND")

    def do_PUT(self):
        if not self.auth():
            return
        m = re.match(r"^/internal/v1/vision/media-assets/([0-9a-f-]+)/content$", self.path)
        data = self.body()
        if not m or m.group(1) not in ASSETS:
            return self.fail(404, "RESOURCE_NOT_FOUND")
        a = ASSETS[m.group(1)]
        if len(data) != a["byte_size"] or hashlib.sha256(data).hexdigest() != a["sha256"]:
            return self.fail(422, "CONTENT_HASH_MISMATCH")
        a["state"] = "STORED"; self.reply(200, {"stored": True, "duplicate": False, "asset_id": m.group(1)})

    def do_POST(self):
        if not self.auth():
            return
        raw = self.body(); j = json.loads(raw) if raw else {}
        try:
            if self.path == "/internal/v1/vision/media-bundles":
                jsonschema.validate(j, schema("Bundle"))
                for f in j["files"]:
                    ASSETS[f["asset_id"]] = {**f, "state": "WRITING", "bundle": j["bundle_id"]}
                BUNDLES[j["bundle_id"]] = {"files": [f["asset_id"] for f in j["files"]], "state": "WRITING"}
                return self.reply(201, {"asset_id": j["bundle_id"]})
            m = re.match(r"^/internal/v1/vision/media-bundles/([0-9a-f-]+)/finalize$", self.path)
            if m:
                b = BUNDLES.get(m.group(1))
                if not b or any(ASSETS[a]["state"] != "STORED" for a in b["files"]):
                    return self.fail(409, "BUNDLE_NOT_READY")
                b["state"] = "PERSISTED"; LOG["uploads"] += 1
                return self.reply(200, {"bundle_id": m.group(1), "persistence_status": "PERSISTED"})
            if self.path == "/internal/v1/vision/reobservations":
                jsonschema.validate(j, schema("Reobserve"))
                with LOCK:
                    if j["candidate_id"] not in CAND:                 # 실제 백엔드: 후보가 없으면 404
                        return self.fail(404, "RESOURCE_NOT_FOUND (reobserve candidate)")
                    LOG["reobservations"] += 1
                    LOG.setdefault("reobserve_reasons", {}); LOG["reobserve_reasons"][j["reason"]] = LOG["reobserve_reasons"].get(j["reason"], 0) + 1
                    LOG.setdefault("reobserve_per_cand", {}); LOG["reobserve_per_cand"][j["candidate_id"]] = LOG["reobserve_per_cand"].get(j["candidate_id"], 0) + 1
                return self.reply(201, {"request_id": j["request_id"]})
            if self.path == "/internal/v1/vision/results":
                jsonschema.validate(j, schema("VisionResult"))
                with LOCK:
                    if j["message_id"] in LOG["messages"]:
                        LOG["duplicates"] += 1; return self.reply(200, {"message_id": j["message_id"], "duplicate": True})
                    f = FRAMES.get(j["frame_id"])
                    if not f or f["rec"]["mission_id"] != j["mission_id"]:
                        return self.fail(422, "FRAME_MISSION_MISMATCH")
                    if f["state"] != "UNASSIGNED":
                        return self.fail(409, "FRAME_ALREADY_ANALYZED")
                    if j["analysis_state"] != "DONE" and (j["observations"] or not j.get("reason")):
                        return self.fail(422, "INVALID_CONTRACT")
                    for o in j["observations"]:
                        b = o["bbox"]
                        if b["x2"] > 1920 or b["y2"] > 1080 or b["x1"] >= b["x2"] or b["y1"] >= b["y2"]:
                            return self.fail(422, "INVALID_BBOX")
                        if int(o["worker_revision"]) <= CAND.get(o["candidate_id"], 0):
                            return self.fail(409, "OLD_WORKER_REVISION")
                        if o["snapshot_asset_id"] and ASSETS.get(o["snapshot_asset_id"], {}).get("state") != "STORED":
                            return self.fail(409, "INPUT_ASSET_NOT_READY")
                        g = o["geo"]
                        if g["status"] == "VALID" and (g["lat"] is None or not g.get("error_ellipse")):
                            return self.fail(422, "GEO_VALID_NEEDS_POSITION")
                        if g["status"] != "VALID" and g["lat"] is not None:
                            return self.fail(422, "GEO_PENDING_HAS_POSITION")
                    for o in j["observations"]:
                        CAND[o["candidate_id"]] = int(o["worker_revision"]); LOG["geo"][o["geo"]["status"]] = LOG["geo"].get(o["geo"]["status"], 0) + 1
                        m = (o.get("appearance") or {}).get("match")
                        if m:
                            LOG.setdefault("appearance_verdicts", {}); LOG["appearance_verdicts"][m["verdict"]] = LOG["appearance_verdicts"].get(m["verdict"], 0) + 1
                    f["state"] = "COMPLETED"; LOG["messages"].add(j["message_id"])
                    LOG["results"] += 1; LOG["observations"] += len(j["observations"])
                return self.reply(200, {"message_id": j["message_id"]})
        except jsonschema.ValidationError as e:
            return self.fail(422, f"SCHEMA {e.message[:160]} @ {list(e.absolute_path)}")
        self.fail(404, "RESOURCE_NOT_FOUND")


def summary():
    d = {k: v for k, v in LOG.items() if k not in ("messages", "reobserve_per_cand")}
    pc = LOG.get("reobserve_per_cand", {})
    d["reobserve_max_per_candidate"] = max(pc.values(), default=0)
    d.update(frames=len(FRAMES), completed=sum(v["state"] == "COMPLETED" for v in FRAMES.values()), candidates=len(CAND))
    return d


if __name__ == "__main__":
    srv = ThreadingHTTPServer(("127.0.0.1", int(OPT.get("port", 18080))), H)
    print(f"가짜 백엔드 · 프레임 {len(FRAMES)} · 임무 {MISSION} · pose {'있음' if OPT.get('pose') else '없음'}", flush=True)
    try:
        import time
        while True:
            srv.handle_request() if False else None
            t = threading.Thread(target=srv.serve_forever, daemon=True); t.start()
            while sum(v["state"] == "COMPLETED" for v in FRAMES.values()) < len(FRAMES):
                time.sleep(0.5)
            time.sleep(1); break
    finally:
        out = summary(); print(json.dumps(out, ensure_ascii=False, indent=1), flush=True)
        if OPT.get("out"):
            Path(OPT["out"]).write_text(json.dumps(out, ensure_ascii=False, indent=1))
        srv.shutdown()
