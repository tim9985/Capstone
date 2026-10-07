"""
ingest.py — 비전 worker v1.0 핵심 (vision-ingest/1.0 · 2026-10-07)

  규격  schema/vision_backend_handoff_v1.0.md · openapi-vision.json (백엔드 drone_backend/real.py 로 동작 확인)
  흐름  GET /internal/v1/frames (미분석 · 최대 20) → GET …/{fid}/input (SHA-256 확인) → 디코드
        → 탐지 (타일 4장) → 좌표 (pose 가 있으면 · 없으면 PENDING/NO_POSE) → 후보 기억 → 상의 색
        → (새 후보 · 확신도 최고 갱신이면) 크롭 업로드 media-bundles → PUT → finalize
        → VisionResult 한 줄을 /scratch/outbox.jsonl 에 (fsync) → 전송 (app.vision_outbox.send_pending 과 같은 규칙)
  규칙  프레임마다 결과 하나 (탐지 0 개도 DONE + []) · 못 읽으면 DROPPED + reason
        후보 id = uuid4 · 관측마다 worker_revision 후보별 1씩 증가 (문자열) · 박스는 폭·높이 안으로
        좌표 VALID 면 위경도 + 오차 타원 · PENDING/INVALID 는 위경도 null · 보류 사유는 코드
        outbox 에 쓴 frame_id 는 다시 처리하지 않는다 (결과가 반영되기 전엔 같은 프레임이 또 나온다)
"""
import hashlib
import json
import math
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from .color import ColorAccumulator, ColorProfiler, frame_stats
from .geo import GeoResolver, Telemetry
from .registry import CandidateRegistry

SCHEMA = "vision-ingest/1.0"
REASON_CODE = (("기울기", "TILT_GT_6"), ("자세 변화", "RATE_GT_1_5"), ("발끝", "FOOT_AT_EDGE"),
               ("지평선", "RAY_ABOVE_HORIZON"), ("거리", "RANGE_GT_200"), ("지형", "NO_TERRAIN_HIT"))


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def telemetry_from_pose(pose):
    """프레임 pose (게이트웨이 형식 · 미확정) → Telemetry · 모자라면 None"""
    if not isinstance(pose, dict):
        return None
    g = lambda *ks: next((pose[k] for k in ks if pose.get(k) is not None), None)
    lat, lon, alt = g("lat"), g("lon"), g("alt_agl_m", "alt_agl", "relative_alt_m")
    if lat is None or lon is None or alt is None:
        return None
    rad = lambda v: math.radians(float(v or 0.0))
    return Telemetry(float(lat), float(lon), float(alt), roll=rad(g("roll_deg")), pitch=rad(g("pitch_deg")), yaw=rad(g("yaw_deg")),
                     gimbal_pitch_deg=float(g("gimbal_pitch_deg") if g("gimbal_pitch_deg") is not None else -45.0),
                     gimbal_yaw_deg=float(g("gimbal_yaw_deg") or 0.0),
                     stabilized=bool(pose.get("gimbal_stabilized", True)),
                     attitude_rate_deg=float(g("attitude_rate_deg") or 0.0))


def geo_json(res):
    if res is None:
        return {"status": "PENDING", "lat": None, "lon": None, "reasons": ["NO_POSE"], "error_ellipse": None, "method": None}
    codes = []
    for txt in res.reasons:
        codes.append(next((c for k, c in REASON_CODE if k in txt), "OTHER"))
    method = "FOOT_RAY_FLAT"
    if res.status == "OK" and res.lat is not None and res.ellipse:
        a, c, b = res.ellipse
        major, minor = max(a, c), min(a, c)
        az = b % 360 if a >= c else (b + 90) % 360
        return {"status": "VALID", "lat": round(res.lat, 7), "lon": round(res.lon, 7), "reasons": [],
                "error_ellipse": {"major_m": round(major, 2), "minor_m": round(minor, 2), "azimuth_deg": round(az, 1), "k": 1}, "method": method}
    return {"status": "PENDING", "lat": None, "lon": None, "reasons": codes[:16] or ["OTHER"], "error_ellipse": None, "method": method}


class Outbox:
    """/scratch/outbox.jsonl 덧붙이기 (한 생산자 · fsync · 한 줄 ≤ 64 KiB)"""
    def __init__(self, root):
        self.path = Path(root) / "outbox.jsonl"

    def append(self, kind, body):
        line = (json.dumps({"type": kind, "body": body}, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
        if len(line) > 65536:
            raise ValueError("outbox 한 줄이 64 KiB 를 넘는다")
        with open(self.path, "ab") as f:
            f.write(line); f.flush(); os.fsync(f.fileno())


class State:
    """재시작해도 이어 가는 것 — sequence · 처리한 frame_id (최근 5만)"""
    def __init__(self, root):
        self.path = Path(root) / "ingest_state.json"
        d = json.loads(self.path.read_text()) if self.path.exists() else {}
        self.sequence = int(d.get("sequence", 0)); self.done = list(d.get("done", [])); self._set = set(self.done)

    def mark(self, fid):
        self.done.append(fid); self._set.add(fid)
        if len(self.done) > 50000:
            old = self.done[:10000]; self.done = self.done[10000:]; self._set.difference_update(old)

    def seen(self, fid):
        return fid in self._set

    def next_seq(self):
        self.sequence += 1; return str(self.sequence)

    def save(self):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"sequence": self.sequence, "done": self.done})); os.replace(tmp, self.path)


class IngestWorker:
    def __init__(self, api, token, detector, model_config_id, scratch="/scratch", upload_crops=True, color_every_s=0.5):
        self.api, self.token, self.det = api, token, detector
        self.model_config_id = str(model_config_id)
        self.session = str(uuid.uuid4())                 # producer_session_id — 실행마다 새로
        self.outbox, self.state = Outbox(scratch), State(scratch)
        self.geo, self.colorer = GeoResolver(), ColorProfiler()
        self.upload_crops, self.color_every_s = upload_crops, color_every_s
        self.reg = {}                                     # 임무 → CandidateRegistry
        self.uid, self.rev = {}, {}                      # (임무, 후보 라벨) → uuid · uuid → revision
        self.stats = {"frames": 0, "observations": 0, "dropped": 0, "uploads": 0, "upload_fail": 0}

    def _h(self):
        return {"Authorization": "Bearer " + self.token()}

    # ── 미디어 업로드 (FRAME 한 장 · 결과보다 먼저 저장 완료) ──
    def upload_crop(self, mission_id, jpg):
        bid, aid = str(uuid.uuid4()), str(uuid.uuid4())
        man = {"bundle_id": bid, "mission_id": mission_id,
               "files": [{"asset_id": aid, "role": "FRAME", "byte_size": len(jpg), "sha256": hashlib.sha256(jpg).hexdigest()}]}
        try:
            self.api.post("/internal/v1/vision/media-bundles", json=man, headers=self._h()).raise_for_status()
            self.api.put(f"/internal/v1/vision/media-assets/{aid}/content", content=jpg,
                         headers=self._h() | {"Content-Type": "application/octet-stream"}).raise_for_status()
            self.api.post(f"/internal/v1/vision/media-bundles/{bid}/finalize", headers=self._h()).raise_for_status()
            self.stats["uploads"] += 1
            return aid
        except Exception:
            self.stats["upload_fail"] += 1
            return None

    def result_body(self, fr, state, obs=(), reason=None):
        return {"schema_version": SCHEMA, "message_id": str(uuid.uuid4()), "producer_session_id": self.session,
                "sequence": self.state.next_seq(), "mission_id": fr["mission_id"], "frame_id": fr["frame_id"],
                "model_config_id": self.model_config_id, "emitted_at": now_iso(), "analysis_state": state,
                "reason": reason, "observations": list(obs)}

    def process(self, fr, data):
        """프레임 한 장 → VisionResult 본문 (outbox 에 쓰기 전)"""
        mid, W, H = fr["mission_id"], int(fr["width"]), int(fr["height"])
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR) if data else None
        if img is None:
            self.stats["dropped"] += 1
            return self.result_body(fr, "DROPPED", reason="DECODE_FAILED")
        if (img.shape[1], img.shape[0]) != (W, H):
            return self.result_body(fr, "DROPPED", reason=f"SIZE_MISMATCH {img.shape[1]}x{img.shape[0]}")
        boxes, confs, ms = self.det.detect(img)
        order = np.argsort(-confs)[:128]
        tel = telemetry_from_pose(fr.get("pose"))
        t = datetime.fromisoformat(fr["capture_at"].replace("Z", "+00:00")).timestamp() if fr.get("capture_at") else time.time()
        reg = self.reg.setdefault(mid, CandidateRegistry(mid))
        stats = frame_stats(img) if len(order) else None
        obs = []
        for k, i in enumerate(order):
            x1, y1, x2, y2 = [float(v) for v in boxes[i]]
            x1, y1 = max(0.0, x1), max(0.0, y1); x2, y2 = min(float(W), x2), min(float(H), y2)
            if x2 <= x1 or y2 <= y1:
                continue
            g = self.geo.to_world((x1, y1, x2, y2), tel) if tel is not None else None
            cand, ev = reg.update(t, (x1, y1, x2, y2), float(confs[i]), g)
            key = (mid, cand.cid)
            cid = self.uid.setdefault(key, str(uuid.uuid4()))
            self.rev[cid] = self.rev.get(cid, 0) + 1
            if t - cand.color_t >= self.color_every_s:
                cand.color_t = t
                cand.color = cand.color or ColorAccumulator()
                cand.color.add(self.colorer.profile(img, (x1, y1, x2, y2), stats=stats))
            col = cand.color.result() if cand.color else {"status": "undetermined", "top": [], "n_obs": 0}
            snap = None
            if self.upload_crops and ev in ("new", "best"):
                pad = max(8, int(0.3 * max(x2 - x1, y2 - y1)))
                crop = img[max(0, int(y1) - pad):int(y2) + pad, max(0, int(x1) - pad):int(x2) + pad]
                ok, enc = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 90]) if crop.size else (False, None)
                if ok:
                    snap = self.upload_crop(mid, enc.tobytes())
            obs.append({"observation_id": str(uuid.uuid4()), "candidate_id": cid, "detection_index": len(obs),
                        "bbox": {"x1": round(x1, 1), "y1": round(y1, 1), "x2": round(x2, 1), "y2": round(y2, 1)},
                        "confidence": round(float(confs[i]), 4), "kind": "DETECT", "track_id": None,
                        "geo": geo_json(g), "auto_confirmed": bool(cand.confirmed),
                        "appearance": {"upper": {"status": "OK" if col.get("status") == "ok" else "UNDETERMINED",
                                                 "top": [{"color": c, "p": p} for c, p in col.get("top", [])], "n_obs": col.get("n_obs", 0)}},
                        "state": None, "worker_revision": str(self.rev[cid]), "snapshot_asset_id": snap, "clip_asset_id": None})
        self.stats["frames"] += 1; self.stats["observations"] += len(obs)
        return self.result_body(fr, "DONE", obs)

    def poll_once(self, send=None):
        """대기 프레임을 한 번 훑는다 · 처리한 수"""
        r = self.api.get("/internal/v1/frames", headers=self._h()); r.raise_for_status()
        n = 0
        for fr in r.json():
            fid = str(fr["frame_id"])
            if self.state.seen(fid):
                continue
            ri = self.api.get(f"/internal/v1/frames/{fid}/input", headers=self._h())
            if ri.status_code != 200:
                body = self.result_body(fr, "DROPPED", reason=f"INPUT_HTTP_{ri.status_code}")
            else:
                data = ri.content
                want = ri.headers.get("X-Content-SHA256")
                body = self.result_body(fr, "DROPPED", reason="INPUT_HASH_MISMATCH") if want and hashlib.sha256(data).hexdigest() != want \
                    else self.process(fr, data)
            # outbox 한 줄 64 KiB 한도 — 관측이 많으면 확신도 낮은 것부터 뺀다 (관측 하나 ~1.2 KB · 보통 50 개 안팎까지 들어감)
            while len(json.dumps({"type": "result", "body": body}, ensure_ascii=False, separators=(",", ":")).encode()) > 60000 and body["observations"]:
                body["observations"].pop()
            for k, o in enumerate(body["observations"]):
                o["detection_index"] = k
            self.outbox.append("result", body)
            self.state.mark(fid); n += 1
        if n:
            self.state.save()
        if send:
            send()
        return n
