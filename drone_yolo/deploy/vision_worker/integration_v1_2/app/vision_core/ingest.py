"""
ingest.py — 비전 worker v1.2 핵심 (vision-ingest/1.0 · 통합 v1.1 판 위에 비전 1.2 기능 · 2026-10-14)

  v1.2 추가 (통합 v1.1 규칙은 그대로 — 빠진 자세를 채우지 않음 · 좌표 기본 PENDING)
    · 추적 · 상태 (BoT-SORT · 누움 점수 · 무동작 · 등급) — /models/posture.json 이 있을 때 · observation.track_id · state
    · 후보 묶기 = 추적 키 → 같은 후보 (attach) · 좌표 주장이 있으면 좌표 문 · 없으면 화면 IoU (옛 동작)
    · 인상착의 비교 — 프레임 기록의 appearance_query 또는 /scratch/appearance_query.json {임무 id | "*": 조건} → appearance.match
    · frame-pose/1.0 → 발끝 광선 좌표 (pose.py) — VISION_GEO_CLAIMS=1 일 때만 VALID 주장 (공식 승격은 백엔드) · 기본은 PENDING + 사유 (계산 사유 덧붙임)
      VISION_GEO_GROUPING=1 (기본) — 보고는 PENDING 이어도 후보 묶기에는 계산한 위치를 씀 (내부용)
      지면: VISION_DEM_DIR (Terrarium 타일 · 고도 MSL) 이 있으면 DEM (평지 가정 없음 · method +DEM) · VISION_GEO_AGL_FLAT=1 이면 AGL 고도로 「기체 아래와 같은 높이」 가정 (method 에 LOCAL_FLAT_AGL) · 아니면 지형 함수가 있어야 함

  규격  schema/vision_backend_handoff_v1.0.md · openapi-vision.json (백엔드 drone_backend/real.py 로 동작 확인)
  흐름  GET /internal/v1/frames (미분석 · 최대 20) → GET …/{fid}/input (SHA-256 확인) → 디코드
        → 탐지 (타일 4장) → pose 보존 · 지리좌표 보류 → 후보 기억 → 상의 색
        → (새 후보 · 확신도 최고 갱신이면) 크롭 업로드 media-bundles → PUT → finalize
        → VisionResult 한 줄을 /scratch/outbox.jsonl 에 (fsync) → 전송 (app.vision_outbox.send_pending 과 같은 규칙)
  규칙  프레임마다 결과 하나 (탐지 0 개도 DONE + []) · 못 읽으면 DROPPED + reason
        후보 id = uuid4 · 관측마다 worker_revision 후보별 1씩 증가 (문자열) · 박스는 폭·높이 안으로
        좌표 VALID 면 위경도 + 오차 타원 · PENDING/INVALID 는 위경도 null · 보류 사유는 코드
        outbox 에 쓴 frame_id 는 다시 처리하지 않는다 (결과가 반영되기 전엔 같은 프레임이 또 나온다)
"""
import hashlib
import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from .appearance import AppearanceComparator
from .color import ColorAccumulator, ColorProfiler, frame_stats
from .geo import CameraModel
from .pose import DemTiles, PoseGeo, as_registry_geo, geo_json
from .registry import CandidateRegistry

SCHEMA = "vision-ingest/1.0"
REASON_CODE = (("기울기", "TILT_GT_6"), ("자세 변화", "RATE_GT_1_5"), ("발끝", "FOOT_AT_EDGE"),
               ("지평선", "RAY_ABOVE_HORIZON"), ("거리", "RANGE_GT_200"), ("지형", "NO_TERRAIN_HIT"))


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class Outbox:
    """/scratch/outbox.jsonl 덧붙이기 (한 생산자 · fsync · 한 줄 ≤ 64 KiB)"""
    def __init__(self, root):
        self.path = Path(root) / "outbox.jsonl"
        if self.path.exists():
            data=self.path.read_bytes()
            if data and not data.endswith(b'\n'):
                end=data.rfind(b'\n')+1
                tail=self.path.with_name('outbox-incomplete-'+str(uuid.uuid4())+'.bin')
                with tail.open('wb') as f:f.write(data[end:]);f.flush();os.fsync(f.fileno())
                with self.path.open('r+b') as f:f.truncate(end);f.flush();os.fsync(f.fileno())

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
        journal=Path(root)/'outbox.jsonl'
        if journal.exists():
            with journal.open('rb') as f:
                for line in f:
                    row=json.loads(line)
                    if row['type']=='result':
                        fid=row['body']['frame_id']
                        if not self.seen(fid):self.mark(fid)

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
    def __init__(self, api, token, detector, model_config_id, scratch="/scratch", upload_crops=True, color_every_s=0.5, posture=None,
                 geo_claims=None, environment=None):
        self.api, self.token, self.det = api, token, detector
        self.model_config_id = str(model_config_id)
        self.session = str(uuid.uuid4())                 # producer_session_id — 실행마다 새로
        self.outbox, self.state = Outbox(scratch), State(scratch)
        self.colorer = ColorProfiler()
        self.upload_crops, self.color_every_s = upload_crops, color_every_s
        self.reg = {}                                     # 임무 → CandidateRegistry
        self.uid, self.rev = {}, {}                      # (임무, 후보 라벨) → uuid · uuid → revision
        self.stats = {"frames": 0, "observations": 0, "dropped": 0, "uploads": 0, "upload_fail": 0}
        self.posture = posture if posture and Path(posture).exists() else None
        self.trk, self.tcand = {}, {}                    # 임무 → StateTracker · (임무, 추적 키) → 후보
        self.appear = AppearanceComparator()
        self.query_file = Path(os.environ.get("INGEST_APPEARANCE_FILE", str(Path(scratch) / "appearance_query.json")))
        env = environment or os.environ.get("INGEST_ENVIRONMENT", "REAL")
        self.geo_claims = (os.environ.get("VISION_GEO_CLAIMS", "0") == "1") if geo_claims is None else bool(geo_claims)
        # 보고는 PENDING 이어도 후보 묶기 (같은 사람 판단) 에는 pose 로 계산한 위치를 쓴다 — 밖으로 내보내는 좌표가 아님 (UE: 사람당 후보 10~19 → 4~5)
        self.geo_grouping = os.environ.get("VISION_GEO_GROUPING", "1") == "1"
        dem_dir = os.environ.get("VISION_DEM_DIR")                          # 10-17 · Terrarium 타일 (z16) · 읽기 전용 · 고도 기준 MSL 일 때 광선 × DEM
        self.pgeo = PoseGeo(CameraModel(hfov_deg=float(os.environ.get("VISION_HFOV_DEG", "80.2"))), environment=env,
                            allow_agl_flat=os.environ.get("VISION_GEO_AGL_FLAT", "0") == "1",
                            dem=DemTiles(dem_dir) if dem_dir and Path(dem_dir).exists() else None)
        # 10-17 · 이착륙 (지면 위 VISION_MIN_AGL_M 아래) 프레임은 탐지만 하고 후보 · 관측을 만들지 않는다 (기체 그림자 오탐 · UE 녹화 29/29) · 0 = 끔
        self.min_agl = float(os.environ.get("VISION_MIN_AGL_M", "10"))
        self.stats["dem_loaded"] = self.pgeo.dem is not None              # 상태 파일 worker-status.json 에 보임 (마운트 확인용)
        self.stats["min_agl_m"] = self.min_agl
        # 10-17 T · 나무 줄기 오탐 (route_132434) — 기본 끔 · 판정 _학습 큐 「10-17 T」
        self.edge_margin = float(os.environ.get("VISION_EDGE_MARGIN_PX", "0"))       # >0 → 경계에 닿은 박스는 후보 · 관측 안 만듦
        self.emit_confirmed_only = os.environ.get("VISION_EMIT_CONFIRMED_ONLY", "0") == "1"   # 후보 확정 (3 s 안 2장) 뒤부터 관측
        self.last_skip = None

    def query_for(self, fr):
        """찾는 사람 외형 조건 — 프레임 기록의 appearance_query · 없으면 파일 {임무 id 또는 "*": 조건}"""
        if fr.get("appearance_query"):
            return fr["appearance_query"]
        try:
            q = json.loads(self.query_file.read_text()) if self.query_file.exists() else {}
        except (OSError, ValueError):
            return None
        return q.get(str(fr.get("mission_id"))) or q.get("*")

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
        # The physical camera/time calibration is unverified; never invent pose defaults or coordinates.
        t = datetime.fromisoformat(fr["capture_at"].replace("Z", "+00:00")).timestamp() if fr.get("capture_at") else time.time()
        obs = self.observe(mid, img, t, fr.get("pose"), query=self.query_for(fr))
        self.stats["frames"] += 1; self.stats["observations"] += len(obs)
        if self.last_skip:
            return self.result_body(fr, "SKIPPED", reason=self.last_skip)
        return self.result_body(fr, "DONE", obs)

    def frame_agl(self, pose):
        """지면 위 높이 · (m, 근거) — ① 고도 기준 MSL + DEM → 고도 − DEM(기체 위경도) ② 고도 기준 AGL → 그 값 ③ 그 밖 → (None, "UNKNOWN")
        동기 오차는 보지 않는다 (높이 판단만 · UE 녹화 333 ms 에 수직 차 ≤ 0.4 m)"""
        g = (pose or {}).get("geographic") or {}
        if g.get("altitude_m") is None:
            return None, "UNKNOWN"
        if g.get("altitude_datum") == "MSL" and self.pgeo.dem and g.get("latitude") is not None:
            z = self.pgeo.dem(float(g["latitude"]), float(g["longitude"]))
            return (float(g["altitude_m"]) - z, "MSL-DEM") if z is not None else (None, "UNKNOWN")
        if g.get("altitude_datum") == "AGL":
            return float(g["altitude_m"]), "AGL"
        return None, "UNKNOWN"

    def observe(self, mid, img, t, pose, query=None, upload_crops=None):
        """디코드된 한 장 → 관측 목록 (vision-ingest/1.0) · SIM worker 도 이걸 쓴다 (프레임은 시각 순서로)
        이착륙 프레임 (지면 위 min_agl 아래) 은 [] · self.last_skip = "LOW_ALTITUDE …" (REAL 은 SKIPPED 사유 · SIM 은 stats)"""
        H, W = img.shape[:2]
        upload_crops = self.upload_crops if upload_crops is None else upload_crops
        boxes, confs, ms = self.det.detect(img)
        self.last_skip = None
        if self.min_agl > 0:
            agl, basis = self.frame_agl(pose)
            if agl is None:                               # 근거 없음 → 막지 않는다 (미탐이 오탐보다 나쁨)
                self.stats["agl_unknown_frames"] = self.stats.get("agl_unknown_frames", 0) + 1
            elif agl < self.min_agl:
                self.stats["low_altitude_frames"] = self.stats.get("low_altitude_frames", 0) + 1
                self.stats["low_altitude_dets"] = self.stats.get("low_altitude_dets", 0) + len(boxes)
                self.last_skip = f"LOW_ALTITUDE agl={agl:.1f}m<{self.min_agl:g} basis={basis}"
                return []
        order = np.argsort(-confs)[:128]
        reg = self.reg.setdefault(mid, CandidateRegistry(mid))
        stats = frame_stats(img) if len(order) else None
        kept = []
        for i in order:
            x1, y1, x2, y2 = [float(v) for v in boxes[i]]
            x1, y1 = max(0.0, x1), max(0.0, y1); x2, y2 = min(float(W), x2), min(float(H), y2)
            if x2 > x1 and y2 > y1:
                kept.append((i, (x1, y1, x2, y2)))
        states = [(None, None)] * len(kept)
        if self.posture:                                  # 추적 · 상태 (프레임마다 · 시각 순서)
            try:
                if mid not in self.trk:
                    from .state import StateTracker          # BoT-SORT 는 `lap` 패키지가 필요 — 없으면 상태만 끄고 탐지 · 후보 · 색은 계속
                    self.trk[mid] = StateTracker(self.posture)
                states = self.trk[mid].update(t, img, [b for _, b in kept], [float(confs[i]) for i, _ in kept])
            except Exception as e:
                self.posture, self.stats["state_disabled"] = None, f"{type(e).__name__}: {e}"[:200]
                states = [(None, None)] * len(kept)
        frame_reasons = self.pgeo.check(pose)
        obs = []
        for (i, (x1, y1, x2, y2)), (tkey, st) in zip(kept, states):
            m = self.edge_margin
            if m > 0 and (x1 <= m or y1 <= m or x2 >= W - m or y2 >= H - m):   # 가장자리에 잘린 박스 (추적은 위에서 이미 갱신)
                self.stats["edge_dets"] = self.stats.get("edge_dets", 0) + 1
                continue
            claim = self.pgeo.locate((x1, y1, x2, y2), pose) if pose else {"status": "PENDING", "reasons": frame_reasons, "source": None}
            g = as_registry_geo(claim) if (self.geo_claims or self.geo_grouping) else None
            known = self.tcand.get((mid, tkey)) if tkey else None
            if known is not None and known.last_t != t:   # 추적기가 같은 사람이라고 한 후보
                cand, ev = reg.attach(known, t, (x1, y1, x2, y2), float(confs[i]), g)
            else:
                cand, ev = reg.update(t, (x1, y1, x2, y2), float(confs[i]), g)
            if tkey:
                self.tcand[(mid, tkey)] = cand
            if self.emit_confirmed_only and not cand.confirmed:   # 한 장짜리 보류 — 후보 기억에는 남김
                self.stats["unconfirmed_dets"] = self.stats.get("unconfirmed_dets", 0) + 1
                continue
            key = (mid, cand.cid)
            cid = self.uid.setdefault(key, str(uuid.uuid4()))
            self.rev[cid] = self.rev.get(cid, 0) + 1
            if t - cand.color_t >= self.color_every_s:
                cand.color_t = t
                cand.color = cand.color or ColorAccumulator()
                cand.color.add(self.colorer.profile(img, (x1, y1, x2, y2), stats=stats))
            col = cand.color.result() if cand.color else {"status": "undetermined", "top": [], "n_obs": 0}
            snap = None
            if upload_crops and (ev in ("new", "best") or (self.emit_confirmed_only and ev == "confirmed")):   # 보류 뒤 첫 관측에도 사진
                pad = max(8, int(0.3 * max(x2 - x1, y2 - y1)))
                crop = img[max(0, int(y1) - pad):int(y2) + pad, max(0, int(x1) - pad):int(x2) + pad]
                ok, enc = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 90]) if crop.size else (False, None)
                if ok:
                    snap = self.upload_crop(mid, enc.tobytes())
            appearance = {"upper": {"status": "OK" if col.get("status") == "ok" else "UNDETERMINED",
                                    "top": [{"color": c, "p": p} for c, p in col.get("top", [])], "n_obs": col.get("n_obs", 0)}}
            if query:
                appearance["match"] = self.appear.compare(col, query)
            obs.append({"observation_id": str(uuid.uuid4()), "candidate_id": cid, "detection_index": len(obs),
                        "bbox": {"x1": round(x1, 1), "y1": round(y1, 1), "x2": round(x2, 1), "y2": round(y2, 1)},
                        "confidence": round(float(confs[i]), 4), "kind": "DETECT",
                        "track_id": f"{self.session[:8]}:{tkey}" if tkey else None,
                        "geo": geo_json(claim, self.geo_claims), "auto_confirmed": bool(cand.confirmed), "appearance": appearance,
                        "state": st, "worker_revision": str(self.rev[cid]), "snapshot_asset_id": snap, "clip_asset_id": None})
        return obs

    def poll_once(self, send=None):
        """대기 프레임을 한 번 훑는다 · 처리한 수"""
        if send:
            send()  # Drain durable results before producing more during a backend outage.
        r = self.api.get("/internal/v1/frames", headers=self._h()); r.raise_for_status()
        n = 0
        for fr in r.json():
            fid = str(fr["frame_id"])
            if self.state.seen(fid):
                continue
            ri = self.api.get(f"/internal/v1/frames/{fid}/input", headers=self._h())
            if ri.status_code in (401,403,409) or ri.status_code>=500:
                ri.raise_for_status()
            if ri.status_code != 200:
                body = self.result_body(fr, "DROPPED", reason=f"INPUT_HTTP_{ri.status_code}")
            else:
                data = ri.content
                want = ri.headers.get("X-Content-SHA256")
                body = self.result_body(fr, "DROPPED", reason="INPUT_HASH_MISMATCH") if not want or hashlib.sha256(data).hexdigest() != want \
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
