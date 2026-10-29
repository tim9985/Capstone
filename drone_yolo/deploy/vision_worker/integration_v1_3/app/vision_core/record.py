"""
record.py — 실비행 프레임 녹화 (worker 1.3 · 2026-10-30) — 비행 뒤 같은 worker 로 다시 돌려 평가 (탐지 · 오탐 · 색 · 자세 · 좌표)

  VISION_RECORD_DIR 가 있을 때만 켜짐 (기본 끔) — worker 가 분석한 그 입력 (JPEG 원본 바이트 · frame 메타 · pose) 을 그대로 저장
    <dir>/<mission_id>/frames/<frame_id>.jpg
    <dir>/<mission_id>/poses.jsonl   한 줄 = {"frame": "<frame_id>.jpg", "frame_id", "capture_at", "width", "height", "pose", "sha256", "analysis": 관측 요약}
    = 노트북 경로 녹화와 같은 형식 → `drone_yolo/ue_route_check.py` · `real_flight_eval.py` 로 다시 돌림
  VISION_RECORD_EVERY (기본 1 = 모든 장) · VISION_RECORD_MAX_MB (기본 3000 · 넘으면 녹화만 멈춤) · 실패해도 분석은 그대로 (stats record_error)
  ⚠ 사람이 찍힌 원본 — 출연 동의한 촬영에만 켜고, 평가 뒤 지운다 · git · 공개 폴더에 올리지 않는다
"""
import hashlib
import json
import os
from pathlib import Path


class FrameRecorder:
    def __init__(self, stats):
        self.dir = os.environ.get("VISION_RECORD_DIR")
        self.every = max(1, int(os.environ.get("VISION_RECORD_EVERY", "1")))
        self.max_bytes = float(os.environ.get("VISION_RECORD_MAX_MB", "3000")) * 1e6
        self.n, self.bytes, self.stats = 0, 0, stats
        if self.dir:
            stats.update(record_dir=self.dir, record_frames=0, record_mb=0.0)

    def save(self, fr, data, obs=()):
        if not self.dir or not data:
            return
        self.n += 1
        if (self.n - 1) % self.every or self.bytes + len(data) > self.max_bytes:
            if self.bytes + len(data) > self.max_bytes:
                self.stats["record_full"] = True
            return
        try:
            root = Path(self.dir) / str(fr.get("mission_id", "unknown"))
            (root / "frames").mkdir(parents=True, exist_ok=True)
            fid = str(fr["frame_id"])
            (root / "frames" / f"{fid}.jpg").write_bytes(data)
            ana = [{"bbox": o.get("bbox") or o.get("box"), "confidence": o.get("confidence"),
                    "upper": ((o.get("appearance") or {}).get("upper") or {}).get("top"),
                    "grade": ((o.get("state") or {}).get("grade"))} for o in obs]
            rec = {"frame": f"{fid}.jpg", "frame_id": fid, "capture_at": fr.get("capture_at"), "width": fr.get("width"), "height": fr.get("height"),
                   "pose": fr.get("pose"), "sha256": hashlib.sha256(data).hexdigest(), "analysis": ana}
            with open(root / "poses.jsonl", "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")
            self.bytes += len(data)
            self.stats["record_frames"] = self.stats.get("record_frames", 0) + 1
            self.stats["record_mb"] = round(self.bytes / 1e6, 1)
        except Exception as e:                            # 녹화 실패는 분석을 막지 않는다
            self.stats["record_error"] = f"{type(e).__name__}: {e}"[:200]
