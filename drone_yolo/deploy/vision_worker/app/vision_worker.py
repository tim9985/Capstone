"""
Vision worker — 비전 파트 (동료 샌드박스 app/vision_worker.py 교체본 · 2026-10-05)

  임무가 지정되면 영상을 읽어 사람 탐지 → 좌표 산정 → 후보 기억 → 임무 폴더에 쓴다.
  비행 명령은 내지 않는다 (flight_enabled=false 유지). 후보를 지우지 않는다.

  상태 파일   $STORAGE_ROOT/worker-status.json — 5초 안에 한 번 이상 갱신 (기존 20초 헬스체크 그대로)
              기존 키 (worker · updated_unix · cuda_available · model_loaded · flight_enabled · state) 유지 + 비전 항목 추가
  임무 지정   $STORAGE_ROOT/vision-control.json  {"mission_id": "m001", "source": "rtsp://… 또는 파일 (없으면 RTSP 기본)"}
              파일을 지우거나 mission_id 를 null 로 하면 멈춘다
  모델        $VISION_MODEL_DIR/best.pt (기본 /data/models/soup_v7r2) · 엔진은 처음 한 번 만들어 같은 폴더에 둔다
  출력        missions/<id>/results/detections.jsonl · candidates.jsonl · crops/<후보>.jpg · logs/vision.log
"""
import json
import os
import time
import traceback
from pathlib import Path

ROOT = Path(os.environ.get("STORAGE_ROOT", "/data"))
MODEL_DIR = Path(os.environ.get("VISION_MODEL_DIR", str(ROOT / "models" / "soup_v7r2")))
CONTROL = Path(os.environ.get("VISION_CONTROL", str(ROOT / "vision-control.json")))
RTSP_BASE = os.environ.get("VISION_RTSP_BASE", "rtsp://127.0.0.1:18554")
FPS = float(os.environ.get("VISION_FPS", "10"))
CONF = float(os.environ.get("VISION_CONF", "0.15"))
BUILD_ENGINE = os.environ.get("VISION_BUILD_ENGINE", "1") == "1"
COLOR_EVERY_S = float(os.environ.get("VISION_COLOR_EVERY_S", "0.5"))
STATUS = ROOT / "worker-status.json"

state = {"worker": "vision", "cuda_available": False, "model_loaded": False, "flight_enabled": False,
         "state": "starting", "model": None, "backend": None, "mission_id": None,
         "fps": 0.0, "latency_ms_p50": None, "last_frame_age_s": None, "time_source": None,
         "frames_processed": 0, "frames_dropped": 0, "detections": 0, "candidates": 0, "confirmed": 0,
         "geo_ok_ratio": None, "error": None}


def write_status(**kw):
    state.update(kw)
    state["updated_unix"] = time.time()
    tmp = STATUS.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    tmp.replace(STATUS)


def read_control():
    try:
        c = json.loads(CONTROL.read_text())
        return c if c.get("mission_id") else None
    except (OSError, ValueError, AttributeError):
        return None


def engine_path():
    import tensorrt as trt
    import torch
    gpu = torch.cuda.get_device_name(0).replace("NVIDIA ", "").replace("GeForce ", "").replace(" ", "")
    return MODEL_DIR / f"best_trt{trt.__version__}_{gpu}.engine"


def load_detector():
    from app.vision.detector import PersonDetector
    import torch
    write_status(cuda_available=torch.cuda.is_available())
    w = MODEL_DIR / "best.pt"
    while not w.exists():
        write_status(state="waiting_for_model", error=f"{w} 없음")
        time.sleep(5)
    eng = None
    try:
        eng = engine_path()
        if not eng.exists() and BUILD_ENGINE:
            write_status(state="building_engine", error=None)
            import threading
            from ultralytics import YOLO
            done = []
            def build():
                out = YOLO(str(w)).export(format="engine", half=True, imgsz=[736, 1280], batch=4, device=0, verbose=False)
                Path(out).replace(eng); done.append(1)
            th = threading.Thread(target=build, daemon=True); th.start()
            while th.is_alive():                  # 빌드 중에도 상태 파일은 계속 (헬스체크 20초)
                write_status(state="building_engine"); time.sleep(5)
            if not done:
                eng = None
    except Exception as e:                        # TensorRT 가 없거나 빌드 실패 → PyTorch FP16 로
        write_status(error=f"엔진 없음 → PyTorch FP16: {type(e).__name__}: {e}"[:300]); eng = None
    det = PersonDetector(w, engine=eng, conf=CONF)
    write_status(model_loaded=True, model=det.name, backend=det.backend, state="idle_no_mission")
    return det


def run_mission(det, ctl):
    import cv2
    import numpy as np
    from app.vision.color import ColorAccumulator, ColorProfiler, frame_stats, match
    from app.vision.geo import GeoResolver
    from app.vision.registry import CandidateRegistry
    from app.vision.sources import FileSource, LiveSource
    from app.vision.telemetry import TelemetryStore

    mid = ctl["mission_id"]
    mdir = ROOT / "missions" / mid
    for sub in ("video", "metadata", "crops", "results", "logs"):
        (mdir / sub).mkdir(parents=True, exist_ok=True)
    src_s = ctl.get("source") or f"{RTSP_BASE}/{mid}"
    live = src_s.startswith(("rtsp://", "srt://", "udp://"))
    src = LiveSource(src_s) if live else FileSource(src_s, FPS)
    tel, geo, reg = TelemetryStore(mdir), GeoResolver(), CandidateRegistry(mid)
    colorer, query = ColorProfiler(), ctl.get("query_color")      # 찾는 사람 상의 색 (UC-0707 · 없으면 판정만)
    det_f = open(mdir / "results" / "detections.jsonl", "a", encoding="utf-8")
    cand_f = open(mdir / "results" / "candidates.jsonl", "a", encoding="utf-8")
    log = open(mdir / "logs" / "vision.log", "a", encoding="utf-8")
    log.write(f"{time.strftime('%F %T')} 시작 · 모델 {det.name} · {det.backend} · 영상 {src_s}\n"); log.flush()
    write_status(state="running", mission_id=mid, frames_processed=0, detections=0, candidates=0, confirmed=0)
    lat_ms, n_geo, n_geo_ok, t_last_status, period = [], 0, 0, 0.0, 1.0 / FPS
    t_frame_prev, fps_ema, result = None, 0.0, "stopped"
    try:
        while True:
            c = read_control()
            if not c or c.get("mission_id") != mid:
                break
            t_loop = time.time()
            got = src.latest(timeout=1.0)
            if got is None:
                if not live:
                    result = "finished_file"; break
                write_status(state="waiting_for_video", last_frame_age_s=None); continue
            frame, pts_ms, recv = got
            tel.refresh()
            t_cap = tel.frame_time(pts_ms)
            tsrc = "sidecar" if t_cap is not None else "receive"
            t_us = t_cap if t_cap is not None else recv * 1e6
            boxes, confs, ms = det.detect(frame)
            lat_ms.append(ms); lat_ms = lat_ms[-200:]
            pose = tel.at(t_cap) if t_cap is not None else None
            fstats = frame_stats(frame) if len(boxes) else None
            recs = []
            for b, cf in zip(boxes, confs):
                g = geo.to_world(tuple(float(v) for v in b), pose) if pose is not None else None
                n_geo += 1; n_geo_ok += int(g is not None and g.status == "OK")
                cand, ev = reg.update(t_us / 1e6, b, float(cf), g)
                if t_us / 1e6 - cand.color_t >= COLOR_EVERY_S:      # 후보마다 0.5초에 한 번 (CPU)
                    cand.color_t = t_us / 1e6
                    if cand.color is None:
                        cand.color = ColorAccumulator()
                    cand.color.add(colorer.profile(frame, b, stats=fstats))
                r = {"bbox": [round(float(v), 1) for v in b], "conf": round(float(cf), 4), "candidate_id": cand.cid,
                     "geo": None if g is None else {"status": g.status, "lat": g.lat, "lon": g.lon,
                                                    "ellipse": g.ellipse, "reasons": g.reasons}}
                recs.append(r)
                if ev in ("new", "best", "confirmed"):
                    x1, y1, x2, y2 = [int(round(v)) for v in b]
                    pad = max(8, int(0.3 * max(x2 - x1, y2 - y1)))
                    crop = frame[max(0, y1 - pad):y2 + pad, max(0, x1 - pad):x2 + pad]
                    if crop.size:
                        cv2.imwrite(str(mdir / "crops" / f"{cand.cid}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    cj = cand.to_json()
                    if query:
                        cj["color_status"], cj["color_score"] = match(cand.color.result(), query)
                    cand_f.write(json.dumps({"event": ev, "t_us": int(t_us), **cj,
                                             "crop": f"crops/{cand.cid}.jpg"}, ensure_ascii=False) + "\n")
            det_f.write(json.dumps({"t_us": int(t_us), "time_source": tsrc, "pts_ms": pts_ms,
                                    "infer_ms": round(ms, 1), "detections": recs}, ensure_ascii=False) + "\n")
            if t_frame_prev is not None:
                fps_ema = 0.9 * fps_ema + 0.1 / max(t_loop - t_frame_prev, 1e-3)
            t_frame_prev = t_loop
            state["frames_processed"] += 1; state["detections"] += len(recs)
            if time.time() - t_last_status >= 1.0:
                det_f.flush(); cand_f.flush()
                write_status(fps=round(fps_ema, 2), latency_ms_p50=round(float(np.median(lat_ms)), 1),
                             last_frame_age_s=round(time.time() - recv, 3), time_source=tsrc,
                             frames_dropped=src.dropped, candidates=len(reg.items),
                             confirmed=sum(c.confirmed for c in reg.items),
                             geo_ok_ratio=round(n_geo_ok / n_geo, 3) if n_geo else None)
                t_last_status = time.time()
            if not live:                          # 파일은 목표 fps 로 건너뛰며 읽으니 쉬지 않는다
                continue
            rest = period - (time.time() - t_loop)
            if rest > 0:
                time.sleep(rest)
    finally:
        src.close(); det_f.close(); cand_f.close()
        log.write(f"{time.strftime('%F %T')} 끝 · 처리 {state['frames_processed']}장 · 후보 {len(reg.items)}\n"); log.close()
        write_status(state="idle_no_mission", mission_id=None)
    return result


def main():
    write_status(state="starting")
    try:
        det = load_detector()
    except Exception as e:
        while True:                               # 모델을 못 올려도 상태는 계속 알린다
            write_status(state="model_error", model_loaded=False, error=f"{type(e).__name__}: {e}"[:300])
            time.sleep(5)
    done = None                                   # 다 읽은 녹화 파일은 지정이 바뀔 때까지 다시 돌지 않는다
    while True:
        ctl = read_control()
        key = None if not ctl else (ctl["mission_id"], ctl.get("source"))
        if ctl and key != done:
            try:
                if run_mission(det, ctl) == "finished_file":
                    done = key
            except Exception as e:
                write_status(state="mission_error", error=traceback.format_exc()[-600:])
                time.sleep(5)
        else:
            if not ctl:
                done = None
            write_status(state="finished_file" if ctl else "idle_no_mission")
            time.sleep(2)


if __name__ == "__main__":
    main()
