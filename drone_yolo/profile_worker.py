"""
profile_worker.py — 비전 worker 단계별 지연 · VRAM · RAM (11-04 · 사용자 「임무별 vram ram 지연」)

  코드  배포 판 vision_core (/home/se/sandboxes/drone-real/app · 읽기만) — 탐지 (TensorRT · 타일 4장) · 추적 + 상태 · 좌표 · 상의 색 (규칙 V10 = 지금 배포)
        + worker 1.3 SigLIP 2 색 (integration_v1_3 · 아직 배포 안 됨) 따로
  입력  UE 비행 S1_a16 프레임 (1920×1080 · 사람 여러 명) · 좌표는 고정 가상 텔레메트리 (16 m · 짐벌 −45°)
  재는 것  단계별 ms (중앙 · P95 · 앞 10장 버림) · 단계마다 쌓인 GPU 메모리 (torch 최대 할당 · nvidia-smi 이 프로세스) · RSS
실행  /home/se/miniconda3/envs/drone/bin/python profile_worker.py [장수 200]
출력  metrics/profile_worker.json
"""
import json, os, subprocess, sys, time
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
DEPLOYED = Path("/home/se/sandboxes/drone-real")
PACK = BASE / "runs_person" / "_pack_v12"
UE = BASE / "ue_flight" / "S1_a16" / "frames"


def rss_mb():
    return int(open(f"/proc/{os.getpid()}/statm").read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**20


def smi_mb():
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    return next((int(l.split(",")[1]) for l in out.splitlines() if l.split(",")[0].strip() == str(os.getpid())), 0)


def stat(a):
    a = np.array(a[10:] if len(a) > 20 else a)
    return {"median_ms": round(float(np.median(a)), 2), "p95_ms": round(float(np.percentile(a, 95)), 2), "n": int(len(a))}


def main(n=200):
    import cv2, torch
    mem = {"start": {"rss_mb": round(rss_mb()), "gpu_mb": smi_mb()}}
    sys.path.insert(0, str(DEPLOYED))
    from app.vision_core.detector import PersonDetector
    from app.vision_core.geo import GeoResolver, Telemetry
    from app.vision_core.state import StateTracker
    from app.vision_core.color import ColorProfiler
    det = PersonDetector(PACK / "v12_f91_swa" / "best.pt", engine=PACK / "v12_f91_swa" / "best_trt11.3.0.99_RTX3090.engine", conf=0.20)
    frames = sorted(UE.glob("*.jpg"))[:n]
    det.detect(cv2.imread(str(frames[0])))
    mem["+탐지 (TensorRT)"] = {"rss_mb": round(rss_mb()), "gpu_mb": smi_mb()}
    trk = StateTracker(str(PACK / "posture.json"))
    geo = GeoResolver(); rule = ColorProfiler()
    tel = Telemetry(lat=37.0, lon=127.0, alt_agl=16.0, gimbal_pitch_deg=-45.0)
    mem["+상태 · 좌표 · 규칙 색"] = {"rss_mb": round(rss_mb()), "gpu_mb": smi_mb()}

    T = {k: [] for k in ("읽기 (JPEG 풀기)", "탐지", "추적 + 상태", "좌표 (사람 전부)", "규칙 색 (사람 전부)", "합계 (배포 판)")}
    T_sig, npers = [], []
    for k, p in enumerate(frames):
        t0 = time.perf_counter(); img = cv2.imread(str(p)); t1 = time.perf_counter()
        r = det.detect(img); t2 = time.perf_counter()
        boxes, confs = (r[0], r[1]) if isinstance(r, tuple) else (r["boxes"], r["confs"])
        boxes = [tuple(map(float, b)) for b in boxes]; confs = [float(c) for c in confs]
        trk.update(k / 10.0, img, boxes, confs); t3 = time.perf_counter()
        for b in boxes:
            geo.to_world(b, tel)
        t4 = time.perf_counter()
        for b in boxes:
            rule.profile(img, b)
        t5 = time.perf_counter()
        for key, v in zip(T, (t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4, t5 - t0)):
            T[key].append(v * 1000)
        npers.append(len(boxes))
    out = {"frames": len(frames), "people_per_frame_median": float(np.median(npers)), "stages": {k: stat(v) for k, v in T.items()}}
    mem["배포 판 다 돈 뒤"] = {"rss_mb": round(rss_mb()), "gpu_mb": smi_mb(), "torch_max_alloc_mb": round(torch.cuda.max_memory_allocated() / 2**20)}

    # worker 1.3 SigLIP 2 색 (배포 전) — 같은 프로세스에 더 얹음
    import importlib.util                                  # 1.3 color.py 는 표준 모듈만 씀 → 파일째 불러옴
    spec = importlib.util.spec_from_file_location("color13", BASE / "deploy" / "vision_worker" / "integration_v1_3" / "app" / "vision_core" / "color.py")
    c13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(c13); SiglipColorProfiler = c13.SiglipColorProfiler
    sig = SiglipColorProfiler(str(BASE / "runs_person" / "_pack_v12_real" / "color_siglip2"))
    kmax = int(np.argmax(npers)); img = cv2.imread(str(frames[kmax])); r = det.detect(img)   # 사람 있는 장으로 데우기
    sig.profile(img, tuple(map(float, r[0][0])))
    mem["+SigLIP 2 색 (1.3)"] = {"rss_mb": round(rss_mb()), "gpu_mb": smi_mb()}
    for p in frames:
        img = cv2.imread(str(p)); r = det.detect(img)
        for b in (r[0] if isinstance(r, tuple) else r["boxes"]):
            t0 = time.perf_counter(); sig.profile(img, tuple(map(float, b))); T_sig.append((time.perf_counter() - t0) * 1000)
    out["siglip_per_person"] = stat(T_sig)
    mem["끝"] = {"rss_mb": round(rss_mb()), "gpu_mb": smi_mb(), "torch_max_alloc_mb": round(torch.cuda.max_memory_allocated() / 2**20)}
    out["memory"] = mem
    json.dump(out, open(BASE / "metrics" / "profile_worker.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 200)
