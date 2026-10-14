"""
make_model_pack.py — 후보 모델을 "갈아 끼우는 폴더" 로 묶는다 (데이터셋은 안 넣음 · 10-07)

  출력 <out>/<이름>/best.pt · model.json (이름 · SHA-256 · 세 평가셋 AP50 · 속도 · 파이프라인 판)  ·  <out>/active.json (지금 쓸 모델)
  model.json 의 내용으로 백엔드에 모델 설정을 등록하고 (POST /api/v1/configs/models · 설정 관리자) 받은 uuid 를 active.json 의 model_config_id 에 적는다
  상태 인지 계수: --posture=<posture.json> (export_posture_models.py 출력) → <out>/posture.json (모델과 상관없이 하나)
실행: python tools/make_model_pack.py <out> v12_f91_swa [soup_v7r2 …] [--active=v12_f91_swa] [--posture=posture.json] [--env=SIM|REAL] [--engine=<엔진 파일>]
  --env    model.json environment (배포 worker 가 INGEST_ENVIRONMENT 와 맞는지 검사 · 10-14 통합 v1.1) · 기본 REAL
  --engine TensorRT 엔진을 꾸러미에 넣고 model.json engine {file, sha256} 기록 (배포 v1.2 는 해시가 맞을 때만 씀) · 첫 모델에만
"""
import csv, hashlib, json, shutil, sys
from pathlib import Path

RUNS = Path(__file__).resolve().parents[3] / "runs_person"
MET = Path(__file__).resolve().parents[3] / "metrics"
args = [a for a in sys.argv[1:] if not a.startswith("--")]
opt = lambda k: next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith(f"--{k}=")), None)
env, engine = (opt("env") or "REAL").upper(), opt("engine")
active = next((a.split("=")[1] for a in sys.argv[1:] if a.startswith("--active=")), args[1] if len(args) > 1 else None)
out = Path(args[0]); out.mkdir(parents=True, exist_ok=True)
ap = lambda f: next(iter(csv.DictReader(open(f))), {}).get("AP50") if Path(f).exists() else None
for name in args[1:]:
    src = RUNS / name / "weights" / "best.pt"; d = out / name; d.mkdir(exist_ok=True)
    shutil.copy2(src, d / "best.pt")
    h = hashlib.sha256((d / "best.pt").read_bytes()).hexdigest()
    card = {"name": name, "sha256": h, "framework": "ultralytics 8.4.102 · YOLO11m", "pipeline_version": "vision_worker 1.2 (타일 4장 · NMS 0.6 + 경계 잘린 박스 지우기 · conf 0.20)",
            "environment": env, "ap50": {t: ap(MET / f"{t}_{name}.csv") for t in ("test_v2", "test_obl", "test_kr", "test_v2_edge_ign", "test_kr_edge")}}
    if engine and name == args[1]:
        shutil.copy2(engine, d / Path(engine).name)
        card["engine"] = {"file": Path(engine).name, "sha256": hashlib.sha256((d / Path(engine).name).read_bytes()).hexdigest(),
                          "tensorrt": "FP16 · [736,1280] · batch 4 · GPU 마다 다시 빌드", "verify_ap50_test_v2": 0.6417}
    (d / "model.json").write_text(json.dumps(card, ensure_ascii=False, indent=1))
    print(name, h[:16], card["ap50"])
pst = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--posture=")), None)
if pst:
    shutil.copy2(pst, out / "posture.json"); print("posture →", out / "posture.json")
if active:
    (out / "active.json").write_text(json.dumps({"name": active, "model_config_id": "<백엔드에 등록한 uuid>", "conf": 0.20, "upload_crops": True}, ensure_ascii=False, indent=1))
    print("active →", active)
