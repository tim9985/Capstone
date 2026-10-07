"""
make_model_pack.py — 후보 모델을 "갈아 끼우는 폴더" 로 묶는다 (데이터셋은 안 넣음 · 10-07)

  출력 <out>/<이름>/best.pt · model.json (이름 · SHA-256 · 세 평가셋 AP50 · 속도 · 파이프라인 판)  ·  <out>/active.json (지금 쓸 모델)
  model.json 의 내용으로 백엔드에 모델 설정을 등록하고 (POST /api/v1/configs/models · 설정 관리자) 받은 uuid 를 active.json 의 model_config_id 에 적는다
실행: python tools/make_model_pack.py <out> soup_v7r2 [soup_v9x2 …] [--active=soup_v7r2]
"""
import csv, hashlib, json, shutil, sys
from pathlib import Path

RUNS = Path(__file__).resolve().parents[3] / "runs_person"
MET = Path(__file__).resolve().parents[3] / "metrics"
args = [a for a in sys.argv[1:] if not a.startswith("--")]
active = next((a.split("=")[1] for a in sys.argv[1:] if a.startswith("--active=")), args[1] if len(args) > 1 else None)
out = Path(args[0]); out.mkdir(parents=True, exist_ok=True)
ap = lambda f: next(iter(csv.DictReader(open(f))), {}).get("AP50") if Path(f).exists() else None
for name in args[1:]:
    src = RUNS / name / "weights" / "best.pt"; d = out / name; d.mkdir(exist_ok=True)
    shutil.copy2(src, d / "best.pt")
    h = hashlib.sha256((d / "best.pt").read_bytes()).hexdigest()
    card = {"name": name, "sha256": h, "framework": "ultralytics 8.4.102 · YOLO11m", "pipeline_version": "vision_worker 1.0 (타일 4장 · NMS 0.6 · conf 0.15)",
            "environment": "REAL", "ap50": {t: ap(MET / f"{t}_{name}.csv") for t in ("test_v2", "test_obl", "test_kr")}}
    (d / "model.json").write_text(json.dumps(card, ensure_ascii=False, indent=1))
    print(name, h[:16], card["ap50"])
if active:
    (out / "active.json").write_text(json.dumps({"name": active, "model_config_id": "<백엔드에 등록한 uuid>", "conf": 0.15, "upload_crops": True}, ensure_ascii=False, indent=1))
    print("active →", active)
