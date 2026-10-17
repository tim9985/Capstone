"""
ue_route_check.py — 10-17 노트북 로컬 녹화 (연결기 형식 · 사람 없음) → 배포 worker 경로 그대로 검사

  ① 파일: sha256 · 장 수 · 해상도 · pose 필드 (frame-pose/1.0 · 쿼터니언 · 고도 기준 · sync_error_ms · missing_reasons)
  ② 탐지 (v12_f91_swa · TensorRT · 0.20): 사람이 없으니 **전부 오탐** · 장당 오탐 · 탐지 ms
  ③ 좌표: 화면 중앙 아래쪽 가상 발끝 → PoseGeo (DEM) 상태 · 막는 사유 집계 (sync 규칙 끄고도 한 번)
  ④ DEM 일관성: GPS 고도 (MSL) − 기체 아래 DEM = AGL → 순항 목표 16 m 와 비교
실행  python ue_route_check.py <route 폴더>
"""
import hashlib, json, math, sys, time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / "deploy" / "vision_worker"))
from app.vision_core.detector import PersonDetector  # noqa: E402
from app.vision_core.geo import CameraModel  # noqa: E402
from app.vision_core import pose as P  # noqa: E402

PACK = BASE / "runs_person" / "_pack_v12_sim" / "v12_f91_swa"
DEM = "/home/se/sandboxes/drone-dev/data/spatial/campus-04e1b46514574d45/terrain"


def q(v, p):
    v = sorted(v)
    return round(v[min(len(v) - 1, int(p * len(v)))], 2) if v else None


def get_pose(rec):
    for k in ("pose", "frame_pose", "frame-pose"):
        if isinstance(rec.get(k), dict):
            return {"schema_version": "frame-pose/1.0", **rec[k]}     # 백엔드 FramePose 기본값과 같게 (연결기 원본엔 없음)
    return rec if rec.get("schema_version") else None


def main(root):
    root = Path(root)
    recs = [json.loads(l) for l in open(root / "poses.jsonl") if l.strip()]
    imgs = sorted((root / "frames").glob("*.jpg"))
    by = {Path(str(r.get("frame"))).stem: r for r in recs}
    det = PersonDetector(PACK / "best.pt", PACK / "best_trt11.3.0.99_RTX3090.engine", conf=0.20)
    dem = P.DemTiles(DEM)
    out = {"run": root.name, "frames": len(imgs), "poses": len(recs), "backend": det.backend}
    sha_bad, sizes, fp, ms, sync, datum, miss, src = 0, Counter(), [], [], [], Counter(), Counter(), Counter()
    st_on, st_off, agl, rng = Counter(), Counter(), [], []
    fp_samples = []
    for im in imgs:
        rec = by.get(im.stem) or by.get(im.name) or {}
        raw = im.read_bytes()
        if rec.get("sha256") and hashlib.sha256(raw).hexdigest() != rec["sha256"]:
            sha_bad += 1
        f = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        h, w = f.shape[:2]; sizes[f"{w}x{h}"] += 1
        b, c, t = det.detect(f); ms.append(t); fp.append(len(b))
        if len(b) and len(fp_samples) < 40:
            fp_samples += [(im.name, [round(float(x)) for x in bb], round(float(s), 3)) for bb, s in zip(b, c)][:3]
        pz = get_pose(rec)
        if not pz:
            st_on["POSE_MISSING"] += 1; continue
        if pz.get("sync_error_ms") is not None:
            sync.append(float(pz["sync_error_ms"]))
        g = pz.get("geographic") or {}
        datum[g.get("altitude_datum")] += 1
        for m in pz.get("missing_reasons") or []:
            miss[m] += 1
        cam = CameraModel(width=w, height=h, hfov_deg=80.2)
        box = (w / 2 - 10, h * 0.6, w / 2 + 10, h * 0.75)
        for pg, st, use_sync in ((P.PoseGeo(cam, dem=dem), st_on, True), (P.PoseGeo(cam, dem=dem), st_off, False)):
            pp = pz if use_sync else {k: v for k, v in pz.items() if k != "sync_error_ms"}
            a = pg.locate(box, pp)
            st[a["status"] if a["status"] == "CLAIM" else "PENDING:" + ",".join(r.split(" ")[0] for r in a["reasons"])] += 1
            if not use_sync:
                src[a.get("source")] += 1
                if a["range_m"] is not None:
                    rng.append(a["range_m"])
        if g.get("altitude_datum") == "MSL" and g.get("latitude") is not None:
            z = dem(float(g["latitude"]), float(g["longitude"]))
            if z is not None:
                agl.append(float(g["altitude_m"]) - z)
    n = max(len(imgs), 1)
    out.update({
        "sha256_mismatch": sha_bad, "sizes": dict(sizes),
        "fp_per_frame": round(sum(fp) / n, 3), "frames_with_fp": sum(1 for x in fp if x), "fp_total": sum(fp),
        "det_ms_p50": q(ms[3:], .5), "det_ms_p95": q(ms[3:], .95),
        "sync_error_ms": {"p50": q(sync, .5), "p95": q(sync, .95), "max": q(sync, 1.0), "over_50": sum(1 for s in sync if s > 50), "n": len(sync)},
        "altitude_datum": dict(datum), "missing_reasons": dict(miss), "attitude_source": dict(src),
        "geo_status_rule_on": dict(st_on), "geo_status_sync_ignored": dict(st_off),
        "center_range_m": {"p50": q(rng, .5), "p95": q(rng, .95)},
        "agl_from_dem_m": {"p50": q(agl, .5), "p90": q(agl, .9), "max": q(agl, 1.0), "min": q(agl, 0.0)},
        "fp_samples": fp_samples,
    })
    print(json.dumps(out, ensure_ascii=False, indent=1))
    (BASE / "metrics" / f"ue_route_check_{root.name}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
