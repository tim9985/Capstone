"""
tests_edge_confirm.py — 10-17 T 나무 줄기 오탐 (VISION_EDGE_MARGIN_PX · VISION_EMIT_CONFIRMED_ONLY) 재생 판정 (기준 = _학습 큐 「10-17 T」)

  ① route_132434 (노트북 Kumoh · 사람 없음 · 0.5 s): 순항 관측 · 보낸 후보 수 (저고도 막기 + DEM 켠 채)
  ② UE S1~S4 · 16/20 m (사람 있음 · 정답 박스) → 0.5 s 간격으로 솎아 (실연 간격): 배우 단위 재현율 · 첫 관측까지 시간 · 박스 단위 관측 재현율 · 오탐 관측
  변형: 기준 · T1 (가장자리 3 px) · T2 (확정된 후보만) · T1+T2
실행  python tests_edge_confirm.py [--step 5]
"""
import argparse, csv, json, os, shutil, statistics, sys, tempfile, uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
DEM = "/home/se/sandboxes/drone-dev/data/spatial/campus-04e1b46514574d45/terrain"
BACKEND = "/home/se/sandboxes/drone-real"
VARIANTS = {"기준": ("0", "0"), "T1 가장자리": ("3", "0"), "T2 확정만": ("0", "1"), "T1+T2": ("3", "1")}
UE_RUNS = ["S1_a16", "S2_a16", "S3_a16", "S4_a16", "S1_a20", "S2_a20", "S3_a20", "S4_a20"]


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix
    return ix / u if u > 0 else 0.0


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--step", type=int, default=5); a = ap.parse_args()
    stage = Path(tempfile.mkdtemp(prefix="edgeconf_"))
    shutil.copytree(Path(BACKEND) / "app", stage / "app", ignore=shutil.ignore_patterns("*.bak-*", "__pycache__"))
    for f in (HERE / "app").rglob("*.py"):
        dst = stage / "app" / f.relative_to(HERE / "app"); dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(f, dst)
    sys.path.insert(0, str(stage)); sys.path.insert(0, str(HERE))
    import cv2
    from app.vision_core.detector import PersonDetector
    from app.vision_core.ingest import IngestWorker
    from tests_v1_2 import frame_pose

    W = REPO / "drone_yolo" / "runs_person" / "_pack_v12_sim" / "v12_f91_swa"
    det = PersonDetector(W / "best.pt", engine=W / "best_trt11.3.0.99_RTX3090.engine", conf=0.20)
    posture = str(REPO / "drone_yolo" / "runs_person" / "_pack_v12" / "posture.json")

    def worker(var, dem):
        os.environ["VISION_EDGE_MARGIN_PX"], os.environ["VISION_EMIT_CONFIRMED_ONLY"] = VARIANTS[var]
        os.environ["VISION_MIN_AGL_M"] = "10"; os.environ["VISION_DEM_DIR"] = dem
        return IngestWorker(None, lambda: "x", det, str(uuid.uuid4()), scratch=tempfile.mkdtemp(prefix="edgeconf_w_"),
                            upload_crops=False, posture=posture, environment="SIM")

    out = {"route_132434": {}, "ue": {}}
    # ① 사람 없는 경로
    run = REPO / "drone_yolo" / "ue_flight" / "route_132434"
    recs = [json.loads(l) for l in open(run / "poses.jsonl")]
    imgs = {r["frame"]: cv2.imread(str(run / "frames" / f"{r['frame']}.jpg")) for r in recs}
    for var in VARIANTS:
        w = worker(var, DEM); mid = str(uuid.uuid4()); n_obs, cands = 0, set()
        for r in recs:
            pose = {"schema_version": "frame-pose/1.0", **r["pose"]}
            obs = w.observe(mid, imgs[r["frame"]], int(pose["source_timestamp_ns"]) / 1e9, pose, upload_crops=False)
            n_obs += len(obs); cands |= {o["candidate_id"] for o in obs}
        row = {"관측": n_obs, "보낸 후보": len(cands), "가장자리 막음": w.stats.get("edge_dets", 0), "확정 전 보류": w.stats.get("unconfirmed_dets", 0)}
        out["route_132434"][var] = row; print("route", var, row)
    # ② 사람 있는 UE 장면 (평지 AGL pose · DEM 없음 → 저고도 막기는 AGL 로)
    t0 = datetime(2026, 10, 17, 6, 0, tzinfo=timezone.utc)
    agg = {v: {"actors": 0, "found": 0, "gt": 0, "hit": 0, "fp": 0, "delay": []} for v in VARIANTS}
    for rn in UE_RUNS:
        d = REPO / "drone_yolo" / "ue_flight" / rn
        if not (d / "flight.csv").exists():
            continue
        meta = json.load(open(d / "run.json")); actors = json.load(open(d / "actors.json"))
        gt = {json.loads(l)["frame"]: json.loads(l)["boxes"] for l in open(d / "boxes.jsonl")}
        rows = list(csv.DictReader(open(d / "flight.csv")))[:: a.step]
        frames = {r["frame"]: cv2.imread(str(d / "frames" / f"{r['frame']}.jpg")) for r in rows}
        out["ue"][rn] = {}
        for var in VARIANTS:
            w = worker(var, ""); mid = str(uuid.uuid4())
            first_vis, first_obs = {}, {}; gt_n = hit = fp = 0
            for r in rows:
                t = float(r["t_s"]); cap = (t0 + timedelta(seconds=t)).isoformat().replace("+00:00", "Z")
                pose = frame_pose(r, meta, str(uuid.uuid4()), cap, "SIM")
                obs = w.observe(mid, frames[r["frame"]], t, pose, upload_crops=False)
                bbs = [(o["bbox"]["x1"], o["bbox"]["y1"], o["bbox"]["x2"], o["bbox"]["y2"]) for o in obs]
                used = set()
                for n, g in (gt.get(r["frame"]) or {}).items():
                    if n not in actors or g["px"] < 300:
                        continue
                    gt_n += 1; first_vis.setdefault(n, t)
                    k = max(range(len(bbs)), key=lambda k: iou(bbs[k], g["bb"]), default=None)
                    if k is not None and iou(bbs[k], g["bb"]) >= 0.3:
                        hit += 1; used.add(k); first_obs.setdefault(n, t)
                fp += len(bbs) - len(used)
            dl = [first_obs[n] - first_vis[n] for n in first_obs]
            row = {"배우": len(first_vis), "찾은 배우": len(first_obs), "박스 재현율": round(hit / max(gt_n, 1), 3), "오탐 관측": fp,
                   "첫 관측 지연 중앙 s": round(statistics.median(dl), 2) if dl else None,
                   "가장자리 막음": w.stats.get("edge_dets", 0), "확정 전 보류": w.stats.get("unconfirmed_dets", 0)}
            out["ue"][rn][var] = row; print(rn, var, row)
            A = agg[var]; A["actors"] += len(first_vis); A["found"] += len(first_obs); A["gt"] += gt_n; A["hit"] += hit; A["fp"] += fp; A["delay"] += dl
    out["ue_total"] = {v: {"배우 재현율": f"{A['found']}/{A['actors']}", "박스 재현율": round(A["hit"] / max(A["gt"], 1), 3), "오탐 관측": A["fp"],
                           "첫 관측 지연 중앙 s": round(statistics.median(A["delay"]), 2) if A["delay"] else None} for v, A in agg.items()}
    print(json.dumps(out["ue_total"], ensure_ascii=False, indent=1))
    (REPO / "drone_yolo" / "metrics" / "edge_confirm_replay.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
