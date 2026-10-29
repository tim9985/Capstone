"""
real_flight_eval.py — 실비행 녹화 평가 (11-25 최종발표 · 계획 obsidian 「20 최종발표 · 실기체 비전 계획」 · 10-30)

  입력 = worker 녹화 (VISION_RECORD_DIR · `<임무>/frames/*.jpg` + `poses.jsonl`) 또는 같은 형식 (노트북 경로 녹화)
  ① 파일: sha256 · 해상도 · pose 필드 · 시각 기준 · sync_error · 고도 기준 · DEM 지면 위 높이
  ② 배포 worker 그대로 다시 돌림 (샌드박스 REAL 판 + integration_v1_3 적용 · process() 경로) → 장당 ms · 관측 · 후보 · 상의 색 · 등급
  ③ 정답이 있으면 (`--gt gt.json`)
       "empty": [[i0, i1], …]   사람 없는 구간 (프레임 순번 · 끝 포함) → 오탐 건/장 (NFR-V06)
       "actors": {"이름": {"upper": "red", "pose": "lying"}, …}   출연자 옷 색 · 자세 기록 (12색 이름)
       "labels": "<YOLO 라벨 폴더>"   (프레임 이름.txt · 사람 0 · 정규화 cx cy w h) → AP50 · 재현율@0.20 (NFR-V03) · 크기별
       "actor_boxes": "<폴더>"       라벨 txt 의 6번째 칸 = 출연자 이름 → 후보 ↔ 출연자 이어 색 1순위 · 찾는 색 질의 · 등급
  ④ 라벨 작업용 내보내기 (`--export N`): N 장마다 그림 + 모델 박스 (YOLO txt · 미리 채움) → FiftyOne · CVAT 로 고쳐 `labels` 로 씀
  출력 metrics/real_flight_<임무>.json (수치만 · git) · 그림 · 라벨은 git 밖 (사람 얼굴)
실행  /home/se/miniconda3/envs/drone/bin/python real_flight_eval.py <녹화 임무 폴더> [--gt gt.json] [--export 10 --export-dir 경로] [--engine]
"""
import argparse, hashlib, json, math, os, shutil, subprocess, sys, tempfile, time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
V13 = BASE / "deploy" / "vision_worker" / "integration_v1_3"
SANDBOX = Path("/home/se/sandboxes/drone-real")
PACK = BASE / "runs_person" / "_pack_v12_real"
DEM = "/home/se/sandboxes/drone-dev/data/spatial/campus-04e1b46514574d45/terrain"


def stage_worker():
    """샌드박스 REAL 배포 판 app 복사 → integration_v1_3 (color · record · ingest 패치) 적용 → import 경로"""
    st = Path(tempfile.mkdtemp(prefix="rfe_"))
    shutil.copytree(SANDBOX / "app", st / "app", ignore=shutil.ignore_patterns("*.bak-*", "__pycache__"))
    for f in ("color.py", "record.py"):
        shutil.copy2(V13 / "app" / "vision_core" / f, st / "app" / "vision_core" / f)
    r = subprocess.run(["patch", "-p1", "-d", str(st), "-i", str(V13 / "ingest_real.patch")], capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError("ingest 패치 실패 (배포 판이 바뀜?) " + r.stdout[-300:])
    dev_pose = Path("/home/se/sandboxes/drone-dev/app/vision_core/pose.py")
    if dev_pose.exists():
        shutil.copy2(dev_pose, st / "app" / "vision_core" / "pose_dem.py")
    sys.path.insert(0, str(st))
    return st


def q(v, p):
    v = sorted(v)
    return round(float(v[min(len(v) - 1, int(p * len(v)))]), 2) if v else None


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u if u > 0 else 0.0


def read_yolo(p, W, H):
    out = []
    if p.exists():
        for l in open(p):
            t = l.split()
            if len(t) >= 5:
                cx, cy, w, h = (float(x) for x in t[1:5])
                out.append(((cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H, t[5] if len(t) > 5 else None))
    return out


def ap50(dets, gts):
    """dets {frame: [(box, conf)]} · gts {frame: [box]} → AP50 (모든 점 보간) · 재현율@0.20 · 크기별 재현율"""
    allp = sorted(((c, f, b) for f, L in dets.items() for b, c in L), key=lambda x: -x[0])
    used = {f: [False] * len(g) for f, g in gts.items()}
    tp, n = [], sum(len(g) for g in gts.values())
    hit20 = {f: [False] * len(g) for f, g in gts.items()}
    for c, f, b in allp:
        g = gts.get(f, []); best, j = 0.0, -1
        for k, gb in enumerate(g):
            v = iou(b, gb)
            if v > best and not used[f][k]:
                best, j = v, k
        if best >= 0.5:
            used[f][j] = True; tp.append(1)
            if c >= 0.20:
                hit20[f][j] = True
        else:
            tp.append(0)
    if not n:
        return None
    tp = np.array(tp); ctp = np.cumsum(tp); prec = ctp / np.arange(1, len(tp) + 1); rec = ctp / n
    mrec = np.concatenate([[0], rec, [1]]); mpre = np.concatenate([[0], prec, [0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    size = defaultdict(lambda: [0, 0])
    for f, g in gts.items():
        for k, gb in enumerate(g):
            h = gb[3] - gb[1]; key = "<24" if h < 24 else "24-50" if h < 50 else "50-100" if h < 100 else "≥100"
            size[key][0] += hit20[f][k]; size[key][1] += 1
    return {"AP50": round(float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])), 4), "사람 박스": n,
            "재현율@0.20": round(sum(sum(v) for v in hit20.values()) / n, 4), "키 px 별 재현율": {k: f"{a}/{b}" for k, (a, b) in sorted(size.items())}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("--gt"); ap.add_argument("--engine", action="store_true")
    ap.add_argument("--export", type=int, default=0); ap.add_argument("--export-dir")
    a = ap.parse_args()
    root = Path(a.root)
    recs = [json.loads(l) for l in open(root / "poses.jsonl") if l.strip()]
    gt = json.load(open(a.gt)) if a.gt else {}
    stage_worker()
    import cv2
    from app.vision_core.detector import PersonDetector
    from app.vision_core.ingest import IngestWorker
    from app.vision_core.color import match
    from app.vision_core import pose as P
    act = json.load(open(PACK / "active.json")); W0 = PACK / act["name"]
    eng = sorted(W0.glob("best_trt*.engine"))
    det = PersonDetector(W0 / "best.pt", engine=eng[0] if (a.engine and eng) else None, conf=float(act.get("conf", 0.20)))
    os.environ.setdefault("VISION_COLOR_PACK", str(PACK / "color_siglip2"))
    os.environ.pop("VISION_RECORD_DIR", None)
    w = IngestWorker(None, lambda: "", det, "eval", scratch=tempfile.mkdtemp(prefix="rfe_scratch_"), upload_crops=False,
                     posture=str(PACK / "posture.json"), environment="REAL")
    DemTiles = getattr(P, "DemTiles", None)
    if DemTiles is None:                                              # REAL 배포 판엔 DEM 이 없음 → SIM 배포 판 pose.py 를 옆에 두고 (지면 위 높이 보고용만)
        try:
            from app.vision_core.pose_dem import DemTiles
        except Exception as e:
            print("DEM 없이 진행:", type(e).__name__, e)
    dem = DemTiles(DEM) if DemTiles and Path(DEM).exists() else None
    out = {"임무": root.name, "장": len(recs), "탐지": det.backend, "상의 색": w.stats.get("color_method")}
    sha_bad, sizes, sync, datum, clock, agl, ms, nobs = 0, Counter(), [], Counter(), Counter(), [], [], []
    per_frame, cand = [], defaultdict(lambda: {"n": 0, "upper": None, "grade": Counter(), "last": None, "boxes": []})
    for k, r in enumerate(recs):
        data = (root / "frames" / r["frame"]).read_bytes()
        if r.get("sha256") and hashlib.sha256(data).hexdigest() != r["sha256"]:
            sha_bad += 1
        pz = r.get("pose") or {}
        clock[pz.get("clock_domain")] += 1
        if pz.get("sync_error_ms") is not None:
            sync.append(float(pz["sync_error_ms"]))
        g = pz.get("geographic") or {}
        datum[g.get("altitude_datum")] += 1
        if dem and g.get("altitude_datum") == "MSL" and g.get("latitude") is not None:
            z = dem(float(g["latitude"]), float(g["longitude"]))
            if z is not None:
                agl.append(float(g["altitude_m"]) - z)
        fr = {"frame_id": r.get("frame_id", Path(r["frame"]).stem), "mission_id": root.name, "width": r.get("width"), "height": r.get("height"),
              "capture_at": r.get("capture_at"), "pose": r.get("pose")}
        if fr["width"] is None:
            im = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR); fr["height"], fr["width"] = im.shape[:2]
        sizes[f"{fr['width']}x{fr['height']}"] += 1
        t0 = time.time(); body = w.process(fr, data); ms.append((time.time() - t0) * 1000)
        obs = body.get("observations", []); nobs.append(len(obs))
        dl = []
        for o in obs:
            bb = o["bbox"]; box = (bb["x1"], bb["y1"], bb["x2"], bb["y2"]); dl.append((box, float(o["confidence"])))
            c = cand[o.get("candidate_id") or o.get("track_id") or id(o)]
            c["n"] += 1; c["boxes"].append((r["frame"], box))
            up = ((o.get("appearance") or {}).get("upper") or {})
            if up.get("top"):
                c["upper"] = up["top"]
            stt = o.get("state") or {}
            if stt.get("grade"):
                c["grade"][stt["grade"]] += 1
                if stt.get("tracked") and (stt.get("observed_s") or 0) >= 3:     # 관찰 3 s 넘은 뒤 등급 (처음 3 s 는 규칙상 CHECK)
                    c["last"] = stt["grade"]
        per_frame.append((r["frame"], fr["width"], fr["height"], dl, body.get("analysis_state"), body.get("reason")))
    out.update({"sha256 틀림": sha_bad, "해상도": dict(sizes), "시각 기준": dict(clock), "고도 기준": dict(datum),
                "sync_error_ms": {"p50": q(sync, .5), "p95": q(sync, .95), "50 초과": sum(1 for s in sync if s > 50)},
                "DEM 지면 위 높이 m": {"p10": q(agl, .1), "p50": q(agl, .5), "p90": q(agl, .9)} if agl else None,
                "장당 처리 ms": {"p50": q(ms[3:], .5), "p95": q(ms[3:], .95)}, "관측/장": round(float(np.mean(nobs)), 3) if nobs else 0,
                "분석 상태": dict(Counter(s for *_, s, _ in per_frame)), "건너뜀 사유": dict(Counter((x or "").split(" ")[0] for *_, s, x in per_frame if x)),
                "후보": len(cand), "후보 상의 1순위": dict(Counter(c["upper"][0]["color"] for c in cand.values() if c["upper"])),
                "후보 마지막 등급 (관찰 3 s 넘은 것)": dict(Counter(c["last"] or "관찰 3 s 미만" for c in cand.values()))})
    # ③ 정답
    if gt.get("empty"):
        idx = {i for s, e in gt["empty"] for i in range(s, e + 1)}
        fp = [len(per_frame[i][3]) for i in idx if i < len(per_frame)]
        out["사람 없는 구간 오탐 건/장 (NFR-V06 ≤ 1)"] = round(float(np.mean(fp)), 3) if fp else None
    lab = Path(gt["labels"]) if gt.get("labels") else None
    if lab and lab.exists():
        gts, dets, who = {}, {}, {}
        for f, W, H, dl, *_ in per_frame:
            if (lab / (Path(f).stem + ".txt")).exists():
                L = read_yolo(lab / (Path(f).stem + ".txt"), W, H)
                gts[f] = [x[:4] for x in L]; who[f] = [x[4] for x in L]; dets[f] = dl
        out["탐지 (NFR-V03)"] = ap50(dets, gts)
        if gt.get("actors") and any(any(v) for v in who.values()):           # 후보 ↔ 출연자 (라벨 6번째 칸) → 색 · 등급
            vote = defaultdict(Counter)
            for cid, c in cand.items():
                for f, b in c["boxes"]:
                    if f in gts:
                        best = max(range(len(gts[f])), key=lambda k: iou(b, gts[f][k]), default=None)
                        if best is not None and iou(b, gts[f][best]) >= 0.5 and who[f][best]:
                            vote[cid][who[f][best]] += 1
            per_actor = defaultdict(list)
            for cid, v in vote.items():
                per_actor[v.most_common(1)[0][0]].append(cid)
            rows = {}
            for name, info in gt["actors"].items():
                cs = per_actor.get(name, [])
                top = [cand[c]["upper"][0]["color"] for c in cs if cand[c]["upper"]]
                hit = [match({"status": "ok", "top": [(x["color"], x["p"]) for x in cand[c]["upper"]], "n_obs": 1}, info["upper"])[0] == "일치" for c in cs if cand[c]["upper"]]
                others = [match({"status": "ok", "top": [(x["color"], x["p"]) for x in cand[c]["upper"]], "n_obs": 1}, info["upper"])[0] == "일치"
                          for n2, cs2 in per_actor.items() if gt["actors"].get(n2, {}).get("upper") != info["upper"] for c in cs2 if cand[c]["upper"]]
                rows[name] = {"정답 색": info["upper"], "자세": info.get("pose"), "후보 수": len(cs), "후보 1순위": dict(Counter(top)),
                              "질의 일치 (자기)": round(float(np.mean(hit)), 2) if hit else None, "질의 일치 (다른 색 사람)": round(float(np.mean(others)), 2) if others else None,
                              "마지막 등급": dict(Counter(cand[c]["last"] or "관찰 3 s 미만" for c in cs))}
            out["출연자별"] = rows
    # ④ 라벨 작업용 내보내기
    if a.export:
        ed = Path(a.export_dir or tempfile.mkdtemp(prefix=f"label_{root.name}_")); (ed / "images").mkdir(parents=True, exist_ok=True); (ed / "labels").mkdir(exist_ok=True)
        for k, (f, W, H, dl, *_ ) in enumerate(per_frame):
            if k % a.export:
                continue
            shutil.copy2(root / "frames" / f, ed / "images" / f)
            (ed / "labels" / (Path(f).stem + ".txt")).write_text("".join(f"0 {(b[0] + b[2]) / 2 / W:.6f} {(b[1] + b[3]) / 2 / H:.6f} {(b[2] - b[0]) / W:.6f} {(b[3] - b[1]) / H:.6f}\n" for b, c in dl))
        out["라벨 내보내기"] = {"폴더 (git 밖)": str(ed), "장": len(list((ed / "images").glob("*")))}
    print(json.dumps(out, ensure_ascii=False, indent=1))
    (BASE / "metrics" / f"real_flight_{root.name}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
