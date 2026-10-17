"""
ue_group_dump.py — 후보 묶기 실험용 덤프 (10-15 G · GPU 한 번 → 규칙은 CPU 로 여러 번)

  UE 기록 한 개 → 프레임마다 [탐지 박스 · 확신 · 추적 키 (worker StateTracker) · 좌표 (GeoResolver · telemetry) · 정답 배우 (IoU ≥ 0.5)]
  worker 와 같은 탐지 (v12_f91_swa · TensorRT · 0.20) · 같은 추적 (posture.json) · 같은 좌표 식
출력  runs_state/group_<run>.pkl (git 밖)
실행  python ue_group_dump.py S1_a16 S2_a16 …
"""
import pickle, sys
from pathlib import Path

import cv2

from ue_vision_eval import BASE, GeoResolver, PersonDetector, iou, load_run, pose_of, telemetry_from_pose

W = BASE / "runs_person" / "v12_f91_swa" / "weights"


def dump(name, reid=None):
    run = BASE / "ue_flight" / name
    rows, gtb, actors, meta = load_run(run)
    det = PersonDetector(W / "best.pt", engine=W / "best.engine", conf=0.20)
    from app.vision_core import state as S
    if reid:                                                       # 10-17 B — BoT-SORT ReID 켜기 (외형 모델 경로 · 문턱은 기본값)
        S.TRACK_ARGS.update(with_reid=True, model=reid)
    trk, geo, out = S.StateTracker(str(BASE / "runs_state" / "posture.json")), GeoResolver(), []
    ms = []
    for r in rows:
        img = cv2.imread(str(run / "frames" / f"{r['frame']}.jpg"))
        if img is None:
            continue
        t = int(r["sim_ts_ns"]) / 1e9 if r.get("sim_ts_ns") else float(r["t_s"])
        tel = telemetry_from_pose(pose_of(r, meta))
        boxes, confs, _ = det.detect(img)
        H, Wd = img.shape[:2]
        kept = []
        for b, c in sorted(zip(boxes, confs), key=lambda x: -x[1]):
            x1, y1, x2, y2 = max(0.0, float(b[0])), max(0.0, float(b[1])), min(float(Wd), float(b[2])), min(float(H), float(b[3]))
            if x2 > x1 and y2 > y1:
                kept.append(((x1, y1, x2, y2), float(c)))
        import time as _t
        t0 = _t.perf_counter()
        st = trk.update(t, img, [k[0] for k in kept], [k[1] for k in kept])
        ms.append((_t.perf_counter() - t0) * 1000)
        gts = {n: g for n, g in (gtb.get(r["frame"]) or {}).items() if n in actors}
        dets, used = [], set()
        for (bb, c), (tkey, _) in zip(kept, st):
            g = geo.to_world(bb, tel) if tel is not None else None
            gt = (g.status, g.lat, g.lon, tuple(g.ellipse) if g.ellipse else None) if g is not None else None
            best, bn = 0.5, None
            for n, gg in gts.items():
                if n not in used and iou(bb, gg["bb"]) >= best:
                    best, bn = iou(bb, gg["bb"]), n
            if bn:
                used.add(bn)
            dets.append({"box": bb, "conf": c, "tkey": tkey, "geo": gt, "actor": bn})
        out.append({"t": t, "dets": dets, "visible": [n for n, g in gts.items() if g["px"] >= 40]})
    tag = "" if not reid else "_reid_" + Path(reid).stem.replace("-", "_")
    p = BASE / "runs_state" / f"group_{name}{tag}.pkl"
    p.write_bytes(pickle.dumps({"run": name, "frames": out, "actors": list(actors), "track_ms_p50": float(sorted(ms)[len(ms) // 2])}))
    print(name, tag, len(out), "장 · 추적 ms 중앙", round(sorted(ms)[len(ms) // 2], 1), "→", p)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--reid=")]
    reid = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--reid=")), None)
    for n in args:
        dump(n, reid)
