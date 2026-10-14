"""
ue_vision_eval.py — 로컬 UE 비행 기록으로 비전 worker 성능을 잰다 (서버 · 2026-10-09)

  입력  ue_flight/<run>/ — ue_flight_sequence.py (노트북 · --passive 면 관제 ROUTE 비행을 옆에서 기록)
        frames/<n>.jpg · flight.csv (프레임마다 위경도 · NED · roll/pitch/yaw · 시뮬 시각) · boxes.jsonl (배우별 정답 박스)
        actors.json (배우 이름 · 자세 · NED 위치) · run.json (시작 NED · 마운트 · 카메라 기체 고정)
  흐름  프레임마다 **배포한 worker 와 같은 코드** IngestWorker.observe — 타일 탐지 · 좌표 · 후보 기억 · 추적 · 상태 · 색 · 인상착의 · 재관측
        촬영 자세 = flight.csv (짐벌 안정화 없음 → stabilized False · 짐벌 피치 −마운트) · 시각 = 시뮬 시각
  지표
    탐지   정답 (분할 마스크 ≥ --min-px 화소) 과 IoU ≥ 0.5 · AP50 · 재현율 · 정밀도 (운용 문턱 0.15) · 사람 키 px 층별 재현율
           사람 없는 장의 오탐 수/장 (NFR-V06 ≤ 1)
    시간   탐지 ms · 한 장 전체 (탐지 + 좌표 + 후보 + 상태) ms 중앙 · P95 (NFR-V01 ≤ 30 · V02 ≥ 5 FPS)
    좌표   맞은 관측의 VALID 좌표 ↔ 배우 정답 수평 오차 중앙 · P95 — 전체 · 안정 프레임 (|roll| · |pitch| ≤ 6°) (NFR-V04 10~15 m)
           같은 자세로 **정답 박스**를 넣은 오차를 같이 → 탐지 박스 탓 vs 자세 · 지면 가정 탓을 나눈다 · 보류 사유 분포
    후보   배우마다 처음 보인 · 처음 잡힌 · 확정된 시각 · 후보 수 (1 이 정답 · 2 이상 = 같은 사람 중복) · 배우 없는 확정 후보 (헛후보)
           후보 융합 위치 오차
    상태   배우마다 마지막 상태 (등급 · 자세 · 누움 점수) ↔ 이름의 자세 (Person_<자세>_<번호>)
    재관측 · 인상착의  이유별 수 · 판정 분포 (--query 가 있으면)
  출력  metrics/ue_vision_<run>.json · .md · (--fo) FiftyOne ue/<run> — 정답 · 탐지 (후보 · 좌표 오차) 를 눈으로
실행  /home/se/miniconda3/envs/drone/bin/python ue_vision_eval.py ue_flight/<run> [--query red] [--fo]
      … ue_vision_eval.py --selftest   (UE 기록 없이 코드 경로만 — level01 사진으로 만든 가짜 기록 · 수치는 의미 없음)
"""
import argparse
import csv
import json
import math
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / "deploy" / "vision_worker"))
from app.vision_core.detector import PersonDetector                 # noqa: E402
from app.vision_core.geo import GeoResolver                         # noqa: E402
from app.vision_core.ingest import IngestWorker, telemetry_from_pose  # noqa: E402

R_EARTH = 6378137.0
STABLE_DEG = 6.0
H_BINS = [(0, 20), (20, 40), (40, 80), (80, 1e9)]
GAP_S = 1.0                     # 이 이상 못 잡혔다가 다시 잡히면 「다시 잇기」 채점 (S2 가림 2~6 s)


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u if u > 0 else 0.0


def ap50(scored, n_gt):
    """(확신도, 맞음) 목록 → 모든 점 보간 AP"""
    if not n_gt:
        return None
    s = sorted(scored, key=lambda x: -x[0]); tp = np.cumsum([x[1] for x in s]); fp = np.cumsum([not x[1] for x in s])
    rec = np.concatenate([[0], tp / n_gt, [tp[-1] / n_gt if len(tp) else 0]]); prec = np.concatenate([[1], tp / np.maximum(tp + fp, 1), [0]])
    for i in range(len(prec) - 2, -1, -1):
        prec[i] = max(prec[i], prec[i + 1])
    return float(np.sum((rec[1:] - rec[:-1]) * prec[1:]))


def ne_offset(lat, lon, lat0, lon0):
    return (math.radians(lat - lat0) * R_EARTH, math.radians(lon - lon0) * R_EARTH * math.cos(math.radians(lat0)))


def pct(v, q):
    return round(float(np.percentile(v, q)), 2) if len(v) else None


def truth_posture(name, raw):
    s = (raw or name).lower()
    return "lying" if ("ly" in s or "crawl" in s) else "sitting" if any(k in s for k in ("sit", "kneel", "crouch")) else "standing"   # 걷기 → 서기


class Timed:
    """메서드 하나를 감싸 호출마다 ms 를 프레임 단위로 모은다 (단계별 시간)"""
    def __init__(self, fn):
        self.fn, self.acc = fn, 0.0

    def __call__(self, *a, **k):
        t0 = time.perf_counter(); r = self.fn(*a, **k); self.acc += (time.perf_counter() - t0) * 1000
        return r

    def take(self):
        v, self.acc = self.acc, 0.0
        return v


class TimedDet:
    """worker 탐지기를 감싸 한 장 탐지 ms 를 남긴다"""
    def __init__(self, det):
        self.det, self.ms = det, []

    def detect(self, img, conf=None):
        b, c, ms = self.det.detect(img, conf)
        self.ms.append(ms); return b, c, ms


def load_run(run):
    rows = list(csv.DictReader(open(run / "flight.csv", encoding="utf-8")))
    boxes = {}
    for line in open(run / "boxes.jsonl", encoding="utf-8"):
        j = json.loads(line); boxes[j["frame"]] = j["boxes"]
    actors = json.loads((run / "actors.json").read_text(encoding="utf-8"))
    meta = json.loads((run / "run.json").read_text(encoding="utf-8"))
    return rows, boxes, actors, meta


def pose_of(r, meta):
    """flight.csv 한 줄 → worker 프레임 pose (기체 고정 카메라)"""
    alt = float(meta["start_ned"][2]) - float(r["ned_z"])                       # 이륙점 지면 기준
    return {"lat": float(r["lat"]), "lon": float(r["lon"]), "alt_agl_m": alt, "roll_deg": float(r["roll_deg"]),
            "pitch_deg": float(r["pitch_deg"]), "yaw_deg": float(r["yaw_deg"]),
            "gimbal_pitch_deg": float(r.get("cam_pitch_rel_body_deg") or -float(meta["mount_deg"])), "gimbal_yaw_deg": 0.0,
            "gimbal_stabilized": bool(meta.get("gimbal_stabilized", False))}


COLOR_PROFILES = {"base": {}, "v14": {"wb": None, "exp_clip": (1.0, 2.0)}}   # v14 = 밝히기만 · 화이트밸런스 끔 (10-14 UE 16 m 로 고름 · eval_color V14)


def evaluate(run, weights, engine, posture, query, min_px, fo_name=None, conf=0.20, color="base"):
    rows, gtb, actors, meta = load_run(run)
    det = TimedDet(PersonDetector(weights, engine=engine, conf=conf))
    scratch = Path(tempfile.mkdtemp(prefix="uevis_"))
    w = IngestWorker(None, None, det, "ue-eval", scratch=scratch, upload_crops=False, posture=posture)
    if color != "base":
        from app.vision_core.color import ColorProfiler
        w.colorer = ColorProfiler(**COLOR_PROFILES[color])
    mid, geo = "ue-" + run.name, GeoResolver()
    from app.vision_core import state as state_mod
    t_color = Timed(w.colorer.profile); w.colorer.profile = t_color
    t_state = Timed(state_mod.StateTracker.update); state_mod.StateTracker.update = lambda self, *a, **k: t_state(self, *a, **k)
    stage = {"color": [], "state": []}
    scored, n_gt, hb = [], 0, {b: [0, 0] for b in H_BINS}
    empty_fp, empty_n = [], 0
    geo_err, geo_err_stable, gt_err, gt_err_stable, pend = [], [], [], [], Counter()
    act = defaultdict(lambda: {"first_vis": None, "first_det": None, "first_conf": None, "cands": Counter(), "last_state": None, "n_vis": 0, "n_det": 0})
    cand_act = defaultdict(Counter); cand_conf = set(); reobs = Counter(); reobs_act = Counter(); verdicts = Counter(); act_verd = defaultdict(Counter)
    st_log = []                                                             # (배우 · t · 누움 항 · 종합 점수 · 등급 · 무동작 s) — 요구조자 판단 채점
    total_ms, samples = [], []
    trk_seq = defaultdict(list); trk_pair = Counter(); n_pred_trk = 0      # 10-14 추적 지표 (정답 id ↔ worker track_id)
    hist = defaultdict(list)                                                # 배우별 (t · 후보 · 추적) — 가림 뒤 다시 잇기
    for r in rows:
        stem = r["frame"]; img = cv2.imread(str(run / "frames" / f"{stem}.jpg"))
        if img is None:
            continue
        t = int(r["sim_ts_ns"]) / 1e9 if r.get("sim_ts_ns") else float(r["t_s"])
        pose = pose_of(r, meta); tel = telemetry_from_pose(pose)
        stable = abs(pose["roll_deg"]) <= STABLE_DEG and abs(pose["pitch_deg"]) <= STABLE_DEG
        t0 = time.perf_counter()
        obs = w.observe(mid, img, t, tel, False, query=query)
        total_ms.append((time.perf_counter() - t0) * 1000); stage["color"].append(t_color.take()); stage["state"].append(t_state.take())
        for q in w.pending_reobs:
            reobs[q["reason"]] += 1
        rq = list(w.pending_reobs); w.pending_reobs.clear()
        gts = {n: g for n, g in (gtb.get(stem) or {}).items() if n in actors}
        valid = {n: g for n, g in gts.items() if g["px"] >= min_px}
        n_gt += len(valid)
        for n in valid:
            a = act[n]; a["n_vis"] += 1; a["first_vis"] = a["first_vis"] if a["first_vis"] is not None else t
            h = valid[n]["bb"][3] - valid[n]["bb"][1]
            for b in H_BINS:
                if b[0] <= h < b[1]:
                    hb[b][1] += 1
        used, matched = set(), {}
        for k, o in sorted(enumerate(obs), key=lambda x: -x[1]["confidence"]):
            bb = [o["bbox"][c] for c in ("x1", "y1", "x2", "y2")]
            best, bn = 0.5, None
            for n, g in gts.items():
                if n not in used and iou(bb, g["bb"]) >= best:
                    best, bn = iou(bb, g["bb"]), n
            if bn is not None:
                used.add(bn)
                if bn in valid:
                    scored.append((o["confidence"], True)); matched[k] = bn
                # 기준 미만 (작은 마스크) 정답과 맞으면 채점에서 뺀다
            else:
                scored.append((o["confidence"], False))
            m = (o.get("appearance") or {}).get("match")
            if m:
                verdicts[m["verdict"]] += 1
                if bn is not None and bn in valid:
                    act_verd[bn][m["verdict"]] += 1
        if not gts:
            empty_n += 1; empty_fp.append(len(obs))
        veh = (float(r["ned_x"]), float(r["ned_y"]))
        n_pred_trk += sum(1 for o in obs if o.get("track_id"))
        for k, n in matched.items():
            tid = norm_tid(obs[k].get("track_id"))
            if tid:
                trk_seq[n].append(tid); trk_pair[(n, tid)] += 1
        for k, n in matched.items():
            o = obs[k]; a = act[n]; a["n_det"] += 1; hist[n].append((t, o["candidate_id"], norm_tid(o.get("track_id"))))
            a["first_det"] = a["first_det"] if a["first_det"] is not None else t
            if o["auto_confirmed"] and a["first_conf"] is None:
                a["first_conf"] = t
            a["cands"][o["candidate_id"]] += 1; a["last_state"] = o.get("state"); st_log.append(state_row(n, t, o.get("state"))); cand_act[o["candidate_id"]][n] += 1
            for b in H_BINS:
                h = valid[n]["bb"][3] - valid[n]["bb"][1]
                if b[0] <= h < b[1]:
                    hb[b][0] += 1
            truth = valid[n].get("ned") or actors[n]["ned"]        # 10-14 움직이는 배우는 boxes.jsonl 의 프레임별 위치
            g = o["geo"]
            if g["status"] == "VALID":
                dn, de = ne_offset(g["lat"], g["lon"], pose["lat"], pose["lon"])
                e = math.hypot(veh[0] + dn - truth[0], veh[1] + de - truth[1])
                geo_err.append(e); (geo_err_stable.append(e) if stable else None)
            else:
                pend.update(g.get("reasons") or ["?"])
            a["last_color"] = ((o.get("appearance") or {}).get("upper") or {}).get("top")
            gr = geo.to_world(tuple(valid[n]["bb"]), tel) if tel else None
            if gr is not None and gr.status == "OK":
                e2 = math.hypot(veh[0] + gr.north - truth[0], veh[1] + gr.east - truth[1])
                gt_err.append(e2); (gt_err_stable.append(e2) if stable else None)
        for o in obs:
            if o["auto_confirmed"]:
                cand_conf.add(o["candidate_id"])
        for q in rq:
            hit = cand_act.get(q["candidate_id"])
            reobs_act["배우 있음" if hit else "배우 없음"] += 1
        if fo_name:
            samples.append((run / "frames" / f"{stem}.jpg", gts, obs, matched, stable, t))
    # 후보 → 배우 (관측 다수결)
    cand_major = {c: cnt.most_common(1)[0][0] for c, cnt in cand_act.items()}
    false_conf = sorted(c for c in cand_conf if c not in cand_major)
    reg = w.reg.get(mid)
    cpos = {}
    if reg:
        for c in reg.items:
            u = w.uid.get((mid, c.cid))
            if u in cand_major and c.lat is not None:
                cpos.setdefault(cand_major[u], []).append((c, u))
    state_mod.StateTracker.update = t_state.fn
    per_actor = []
    for n, a in sorted(act.items()):
        if not a["n_vis"]:
            continue
        own = [c for c, k in cand_major.items() if k == n]
        st = a["last_state"] or {}
        perr = None
        if n in cpos and rows:
            r0 = rows[0]; truth = actors[n]["ned"]
            # 후보 융합 위치: 위경도 → 첫 프레임 기체 위치 기준 NED
            c, _ = max(cpos[n], key=lambda x: len(x[0].hits))
            dn, de = ne_offset(c.lat, c.lon, float(r0["lat"]), float(r0["lon"]))
            perr = round(math.hypot(float(r0["ned_x"]) + dn - truth[0], float(r0["ned_y"]) + de - truth[1]), 2)
        seq = trk_seq.get(n, [])
        gaps = [{"gap_s": round(t1 - t0, 2), "same_candidate": c0 == c1, "same_track": bool(k0) and k0 == k1}
                for (t0, c0, k0), (t1, c1, k1) in zip(hist[n], hist[n][1:]) if t1 - t0 >= GAP_S]
        sw = sum(1 for i in range(1, len(seq)) if seq[i] != seq[i - 1])
        tc = actors[n].get("upper_color"); top = [c["color"] for c in (a.get("last_color") or [])]
        per_actor.append({"actor": n, "truth_posture": truth_posture(n, actors[n].get("pose_raw")), "frames_visible": a["n_vis"],
                          "tracks": len(set(seq)), "id_switches": sw, "gaps": gaps, "track_runs": runs(hist[n]), "truth_color": tc, "color_top2": top,
                          "color_top1_ok": (top[0] == tc) if (tc and top) else None, "color_top2_ok": (tc in top) if (tc and top) else None,
                          "frames_detected": a["n_det"], "first_visible_s": a["first_vis"], "first_detect_s": a["first_det"],
                          "first_confirm_s": a["first_conf"], "candidates": len(own), "merged_into_other": a["n_det"] > 0 and not own, "confirmed": sum(c in cand_conf for c in own),
                          "candidate_pos_err_m": perr, "grade": st.get("grade"), "posture": (st.get("posture") or {}).get("label"),
                          "lying_score": (st.get("score_terms") or {}).get("lying")})
    tp_at = [s for s in scored if s[0] >= conf]
    warm = slice(1, None) if len(total_ms) > 1 else slice(None)        # 첫 장 = 엔진 · 추적기 예열 → 시간에서 뺀다
    det_ms, total_ms = det.ms[warm], total_ms[warm]
    res = {"run": run.name, "frames": len(total_ms), "actors": len(actors), "actors_visible": len(per_actor),
           "detection": {"gt_boxes": n_gt, "min_px": min_px, "AP50": None if not n_gt else round(ap50(scored, n_gt), 4),
                         "recall@0.15": round(sum(x[1] for x in tp_at) / n_gt, 4) if n_gt else None,
                         "precision@0.15": round(sum(x[1] for x in tp_at) / max(len(tp_at), 1), 4),
                         "recall_by_height_px": {f"{a}-{b if b < 1e9 else '∞'}": (round(v[0] / v[1], 3) if v[1] else None, v[1]) for (a, b), v in hb.items()},
                         "empty_frames": empty_n, "fp_per_empty_frame": round(float(np.mean(empty_fp)), 3) if empty_n else None},
           "latency_ms": {"detect_p50": pct(det_ms, 50), "detect_p95": pct(det_ms, 95),
                          "color_p50": pct(stage["color"][warm], 50), "state_p50": pct(stage["state"][warm], 50), "frame_p50": pct(total_ms, 50), "frame_p95": pct(total_ms, 95),
                          "fps_at_p50": round(1000 / np.median(total_ms), 1) if total_ms else None, "backend": "tensorrt" if engine else "pytorch_fp16"},
           "geo": {"det_valid": len(geo_err), "det_err_p50": pct(geo_err, 50), "det_err_p95": pct(geo_err, 95),
                   "det_stable_n": len(geo_err_stable), "det_stable_p50": pct(geo_err_stable, 50), "det_stable_p95": pct(geo_err_stable, 95),
                   "gtbox_err_p50": pct(gt_err, 50), "gtbox_stable_p50": pct(gt_err_stable, 50), "pending_reasons": dict(pend.most_common())},
           "candidates": {"total": len(w.uid), "confirmed": len(cand_conf), "confirmed_without_actor": len(false_conf),
                          "actors_found": sum(p["frames_detected"] > 0 for p in per_actor), "actors_confirmed": sum(p["confirmed"] > 0 for p in per_actor),
                          "actors_with_duplicates": sum(p["candidates"] > 1 for p in per_actor),
                          "actors_merged_into_other": sum(p["merged_into_other"] for p in per_actor)},
           "rescue": rescue_metrics(st_log, actors),
           "tracking": {**tracking_metrics(trk_pair, trk_seq, n_gt, n_pred_trk), **reassoc(per_actor)},
           "color": {"actors_with_truth": sum(p["color_top1_ok"] is not None for p in per_actor),
                     "top1_acc": _rate([p["color_top1_ok"] for p in per_actor]), "top2_acc": _rate([p["color_top2_ok"] for p in per_actor])},
           "reobservation": {"by_reason": dict(reobs), "target": dict(reobs_act)}, "appearance": {"query": query, "verdicts": dict(verdicts), **appearance_by_actor(act_verd, query, actors)},
           "per_actor": per_actor}
    if fo_name:
        to_fiftyone(fo_name, samples, res)
    return res


def _rate(v):
    v = [x for x in v if x is not None]
    return round(sum(v) / len(v), 3) if v else None


def state_row(n, t, st):
    st = st or {}
    return (n, t, (st.get("score_terms") or {}).get("lying"), st.get("score"), st.get("grade"), (st.get("motion") or {}).get("still_s"))


def auc_ties(pos, neg):
    """Mann-Whitney AUROC (동점 0.5)"""
    if not pos or not neg:
        return None
    a = np.array(pos)[:, None]; b = np.array(neg)[None, :]
    return round(float(((a > b).sum() + 0.5 * (a == b).sum()) / (a.size * b.size)), 4)


def rescue_metrics(rows, actors):
    """요구조자 판단 (판정 기준 U) — 장 단위 누움 항 · 종합 점수 AUROC (누움 · 기어감 vs 그 밖) · 배우별 등급 시간 비율 · 최고 등급"""
    lab = {n: truth_posture(n, actors[n].get("pose_raw")) == "lying" for n in actors}
    out = {}
    for k, key in ((2, "lying_term"), (3, "score")):
        v = [(lab[r[0]], r[k]) for r in rows if r[k] is not None]
        out[f"auroc_{key}"] = auc_ties([x for l, x in v if l], [x for l, x in v if not l])
    order = ["URGENT", "HIGH", "CHECK", "NORMAL"]
    per = {}
    for n in actors:
        g = Counter(r[4] for r in rows if r[0] == n and r[4])
        if not g:
            continue
        tot = sum(g.values())
        per[n] = {"lying": lab[n], "max_grade": next(x for x in order if g.get(x)), "frac": {x: round(g[x] / tot, 3) for x in order if g.get(x)},
                  "max_still_s": max((r[5] or 0) for r in rows if r[0] == n), "lying_term_p50": round(float(np.median([r[2] for r in rows if r[0] == n and r[2] is not None] or [np.nan])), 3)}
    out["per_actor"] = per
    ly = [p for p in per.values() if p["lying"]]
    out["lying_reached_urgent_or_high"] = f"{sum(p['max_grade'] in ('URGENT', 'HIGH') for p in ly)}/{len(ly)}" if ly else None
    oth = [p for p in per.values() if not p["lying"]]
    out["others_reached_urgent"] = f"{sum(p['max_grade'] == 'URGENT' for p in oth)}/{len(oth)}" if oth else None
    return out


def appearance_by_actor(av, query, actors):
    """찾는 색과 상의색이 같은 배우 = 대상 · 대상 MATCH 율 vs 다른 사람 MATCH 율 (판정 기준 U · 장 단위)"""
    if not query:
        return {}
    want = set(query["upper"])
    rate = lambda c: round(c["MATCH"] / sum(c.values()), 3) if sum(c.values()) else None
    tgt = [n for n in av if actors[n].get("upper_color") in want]
    oth = [n for n in av if n not in tgt]
    return {"by_actor": {n: dict(av[n]) for n in av}, "targets": tgt,
            "target_match": rate(sum((av[n] for n in tgt), Counter())), "others_match": rate(sum((av[n] for n in oth), Counter())),
            "others_match_max": max((rate(av[n]) or 0 for n in oth), default=None)}


def norm_tid(k):
    """worker 추적 id 는 이력이 생기기 전 '임무:14' · 뒤 '임무:14.0' (점 뒤 = ID 바뀜 횟수) → 같은 추적으로 맞춘다"""
    if not k:
        return k
    return k if "." in k.rsplit(":", 1)[-1] else k + ".0"


def runs(h):
    """(시작 s · 끝 s · 장 수 · 추적 · 후보 앞 8자) — 추적이 어디서 끊기는지 보기용"""
    out = []
    for t, c, k in h:
        if out and out[-1][3] == k and out[-1][4] == c[:8]:
            out[-1][1] = round(t, 1); out[-1][2] += 1
        else:
            out.append([round(t, 1), round(t, 1), 1, k, c[:8]])
    return out


def reassoc(per_actor):
    """놓친 뒤 (GAP_S 이상) 다시 잡혔을 때 같은 후보 · 같은 추적으로 이어졌나 — S2 가림 · 판정 기준 U"""
    g = [x for a in per_actor for x in a["gaps"]]
    return {"gaps": len(g), "gap_same_candidate": round(np.mean([x["same_candidate"] for x in g]), 3) if g else None,
            "gap_same_track": round(np.mean([x["same_track"] for x in g]), 3) if g else None}


def tracking_metrics(pair, seq, n_gt, n_pred):
    """IDF1 (정답 id ↔ track id 를 겹친 장 수로 1:1 짝 · 탐욕) · ID 바뀜 · 배우당 추적 수"""
    used_g, used_t, idtp = set(), set(), 0
    for (g, t), c in sorted(pair.items(), key=lambda x: -x[1]):
        if g in used_g or t in used_t:
            continue
        used_g.add(g); used_t.add(t); idtp += c
    idfp, idfn = max(n_pred - idtp, 0), max(n_gt - idtp, 0)
    sw = sum(sum(1 for i in range(1, len(s)) if s[i] != s[i - 1]) for s in seq.values())
    return {"IDF1": round(2 * idtp / max(2 * idtp + idfp + idfn, 1), 4), "id_switches": sw,
            "tracks_per_actor": round(float(np.mean([len(set(s)) for s in seq.values()])), 2) if seq else None}


def to_fiftyone(name, samples, res):
    import fiftyone as fo
    if name in fo.list_datasets():
        fo.delete_dataset(name)
    ds = fo.Dataset(name, persistent=True)
    out = []
    for path, gts, obs, matched, stable, t in samples:
        hit = set(matched.values())
        gt = [fo.Detection(label="person", bounding_box=[g["bb"][0] / 1920, g["bb"][1] / 1080, (g["bb"][2] - g["bb"][0]) / 1920, (g["bb"][3] - g["bb"][1]) / 1080],
                           actor=n, px=g["px"], missed=n not in hit, tags=[] if n in hit else ["missed"]) for n, g in gts.items()]
        pr = []
        for k, o in enumerate(obs):
            b = o["bbox"]
            pr.append(fo.Detection(label="person", confidence=o["confidence"], candidate=o["candidate_id"][:8], actor=matched.get(k),
                                   geo=o["geo"]["status"], grade=(o.get("state") or {}).get("grade"),
                                   bounding_box=[b["x1"] / 1920, b["y1"] / 1080, (b["x2"] - b["x1"]) / 1920, (b["y2"] - b["y1"]) / 1080],
                                   tags=[] if k in matched else ["false_positive"]))
        tags = (["stable"] if stable else ["tilted"]) + (["has_miss"] if any(n not in hit for n in gts) else []) + (["has_fp"] if len(pr) > len(matched) else [])
        out.append(fo.Sample(filepath=str(path), ground_truth=fo.Detections(detections=gt), predictions=fo.Detections(detections=pr), t_s=t, tags=tags))
    ds.add_samples(out)
    ds.info = {"ue_vision_eval": {k: v for k, v in res.items() if k != "per_actor"}}
    ds.save(); print(f"FiftyOne: {name} · {len(out)} 장")


def selftest(tmp):
    """UE 기록 없이 코드 경로만 — level01 사진 (정답 박스 · 마운트 · 고도) 으로 가짜 기록 · 배우 정답 = 정답 박스의 좌표"""
    src = BASE / "data" / "pose_cls" / "ue_level01"
    meta_rows = {r["image"]: r for r in csv.DictReader(open(src / "meta.csv", encoding="utf-8"))}
    run = Path(tmp) / "selftest"; (run / "frames").mkdir(parents=True)
    names = sorted(meta_rows)[:12]
    rows, actors, geo, fb = [], {}, GeoResolver(), open(run / "boxes.jsonl", "w")
    for i, nm in enumerate(names):
        m = meta_rows[nm]; stem = f"{i + 1:05d}"
        (run / "frames" / f"{stem}.jpg").write_bytes((src / "images" / nm).read_bytes())
        lat, lon = 36.145 + i * 1e-4, 128.393
        r = {"frame": stem, "t_s": i * 1.0, "sim_ts_ns": int(i * 1e9), "ned_x": i * 11.1, "ned_y": 0.0, "ned_z": -float(m["alt"]),
             "lat": lat, "lon": lon, "roll_deg": 0.0, "pitch_deg": 0.0, "yaw_deg": float(m["yaw_deg"]), "cam_pitch_rel_body_deg": -float(m["mount"])}
        rows.append(r); tel = telemetry_from_pose(pose_of(r, {"start_ned": [0, 0, 0], "mount_deg": m["mount"], "gimbal_stabilized": False}))
        bx = {}
        for k, line in enumerate(open(src / "labels" / nm.replace(".jpg", ".txt"))):
            _, cx, cy, bw, bh = map(float, line.split())
            bb = [(cx - bw / 2) * 1920, (cy - bh / 2) * 1080, (cx + bw / 2) * 1920, (cy + bh / 2) * 1080]
            g = geo.to_world(tuple(bb), tel); an = f"Person_Standing_{i:02d}{k:02d}"
            if g.status == "OK":
                actors[an] = {"pose_raw": "standing", "ned": [r["ned_x"] + g.north, r["ned_y"] + g.east, 0.0], "yaw_deg": 0}
                bx[an] = {"bb": bb, "px": int(bw * 1920 * bh * 1080 * 0.5), "cut": False}
        fb.write(json.dumps({"frame": stem, "sim_ts_ns": r["sim_ts_ns"], "boxes": bx}) + "\n")
    fb.close()
    with open(run / "flight.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    (run / "actors.json").write_text(json.dumps(actors)); (run / "run.json").write_text(json.dumps({"start_ned": [0, 0, 0], "mount_deg": 45, "gimbal_stabilized": False}))
    return run


def report_md(res):
    d, l, g, c = res["detection"], res["latency_ms"], res["geo"], res["candidates"]
    lines = [f"# UE 비전 평가 — {res['run']}", "", f"- 프레임 {res['frames']} · 배우 {res['actors']} (화면에 보인 {res['actors_visible']})", "",
             "| 항목 | 값 | 기준 |", "|---|---|---|",
             f"| AP50 · 재현율@운용 · 정밀도@운용 | {d['AP50']} · {d['recall@0.15']} · {d['precision@0.15']} | NFR-V03 AP50 ≥ 0.80 (공식은 실사진) |",
             f"| 사람 없는 장 오탐/장 | {d['fp_per_empty_frame']} ({d['empty_frames']} 장) | NFR-V06 ≤ 1 |",
             f"| 탐지 ms p50 · p95 | {l['detect_p50']} · {l['detect_p95']} ({l['backend']}) | NFR-V01 ≤ 30 |",
             f"| 한 장 전체 ms p50 · p95 | {l['frame_p50']} · {l['frame_p95']} (≈ {l['fps_at_p50']} FPS) — 색 {l['color_p50']} · 추적·상태 {l['state_p50']} | NFR-V02 ≥ 5 FPS |",
             f"| 좌표 오차 중앙 · P95 (전체 {g['det_valid']}) | {g['det_err_p50']} · {g['det_err_p95']} m | |",
             f"| 좌표 오차 안정 프레임 중앙 · P95 ({g['det_stable_n']}) | {g['det_stable_p50']} · {g['det_stable_p95']} m | NFR-V04 10~15 m |",
             f"| 정답 박스로 좌표 (중앙 · 안정) | {g['gtbox_err_p50']} · {g['gtbox_stable_p50']} m | 탐지 박스 탓 분리 |",
             f"| 배우 찾음 · 확정 · 중복 후보 · 남의 후보에 합쳐짐 | {c['actors_found']} · {c['actors_confirmed']} · {c['actors_with_duplicates']} · {c['actors_merged_into_other']} / {res['actors_visible']} | 중복 · 합쳐짐 0 이 정답 |",
             f"| 후보 전체 · 확정 · 배우 없는 확정 | {c['total']} · {c['confirmed']} · {c['confirmed_without_actor']} | |",
             f"| 추적 IDF1 · ID 바뀜 · 배우당 추적 수 | {res['tracking']['IDF1']} · {res['tracking']['id_switches']} · {res['tracking']['tracks_per_actor']} | 1 · 0 · 1 이 정답 |",
             f"| 가림 뒤 다시 잇기 (≥ {GAP_S:.0f} s · 같은 후보 · 같은 추적) | {res['tracking']['gaps']} 번 · {res['tracking']['gap_same_candidate']} · {res['tracking']['gap_same_track']} | 1 이 정답 |",
             f"| 상의색 1순위 · 1·2순위 맞음 ({res['color']['actors_with_truth']} 명) | {res['color']['top1_acc']} · {res['color']['top2_acc']} | |",
             f"| 요구조자 — 누움 AUROC (누움 항 · 종합) · 누운 사람 URGENT/HIGH 도달 · 다른 사람 URGENT | {res['rescue']['auroc_lying_term']} · {res['rescue']['auroc_score']} · {res['rescue']['lying_reached_urgent_or_high']} · {res['rescue']['others_reached_urgent']} | AUROC ≥ 0.85 · 누운 사람 전부 |",
             f"| 재관측 | {res['reobservation']['by_reason']} | |", f"| 인상착의 ({res['appearance']['query']}) | {res['appearance']['verdicts']} · 대상 MATCH {res['appearance'].get('target_match')} · 다른 사람 MATCH {res['appearance'].get('others_match')} (최대 {res['appearance'].get('others_match_max')}) | 대상 MATCH · 다른 사람 ≤ 0.20 |", "",
             f"- 사람 키 px 별 재현율 (재현율, 정답 수): {d['recall_by_height_px']}", f"- 좌표 보류 사유: {g['pending_reasons']}", "",
             "| 배우 | 정답 자세 | 보임 · 잡힘 | 처음 보임 · 잡힘 · 확정 (s) | 후보 · 확정 | 후보 위치 오차 m | 등급 · 자세 · 누움 |", "|---|---|---|---|---|---|---|"]
    for p in res["per_actor"]:
        f = lambda v: "—" if v is None else f"{v:.1f}"
        lines.append(f"| {p['actor']} | {p['truth_posture']} | {p['frames_visible']} · {p['frames_detected']} | {f(p['first_visible_s'])} · {f(p['first_detect_s'])} · {f(p['first_confirm_s'])} "
                     f"| {p['candidates']} · {p['confirmed']} | {p['candidate_pos_err_m']} | {p['grade']} · {p['posture']} · {p['lying_score']} |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?")
    ap.add_argument("--weights", default=str(BASE / "runs_person" / "soup_v7r2" / "weights" / "best.pt"))
    ap.add_argument("--engine", default=str(BASE / "runs_person" / "soup_v7r2" / "weights" / "best.engine"), help="'' 이면 PyTorch FP16")
    ap.add_argument("--posture", default=str(BASE / "runs_state" / "posture.json"), help="없으면 상태 없이")
    ap.add_argument("--query", default=None, help="찾는 사람 상의 색 (예: red · 남색)")
    ap.add_argument("--min-px", type=int, default=40, help="정답으로 칠 최소 마스크 화소")
    ap.add_argument("--conf", type=float, default=0.20, help="운용 임계값 (10-13 스윕 제안 0.20)")
    ap.add_argument("--fo", action="store_true", help="FiftyOne ue/<run> 으로 올림")
    ap.add_argument("--color", default="base", choices=list(COLOR_PROFILES), help="상의색 설정 (base = worker 그대로)")
    ap.add_argument("--tag", default="", help="결과 파일 이름 뒤에 붙임 (같은 기록을 질의 색만 바꿔 여러 번)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    tmp = tempfile.mkdtemp(prefix="uevis_self_") if a.selftest else None
    run = selftest(tmp) if a.selftest else Path(a.run)
    posture = a.posture if Path(a.posture).exists() else None
    q = {"upper": [a.query]} if a.query else None
    res = evaluate(run, a.weights, a.engine or None, posture, q, a.min_px, fo_name=(f"ue/{run.name}{a.tag}" if a.fo else None), conf=a.conf, color=a.color)
    res["color_profile"] = a.color
    res["posture_model"] = bool(posture)
    md = report_md(res)
    if not a.selftest:
        (BASE / "metrics" / f"ue_vision_{run.name}{a.tag}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
        (BASE / "metrics" / f"ue_vision_{run.name}{a.tag}.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
