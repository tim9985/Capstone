"""
ue_botsort_reid_cmp.py — 10-17 B · BoT-SORT ReID 끔 / 켬 비교 (덤프 = ue_group_dump.py · 판정 기준 = _학습 큐 「10-17 B」)

  추적 IDF1 (배우 ↔ 추적 키 1:1 · 겹친 장 수) · ID 바뀜 · 후보 지표 (지금 worker 규칙 · ue_group_eval.BASECFG)
실행  python ue_botsort_reid_cmp.py
"""
import json, pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import ue_group_eval as G

BASE = Path(__file__).resolve().parent
RUNS = {"16 m": ["S1_a16", "S2_a16", "S3_a16", "S4_a16"], "20 m": ["S1_a20", "S2_a20", "S3_a20", "S4_a20"]}
VARIANTS = {"끔 (지금)": "", "켬 · yolo26s-reid": "_reid_yolo26s_reid", "켬 · yolo11s-cls": "_reid_yolo11s_cls"}


def track_metrics(d):
    pair, seq, n_gt, n_pred = Counter(), defaultdict(list), 0, 0
    for f in d["frames"]:
        n_gt += len(f["visible"])
        for det in f["dets"]:
            if det["tkey"]:
                n_pred += 1
                if det["actor"]:
                    pair[(det["actor"], det["tkey"])] += 1; seq[det["actor"]].append(det["tkey"])
    ua, ut, idtp = set(), set(), 0
    for (a, t), c in sorted(pair.items(), key=lambda x: -x[1]):
        if a not in ua and t not in ut:
            ua.add(a); ut.add(t); idtp += c
    sw = sum(sum(1 for i in range(1, len(s)) if s[i] != s[i - 1]) for s in seq.values())
    return 2 * idtp / max(n_gt + n_pred, 1), sw


def main():
    out = {}
    for alt, runs in RUNS.items():
        for name, tag in VARIANTS.items():
            idf, sw, cpa, mixed, cidf, ms = [], 0, [], 0, [], []
            for r in runs:
                p = BASE / "runs_state" / f"group_{r}{tag}.pkl"
                if not p.exists():
                    continue
                d = pickle.loads(p.read_bytes())
                a, b = track_metrics(d); idf.append(a); sw += b
                e = G.evaluate(d, G.BASECFG)
                cpa.append(e["cand_per_actor"]); mixed += e["mixed"]; cidf.append(e["idf1"])
                if d.get("track_ms_p50") is not None:
                    ms.append(d["track_ms_p50"])
            if not idf:
                continue
            row = {"추적 IDF1": round(float(np.mean(idf)), 3), "ID 바뀜": sw, "사람당 후보": round(float(np.mean(cpa)), 2),
                   "섞인 후보": mixed, "후보 IDF1": round(float(np.mean(cidf)), 3), "추적 ms 중앙": round(float(np.median(ms)), 1) if ms else None}
            out[f"{alt} · {name}"] = row
            print(alt, name, row)
    (BASE / "metrics" / "ue_botsort_reid_cmp.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
