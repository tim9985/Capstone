"""
state_diag.py — 상태 파이프라인 1초 표본 진단: 왜 5초 중앙값이 탐지 박스에서 누움 판별을 떨어뜨리나 (2026-10-03)

  10-03 P5: 탐지 박스 한 장 누움 AUROC 0.983 → 5초 중앙값 0.800 (정답 박스 P0 누적은 오히려 +0.006~0.03)
  입력: runs_state/samples_<tag>.json (state_pipeline.py --dump)
  본다
    ① 누움 표본 중 한 장은 누움(≥ 0.5)인데 중앙값은 아님(< 0.5) — 얼마나 · 왜
       · 전환: 최근 5표본 안에 같은 사람이 아직 누워 있지 않던 표본이 섞임 (누운 지 몇 초째인가)
       · ID 바뀜: 같은 추적 ID 의 최근 표본이 다른 정답 사람에게 붙어 있었음
       · 공백: 최근 5표본이 5초보다 넓게 퍼짐 (놓쳤다가 다시 잡힘)
    ② 정답 추적 단위 누움 구간 길이 (초)
    ③ 창 규칙별 누움 AUROC — 한 장 · 중앙값 3/5 · 평균 · 최댓값 (관찰 · 여기서 고르지 않는다)
  ⚠ Okutama 는 평가셋 — 이 진단으로 창 규칙을 고르지 않는다. 고르려면 영상을 나눠 한쪽에서만 (→ 볼트)
실행: python state_diag.py [tag=v9x2_pose_box_dump]
"""
import collections
import json
import os
import sys

import numpy as np

from posture_v2 import auc

BASE = os.path.dirname(os.path.abspath(__file__))


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "v9x2_pose_box_dump"
    S = json.load(open(os.path.join(BASE, "runs_state", f"samples_{tag}.json")))
    print(f"1초 표본 {len(S):,} · 누움 {sum(s['pose'] == 'Lying' for s in S)}")
    # 같은 추적 ID 의 표본을 시간순으로 — 정답 ID 이력
    by_trk = collections.defaultdict(list)
    for s in S:
        by_trk[(s["vid"], s["t"])].append(s)
    gt_at = {(s["vid"], s["t"], s["fr"]): (s["gt"], s["pose"]) for s in S}
    # 정답 사람별 누움 시작 시각 (표본 기준 — 탐지에 잡힌 초만)
    lie_start = {}
    for s in sorted(S, key=lambda s: s["fr"]):
        k = (s["vid"], s["gt"])
        if s["pose"] == "Lying" and k not in lie_start:
            lie_start[k] = s["fr"]
    lost, cause = [], collections.Counter()
    for s in S:
        if s["pose"] != "Lying" or not (s["p"][0] >= 0.5 > s["agg"][0]):
            continue
        hist = [gt_at.get((s["vid"], s["t"], f), (None, None)) for f in s["h_fr"]]
        other_id = any(g is not None and g != s["gt"] for g, _ in hist)
        not_lying = sum(1 for g, p in hist if g == s["gt"] and p != "Lying")
        unmatched = sum(1 for g, _ in hist if g is None)
        span = (s["h_fr"][-1] - s["h_fr"][0]) / 30 if len(s["h_fr"]) > 1 else 0
        since = (s["fr"] - lie_start.get((s["vid"], s["gt"]), s["fr"])) / 30
        c = "ID 바뀜" if other_id else ("전환 (같은 사람 · 눕기 전 표본)" if not_lying else ("정답 미매칭 표본 섞임" if unmatched else "누움인데 한 장 확률이 흔들림"))
        cause[c] += 1
        lost.append({"vid": s["vid"], "t": s["t"], "fr": s["fr"], "p": s["p"][0], "agg": s["agg"][0], "h_ly": s["h_ly"],
                     "누운 지 초": since, "창 폭 초": span, "원인": c})
    n_ly = sum(s["pose"] == "Lying" for s in S)
    print(f"\n① 한 장은 누움 · 중앙값은 아님: {len(lost)} / 누움 {n_ly} — 원인 {dict(cause)}")
    for r in lost[:12]:
        print("   ", r)
    # 반대 — 서기·걷기인데 중앙값만 누움 (일어난 직후 등)
    fp = [s for s in S if s["pose"] not in ("Lying",) and s["agg"][0] >= 0.5 > s["p"][0]]
    print(f"   반대 (누움 아님 · 중앙값만 누움): {len(fp)}")
    # ② 정답 사람별 누움 표본 수 (탐지에 잡힌 초)
    per = collections.Counter((s["vid"], s["gt"]) for s in S if s["pose"] == "Lying")
    v = np.array(list(per.values()))
    print(f"\n② 누운 사람 {len(v)} 명 · 잡힌 누움 초 중앙값 {np.median(v):.0f} · 분포 {sorted(v.tolist())}")
    trk_ly = collections.Counter((s["vid"], s["t"]) for s in S if s["pose"] == "Lying")
    print(f"   누움 표본이 붙은 추적 ID {len(trk_ly)} 개 (사람 {len(v)} 명) → 사람당 추적 {len(trk_ly) / max(len(v), 1):.1f} 개")
    # ③ 창 규칙 (관찰)
    y = np.array([s["pose"] == "Lying" for s in S])
    rules = {"한 장": [s["p"][0] for s in S], "중앙값 5 (현행)": [s["agg"][0] for s in S],
             "중앙값 3": [float(np.median(s["h_ly"][-3:])) for s in S], "평균 5": [float(np.mean(s["h_ly"])) for s in S],
             "최댓값 3": [float(np.max(s["h_ly"][-3:])) for s in S], "최댓값 5": [float(np.max(s["h_ly"])) for s in S]}
    print("\n③ 창 규칙별 누움 AUROC (관찰 · 고르지 않음)")
    res = {}
    for k, sc in rules.items():
        sc = np.array(sc); res[k] = round(auc(sc[y], sc[~y]), 3)
        print(f"   {k:12s} {res[k]}")
    out = {"표본": len(S), "누움": int(n_ly), "한장누움_중앙값아님": len(lost), "원인": dict(cause), "반대": len(fp),
           "누운 사람": int(len(v)), "잡힌 누움 초 중앙값": float(np.median(v)), "사람당 추적 ID": round(len(trk_ly) / max(len(v), 1), 2),
           "창 규칙 누움 AUROC (관찰)": res}
    json.dump(out, open(os.path.join(BASE, "metrics", f"state_diag_{tag}.json"), "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
