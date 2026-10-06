"""
posture_archangel_a2.py — 지금 자세 판정기 (박스 모양 로지스틱) 를 Archangel-Real 에 그대로 → 고도 · 각도 · 방위별 오차 (A2 · 2026-10-07)

  판정기  posture_fusion 과 같은 "box" — 특징 [log(h/w), log(긴 변 px)] (1080p 환산) · SARD + NOMAD 비스듬 크롭으로 맞춘 3자세 로지스틱
  평가    data/pose_eval/archangel_real/crops.csv (A1 · 실제 사람 · 영상 80 · 0.5 초 간격)
          누움 vs 서기 AUROC (P(누움)) · 누움을 "서기" 로 판정한 비율 (가장 높은 확률) · 무릎 · 기어감은 어디로 가나
  층      고도 · 반경 · 내려다보는 각 (atan(고도/반경)) · x_code (22.5/45/67.5) · 화면 세로 위치 · 움직이는 대상
  방위    드론이 사람 둘레를 돈다 → 영상 안 위치 (frame / frame_n) = 궤도 위상 (한 바퀴 가정 · 확인용 출력)
          누운 사람마다 세로/가로 비가 가장 큰 프레임 = "시선 방향" 으로 두고 그로부터의 위상 차 (0~90°) 별 오판
  블록 부트스트랩 = 영상 단위 1,000회

실행: /home/se/venvs/state/bin/python posture_archangel_a2.py
출력: metrics/posture_archangel_a2.json · obsidian/_첨부/fig_archangel_A2.svg
"""
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from posture_fusion import CLS, box_feat, rows_of
from posture_v2 import auc

BASE = Path(__file__).resolve().parent
SRC = BASE / "data" / "pose_eval" / "archangel_real" / "crops.csv"
FIG = BASE.parent / "obsidian" / "_첨부" / "fig_archangel_A2.svg"


def boot(ly, st, blocks_ly, blocks_st, rng, n=1000):
    ub = sorted(set(blocks_ly) | set(blocks_st)); il = defaultdict(list); is_ = defaultdict(list)
    for i, b in enumerate(blocks_ly): il[b].append(i)
    for i, b in enumerate(blocks_st): is_[b].append(i)
    v = []
    for _ in range(n):
        pick = rng.choice(ub, len(ub))
        a = [ly[i] for b in pick for i in il[b]]; c = [st[i] for b in pick for i in is_[b]]
        if a and c: v.append(auc(np.array(a), np.array(c)))
    return [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)] if v else None


def main():
    parts = [rows_of(CLS / p / "train") for p in ("sard", "nomad")]
    X = box_feat(np.concatenate([p[2] for p in parts])); y = np.concatenate([p[1] for p in parts])
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X, y)     # 0 lying · 1 sitting · 2 standing

    R = list(csv.DictReader(open(SRC)))
    wh = np.array([[float(r["w1080"]), float(r["h1080"])] for r in R])
    P = clf.predict_proba(box_feat(wh)); pred = P.argmax(1)
    for r, p, k, (w, h) in zip(R, P, pred, wh):
        r["p_ly"], r["pred"], r["ratio"] = float(p[0]), ("lying", "sitting", "standing")[k], h / w
        r["phi"] = math.degrees(math.atan2(int(r["alt"]), int(r["radius"])))
        r["phase"] = 360.0 * int(r["frame"]) / max(int(r["frame_n"]), 1)

    rng = np.random.default_rng(0)

    def summary(rows):
        ly = [r for r in rows if r["pose"] == "lying"]; st = [r for r in rows if r["pose"] == "standing"]
        out = {"n": {k: sum(r["pose"] == k for r in rows) for k in ("lying", "standing", "kneeling", "crawling")}}
        if ly and st:
            out["누움AUROC"] = round(auc(np.array([r["p_ly"] for r in ly]), np.array([r["p_ly"] for r in st])), 3)
            out["누움→서기 비율"] = round(float(np.mean([r["pred"] == "standing" for r in ly])), 3)
            out["누움 재현율 (가장 높은 확률)"] = round(float(np.mean([r["pred"] == "lying" for r in ly])), 3)
            out["서기→누움 비율"] = round(float(np.mean([r["pred"] == "lying" for r in st])), 3)
        for k in ("kneeling", "crawling"):
            K = [r for r in rows if r["pose"] == k]
            if K:
                out[f"{k} 판정 분포"] = {c: round(float(np.mean([r["pred"] == c for r in K])), 3) for c in ("lying", "sitting", "standing")}
        out["세로/가로 중앙 (누움 · 서기)"] = [round(float(np.median([r["ratio"] for r in ly])), 2) if ly else None,
                                       round(float(np.median([r["ratio"] for r in st])), 2) if st else None]
        return out

    res = {"전체": summary(R)}
    ly = [r for r in R if r["pose"] == "lying"]; st = [r for r in R if r["pose"] == "standing"]
    res["전체"]["누움AUROC 95%"] = boot([r["p_ly"] for r in ly], [r["p_ly"] for r in st], [r["place"] for r in ly], [r["place"] for r in st], rng)
    for key, f in (("고도", lambda r: int(r["alt"])), ("반경", lambda r: int(r["radius"])),
                   ("내려다보는각", lambda r: f"{int(r['phi'] // 10 * 10)}~{int(r['phi'] // 10 * 10 + 10)}°"),
                   ("x_code", lambda r: r["x_code"]), ("움직임", lambda r: "moving" if r["moving"] == "1" else "static"),
                   ("화면세로", lambda r: ["위 0~0.33", "가운데", "아래 0.67~1"][min(int(float(r["y_bottom"]) * 3), 2)])):
        g = defaultdict(list)
        for r in R: g[f(r)].append(r)
        res[key] = {str(k): summary(v) for k, v in sorted(g.items())}

    # 방위 — 누운 사람마다 세로/가로가 가장 큰 위상 = 시선 방향 (가정) · 거기서 떨어진 위상 차별 오판
    tr = defaultdict(list)
    for r in ly: tr[(r["place"], r["track"])].append(r)
    rel = []
    for k, rows in tr.items():
        if len(rows) < 8: continue
        top = max(rows, key=lambda r: r["ratio"])["phase"]
        for r in rows:
            d = abs((r["phase"] - top + 90) % 180 - 90)          # 0 = 시선 방향 · 90 = 직각
            rel.append((d, r))
    bins = [(0, 15), (15, 30), (30, 45), (45, 60), (60, 75), (75, 91)]
    res["방위 (누움 · 시선 방향에서 위상 차)"] = {
        f"{a}~{b}°": {"n": len(sel := [r for d, r in rel if a <= d < b]),
                      "누움→서기": round(float(np.mean([r["pred"] == "standing" for r in sel])), 3) if sel else None,
                      "세로/가로 중앙": round(float(np.median([r["ratio"] for r in sel])), 2) if sel else None,
                      "누움AUROC (vs 전체 서기)": round(auc(np.array([r["p_ly"] for r in sel]), np.array([r["p_ly"] for r in st])), 3) if sel else None}
        for a, b in bins}
    # 한 바퀴 가정 확인: 누운 사람의 세로/가로 비가 영상 안에서 몇 번 크게 오르내리나 (180° 주기면 2번)
    cyc = []
    for k, rows in tr.items():
        if len(rows) < 20: continue
        rr = np.array([r["ratio"] for r in sorted(rows, key=lambda r: r["phase"])])
        z = rr - np.median(rr)
        cyc.append(int(np.sum((z[:-1] < 0) & (z[1:] >= 0))))
    res["확인 — 누움 추적의 비 오르내림 횟수 (중앙값 위로 교차)"] = {"추적 수": len(cyc), "중앙값": float(np.median(cyc)) if cyc else None}

    (BASE / "metrics" / "posture_archangel_a2.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps({k: res[k] for k in ("전체", "방위 (누움 · 시선 방향에서 위상 차)", "확인 — 누움 추적의 비 오르내림 횟수 (중앙값 위로 교차)")}, ensure_ascii=False, indent=1))
    for key in ("고도", "내려다보는각", "x_code", "화면세로", "움직임"):
        print(f"\n[{key}]")
        for k, v in res[key].items():
            print(f"  {k:12s} n누움 {v['n']['lying']:5d} · 누움AUROC {v.get('누움AUROC')} · 누움→서기 {v.get('누움→서기 비율')} · 비(누움/서기) {v['세로/가로 중앙 (누움 · 서기)']}")

    import matplotlib
    matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"svg.fonttype": "none", "font.size": 9})
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    b = res["방위 (누움 · 시선 방향에서 위상 차)"]
    xs = [int(k.split("~")[0]) + 7 for k in b]
    ax[0].bar(xs, [v["누움→서기"] or 0 for v in b.values()], width=12, color="#c0392b")
    ax[0].set_xlabel("azimuth offset from 'along view' (deg, per lying track)"); ax[0].set_ylabel("lying judged as standing")
    ax[0].set_title("Archangel-Real: lying misread as standing, by azimuth", fontsize=9); ax[0].grid(alpha=.3)
    g = res["내려다보는각"]
    ax[1].bar(range(len(g)), [v.get("누움→서기 비율") or 0 for v in g.values()], color="#2471a3")
    ax[1].set_xticks(range(len(g))); ax[1].set_xticklabels([k.replace("°", "") for k in g], rotation=0)
    ax[1].set_xlabel("look-down angle atan(alt/radius) (deg)"); ax[1].set_title("lying misread as standing, by look-down angle", fontsize=9); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(FIG)


if __name__ == "__main__":
    main()
