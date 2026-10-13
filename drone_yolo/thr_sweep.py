"""
thr_sweep.py — 운용 임계값 다시 보기 (10-13 · 사용자 「0.15 에 묶이지 말고 모든 가능성」)

  입력  runs_person/dumps/<모델>_<셋>.npz (dump_dets.py · 확신 0.01 까지) — 합치기 = 타일 셋은 경계 개선 (edge) · test_v2 는 무시 영역
  임계값 0.01 ~ 0.90 마다
    정밀도 · 재현율 · F1 · F2 (IoU 0.5 · 점수 순 탐욕 짝) · 사람 없는 장 오탐 (건/장 — test_v2 음성 41장 · AU-AIR 사람 없는 장)
    크기별 재현율 (< 24 · 24~50 · 50~80 · 80+ px)
  고르는 법 (모두 같은 표에)
    ① val_v12 F2 최대 (NFR-V05 방식 · 평가셋을 안 보고 고름 — 정직한 값)
    ② 평가셋마다 F2 최대 (참고 · 평가셋에 맞춘 상한)
    ③ 오탐 제약 — AU-AIR 사람 없는 장 ≤ 1 건/장 (NFR-V06) 을 지키는 가장 낮은 임계값
    ④ 재현율 목표 — 세 평가셋 평균 재현율 0.80 을 지키는 가장 높은 임계값
출력  metrics/thr_sweep.json · 그림 obsidian/_첨부/fig_임계값_스윕.png
실행: /home/se/miniconda3/envs/drone/bin/python thr_sweep.py v12_f91_swa v12_f92_swa soup_v7r2
"""
import json
import sys
from pathlib import Path

import numpy as np

from analyze_dets import load, nms
from eval_test_v2 import iou_mat
from ignore_regions import Ignore
from tile_merge import edge_merge

BASE = Path(__file__).resolve().parent
TH = [0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9]
SETS = ("test_v2", "test_obl", "test_kr", "test_auair", "val_v12")
BINS = [(0, 24), (24, 50), (50, 80), (80, 1e9)]
IGN = Ignore()


def per_image(model, tag):
    d = load(model, tag); gts = list(d["gt"]); names = [str(n) for n in d["names"]]; wh = d["wh"]
    img, box, sc, tl = d["img"], d["box"], d["score"], d["tile"]
    o = np.argsort(img, kind="stable"); img, box, sc, tl = img[o], box[o], sc[o], tl[o]
    dets = {i: (np.zeros((0, 4), np.float32), np.zeros(0, np.float32)) for i in range(len(gts))}
    for seg in np.split(np.arange(len(img)), np.flatnonzero(np.diff(img)) + 1):
        if not len(seg):
            continue
        i = int(img[seg[0]]); b, s, t = box[seg], sc[seg], tl[seg]
        k = nms(b, s, 0.6); b, s, t = b[k], s[k].copy(), t[k]
        b, s = edge_merge(b, s, t, *wh[i])
        dets[i] = (b, s)
    return dets, gts, names


def stats(dets, gts, names, tag, t):
    tp = fp = fn = 0; negfp = 0; nneg = 0; hit = {x: [0, 0] for x in BINS}
    for i, gt in enumerate(gts):
        b, s = dets[i]; k = s >= t; b, s = b[k], s[k]; o = np.argsort(-s); b, s = b[o], s[o]
        out = np.zeros(len(b), bool)
        if tag == "test_v2":
            b, out = IGN.prepare(names[i], b)
        M = iou_mat(gt, b) if len(gt) else np.zeros((0, len(b))); used = np.zeros(len(gt), bool)
        for j in range(len(b)):
            c = np.where((M[:, j] >= 0.5) & ~used)[0] if len(gt) else []
            if len(c):
                used[c[np.argmax(M[c, j])]] = True; tp += 1
            elif tag == "test_v2" and IGN.skip(names[i], b[j], out[j]):
                continue
            else:
                fp += 1
                if not len(gt):
                    negfp += 1
        if not len(gt):
            nneg += 1
        fn += int((~used).sum())
        L = np.maximum(gt[:, 2] - gt[:, 0], gt[:, 3] - gt[:, 1]) if len(gt) else np.zeros(0)
        for lo, hi in BINS:
            m = (L >= lo) & (L < hi); hit[(lo, hi)][0] += int(used[m].sum()); hit[(lo, hi)][1] += int(m.sum())
    P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1)
    f = lambda beta: (1 + beta ** 2) * P * R / max(beta ** 2 * P + R, 1e-9)
    return {"P": round(P, 4), "R": round(R, 4), "F1": round(f(1), 4), "F2": round(f(2), 4), "TP": tp, "FP": fp, "FN": fn,
            "음성장_오탐": round(negfp / max(nneg, 1), 3), "음성장": nneg, "오탐_장당": round(fp / max(len(gts), 1), 3),
            "크기별_R": {f"{lo}-{hi if hi < 1e9 else '∞'}": round(h / max(n, 1), 3) for (lo, hi), (h, n) in hit.items()}}


def main():
    models = sys.argv[1:] or ["v12_f91_swa", "v12_f92_swa", "soup_v7r2"]
    res = {}
    for m in models:
        res[m] = {}
        for tag in SETS:
            try:
                d, g, n = per_image(m, tag)
            except FileNotFoundError:
                continue
            res[m][tag] = {str(t): stats(d, g, n, tag, t) for t in TH}
            print(m, tag, "done", flush=True)
        R = res[m]; pick = {}
        if "val_v12" in R:
            pick["① val F2 최대"] = max(TH, key=lambda t: R["val_v12"][str(t)]["F2"])
        for tag in ("test_v2", "test_obl", "test_kr"):
            if tag in R:
                pick[f"② {tag} F2 최대"] = max(TH, key=lambda t: R[tag][str(t)]["F2"])
        if "test_auair" in R:
            ok = [t for t in TH if R["test_auair"][str(t)]["음성장_오탐"] <= 1.0]
            pick["③ AU-AIR 오탐 ≤ 1"] = min(ok) if ok else None
        main3 = [x for x in ("test_v2", "test_obl", "test_kr") if x in R]
        ok = [t for t in TH if np.mean([R[x][str(t)]["R"] for x in main3]) >= 0.80]
        pick["④ 평균 재현율 ≥ 0.80"] = max(ok) if ok else None
        res[m]["_고른_임계값"] = pick
        print(m, pick, flush=True)
    (BASE / "metrics" / "thr_sweep.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    # 그림: 세트별 정밀도 · 재현율 · F2 · 사람 없는 장 오탐 (임계값 축)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 4, figsize=(18, 4.2))
    cols = {"test_v2": "#2a9d8f", "test_obl": "#e76f51", "test_kr": "#264653", "val_v12": "#8d99ae"}
    for m, ls in zip(models, ("-", "--", ":")):
        for tag, c in cols.items():
            if tag not in res[m]:
                continue
            y = [res[m][tag][str(t)] for t in TH]
            ax[0].plot(TH, [v["P"] for v in y], ls, color=c, label=f"{m} {tag}" if ls == "-" else None)
            ax[1].plot(TH, [v["R"] for v in y], ls, color=c)
            ax[2].plot(TH, [v["F2"] for v in y], ls, color=c)
        if "test_auair" in res[m]:
            ax[3].plot(TH, [res[m]["test_auair"][str(t)]["음성장_오탐"] for t in TH], ls, color="k", label=m)
    for a, tt in zip(ax, ("Precision", "Recall", "F2", "AU-AIR FP per person-free frame")):
        a.set_title(tt); a.set_xlabel("confidence threshold"); a.grid(alpha=.3); a.axvline(0.15, color="gray", lw=.8)
    ax[3].axhline(1.0, color="r", lw=.8); ax[3].set_ylim(0, 5); ax[3].legend(fontsize=7); ax[0].legend(fontsize=6)
    out = BASE.parent / "obsidian" / "_첨부" / "fig_임계값_스윕.png"
    fig.tight_layout(); fig.savefig(out, dpi=110); print("그림", out)


if __name__ == "__main__":
    main()
