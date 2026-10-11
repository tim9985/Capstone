"""
analyze_dets.py — dump_dets 결과로 오프라인 진단 (10-11 · v7 넘기 검토) → metrics/analyze_dets.json

  ① 합치기 (타일 경계) — NMS 0.6 (공식 · AP 재현 검산) · 0.5 · 0.7
  ② IoU 0.3 · 0.5 · 0.7 AP — 찾기 문제 vs 박스 위치 문제
  ③ 확신 ≥ 0.5 오탐 종류 — 중복 (이미 짝지어진 정답과 IoU ≥ 0.5) · 위치 어긋남 (어떤 정답과 0.1~0.5) · 배경 (< 0.1)
  ④ 장소 (이름 앞부분) 별 AP50 · 크기별 재현율@운용 (우리 0.15 · FlyPose 0.5)
  ⑤ 앙상블 상한 — 두 모델 상자 융합 (WBF 간이 · 한 모델만 찾은 상자는 점수 ½)
실행: /home/se/miniconda3/envs/drone/bin/python analyze_dets.py
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from eval_test_v2 import ap101, iou_mat

BASE = Path(__file__).resolve().parent
D = BASE / "runs_person" / "dumps"
SETS = ("test_v2", "test_obl", "test_kr")
OP = {"soup_v7r2": 0.15, "flypose": 0.5}


def load(model, tag):
    z = np.load(D / f"{model}_{tag}.npz", allow_pickle=True)
    return {k: z[k] for k in z.files}


def nms(b, s, thr):
    o = np.argsort(-s); keep = []
    while len(o):
        i = o[0]; keep.append(i)
        if len(o) == 1:
            break
        m = iou_mat(b[i:i + 1], b[o[1:]])[0]; o = o[1:][m <= thr]
    return np.array(keep, int)


def per_image(d, merge=0.6, score_scale=1.0):
    out = defaultdict(lambda: (np.zeros((0, 4), np.float32), np.zeros(0, np.float32)))
    img, box, sc = d["img"], d["box"], d["score"] * score_scale
    order = np.argsort(img, kind="stable"); img, box, sc = img[order], box[order], sc[order]
    cuts = np.flatnonzero(np.diff(img)) + 1
    for seg in np.split(np.arange(len(img)), cuts):
        if not len(seg):
            continue
        b, s = box[seg], sc[seg]
        k = nms(b, s, merge) if merge is not None else np.arange(len(b))
        out[int(img[seg[0]])] = (b[k], s[k])
    return out


def match(gt, b, s, thr):
    """점수 순 탐욕 짝 → (점수, 맞음, 최대 IoU, 짝 지은 정답 번호)"""
    o = np.argsort(-s); b, s = b[o], s[o]
    M = iou_mat(gt, b) if len(gt) else np.zeros((0, len(b))); used = np.zeros(len(gt), bool); rows = []
    for j in range(len(b)):
        k = -1
        if len(gt):
            c = np.where((M[:, j] >= thr) & ~used)[0]
            if len(c):
                k = c[np.argmax(M[c, j])]; used[k] = True
        rows.append((s[j], k >= 0, float(M[:, j].max()) if len(gt) else 0.0, k))
    return rows, used


def ap_at(dets, gts, thr, sel=None):
    S, T, n = [], [], 0
    for i, gt in enumerate(gts):
        if sel is not None and not sel(i):
            continue
        b, s = dets[i]; n += len(gt); rows, _ = match(gt, b, s, thr)
        S += [r[0] for r in rows]; T += [r[1] for r in rows]
    return round(ap101(np.array(S), np.array(T, bool), n), 4) if n else None


def wbf(d1, d2, thr=0.55):
    """두 모델 상자 융합 — 겹치는 묶음은 점수 가중 평균 상자 · 점수 = 합 / 2"""
    out = {}
    for i in set(d1) | set(d2):
        b = np.concatenate([d1[i][0], d2[i][0]]); s = np.concatenate([d1[i][1], d2[i][1]]); src = np.r_[np.zeros(len(d1[i][1])), np.ones(len(d2[i][1]))]
        o = np.argsort(-s); used = np.zeros(len(b), bool); fb, fs = [], []
        for j in o:
            if used[j]:
                continue
            m = (iou_mat(b[j:j + 1], b)[0] >= thr) & ~used
            grp = np.flatnonzero(m); used[grp] = True
            best = {}                                         # 모델마다 가장 높은 상자 하나
            for g in grp:
                if src[g] not in best or s[g] > s[best[src[g]]]:
                    best[src[g]] = g
            gi = np.array(list(best.values())); w = s[gi]
            fb.append((b[gi] * w[:, None]).sum(0) / w.sum()); fs.append(w.sum() / 2)
        out[i] = (np.array(fb, np.float32).reshape(-1, 4), np.array(fs, np.float32))
    return out


def main():
    res = {}
    for tag in SETS:
        R = {}; data = {m: load(m, tag) for m in ("soup_v7r2", "flypose")}
        gts = list(data["soup_v7r2"]["gt"]); names = list(data["soup_v7r2"]["names"])
        groups = sorted({n.rsplit("_", 1)[0] for n in names})
        for m, d in data.items():
            r = {}
            for mg in (0.5, 0.6, 0.7):
                r[f"AP50 (합치기 NMS {mg})"] = ap_at(per_image(d, mg), gts, 0.5)
            dets = per_image(d, 0.6)
            r["AP@IoU 0.3 · 0.5 · 0.7"] = [ap_at(dets, gts, t) for t in (0.3, 0.5, 0.7)]
            fp = defaultdict(int); tp = 0
            for i, gt in enumerate(gts):
                b, s = dets[i]; keep = s >= 0.5; rows, _ = match(gt, b[keep], s[keep], 0.5)
                for sc, ok, miou, k in rows:
                    if ok:
                        tp += 1
                    elif miou >= 0.5:
                        fp["중복"] += 1
                    elif miou >= 0.1:
                        fp["위치 어긋남"] += 1
                    else:
                        fp["배경"] += 1
            r["확신 ≥ 0.5 맞음 · 오탐 종류"] = {"맞음": tp, **fp}
            if len(groups) > 1 and tag != "test_kr":              # test_kr 장소 이름에는 사람 이름이 들어 있다 — 결과 파일에 남기지 않는다
                r["장소별 AP50"] = {g: ap_at(dets, gts, 0.5, sel=lambda i, g=g: names[i].rsplit("_", 1)[0] == g) for g in groups}
            # 크기별 재현율@운용
            bins = [(0, 24), (24, 36), (36, 50), (50, 80), (80, 1e9)]; hit = defaultdict(lambda: [0, 0])
            for i, gt in enumerate(gts):
                b, s = dets[i]; keep = s >= OP[m]; rows, used = match(gt, b[keep], s[keep], 0.5)
                L = np.maximum(gt[:, 2] - gt[:, 0], gt[:, 3] - gt[:, 1]) if len(gt) else np.zeros(0)
                for (lo, hi) in bins:
                    sel = (L >= lo) & (L < hi); hit[(lo, hi)][0] += int(used[sel].sum()); hit[(lo, hi)][1] += int(sel.sum())
            r[f"크기별 재현율@{OP[m]}"] = {f"{lo}-{hi if hi < 1e9 else '∞'}": f"{v[0] / max(v[1], 1):.3f} ({v[1]})" for (lo, hi), v in hit.items()}
            R[m] = r
        ens = wbf(per_image(data["soup_v7r2"], 0.6), per_image(data["flypose"], 0.6))
        R["앙상블 WBF (soup_v7r2 + flypose)"] = {"AP@IoU 0.3 · 0.5 · 0.7": [ap_at(ens, gts, t) for t in (0.3, 0.5, 0.7)]}
        if len(groups) > 1 and tag != "test_kr":
            R["앙상블 WBF (soup_v7r2 + flypose)"]["장소별 AP50"] = {g: ap_at(ens, gts, 0.5, sel=lambda i, g=g: names[i].rsplit("_", 1)[0] == g) for g in groups}
        res[tag] = R; print(tag, json.dumps(R, ensure_ascii=False)[:2500], flush=True)
    (BASE / "metrics" / "analyze_dets.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
