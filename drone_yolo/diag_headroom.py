"""
diag_headroom.py — 사람 탐지 더 올릴 여지가 어디 있나 (11-08 · 사용자 「수단과 방법 가리지 말고」) → metrics/diag_headroom.json

  원자료  runs_person/dumps (dump_dets.py · 박스째) — v12_f91 · f92 · f93 swa (운용 타일 4장 · conf ≥ 0.01) · FlyPose 검출기 (다른 데이터 11종 · 통째 1280)
  ① 놓친 사람 나누기 (v12_f91 · 문턱 0.20) — 아예 후보 없음 (conf ≥ 0.01 에서도 IoU ≥ 0.3 박스 없음) · 확신 낮음 (0.01~0.20 에만) · 위치만 어긋남 (0.20 이상인데 IoU 0.3~0.5)
  ② 크기별 (긴 변 px · 원본 그림 기준) 재현율@0.20 · 상한 (conf ≥ 0.01)
  ③ 합치기 상한 — 같은 데이터 반복끼리 (f91 + f92 + f93) vs 다른 데이터 모델 (f91 + FlyPose) · WBF 간이 (analyze_dets.wbf 와 같은 식 · n 개)
     → 반복끼리로 안 오르고 FlyPose 와는 오르면 = 모델 · 학습 문제가 아니라 **데이터 다양성** 문제
  ⑤ 박스 습관과 상관없는 「중심 일치」 AP — 예측 박스 중심이 정답 박스 안 (또는 정답 중심이 예측 안) 이면 맞음 · 정답 박스가 헐렁하거나 (Karen 2 배) 몸 일부만 (다리) 인 라벨 영향 빼기
  ④ FlyPose 만 찾은 사람 (v12 놓침 @0.20 · FlyPose @0.5 찾음) 크기 · 장소
  ⚠ test_v2 무시 영역 · 가장자리 합치기 안 씀 (상대 비교용 · 판정 수치는 eval_test_v2)
실행  /home/se/miniconda3/envs/drone/bin/python diag_headroom.py
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from analyze_dets import ap_at, load, match, per_image
from eval_test_v2 import iou_mat

BASE = Path(__file__).resolve().parent
D = BASE / "runs_person" / "dumps"
SETS = ("test_v2", "test_obl", "test_kr")
BINS = [(0, 24), (24, 36), (36, 50), (50, 80), (80, 1e9)]
OP = 0.20


def wbf_n(ds, thr=0.55):
    """n 모델 상자 융합 — 겹치는 묶음은 모델마다 가장 높은 상자 하나 · 점수 가중 평균 상자 · 점수 = 합 / n"""
    out = {}; n = len(ds)
    for i in set().union(*[set(d) for d in ds]):
        b = np.concatenate([d[i][0] for d in ds]); s = np.concatenate([d[i][1] for d in ds])
        src = np.concatenate([np.full(len(d[i][1]), k) for k, d in enumerate(ds)])
        o = np.argsort(-s); used = np.zeros(len(b), bool); fb, fs = [], []
        for j in o:
            if used[j]:
                continue
            m = (iou_mat(b[j:j + 1], b)[0] >= thr) & ~used; m[j] = True; grp = np.flatnonzero(m); used[grp] = True   # 넓이 0 박스도 자기 자신은 묶음에
            best = {}
            for g in grp:
                if src[g] not in best or s[g] > s[best[src[g]]]:
                    best[src[g]] = g
            gi = np.array(list(best.values()), int); w = np.maximum(s[gi], 1e-9)
            fb.append((b[gi] * w[:, None]).sum(0) / w.sum()); fs.append(w.sum() / n)
        out[i] = (np.array(fb, np.float32).reshape(-1, 4), np.array(fs, np.float32))
    return out


def ap_center(dets, gts):
    """중심 일치 AP — 점수 순 탐욕 짝 · 정답 하나에 예측 하나"""
    from eval_test_v2 import ap101
    S, T, n = [], [], 0
    for i, gt in enumerate(gts):
        b, s = dets[i]; n += len(gt); o = np.argsort(-s); b, s = b[o], s[o]; used = np.zeros(len(gt), bool)
        for j in range(len(b)):
            cx, cy = (b[j, 0] + b[j, 2]) / 2, (b[j, 1] + b[j, 3]) / 2
            ok = -1
            if len(gt):
                gx, gy = (gt[:, 0] + gt[:, 2]) / 2, (gt[:, 1] + gt[:, 3]) / 2
                hit = (((gt[:, 0] <= cx) & (cx <= gt[:, 2]) & (gt[:, 1] <= cy) & (cy <= gt[:, 3])) |
                       ((b[j, 0] <= gx) & (gx <= b[j, 2]) & (b[j, 1] <= gy) & (gy <= b[j, 3]))) & ~used
                if hit.any():
                    c = np.flatnonzero(hit); ok = c[np.argmin((gx[c] - cx) ** 2 + (gy[c] - cy) ** 2)]; used[ok] = True
            S.append(s[j]); T.append(ok >= 0)
    return round(ap101(np.array(S), np.array(T, bool), n), 4) if n else None


def best_iou(gt, b, s, t):
    """정답마다 확신 ≥ t 박스와의 최대 IoU"""
    k = s >= t
    if not len(gt):
        return np.zeros(0)
    if not k.any():
        return np.zeros(len(gt))
    return iou_mat(gt, b[k]).max(1)


def main():
    res = {}
    for tag in SETS:
        R = {}; have = {m: (D / f"{m}_{tag}.npz").exists() for m in ("v12_f91_swa", "v12_f92_swa", "v12_f93_swa", "flypose")}
        dat = {m: load(m, tag) for m, ok in have.items() if ok}
        gts = list(dat["v12_f91_swa"]["gt"]); names = list(dat["v12_f91_swa"]["names"])
        det = {m: per_image(d, 0.6) for m, d in dat.items()}
        # ① ② 놓친 사람 나누기 · 크기별
        cnt = defaultdict(int); size = defaultdict(lambda: [0, 0, 0])
        for i, gt in enumerate(gts):
            if not len(gt):
                continue
            b, s = det["v12_f91_swa"][i]; _, used = match(gt, b[s >= OP], s[s >= OP], 0.5)
            lo = best_iou(gt, b, s, 0.01); hi = best_iou(gt, b, s, OP)
            L = np.maximum(gt[:, 2] - gt[:, 0], gt[:, 3] - gt[:, 1])
            for g in range(len(gt)):
                if used[g]:
                    cnt["찾음"] += 1
                elif lo[g] < 0.3:
                    cnt["아예 후보 없음"] += 1
                elif hi[g] >= 0.3:
                    cnt["위치만 어긋남 (IoU 0.3~0.5)"] += 1
                else:
                    cnt["확신 낮음 (0.01~0.20 에만)"] += 1
                for a, z in BINS:
                    if a <= L[g] < z:
                        size[(a, z)][0] += int(used[g]); size[(a, z)][1] += int(lo[g] >= 0.5); size[(a, z)][2] += 1
        n = sum(cnt.values())
        R["놓친 사람 나누기 (v12_f91 @0.20 · 사람 수 · 비율)"] = {k: f"{v} ({v / n:.1%})" for k, v in cnt.items()}
        R["크기별 재현율@0.20 · 상한@0.01 (사람 수)"] = {f"{a}-{z if z < 1e9 else '∞'}": f"{v[0] / max(v[2], 1):.3f} · {v[1] / max(v[2], 1):.3f} ({v[2]})" for (a, z), v in size.items()}
        # ③ 합치기 상한
        ap = {}
        for m in det:
            ap[m] = [ap_at(det[m], gts, t) for t in (0.3, 0.5)] + [ap_center(det[m], gts)]
        reps = [m for m in ("v12_f91_swa", "v12_f92_swa", "v12_f93_swa") if m in det]
        if len(reps) >= 2:
            e = wbf_n([det[m] for m in reps]); ap[f"반복 {len(reps)}개 합침"] = [ap_at(e, gts, t) for t in (0.3, 0.5)] + [ap_center(e, gts)]
            e = wbf_n([det["v12_f91_swa"], det["v12_f92_swa"]]); ap["반복 2개 합침 (f91+f92)"] = [ap_at(e, gts, t) for t in (0.3, 0.5)] + [ap_center(e, gts)]
        if "flypose" in det:
            e = wbf_n([det["v12_f91_swa"], det["flypose"]]); ap["f91 + FlyPose 합침"] = [ap_at(e, gts, t) for t in (0.3, 0.5)] + [ap_center(e, gts)]
        R["AP (IoU 0.3 · 0.5 · 중심 일치)"] = ap
        # ④ FlyPose 만 찾은 사람
        if "flypose" in det:
            only = defaultdict(lambda: [0, 0]); place = defaultdict(lambda: [0, 0])
            for i, gt in enumerate(gts):
                if not len(gt):
                    continue
                b, s = det["v12_f91_swa"][i]; _, uo = match(gt, b[s >= OP], s[s >= OP], 0.5)
                bf, sf = det["flypose"][i]; _, uf = match(gt, bf[sf >= 0.5], sf[sf >= 0.5], 0.5)
                L = np.maximum(gt[:, 2] - gt[:, 0], gt[:, 3] - gt[:, 1]); pl = names[i].rsplit("_", 1)[0]
                for g in range(len(gt)):
                    for a, z in BINS:
                        if a <= L[g] < z:
                            only[(a, z)][0] += int(uf[g] and not uo[g]); only[(a, z)][1] += int(uo[g] and not uf[g])
                    place[pl][0] += int(uf[g] and not uo[g]); place[pl][1] += int(uo[g] and not uf[g])
            R["FlyPose 만 찾음 · v12 만 찾음 (크기별 사람 수)"] = {f"{a}-{z if z < 1e9 else '∞'}": v for (a, z), v in only.items()}
            if tag != "test_kr":                                  # test_kr 장소 이름에 사람 이름 — 남기지 않음
                R["FlyPose 만 · v12 만 (장소별)"] = dict(place)
        res[tag] = R
        print(f"\n== {tag}"); [print(" ", k, json.dumps(v, ensure_ascii=False)) for k, v in R.items()]
    (BASE / "metrics" / "diag_headroom.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
