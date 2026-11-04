"""
fair_viz_data.py — 엔지니어링 페어 시각화 원자료 (11-04) → metrics/fair_viz.json

  탐지   셋마다 v12_f91_swa (운용 틀 = 창 4장) · 기성 최고 1개 (bench_all · bench_native 에서 가장 좋았던 방식)
         PR 곡선 (재현율 0~1 · 101점 · AP 와 같은 포락선) · AP50 · @0.20 맞음 · 헛탐지 · 놓침 · 사람 키별 재현율 · 확신 문턱별 재현율 · 정밀도
  도로   AU-AIR 사람 없는 장 헛탐지 @0.20 (둘 다)
  자세   배포 posture.json (박스 2특징 로지스틱 · score) — Okutama 비스듬 정답 크롭 · Archangel 실사람 → ROC (누움 vs 나머지) · 3자세 혼동행렬 (argmax)
실행  /home/se/miniconda3/envs/drone/bin/python fair_viz_data.py
"""
import csv, json, sys
from pathlib import Path

import cv2
import numpy as np

from bench_all import BASE, EXTS, MAXN, Model
from bench_native import Full
from eval_external_det import ap101, iou_mat, load_gt

OUT = BASE / "metrics" / "fair_viz.json"
# 셋 → (표시 이름, 기성 최고 이름, 방식, imgsz)  · 11-04 bench_all + bench_native 기준
SETS = {"test_sar": ("UAV-SAR", "aeronova", "full", 1280), "test_v2": ("WiSARD 숲", "visdrone_yolo11x", "tile", 640),
        "test_obl": ("Okutama 비스듬", "aeronova", "full", 1280), "test_kr": ("AI-Hub 한국", "aeronova", "tile", 640),
        "test_nomad": ("NOMAD", "ap_mix_26l", "full", 1280), "test_visdrone": ("VisDrone", "visdrone_yolo26x", "full", 1280),
        "test_heridal": ("HERIDAL", "ap_heridal_26l", "tile", 640), "test_sard": ("SARD", "ap_sard_26l", "full", 1280),
        "test_ladd_h": ("LADD", "ap_ladd_26l", "tile", 640)}
HBINS = [(0, 24), (24, 48), (48, 96), (96, 9999)]
THRS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


def run(model, data):
    imgs = sorted((data / "images").glob("*.jpg"))
    if len(imgs) > MAXN:
        imgs = [imgs[i] for i in np.linspace(0, len(imgs) - 1, MAXN).astype(int)]
    S, T, GH, GB = [], [], [], []           # GB = GT 마다 가장 높은 맞은 확신 (못 맞으면 0)
    n_neg = fp_neg = 0
    for p in imgs:
        img = cv2.imread(str(p)); ih, iw = img.shape[:2]; gt = load_gt(p, iw, ih)
        box, cf = model(img)
        if not len(gt):
            n_neg += 1; fp_neg += int((cf >= 0.2).sum())
        o = np.argsort(-cf); box, cf = box[o], cf[o]; M = iou_mat(gt, box); used = np.zeros(len(gt), bool); best = np.zeros(len(gt))
        for j in range(len(box)):
            k = -1
            if len(gt):
                cand = np.where((M[:, j] >= 0.5) & ~used)[0]
                if len(cand):
                    k = cand[np.argmax(M[cand, j])]; used[k] = True; best[k] = cf[j]
            S.append(float(cf[j])); T.append(k >= 0)
        if len(gt):
            GH += list((gt[:, 3] - gt[:, 1]) * 1080 / ih); GB += list(best)
    S, T, GH, GB = np.array(S), np.array(T, bool), np.array(GH), np.array(GB); n = len(GH)
    o = np.argsort(-S); tp = np.cumsum(T[o]); fp = np.cumsum(~T[o])
    rec, prec = tp / max(n, 1), tp / np.maximum(tp + fp, 1)
    env = np.maximum.accumulate(prec[::-1])[::-1]
    grid = np.linspace(0, 1, 101); pr = [float(env[np.searchsorted(rec, r, side="left")]) if r <= rec[-1] else 0.0 for r in grid]
    sel = S >= 0.2; TP = int(T[sel].sum())
    return {"imgs": len(imgs), "n_gt": n, "AP50": round(ap101(S, T, n), 4), "pr": [round(x, 4) for x in pr], "max_recall": round(float(rec[-1]), 4),
            "at020": {"TP": TP, "FP": int((~T[sel]).sum()), "FN": n - TP, "R": round(TP / n, 4), "P": round(TP / max(sel.sum(), 1), 4)},
            "thr": {str(t): {"R": round(float((GB >= t).sum() / n), 4), "P": round(float(T[S >= t].mean()) if (S >= t).any() else 1.0, 4)} for t in THRS},
            "by_height": {f"{a}-{b}": {"n": int(((GH >= a) & (GH < b)).sum()), "R": round(float((GB[(GH >= a) & (GH < b)] >= 0.2).mean()), 4) if ((GH >= a) & (GH < b)).any() else None}
                          for a, b in HBINS},
            "neg_imgs": n_neg, "FP_neg": round(fp_neg / n_neg, 3) if n_neg else None}


def posture():
    sys.path.insert(0, str(BASE / "deploy" / "vision_worker"))
    from app.vision_core.state import PostureLR
    P = PostureLR(json.load(open(BASE / "runs_person" / "_pack_v12" / "posture.json"))["score"])
    out = {}
    for s, k, name in (("okutama_obl", 1.5, "Okutama 비스듬 (실제 드론 · 처음 보는 장소)"), ("archangel_real", 1.0, "Archangel 실사람 (45° · 원 궤도)")):
        rows = list(csv.DictReader(open(BASE / "data" / "pose_eval" / s / "crops.csv")))
        wh = np.array([[float(r["w1080"]) * k, float(r["h1080"]) * k] for r in rows]); pr = P.proba(wh)
        y = np.array([{"lying": 0, "sitting": 1, "kneeling": 1, "standing": 2, "crawling": 0}[r["pose"]] for r in rows])
        s_ly = pr[:, 0]; pos = y == 0
        o = np.argsort(-s_ly); tpr = np.r_[0, np.cumsum(pos[o]) / pos.sum()]; fpr = np.r_[0, np.cumsum(~pos[o]) / (~pos).sum()]
        idx = np.unique(np.linspace(0, len(tpr) - 1, 200).astype(int))
        r = np.argsort(np.argsort(s_ly)) + 1.0
        auc = (r[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum())
        cm = np.zeros((3, 3), int)
        for t, p in zip(y, pr.argmax(1)):
            cm[t, p] += 1
        out[s] = {"name": name, "n": int(len(y)), "n_lying": int(pos.sum()), "AUROC": round(float(auc), 3),
                  "roc": [[round(float(fpr[i]), 4), round(float(tpr[i]), 4)] for i in idx], "classes": ["누움", "앉음 · 무릎", "서기"], "confusion": cm.tolist()}
        print(s, out[s]["AUROC"], cm.tolist(), flush=True)
    return out


def main():
    D = json.load(open(OUT)) if OUT.exists() else {}
    D["posture"] = posture(); json.dump(D, open(OUT, "w"), ensure_ascii=False)
    det = D.setdefault("det", {})
    ours = Model(BASE / "runs_person" / "v12_f91_swa" / "weights" / "best.pt", 1280)
    for s, (label, ext, mode, z) in list(SETS.items()) + [("test_auair", ("AU-AIR 도로 (사람 없는 장)", None, None, None))]:
        if s in det:
            continue
        r = {"label": label, "ours": run(ours, BASE / "data" / s)}
        if ext:
            m = (Full if mode == "full" else Model)(EXTS[ext], z)
            r["ext"] = run(m, BASE / "data" / s) | {"name": ext, "mode": mode, "imgsz": z}; del m
        det[s] = r; json.dump(D, open(OUT, "w"), ensure_ascii=False)
        print(s, r["ours"]["AP50"], r.get("ext", {}).get("AP50"), flush=True)
    # 도로 헛탐지 — 기성 최고들
    road = D.setdefault("road_ext", {})
    for ext in sorted({v[1] for v in SETS.values()}):
        if ext not in road:
            m = Model(EXTS[ext], 640); road[ext] = run(m, BASE / "data" / "test_auair")["FP_neg"]; del m
            json.dump(D, open(OUT, "w"), ensure_ascii=False); print("road", ext, road[ext], flush=True)


if __name__ == "__main__":
    main()
