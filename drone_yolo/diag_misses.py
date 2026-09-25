"""
diag_misses.py — 어디서 · 왜 못 찾나 (2026-09-25)

  모델 예측을 느슨하게(conf 0.01 · NMS 0.9) 한 번만 뽑아 두고, 오프라인에서
  ① NMS 기준 0.5~0.8 을 바꿔 가며 AP50 (추론만 바꿔 얻는 공짜 이득이 있나)
  ② 정답 한 명씩 놓친 원인을 가른다 (운용 NMS 기준 · conf 0.15)
       찾음            IoU ≥ 0.5 로 짝지어짐 (conf ≥ 0.15)
       확신도 낮음      conf 0.01~0.15 에선 짝지어짐
       위치 어긋남      가장 가까운 예측 IoU 0.3~0.5
       묶임            다른 사람과 한 박스로 — 한 예측 안에 정답 ≥2명이 절반 이상 들어감
       아예 못 봄       IoU 0.3 이상 예측 없음
  ③ 크기 · 붙어 있음 · (Okutama) 가림 · 자세별 재현율

실행: python diag_misses.py --weights runs_person/v6_obl/weights/best.pt
"""
import argparse, glob, json
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np

BASE = Path(__file__).resolve().parent
TW, TH = 1280, 720
SETS = {"test_obl": ("single", 0.7), "test_v2": ("tile", 0.6)}   # 운용·판정과 같은 NMS 기준


def gt_boxes(p, w, h):
    t = Path(str(p).replace("/images/", "/labels/")).with_suffix(".txt")
    if not t.exists():
        return np.zeros((0, 4), np.float32)
    a = np.array([[float(v) for v in r.split()[1:5]] for r in t.read_text().splitlines() if r.strip()], np.float32).reshape(-1, 4)
    return np.c_[(a[:, 0] - a[:, 2] / 2) * w, (a[:, 1] - a[:, 3] / 2) * h, (a[:, 0] + a[:, 2] / 2) * w, (a[:, 1] + a[:, 3] / 2) * h]


def iou(a, b):
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)), np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    it = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    ar = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])
    return it / (ar(a)[:, None] + ar(b)[None, :] - it + 1e-9)


def ioa_gt(g, p):
    """정답 박스 중 예측 박스 안에 들어간 비율 (gt × pred)"""
    x1 = np.maximum(g[:, None, 0], p[None, :, 0]); y1 = np.maximum(g[:, None, 1], p[None, :, 1])
    x2 = np.minimum(g[:, None, 2], p[None, :, 2]); y2 = np.minimum(g[:, None, 3], p[None, :, 3])
    it = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    return it / (((g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1]))[:, None] + 1e-9)


def nms(b, c, thr):
    import cv2
    if not len(b):
        return np.array([], int)
    k = cv2.dnn.NMSBoxes([[float(x1), float(y1), float(x2 - x1), float(y2 - y1)] for x1, y1, x2, y2 in b], c.tolist(), 0.01, thr)
    return np.array(k).ravel().astype(int) if len(k) else np.array([], int)


def greedy(G, P, C):
    """eval_test_v2 와 같은 짝짓기 — (예측별 정답여부, 정답별 짝지어진 예측의 conf)"""
    o = np.argsort(-C); P, C = P[o], C[o]
    M = iou(G, P); used = np.zeros(len(G), bool); tp = []; gconf = np.full(len(G), -1.0)
    for j in range(len(P)):
        k = -1
        if len(G):
            cand = np.where((M[:, j] >= 0.5) & ~used)[0]
            if len(cand):
                k = cand[np.argmax(M[cand, j])]
        tp.append(k >= 0)
        if k >= 0:
            used[k] = True; gconf[k] = C[j]
    return np.array(tp, bool), C, gconf


def ap101(s, t, n):
    if not len(s) or n == 0:
        return 0.0
    o = np.argsort(-s); t = t[o]; tp = np.cumsum(t); fp = np.cumsum(~t)
    rec = tp / n; prec = tp / np.maximum(tp + fp, 1)
    return float(np.mean([prec[rec >= r].max() if (rec >= r).any() else 0 for r in np.linspace(0, 1, 101)]))


def okutama_meta():
    """test_obl 정답 순서와 같은 (가림, 자세) — make_testset_obl.py 와 같은 걸러내기"""
    meta = {}
    files = sorted(glob.glob(str(BASE / "data/raw/okutama/**/Labels/SingleActionLabels/3840x2160/*.txt"), recursive=True),
                   key=lambda f: "TrainSetVideos" in f)
    seen = set()
    for lp in files:
        seq = Path(lp).stem
        if seq in seen:
            continue
        seen.add(seq); per = defaultdict(list)
        for ln in open(lp, encoding="utf-8", errors="replace"):
            t = ln.split()
            if len(t) < 7:
                continue
            try:
                x1, y1, x2, y2, fr, lost, occ = (int(t[i]) for i in (1, 2, 3, 4, 5, 6, 7))
            except ValueError:
                continue
            if lost:
                continue
            s = TW / 3840
            if (x2 - x1) * s > 3 and (y2 - y1) * s > 3:
                act = t[-1].strip('"') if len(t) >= 11 else "?"
                per[fr].append((occ, act))
        for fr, v in per.items():
            meta[f"okutama_{seq}_{fr:05d}"] = v
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--sets", default="test_obl,test_v2")
    args = ap.parse_args()
    import cv2
    from ultralytics import YOLO
    model = YOLO(args.weights); name = Path(args.weights).parent.parent.name
    report = {}
    omet = okutama_meta()
    for tag in args.sets.split(","):
        mode, op_nms = SETS[tag]
        imgs = sorted((BASE / "data" / tag / "images").glob("*.jpg"))
        recs = []
        for p in imgs:
            img = cv2.imread(str(p)); h, w = img.shape[:2]
            if mode == "tile":
                offs = [(x, y) for y in (0, h - TH) for x in (0, w - TW)]
                tiles = [img[y:y + TH, x:x + TW] for x, y in offs]
            else:
                offs, tiles = [(0, 0)], [img]
            res = model.predict(tiles, imgsz=1280, conf=0.01, iou=0.9, max_det=1000, batch=len(tiles),
                                quantize="fp16", verbose=False)
            B, C = [], []
            for (ox, oy), r in zip(offs, res):
                if len(r.boxes):
                    B.append(r.boxes.xyxy.cpu().numpy() + np.array([ox, oy, ox, oy], np.float32)); C.append(r.boxes.conf.cpu().numpy())
            B = np.concatenate(B) if B else np.zeros((0, 4), np.float32); C = np.concatenate(C) if C else np.zeros(0, np.float32)
            recs.append((p.stem, gt_boxes(p, w, h), B, C))
        # ① NMS 기준 스윕
        sweep = {}
        for thr in (0.5, 0.6, 0.7, 0.8):
            S, T, n = [], [], 0
            for _, G, B, C in recs:
                k = nms(B, C, thr); tp, cs, _ = greedy(G, B[k], C[k]); S += list(cs); T += list(tp); n += len(G)
            sweep[thr] = round(ap101(np.array(S), np.array(T, bool), n), 4)
        # ② 놓친 원인 (운용 NMS)
        cause = Counter(); strata = defaultdict(lambda: [0, 0])
        fp_hi = Counter()
        for stem, G, B, C in recs:
            k = nms(B, C, op_nms); B2, C2 = B[k], C[k]
            _, _, gconf = greedy(G, B2, C2)
            hi = C2 >= 0.15; Bh = B2[hi]
            M_all = iou(G, B2)
            IA = ioa_gt(G, Bh) if len(Bh) and len(G) else np.zeros((len(G), len(Bh)))
            merged_pred = set(np.where((IA >= 0.5).sum(0) >= 2)[0]) if len(Bh) else set()
            GG = iou(G, G); np.fill_diagonal(GG, 0) if len(G) else None
            crowd = (GG.max(1) >= 0.1) if len(G) > 1 else np.zeros(len(G), bool)
            L = np.maximum(G[:, 2] - G[:, 0], G[:, 3] - G[:, 1]) if len(G) else np.zeros(0)
            meta = omet.get(stem) if tag == "test_obl" else None
            for i in range(len(G)):
                if gconf[i] >= 0.15:
                    c = "찾음"
                elif gconf[i] >= 0:
                    c = "확신도 낮음"
                elif len(Bh) and any(IA[i, j] >= 0.5 for j in merged_pred):
                    c = "묶임"
                elif len(B2) and M_all[i].max() >= 0.3:
                    c = "위치 어긋남"
                else:
                    c = "아예 못 봄"
                cause[c] += 1
                hit = c == "찾음"
                size = "<24" if L[i] < 24 else "24-36" if L[i] < 36 else "36-50" if L[i] < 50 else "50+"
                strata[("크기", size)][0] += hit; strata[("크기", size)][1] += 1
                strata[("붙어 있음", "예" if crowd[i] else "아니오")][0] += hit; strata[("붙어 있음", "예" if crowd[i] else "아니오")][1] += 1
                if meta and len(meta) == len(G):
                    occ, act = meta[i]
                    strata[("가림", "예" if occ else "아니오")][0] += hit; strata[("가림", "예" if occ else "아니오")][1] += 1
                    strata[("자세", act)][0] += hit; strata[("자세", act)][1] += 1
                strata[("그룹", stem.rsplit("_", 1)[0])][0] += hit; strata[("그룹", stem.rsplit("_", 1)[0])][1] += 1
            # 고신뢰 오탐 중 묶음 박스 비율
            if len(Bh):
                Mh = iou(G, Bh) if len(G) else np.zeros((0, len(Bh)))
                for j in range(len(Bh)):
                    if len(G) and Mh[:, j].max() >= 0.5:
                        continue
                    fp_hi["묶음" if j in merged_pred else "기타"] += 1
        n = sum(cause.values())
        report[tag] = {"NMS_스윕_AP50": sweep, "운용_NMS": op_nms,
                       "놓친_원인": {k: f"{v} ({v / n * 100:.1f} %)" for k, v in cause.most_common()},
                       "고신뢰_오탐": dict(fp_hi),
                       "층별_재현율@0.15": {f"{a}:{b}": f"{h / t:.3f} (n={t})" for (a, b), (h, t) in sorted(strata.items()) if t >= 30}}
        print(f"\n===== {name} · {tag} =====")
        print(json.dumps(report[tag], ensure_ascii=False, indent=1))
    out = BASE / "metrics" / f"diag_misses_{name}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print("→", out)


if __name__ == "__main__":
    main()
