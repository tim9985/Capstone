"""
lying_diag.py — 누운 사람 미탐 진단 (10-30 · 사용자 「누운 사람 강건성 어떻게든」) — Okutama 비스듬 23편 · 1초 표본 · worker 타일 탐지 (1080p)
  정답 Lying 마다 (720p ×1.5): 찾음 (IoU ≥ 0.5 · 확신 ≥ 0.20) · 확신 낮음 (IoU ≥ 0.5 박스는 있는데 < 0.20) · 위치 어긋남 (확신 ≥ 0.20 박스 IoU 0.3~0.5) · 못 봄
  층: 키 px · 가로/세로 · 정답의 occluded · generated (보간) 표시 · 맞은 박스의 가로 · 세로 비 (모델 / 정답)
  출력 metrics/lying_diag.json · runs_state/lying_diag_<모델>.pkl (그림 크롭 · git 밖)
실행  python lying_diag.py soup_v7r2 v12_f91_swa v12ly_swa
"""
import glob, json, os, pickle, sys
from collections import Counter, defaultdict
import cv2, numpy as np
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE, "deploy", "vision_worker")); sys.path.insert(0, BASE)
from app.vision_core.detector import PersonDetector  # noqa: E402
from okutama_motion import FPS, LAB, OK, S as S4K  # noqa: E402   (S4K = 원본 4K 주석 → 1280)
from okutama_motion_det import iou  # noqa: E402
from state_trackfix import videos  # noqa: E402

UP = 1.5


def lying_gt(vid):
    """프레임 → [(박스 1080p, occluded, generated)] · Lying · lost 아님 · 1초마다"""
    out = defaultdict(list)
    for ln in open(f"{LAB}/{vid}.txt"):
        p = ln.split()
        if len(p) < 10 or int(p[6]) or '"Lying"' not in ln:
            continue
        tid, x1, y1, x2, y2, fr = map(int, p[:6])
        if fr % FPS == 0:
            k = S4K * UP
            out[fr].append(((x1 * k, y1 * k, x2 * k, y2 * k), int(p[7]), int(p[8])))
    return out


def main(names):
    allv, _, _ = videos()
    fd = {os.path.basename(d): d for d in glob.glob(f"{OK}/Drone*/*/Extracted-Frames-1280x720/*")}
    dets = {n: PersonDetector(os.path.join(BASE, "runs_person", n, "weights", "best.pt"), conf=0.01) for n in names}
    rows = {n: [] for n in names}; crops = defaultdict(list)
    for v in allv:
        for fr, G in sorted(lying_gt(v).items()):
            img = cv2.imread(f"{fd[v]}/{fr}.jpg")
            if img is None:
                continue
            img = cv2.resize(img, (1920, 1080))
            for n, d in dets.items():
                bx, cf, _ = d.detect(img)
                for gb, occ, gen in G:
                    ious = [iou(tuple(b), gb) for b in bx]
                    best_any = max([(u, float(c), tuple(b)) for u, c, b in zip(ious, cf, bx)] or [(0.0, 0.0, None)])
                    hi = [(u, float(c), tuple(b)) for u, c, b in zip(ious, cf, bx) if c >= 0.20]
                    best_hi = max(hi or [(0.0, 0.0, None)])
                    if best_hi[0] >= 0.5:
                        cat = "찾음"
                    elif any(u >= 0.5 for u in ious):
                        cat = "확신 낮음"
                    elif best_hi[0] >= 0.3:
                        cat = "위치 어긋남"
                    else:
                        cat = "못 봄"
                    w, h = gb[2] - gb[0], gb[3] - gb[1]
                    ctr = any(c >= 0.20 and ((gb[0] <= (b[0] + b[2]) / 2 <= gb[2] and gb[1] <= (b[1] + b[3]) / 2 <= gb[3]) or
                                             (b[0] <= (gb[0] + gb[2]) / 2 <= b[2] and b[1] <= (gb[1] + gb[3]) / 2 <= b[3])) for b, c in zip(bx, cf))
                    r = {"vid": v, "fr": fr, "cat": cat, "h": h, "w": w, "ar": w / max(h, 1), "occ": occ, "gen": gen, "hit": bool(ctr), "hit30": best_hi[0] >= 0.3,
                         "best_iou_hi": best_hi[0], "best_conf_iou50": max([float(c) for u, c in zip(ious, cf) if u >= 0.5] or [0.0])}
                    if best_hi[2] is not None and best_hi[0] >= 0.3:
                        pb = best_hi[2]; r["pw_gw"] = (pb[2] - pb[0]) / max(w, 1); r["ph_gh"] = (pb[3] - pb[1]) / max(h, 1)
                    rows[n].append(r)
                    if n == names[1] and len(crops[cat]) < 60:
                        cx, cy, s = (gb[0] + gb[2]) / 2, (gb[1] + gb[3]) / 2, max(w, h) * 2.0
                        c = img[int(max(0, cy - s / 2)):int(cy + s / 2), int(max(0, cx - s / 2)):int(cx + s / 2)].copy()
                        crops[cat].append((v, fr, cv2.resize(c, (160, 160)) if c.size else None, best_hi[2], gb))
    out = {}
    for n, R in rows.items():
        cats = Counter(r["cat"] for r in R); N = len(R)
        o = {"정답": N, "분류": {k: f"{c} ({c / N:.1%})" for k, c in cats.most_common()},
             "사람 위에 뜸 (박스 중심 ↔ 정답 · 확신 ≥ 0.20)": round(float(np.mean([r["hit"] for r in R])), 3),
             "IoU ≥ 0.3 · 확신 ≥ 0.20": round(float(np.mean([r["hit30"] for r in R])), 3)}
        g2 = defaultdict(list)
        for r in R:
            g2["<30" if r["h"] < 30 else "30-50" if r["h"] < 50 else "≥50"].append(r["hit"])
        o["키 px별 사람 위에 뜸"] = {k: f"{np.mean(x):.3f} (n={len(x)})" for k, x in sorted(g2.items())}
        for name, f in (("키 px", lambda r: "<30" if r["h"] < 30 else "30-50" if r["h"] < 50 else "≥50"),
                        ("가로/세로", lambda r: "<1 (세로로 김)" if r["ar"] < 1 else "1-2" if r["ar"] < 2 else "≥2 (가로로 김)"),
                        ("가림 표시", lambda r: "가림" if r["occ"] else "안 가림"), ("보간 박스", lambda r: "보간" if r["gen"] else "손 라벨")):
            g = defaultdict(list)
            for r in R:
                g[f(r)].append(r["cat"] == "찾음")
            o[f"{name}별 찾음"] = {k: f"{np.mean(x):.3f} (n={len(x)})" for k, x in sorted(g.items())}
        m = [r for r in R if "pw_gw" in r]
        if m:
            o["맞춘 박스 크기 (모델/정답) 중앙"] = {"가로": round(float(np.median([r["pw_gw"] for r in m])), 3), "세로": round(float(np.median([r["ph_gh"] for r in m])), 3), "n": len(m)}
        out[n] = o
        print(n, json.dumps(o, ensure_ascii=False), flush=True)
    json.dump(out, open(os.path.join(BASE, "metrics", "lying_diag.json"), "w"), ensure_ascii=False, indent=1)
    pickle.dump({"rows": rows, "crops": dict(crops)}, open(os.path.join(BASE, "runs_state", "lying_diag.pkl"), "wb"))


if __name__ == "__main__":
    main(sys.argv[1:])
