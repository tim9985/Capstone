"""
shape_thr_eval.py — 모양별 문턱 (10-30 · 누운 사람 · 추론 쪽 · 학습 없음)
  규칙 W(a, t): 확신 ≥ 0.20 전부 + **가로/세로 ≥ a 인 박스는 확신 ≥ t 까지** (비스듬 시점의 누운 모양)
  재는 것: Okutama 누움 (벤치 뺌) · 전체 사람 1초 표본 「사람 위에 뜸」 · 빈 장면 오탐 (UE S5 · route_132434 · AU-AIR 도로 사람 없는 장)
출력 metrics/shape_thr_eval_<모델>.json
"""
import glob, json, os, sys
from collections import defaultdict
import cv2, numpy as np
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE, "deploy", "vision_worker")); sys.path.insert(0, BASE)
from app.vision_core.detector import PersonDetector  # noqa: E402
from okutama_motion import FPS, LAB, OK, load_boxes  # noqa: E402
from state_trackfix import videos  # noqa: E402
from lying_rescue_check import hit  # noqa: E402
import lying_diag as LD  # noqa: E402

RULES = {"base": None, "W1.2@0.10": (1.2, 0.10), "W1.5@0.10": (1.5, 0.10), "W1.2@0.15": (1.2, 0.15), "all@0.10": (0.0, 0.10)}


def keep(rule, bx, cf):
    p = RULES[rule]
    out = []
    for b, c in zip(bx, cf):
        w, h = b[2] - b[0], b[3] - b[1]
        if c >= 0.20 or (p and c >= p[1] and w >= p[0] * h):
            out.append((b, c))
    return [b for b, _ in out], [c for _, c in out]


def main(name):
    det = PersonDetector(os.path.join(BASE, "runs_person", name, "weights", "best.pt"), conf=0.05)
    acc = {r: defaultdict(list) for r in RULES}
    allv, _, _ = videos()
    fd = {os.path.basename(d): d for d in glob.glob(f"{OK}/Drone*/*/Extracted-Frames-1280x720/*")}
    for v in allv:
        gt = load_boxes(f"{LAB}/{v}.txt"); ly = LD.lying_gt(v)
        for fr in sorted(gt):
            if fr % FPS or fr % (FPS * 2):                   # 2초마다 (전체 사람은 많음)
                continue
            img = cv2.imread(f"{fd[v]}/{fr}.jpg")
            if img is None:
                continue
            bx, cf, _ = det.detect(cv2.resize(img, (1920, 1080)))
            bx = [tuple(float(x) for x in b) for b in bx]; cf = [float(c) for c in cf]
            for r in RULES:
                B, C = keep(r, bx, cf)
                if v != "2.1.7":
                    for g in ly.get(fr, []):
                        acc[r]["누움 (벤치 뺌)"].append(hit(B, C, g[0], 0.0))
                for (x1, y1, x2, y2, p) in gt[fr].values():
                    acc[r]["전체"].append(hit(B, C, (x1 * 1.5, y1 * 1.5, x2 * 1.5, y2 * 1.5), 0.0))
    for tag, files in (("빈 UE S5", sorted(glob.glob(f"{BASE}/ue_flight/S5_a*/frames/*.jpg"))[::4]),
                       ("빈 캠퍼스 route", sorted(glob.glob(f"{BASE}/ue_flight/route_132434/frames/*.jpg"))),
                       ("도로 AU-AIR (사람 없는 장)", None)):
        if files is None:
            files = []
            for p in sorted(glob.glob(f"{BASE}/data/test_auair/images/*"))[:: 3]:
                lab = p.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
                if not os.path.exists(lab) or not open(lab).read().strip():
                    files.append(p)
        for f in files:
            img = cv2.imread(f)
            if img is None:
                continue
            if img.shape[:2] != (1080, 1920):
                img = cv2.resize(img, (1920, 1080))
            bx, cf, _ = det.detect(img)
            bx = [tuple(float(x) for x in b) for b in bx]; cf = [float(c) for c in cf]
            for r in RULES:
                acc[r][tag + " 오탐 건/장"].append(len(keep(r, bx, cf)[0]))
    out = {r: {k: round(float(np.mean(v)), 3) for k, v in M.items()} for r, M in acc.items()}
    for r, o in out.items():
        print(r, o, flush=True)
    json.dump(out, open(os.path.join(BASE, "metrics", f"shape_thr_eval_{name}.json"), "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
