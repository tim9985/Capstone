"""
mannequin_bench.py — Archangel-Mannequin 으로 자세 · 누움 탐지 시험 (11-01 · 사용자 「누움 라벨 데이터로 자세 분류 테스트」 · 평가 전용)

  데이터  data/raw/archangel/mannequin/<고도>/*.mp4 + 같은 이름 .json (프레임마다 박스 · category = mannequin - lying down · kneeling · standing)
          실제 드론 · 1920×1080 · 45° · 원 궤도 (모든 방위) · 고도 15~50 m · v12 학습에 안 씀 (Archangel-Real 만 씀)
  표본    영상마다 STEP 프레임마다 (순차 디코딩 — H.264 건너뛰기는 프레임이 어긋남) · 고도 묶음별
  지표    ① 탐지 = 「뜸」 (박스 중심 ↔ 정답 · 확신 ≥ 0.20 · lying_bench 와 같음) 자세별
          ② 자세 (worker 박스 2특징 로지스틱 · posture.json score) — 정답 박스 · 탐지 박스 (맞은 것) 각각 P(누움) 의 누움 vs 서기 AUROC · 무릎 vs 서기 AUROC(P 누움+앉음)
실행  python mannequin_bench.py [모델 v12_f91_swa] [고도 묶음 15m,15-20m,20m,20-25m] [STEP 60]
출력  metrics/mannequin_bench_<모델>.json
"""
import glob, json, os, sys
from collections import defaultdict
import cv2, numpy as np
BASE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, BASE)
from bench_all import Model  # noqa: E402
from lying_rescue_check import hit  # noqa: E402
from okutama_motion_det import iou  # noqa: E402
sys.path.insert(0, os.path.join(BASE, "deploy", "vision_worker"))
from app.vision_core.state import PostureLR  # noqa: E402

RAW = os.path.join(BASE, "..", "data", "raw", "archangel", "mannequin")
POSE = {"mannequin - lying down": "누움", "mannequin - kneeling": "무릎", "mannequin - standing": "서기"}


def auc(pos, neg):
    if not pos or not neg:
        return None
    a = np.r_[pos, neg]; r = a.argsort().argsort() + 1.0
    for v in np.unique(a):                                  # 동점 평균 순위
        m = a == v; r[m] = r[m].mean()
    return round(float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))), 3)


def main(name="v12_f91_swa", groups=("15m", "15-20m", "20m", "20-25m"), step=60):
    det = Model(os.path.join(BASE, "runs_person", name, "weights", "best.pt"), 1280)
    pj = json.load(open(os.path.join(BASE, "runs_person", "_pack_v12", "posture.json"))); P = PostureLR(pj["score"])
    out = {}
    for g in groups:
        acc = defaultdict(list); hs = defaultdict(list); pg = defaultdict(list); pd = defaultdict(list); nfr = 0
        for v in sorted(glob.glob(f"{RAW}/Archangel_mannequin_{g}/*.mp4")):
            ann = json.load(open(v[:-4] + ".json")); cap = cv2.VideoCapture(v); k = -1
            while True:
                ok, img = cap.read(); k += 1
                if not ok or k >= len(ann):
                    break
                if k % step:
                    continue
                G = [(POSE[o["category"]], (o["x"], o["y"], o["x"] + o["width"], o["y"] + o["height"])) for o in (ann[k] or []) if o.get("category") in POSE]
                if not G:
                    continue
                nfr += 1; bx, cf = det(img)
                for pose, gb in G:
                    acc[pose].append(hit(bx, cf, gb, 0.20)); hs[pose].append(max(gb[2] - gb[0], gb[3] - gb[1]))
                    pr = P.proba([[gb[2] - gb[0], gb[3] - gb[1]]])[0]; pg[pose].append(pr)          # 클래스 순서 lying · sitting · standing
                    best = max(((iou(tuple(b), gb), b) for b, c in zip(bx, cf) if c >= 0.20), default=(0, None), key=lambda t: t[0])
                    if best[0] >= 0.3:
                        b = best[1]; pd[pose].append(P.proba([[b[2] - b[0], b[3] - b[1]]])[0])
            cap.release()
        r = {"프레임": nfr}
        for pose in ("누움", "무릎", "서기"):
            if acc[pose]:
                r[pose] = {"n": len(acc[pose]), "뜸@0.20": round(float(np.mean(acc[pose])), 3), "긴 변 중앙 px": float(np.median(hs[pose]))}
        for src, D in (("정답 박스", pg), ("탐지 박스", pd)):
            r[f"자세 AUROC ({src})"] = {"누움 vs 서기": auc([p[0] for p in D["누움"]], [p[0] for p in D["서기"]]),
                                        "무릎 vs 서기": auc([p[0] + p[1] for p in D["무릎"]], [p[0] + p[1] for p in D["서기"]]),
                                        "누움을 서기로 (argmax)": round(float(np.mean([np.argmax(p) == 2 for p in D["누움"]])), 3) if D["누움"] else None}
        out[g] = r; print(g, json.dumps(r, ensure_ascii=False), flush=True)
    json.dump(out, open(os.path.join(BASE, "metrics", f"mannequin_bench_{name}.json"), "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0] if a else "v12_f91_swa", tuple(a[1].split(",")) if len(a) > 1 else ("15m", "15-20m", "20m", "20-25m"), int(a[2]) if len(a) > 2 else 60)
