"""
rescue_eval.py — 약한 탐지 살리기 (10-30 · 누운 사람 강건성 · 추론 쪽 · 학습 없음)

  규칙
    base   지금 worker — 탐지 확신 ≥ 0.20 만 (추적 문턱 0.15 · 낮은 짝 0.10)
    Ra     탐지 0.05 까지 추적기에 넣고 (낮은 짝 0.05) · 내보내기 = 확신 ≥ 0.20 또는 「이미 추적 중 (관찰 ≥ T 초) 인 추적의 이어진 탐지」 (T = 1 · 2)
           → 한 번 확실히 잡힌 사람이 누워 · 가려 확신이 떨어져도 계속 보임 · 처음부터 약한 것은 새 후보 안 만듦
  재는 것
    Okutama 비스듬 23편 (10 fps · 3장마다) — 1초 표본 정답마다 내보낸 박스가 사람 위에 뜸 (박스 중심 ↔ 정답) · 누움 (Lying) · 전체
    빈 장면 오탐 — UE S5_a16 · S5_a20 (배우 0) · route_132434 (노트북 캠퍼스 비행 · 사람 없음) — 내보낸 박스 / 장
출력 metrics/rescue_eval_<모델>.json
실행  python rescue_eval.py v12_f91_swa
"""
import glob, json, os, sys
from collections import defaultdict
import cv2, numpy as np
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE, "deploy", "vision_worker")); sys.path.insert(0, BASE)
from app.vision_core.detector import PersonDetector  # noqa: E402
import app.vision_core.state as ST  # noqa: E402
from okutama_motion import FPS, LAB, OK, load_boxes  # noqa: E402
from state_trackfix import videos  # noqa: E402
from lying_rescue_check import hit  # noqa: E402

POST = os.path.join(BASE, "runs_person", "_pack_v12", "posture.json")
RULES = {"base": None, "Ra_T1": 1, "Ra_T2": 2}


def emit(rule, confs, out):
    T = RULES[rule]
    keep = []
    for j, c in enumerate(confs):
        if c >= 0.20:
            keep.append(j)
        elif T is not None and c >= 0.05 and out[j][0] and out[j][1] and out[j][1].get("tracked") and (out[j][1].get("observed_s") or 0) >= T:
            keep.append(j)
    return keep


def run_seq(det, frames, fps, per_tick=None):
    """frames: [(t, img)] — 규칙마다 (t, 내보낸 박스 목록) · 추적기는 규칙마다 따로 (낮은 짝 문턱이 다름)"""
    res = {r: [] for r in RULES}
    trk = {}
    for r in RULES:
        if r == "base":
            ST.TRACK_ARGS.update(track_low_thresh=0.1)
        else:
            ST.TRACK_ARGS.update(track_low_thresh=0.05)
        trk[r] = ST.StateTracker(POST)
    ST.TRACK_ARGS.update(track_low_thresh=0.1)
    for t, img in frames:
        bx, cf, _ = det.detect(img)
        bx = [tuple(float(v) for v in b) for b in bx]; cf = [float(c) for c in cf]
        for r in RULES:
            if r == "base":
                idx = [j for j, c in enumerate(cf) if c >= 0.20]
                b2, c2 = [bx[j] for j in idx], [cf[j] for j in idx]
                out = trk[r].update(t, img, b2, c2) if b2 else []
                res[r].append((t, [b2[j] for j in range(len(b2))], [c2[j] for j in range(len(b2))]))
            else:
                out = trk[r].update(t, img, bx, cf) if bx else []
                k = emit(r, cf, out)
                res[r].append((t, [bx[j] for j in k], [cf[j] for j in k]))
    return res


def main(name):
    det = PersonDetector(os.path.join(BASE, "runs_person", name, "weights", "best.pt"), conf=0.05)
    allv, _, _ = videos()
    fd = {os.path.basename(d): d for d in glob.glob(f"{OK}/Drone*/*/Extracted-Frames-1280x720/*")}
    import lying_diag as LD
    acc = {r: defaultdict(list) for r in RULES}
    for v in allv:
        gt = load_boxes(f"{LAB}/{v}.txt"); ly = LD.lying_gt(v)
        n = len(glob.glob(f"{fd[v]}/*.jpg"))
        frames = []
        for fr in range(0, n, 3):
            img = cv2.imread(f"{fd[v]}/{fr}.jpg")
            if img is not None:
                frames.append((fr / FPS, cv2.resize(img, (1920, 1080)), fr))
        res = run_seq(det, [(t, im) for t, im, _ in frames], FPS)
        for r in RULES:
            for (t, B, C), (_, _, fr) in zip(res[r], frames):
                if fr % FPS:
                    continue
                for g in ly.get(fr, []):
                    acc[r]["누움"].append(hit(B, C, g[0], 0.0))
                    if v != "2.1.7":
                        acc[r]["누움 (벤치 뺌)"].append(hit(B, C, g[0], 0.0))
                for (x1, y1, x2, y2, p) in gt.get(fr, {}).values():
                    acc[r]["전체"].append(hit(B, C, (x1 * 1.5, y1 * 1.5, x2 * 1.5, y2 * 1.5), 0.0))
        print(v, {r: round(float(np.mean(acc[r]["누움"])) if acc[r]["누움"] else -1, 3) for r in RULES}, flush=True)
    # 빈 장면
    fp = {r: [] for r in RULES}
    for run in ("S5_a16", "S5_a20", "route_132434"):
        R = os.path.join(BASE, "ue_flight", run)
        fs = sorted(glob.glob(f"{R}/frames/*.jpg"))
        frames = []
        for k, f in enumerate(fs):
            img = cv2.imread(f)
            if img is not None:
                frames.append(((k / 10.0) if run.startswith("S5") else k * 0.5, img if img.shape[1] == 1920 else cv2.resize(img, (1920, 1080))))
        res = run_seq(det, frames, 10)
        for r in RULES:
            fp[r] += [len(B) for _, B, _ in res[r]]
    out = {r: {k: round(float(np.mean(v)), 3) for k, v in acc[r].items()} | {"빈 장면 오탐 건/장": round(float(np.mean(fp[r])), 3)} for r in RULES}
    print(json.dumps(out, ensure_ascii=False, indent=1))
    json.dump(out, open(os.path.join(BASE, "metrics", f"rescue_eval_{name}.json"), "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
