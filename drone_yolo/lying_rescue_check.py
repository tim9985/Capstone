"""
lying_rescue_check.py — 누운 사람: 확신도를 낮추면 얼마나 살아나나 + 운용 크기 (NOMAD 검증 배우) 누움 재현 (10-30 · 진단)
  ① Okutama 비스듬 23편 1초 표본 Lying (lying_diag.lying_gt) — 사람 위에 뜸 (박스 중심 ↔ 정답) @ 확신 0.20 · 0.10 · 0.05 · 벤치 위 누움 (2.1.7) 뺀 값
  ② NOMAD 검증 배우 96~100 (학습에 없음) — 행동 Laying · Hiding (Laying) vs Walking · 거리 a10 · a30 · a50 · 1080p 로 줄임 · 가림 ≥ 50 %
     찾음 (IoU ≥ 0.5 · 0.20) · 사람 위에 뜸 (0.20) · 키 px
출력 metrics/lying_rescue_check.json
"""
import glob, json, os, re, sys
from collections import defaultdict
import cv2, numpy as np
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE, "deploy", "vision_worker")); sys.path.insert(0, BASE)
from app.vision_core.detector import PersonDetector  # noqa: E402
from okutama_motion import OK  # noqa: E402
from okutama_motion_det import iou  # noqa: E402
from state_trackfix import videos  # noqa: E402
from make_posture_crops import NOMAD, _nomad_act  # noqa: E402
import lying_diag as LD  # noqa: E402


def hit(bx, cf, gb, th):
    for b, c in zip(bx, cf):
        if c < th:
            continue
        if (gb[0] <= (b[0] + b[2]) / 2 <= gb[2] and gb[1] <= (b[1] + b[3]) / 2 <= gb[3]) or (b[0] <= (gb[0] + gb[2]) / 2 <= b[2] and b[1] <= (gb[1] + gb[3]) / 2 <= b[3]):
            return True
    return False


def main(names):
    dets = {n: PersonDetector(os.path.join(BASE, "runs_person", n, "weights", "best.pt"), conf=0.01) for n in names}
    out = {n: {} for n in names}
    # ① Okutama
    allv, _, _ = videos()
    fd = {os.path.basename(d): d for d in glob.glob(f"{OK}/Drone*/*/Extracted-Frames-1280x720/*")}
    R = {n: [] for n in names}
    for v in allv:
        for fr, G in sorted(LD.lying_gt(v).items()):
            img = cv2.imread(f"{fd[v]}/{fr}.jpg")
            if img is None:
                continue
            img = cv2.resize(img, (1920, 1080))
            for n, d in dets.items():
                bx, cf, _ = d.detect(img)
                for gb, occ, gen in G:
                    R[n].append((v, {th: hit(bx, cf, gb, th) for th in (0.20, 0.10, 0.05)}))
    for n in names:
        out[n]["Okutama 누움 사람 위에 뜸"] = {f"@{th}": round(float(np.mean([r[1][th] for r in R[n]])), 3) for th in (0.20, 0.10, 0.05)}
        out[n]["Okutama 누움 (벤치 2.1.7 뺌)"] = {f"@{th}": round(float(np.mean([r[1][th] for r in R[n] if r[0] != "2.1.7"])), 3) for th in (0.20, 0.10, 0.05)}
    # ② NOMAD 검증 배우
    acts = {x["id"]: x["labels"] for x in json.load(open(NOMAD / "activityLabels.json"))}
    rows = defaultdict(list)
    for r in json.load(open(NOMAD / "annotations.json")):
        actor_s, dist_s, f_s = r["file_name"][:-4].split("_")
        a = int(actor_s[5:])
        if a < 96 or dist_s not in ("a10", "a30", "a50"):
            continue
        act = _nomad_act(acts, a, dist_s[1:], int(f_s[1:]))
        kind = "누움" if act in ("Laying", "Hiding (Laying)") else "걷기" if act == "Walking" else None
        boxes = [b["bbox"] for b in r["annotations"] if int(b.get("visibility", 100)) >= 50]
        if kind and len(boxes) == 1:
            rows[(kind, dist_s)].append((NOMAD / "images" / actor_s / f"{actor_s}_{dist_s}" / r["file_name"], boxes[0]))
    rng = np.random.default_rng(0)
    for key, L in sorted(rows.items()):
        pick = [L[i] for i in sorted(rng.choice(len(L), min(60, len(L)), replace=False))]
        stat = {n: [] for n in names}; hs = []
        for p, (x, y, w, h) in pick:
            img = cv2.imread(str(p))
            if img is None:
                continue
            s = 1080 / img.shape[0]
            img = cv2.resize(img, (round(img.shape[1] * s), 1080), interpolation=cv2.INTER_AREA)
            if img.shape[1] != 1920:
                img = cv2.resize(img, (1920, 1080))
            sx = 1920 / (round(img.shape[1]) if False else 1920)
            gb = (x * s, y * s, (x + w) * s, (y + h) * s); hs.append(gb[3] - gb[1])
            for n, d in dets.items():
                bx, cf, _ = d.detect(img)
                strict = any(iou(tuple(b), gb) >= 0.5 and c >= 0.20 for b, c in zip(bx, cf))
                stat[n].append((strict, hit(bx, cf, gb, 0.20)))
        for n in names:
            a = np.array(stat[n])
            out[n][f"NOMAD 검증 {key[0]} {key[1]} (키 중앙 {np.median(hs):.0f} px · n={len(a)})"] = {"찾음 IoU0.5": round(float(a[:, 0].mean()), 3), "사람 위에 뜸": round(float(a[:, 1].mean()), 3)}
    for n in names:
        print(n, json.dumps(out[n], ensure_ascii=False), flush=True)
    json.dump(out, open(os.path.join(BASE, "metrics", "lying_rescue_check.json"), "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main(sys.argv[1:])
