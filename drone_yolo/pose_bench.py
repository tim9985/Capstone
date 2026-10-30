"""
pose_bench.py — 자세별 「뜸」 · 전송 화질 민감도 (10-31 · 다음 학습 검토)

  셋 (학습에 안 쓴 것만 · lying_bench 와 같은 운용 탐지 = worker 타일 · 1080p)
    OK-<자세>  Okutama 비스듬 23편 · 1초 표본 · 자세 = Lying · Sitting · Standing · Walking · Running (load_boxes · lost 제외)
               누움은 벤치 영상 2.1.7 을 뺌 (lying_bench OK 와 같은 셈)
    UE         UE S1 · S3 (16 · 20 m) 배우 전체 · 4장마다 (잘림 없음 · 키 ≥ 12 px)
  지표  엄격 = IoU ≥ 0.5 · 확신 ≥ 0.20   ·   뜸 = 박스 중심 ↔ 정답 (서로 안) · 확신 ≥ 0.20   — lying_bench 와 같음
        뜸곡선 = 문턱 0.05 · 0.10 · 0.15 · 0.20 · 0.25 · 0.30 · 0.40 · 0.50 의 뜸 평균 (문턱 근처 깜빡임에 덜 민감 · 10-31 v12 반복 3개 누움 뜸 0.58~0.73)
  화질  --cond=raw,j50,j30,720  (j<q> = 1080p 를 JPEG 품질 q 로 다시 저장해 읽음 = 실시간 전송 압축 손실의 대용 ·
        720 = 1280×720 로 줄였다 1080p 로 키움 = 720p 전송 · g<σ×10> = 가우시안 흐림 · w<q> = WebP · a+b = 차례대로) · ffmpeg 가 없어 H.264 대신 JPEG
출력 metrics/pose_bench_<모델>.json · 콘솔 표
실행  python pose_bench.py <모델 …> [--cond=raw,j50,j30,720]
"""
import json, os, sys
from collections import defaultdict
import cv2, numpy as np
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE, "deploy", "vision_worker")); sys.path.insert(0, BASE)
from app.vision_core.detector import PersonDetector  # noqa: E402
from okutama_motion import FPS, LAB, OK, load_boxes  # noqa: E402
from okutama_motion_det import iou  # noqa: E402
from state_trackfix import videos  # noqa: E402
from lying_rescue_check import hit  # noqa: E402
import glob  # noqa: E402

UP = 1.5            # 1280 → 1080p
POSES = ("Lying", "Sitting", "Standing", "Walking", "Running")
TS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)


def items():
    """[(그림 경로, 종류, [(셋, 정답 박스 1080p)])]"""
    out = []
    allv, _, _ = videos()
    fd = {os.path.basename(d): d for d in glob.glob(f"{OK}/Drone*/*/Extracted-Frames-1280x720/*")}
    for v in allv:
        for fr, B in sorted(load_boxes(f"{LAB}/{v}.txt").items()):
            if fr % FPS:
                continue
            G = [(f"OK-{b[4]}", tuple(x * UP for x in b[:4])) for b in B.values() if not (b[4] == "Lying" and v == "2.1.7")]
            if G:
                out.append((f"{fd[v]}/{fr}.jpg", "OK", G))
    for run in ("S1_a16", "S1_a20", "S3_a16", "S3_a20"):
        R = os.path.join(BASE, "ue_flight", run)
        A = json.load(open(f"{R}/actors.json"))
        for k, l in enumerate(open(f"{R}/boxes.jsonl")):
            if k % 4:
                continue
            r = json.loads(l)
            G = [("UE", tuple(b["bb"])) for a, b in r["boxes"].items() if a in A and not b.get("cut") and b["bb"][3] - b["bb"][1] >= 12]
            if G:
                out.append((f"{R}/frames/{r['frame']}.jpg", "UE", G))
    return out


def degrade(img, cond):
    if "+" in cond:                   # j30+g06 = JPEG 30 뒤 흐림 (차례대로)
        for c in cond.split("+"):
            img = degrade(img, c)
        return img
    if cond == "raw":
        return img
    if cond == "720":
        return cv2.resize(cv2.resize(img, (1280, 720), interpolation=cv2.INTER_AREA), (1920, 1080), interpolation=cv2.INTER_LINEAR)
    if cond.startswith("j"):
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(cond[1:])])
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if cond.startswith("w"):          # w30 = WebP 품질 30 — 학습 증강 (JPEG) 에 안 쓰는 압축 (판정 때 「시험 문제로 공부」 막기)
        ok, buf = cv2.imencode(".webp", img, [cv2.IMWRITE_WEBP_QUALITY, int(cond[1:])])
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if cond.startswith("g"):          # g10 = 가우시안 흐림 σ 1.0 (1080p) — 「720 왕복이 Okutama 를 올림」 확인용 (10-31)
        return cv2.GaussianBlur(img, (0, 0), int(cond[1:]) / 10)
    raise ValueError(cond)


def main(names, conds):
    its = items()
    print("그림", len(its), "· 정답", sum(len(g) for _, _, g in its), flush=True)
    for n in names:
        d = PersonDetector(os.path.join(BASE, "runs_person", n, "weights", "best.pt"), conf=0.05)
        res = {}
        for cond in conds:
            acc = defaultdict(lambda: defaultdict(list)); hs = defaultdict(list)
            for path, kind, G in its:
                img = cv2.imread(path)
                if img is None:
                    continue
                if kind == "OK":
                    img = cv2.resize(img, (1920, 1080))
                bx, cf, _ = d.detect(degrade(img, cond))
                for s, gb in G:
                    acc[s]["엄격"].append(any(iou(tuple(b), gb) >= 0.5 and c >= 0.20 for b, c in zip(bx, cf)))
                    acc[s]["뜸"].append(hit(bx, cf, gb, 0.20))
                    acc[s]["뜸곡선"].append(float(np.mean([hit(bx, cf, gb, t) for t in TS])))
                    hs[s].append(gb[3] - gb[1])
            res[cond] = {s: {m: round(float(np.mean(v)), 3) for m, v in M.items()} | {"n": len(M["뜸"]), "키 중앙 px": round(float(np.median(hs[s])), 1)}
                         for s, M in sorted(acc.items())}
            print(n, cond, " | ".join(f"{s} {v['엄격']:.3f}/{v['뜸']:.3f}/{v['뜸곡선']:.3f} (n {v['n']} · {v['키 중앙 px']} px)" for s, v in res[cond].items()), flush=True)
        out = os.path.join(BASE, "metrics", f"pose_bench_{n}.json")
        old = json.load(open(out)) if os.path.exists(out) else {}       # 조건을 나눠 돌려도 합침 (같은 조건은 새 값)
        json.dump(old | res, open(out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--cond")]
    cond = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--cond=")), "raw,j50,j30,720")
    main(args, cond.split(","))
