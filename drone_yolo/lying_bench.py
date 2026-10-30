"""
lying_bench.py — 누운 사람 평가 묶음 v1 (10-30 · 앞으로 탐지 모델 결정에 같이 봄 · _학습 큐 「10-30 CP」)

  셋 (학습에 안 쓴 것만)
    OK   Okutama 비스듬 23편 · 1초 표본 · Lying 정답 (lying_diag.lying_gt) — 벤치 위 누움 영상 2.1.7 은 따로 (OK-벤치)
    UE   UE S1 · S3 (16 · 20 m) 누움 · 기어감 배우 · 정답 박스 (잘림 없음 · 키 ≥ 12 px) · 4장마다
    NM   NOMAD 검증 배우 96~100 · Laying · Hiding (Laying) · a50 · a70 (작은 누움) · 가림 ≥ 50 % · 묶음당 ≤ 60
  지표 (운용 탐지 = worker 타일 · 1080p)
    엄격 = IoU ≥ 0.5 · 확신 ≥ 0.20   ·   뜸 = 박스 중심 ↔ 정답 (서로 안) · 확신 ≥ 0.20 (운영자가 그 사람을 보게 되는가)   ·   뜸@0.10
  출력 metrics/lying_bench_<모델>.json · 콘솔 한 줄
실행  python lying_bench.py <모델 이름 …>
"""
import csv, glob, json, os, sys
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
from lying_rescue_check import hit  # noqa: E402


def items():
    """(셋, 그림 읽기 함수, [정답 박스 1080p]) 목록"""
    out = []
    allv, _, _ = videos()
    fd = {os.path.basename(d): d for d in glob.glob(f"{OK}/Drone*/*/Extracted-Frames-1280x720/*")}
    for v in allv:
        for fr, G in sorted(LD.lying_gt(v).items()):
            out.append(("OK-벤치" if v == "2.1.7" else "OK", f"{fd[v]}/{fr}.jpg", [g[0] for g in G]))
    for run in ("S1_a16", "S1_a20", "S3_a16", "S3_a20"):
        R = os.path.join(BASE, "ue_flight", run)
        A = json.load(open(f"{R}/actors.json"))
        ly = {a for a, v in A.items() if v["pose_raw"] in ("lying", "crawling")}
        for k, l in enumerate(open(f"{R}/boxes.jsonl")):
            r = json.loads(l)
            if k % 4:
                continue
            G = [b["bb"] for a, b in r["boxes"].items() if a in ly and not b.get("cut") and b["bb"][3] - b["bb"][1] >= 12]
            if G:
                out.append(("UE", f"{R}/frames/{r['frame']}.jpg", G))
    acts = {x["id"]: x["labels"] for x in json.load(open(NOMAD / "activityLabels.json"))}
    per = defaultdict(int)
    for r in json.load(open(NOMAD / "annotations.json")):
        actor_s, dist_s, f_s = r["file_name"][:-4].split("_")
        a = int(actor_s[5:])
        if a < 96 or dist_s not in ("a50", "a70") or per[dist_s] >= 60:
            continue
        if _nomad_act(acts, a, dist_s[1:], int(f_s[1:])) not in ("Laying", "Hiding (Laying)"):
            continue
        bb = [b["bbox"] for b in r["annotations"] if int(b.get("visibility", 100)) >= 50]
        if len(bb) != 1:
            continue
        per[dist_s] += 1
        out.append(("NM", str(NOMAD / "images" / actor_s / f"{actor_s}_{dist_s}" / r["file_name"]), [bb[0]], ))
    return out


def load(path, kind):
    img = cv2.imread(path)
    if img is None:
        return None, 1.0
    if kind == "OK":
        return cv2.resize(img, (1920, 1080)), 1.0
    if kind == "NM":
        s = 1080 / img.shape[0]
        return cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_AREA), s
    return img, 1.0


def main(names):
    its = items()
    for n in names:
        d = PersonDetector(os.path.join(BASE, "runs_person", n, "weights", "best.pt"), conf=0.05)
        acc = defaultdict(lambda: defaultdict(list)); hs = defaultdict(list)
        for kind, path, G in its:
            img, s = load(path, "OK" if kind.startswith("OK") else kind)
            if img is None:
                continue
            bx, cf, _ = d.detect(img)
            for g in G:
                gb = (g[0] * s, g[1] * s, (g[0] + g[2]) * s, (g[1] + g[3]) * s) if kind == "NM" else tuple(g)
                acc[kind]["엄격"].append(any(iou(tuple(b), gb) >= 0.5 and c >= 0.20 for b, c in zip(bx, cf)))
                acc[kind]["뜸"].append(hit(bx, cf, gb, 0.20)); acc[kind]["뜸@0.10"].append(hit(bx, cf, gb, 0.10))
                hs[kind].append(gb[3] - gb[1])
        out = {k: {m: round(float(np.mean(v)), 3) for m, v in M.items()} | {"n": len(M["뜸"]), "키 중앙 px": round(float(np.median(hs[k])), 1)} for k, M in acc.items()}
        json.dump(out, open(os.path.join(BASE, "metrics", f"lying_bench_{n}.json"), "w"), ensure_ascii=False, indent=1)
        print(n, " | ".join(f"{k} 엄격 {v['엄격']:.3f} · 뜸 {v['뜸']:.3f} · 뜸@0.10 {v['뜸@0.10']:.3f} (n {v['n']} · {v['키 중앙 px']} px)" for k, v in sorted(out.items())), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
