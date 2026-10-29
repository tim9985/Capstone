"""
lying_det_check.py — Okutama 누운 사람 탐지 (10-30 S · L 판정 ①) — 상태 판정 (state_worker_check) 과 같은 영상 · 1초 표본
  B 11편 · 정답 Lying 박스 (720p ×1.5) 마다 worker 타일 탐지 (1080p) 의 가장 겹치는 박스 → IoU ≥ 0.5 · 확신 ≥ 0.20 (배포) 비율
실행  python lying_det_check.py <모델 이름 …>   (runs_person/<이름>/weights/best.pt)
"""
import glob, json, os, sys
import cv2, numpy as np
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE, "deploy", "vision_worker")); sys.path.insert(0, BASE)
from app.vision_core.detector import PersonDetector  # noqa: E402
from okutama_motion import FPS, LAB, OK, load_boxes  # noqa: E402
from okutama_motion_det import iou  # noqa: E402
from state_trackfix import videos  # noqa: E402


def main(names):
    _, _, B = videos()
    fd = {os.path.basename(d): d for d in glob.glob(f"{OK}/Drone*/*/Extracted-Frames-1280x720/*")}
    dets = {n: PersonDetector(os.path.join(BASE, "runs_person", n, "weights", "best.pt"), conf=0.05) for n in names}
    res = {n: [] for n in names}
    for v in B:
        gt = load_boxes(f"{LAB}/{v}.txt")
        for fr, g in sorted(gt.items()):
            if fr % FPS:
                continue
            ly = [(b[0] * 1.5, b[1] * 1.5, b[2] * 1.5, b[3] * 1.5) for b in g.values() if b[4] == "Lying"]
            img = cv2.imread(f"{fd[v]}/{fr}.jpg") if ly else None
            if img is None:
                continue
            img = cv2.resize(img, (1920, 1080))
            for n, d in dets.items():
                bx, cf, _ = d.detect(img)
                for gb in ly:
                    res[n].append(max([(iou(tuple(b), gb), float(c)) for b, c in zip(bx, cf)] or [(0.0, 0.0)]))
    out = {}
    for n, r in res.items():
        r = np.array(r)
        out[n] = {"누움 정답": len(r), "IoU≥0.5·conf≥0.20": round(float(np.mean((r[:, 0] >= 0.5) & (r[:, 1] >= 0.20))), 3),
                  "IoU≥0.3·conf≥0.20": round(float(np.mean((r[:, 0] >= 0.3) & (r[:, 1] >= 0.20))), 3)}
        print(n, out[n], flush=True)
    json.dump(out, open(os.path.join(BASE, "metrics", f"lying_det_{'_vs_'.join(names)}.json"), "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main(sys.argv[1:])
