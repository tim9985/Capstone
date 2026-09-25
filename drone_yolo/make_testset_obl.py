"""
make_testset_obl.py — test_obl: 60° 운용 판정용 비스듬 평가셋 (2026-09-24)

  Okutama 비스듬 시퀀스 16개 (박스 크기–y 상관 r ≥ 0.60 · metrics/survey_tilt2.csv)
  · 1280×720 추출 프레임 · 시퀀스당 균등 표집 60장
  · 라벨은 4K 좌표 → 1280×720 · lost=1(화면 밖) 박스 제외 · 3 px 이하 제외
  · ⚠ 09-25 수정: 라벨 파일이 두 폴더에 중복 → 하나만 읽는다 (그전 test_obl 은 정답이 전부 2번씩 — 재현율 상한 50 %)
  · Okutama 는 학습에 쓰지 않는다 (결정 - Okutama 평가 전용)
  평가: python eval_test_v2.py --data data/test_obl --mode single --tag test_obl --weights …
"""
import csv, glob, json, shutil
from collections import defaultdict
from pathlib import Path
import numpy as np

BASE = Path(__file__).resolve().parent
OUT = BASE / "data" / "test_obl"
W, H, SW = 1280, 720, 3840
PER_SEQ = 60


def labels(seq):
    per = defaultdict(list)
    # 같은 라벨 파일이 Labels/ 와 TrainSetVideos (2)/Labels/ 두 곳에 있다 → 하나만 읽는다 (09-25 중복 버그 수정)
    files = sorted(glob.glob(str(BASE / f"data/raw/okutama/**/Labels/SingleActionLabels/3840x2160/{seq}.txt"), recursive=True),
                   key=lambda f: "TrainSetVideos" in f)
    for lp in files[:1]:
        for ln in open(lp, encoding="utf-8", errors="replace"):
            t = ln.split()
            if len(t) < 7:
                continue
            try:
                x1, y1, x2, y2, fr, lost = (int(t[i]) for i in (1, 2, 3, 4, 5, 6))
            except ValueError:
                continue
            if lost:
                continue
            s = W / SW
            b = [x1 * s, y1 * s, x2 * s, y2 * s]
            if b[2] - b[0] > 3 and b[3] - b[1] > 3:
                per[fr].append(b)
    return per


def main():
    rows = list(csv.DictReader(open(BASE / "metrics/survey_tilt2.csv", encoding="utf-8")))
    seqs = [r["group"].split()[1] for r in rows
            if r["group"].startswith("Okutama") and float(r["corr_cy_px"]) >= 0.60]
    for d in ("images", "labels"):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    stat, px = {}, []
    for seq in seqs:
        lab = labels(seq)
        dirs = glob.glob(str(BASE / f"data/raw/okutama/**/Extracted-Frames-1280x720/{seq}"), recursive=True)
        if not lab or not dirs:
            stat[seq] = 0
            continue
        frames = [f for f in sorted(lab) if (Path(dirs[0]) / f"{f}.jpg").exists()]
        pick = [frames[int(i)] for i in np.linspace(0, len(frames) - 1, min(PER_SEQ, len(frames)))]
        for fr in pick:
            name = f"okutama_{seq}_{fr:05d}"
            shutil.copy(Path(dirs[0]) / f"{fr}.jpg", OUT / "images" / f"{name}.jpg")
            with open(OUT / "labels" / f"{name}.txt", "w") as f:
                for x1, y1, x2, y2 in lab[fr]:
                    x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, W), min(y2, H)
                    f.write(f"0 {(x1+x2)/2/W:.6f} {(y1+y2)/2/H:.6f} {(x2-x1)/W:.6f} {(y2-y1)/H:.6f}\n")
                    px.append(max(x2 - x1, y2 - y1))
        stat[seq] = len(pick)
    q = np.percentile(px, [5, 25, 50, 75, 95]).round(1).tolist() if px else []
    man = {"목적": "60° 운용 판정 — Okutama 비스듬 시퀀스 (학습 미사용)", "입력": "1280x720 단일 추론",
           "시퀀스별_장수": stat, "장수": sum(stat.values()), "사람": len(px), "사람_px_분위수_5_25_50_75_95": q}
    (OUT / "manifest.json").write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(man, ensure_ascii=False))


if __name__ == "__main__":
    main()
