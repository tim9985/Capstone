"""
make_v14_hn.py — 하드 네거티브 (틀린 탐지 되먹이기) 창 만들기 (10-15 H · 판정 기준 = _학습 큐 「10-15 H」)

  ① 후보 음성 창: 학습 분할 프레임에서 사람과 안 겹치는 1280×720 창 (make_v12_data.window 와 같은 규칙 · 무시 영역 가림 · k = 7~14)
     출처 = 라벨이 빠짐없는 VisDrone · Unicamp · NII-CU · LADD · AI-Hub + WiSARD 라벨된 프레임의 빈 자리
  ② v12_f91_swa 로 훑기 (conf 0.25) → 최고 확신이 [0.25, 0.5) 인 창만 = 하드 네거티브 (0.5 이상은 라벨 없는 사람일 수 있어 버림)
  ③ 출처당 상한 → 합계 ≈ 600 · 출력 data/det_v14hn · 목록 train_v14hn_f91/f92 = v12 같은 95 % 목록 + 하드 네거티브 전부
  ④ 표본 크롭 (검출 자리) → logs/v14hn_preview.jpg (눈으로 확인)
실행  python make_v14_hn.py
"""
import json, random
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

import make_v12_data as M

BASE = Path(__file__).resolve().parent
OUT = BASE / "data" / "det_v14hn"
PER_SRC = {"visdrone_obl": 140, "visdrone_nadir": 90, "unicamp": 40, "niicu": 40, "ladd": 60, "aihub": 150, "wisard": 120}
LO, HI, TOTAL = 0.25, 0.5, 600


def main():
    M.OUT = OUT
    for sp in ("train",):
        (OUT / "images" / sp).mkdir(parents=True, exist_ok=True); (OUT / "labels" / sp).mkdir(parents=True, exist_ok=True)
    rng = random.Random(1415)
    vis = M.src_visdrone()
    pools = {"visdrone_obl": [x for x in vis if x["src"] == "visdrone_obl"], "visdrone_nadir": [x for x in vis if x["src"] == "visdrone_nadir"],
             "unicamp": M.src_unicamp(), "niicu": M.src_niicu(), "ladd": M.src_ladd(), "aihub": M.src_aihub(), "wisard": M.src_wisard()}
    jobs = []
    for name, recs in pools.items():
        tr = [x for x in recs if x["split"] == "train"]
        if name == "wisard":
            tr = [x for x in tr if M.has_person(x)]               # 빈 프레임은 라벨 없는 사람 → 라벨된 프레임의 빈 자리만
        for x in rng.sample(tr, min(len(tr), PER_SRC[name] * 40)):    # 하드 비율 ≈ 8 % → 후보를 넉넉히 (10-15 첫 시도 ×12 는 146장뿐)
            jobs.append((x, rng.choice(range(7, 15)), True))
    cv2.setNumThreads(1)
    with Pool(12) as pool:
        made = [x for x in pool.imap_unordered(M.window, jobs, chunksize=8) if x]
    print(f"후보 음성 창 {len(made)}")
    from ultralytics import YOLO
    m = YOLO(str(BASE / "runs_person/v12_f91_swa/weights/best.pt"))
    hard, drop_hi, none = {}, Counter(), Counter()
    paths = [x["path"] for x in made]
    for i in range(0, len(paths), 16):
        for p, r in zip(paths[i:i + 16], m.predict(paths[i:i + 16], imgsz=1280, conf=LO, verbose=False, half=True)):
            src = Path(p).name.split("_")[0] if not Path(p).name.startswith("visdrone") else "_".join(Path(p).name.split("_")[:2])
            c = r.boxes.conf.cpu().numpy() if len(r.boxes) else np.zeros(0)
            if not len(c):
                none[src] += 1
            elif c.max() >= HI:
                drop_hi[src] += 1
            else:
                hard.setdefault(src, []).append((p, r.boxes.xyxy.cpu().numpy(), c))
    keep = []
    for src, lst in hard.items():
        rng.shuffle(lst); keep += lst[:PER_SRC.get(src, 60)]
    keep = keep[:TOTAL]
    kept = {p for p, _, _ in keep}
    for x in made:                                                 # 안 쓰는 후보 창은 지움 (이 스크립트가 만든 것만)
        if x["path"] not in kept:
            Path(x["path"]).unlink(missing_ok=True)
            Path(x["path"].replace("/images/", "/labels/")).with_suffix(".txt").unlink(missing_ok=True)
    hn = sorted(kept)
    for s in (91, 92):
        base = [l.strip() for l in open(BASE / f"configs/lists/train_v12_f{s}.txt") if l.strip()]
        (BASE / f"configs/lists/train_v14hn_f{s}.txt").write_text("\n".join(base + hn) + "\n")
        (BASE / f"configs/data_v14hn_f{s}.yaml").write_text(
            f"# v14hn_f{s} (10-15 H) — v12_f{s} 와 같은 95 % 목록 + 하드 네거티브 {len(hn)} · val = val_v12\n"
            f"train: {BASE}/configs/lists/train_v14hn_f{s}.txt\nval: {BASE}/configs/lists/val_v12.txt\nnc: 1\nnames: ['person']\n")
        neg0 = sum(1 for p in base if Path(p.replace("/images/", "/labels/")).with_suffix(".txt").stat().st_size == 0)
        print(f"f{s}: 양성+음성 {len(base)} (음성 {neg0}) + 하드 {len(hn)} → 음성 비율 {(neg0 + len(hn)) / (len(base) + len(hn)):.3f}")
    st = {"후보 창": len(made), "하드 (0.25~0.5)": {k: len(v) for k, v in hard.items()}, "버림 ≥ 0.5": dict(drop_hi), "검출 없음": dict(none),
          "쓴 하드 네거티브": dict(Counter(Path(p).name.split("_")[0] for p in hn)), "합계": len(hn)}
    (BASE / "metrics/v14hn_stats.json").write_text(json.dumps(st, ensure_ascii=False, indent=1))
    print(json.dumps(st, ensure_ascii=False))
    tiles = []
    for p, xy, c in keep[:: max(1, len(keep) // 32)][:32]:
        img = cv2.imread(p); j = int(np.argmax(c)); x1, y1, x2, y2 = xy[j].astype(int)
        m_ = int(max(x2 - x1, y2 - y1) * 1.5) + 16
        X1, Y1, X2, Y2 = max(0, x1 - m_), max(0, y1 - m_), min(img.shape[1], x2 + m_), min(img.shape[0], y2 + m_)
        t = img[Y1:Y2, X1:X2].copy(); cv2.rectangle(t, (x1 - X1, y1 - Y1), (x2 - X1, y2 - Y1), (0, 0, 255), 1)
        t = cv2.resize(t, (200, 200)); cv2.putText(t, f"{Path(p).name.split('_')[0]} {c[j]:.2f}", (3, 14), 0, 0.45, (255, 255, 255), 1)
        tiles.append(t)
    rows = [np.hstack(tiles[i:i + 8]) for i in range(0, len(tiles) - len(tiles) % 8, 8)]
    if rows:
        cv2.imwrite(str(BASE / "logs/v14hn_preview.jpg"), np.vstack(rows))


if __name__ == "__main__":
    main()
