"""
make_v12_neg.py — v12 음성 창 다시 고르기 (10-12 · make_v12_data.py 다음 단계)

  발견  NOMAD · WiSARD · HERIDAL 의 「라벨 없는 프레임」 은 사람이 없는 게 아니라 **라벨을 안 단 프레임** —
        soup_v7r2 확신 ≥ 0.5 검출이 NOMAD 음성 22 % · WiSARD 15 % · HERIDAL 13 % 창에 있고 눈으로 보면 거의 다 사람
        (v7 도 wisard_prep 이 빈 프레임을 음성 10 % 로 섞었다 → 같은 오염)
  규칙  ① 빈 프레임 음성은 라벨이 빠짐없는 출처만 (VisDrone · Unicamp · NII-CU · LADD · AI-Hub) — NOMAD · WiSARD · HERIDAL · SARD 빈 프레임 음성은 뺌
        ② 모자란 만큼 **라벨된 프레임의 사람 없는 자리** (AI-Hub · WiSARD — 양성 창 안 라벨 없는 검출 0.2 · 0.8 %) 에서 더 자름
        ③ 모든 음성 창을 soup_v7r2 로 훑어 확신 ≥ 0.5 검출이 있으면 버림 (라벨 없는 사람일 수 있음 · 물체 음성 일부도 같이 버려짐)
  목표 음성 ≈ 6 % · 결과는 configs/lists/train_v12.txt 를 고쳐 쓴다 (전 목록은 train_v12_raw.txt 로 남김)
실행: /home/se/miniconda3/envs/drone/bin/python make_v12_neg.py
"""
import json
import random
import shutil
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import make_v12_data as M

BASE = Path(__file__).resolve().parent
LISTS = BASE / "configs" / "lists"
DROP_EMPTY = ("nomad", "wisard", "heridal", "sard")
EXTRA = {"aihub": 380, "wisard": 380}
TARGET = 0.06


def main():
    raw = LISTS / "train_v12_raw.txt"
    if not raw.exists():
        shutil.copy(LISTS / "train_v12.txt", raw)
    L = [l.strip() for l in open(raw) if l.strip()]
    pos = [p for p in L if not p.endswith("_n.jpg")]
    neg = [p for p in L if p.endswith("_n.jpg") and not Path(p).name.startswith(DROP_EMPTY)]
    rng = random.Random(121); jobs = []
    for name, n in EXTRA.items():
        recs = [x for x in {"aihub": M.src_aihub, "wisard": M.src_wisard}[name]() if x["split"] == "train" and M.has_person(x)]
        jobs += [(x, 3, True) for x in rng.sample(recs, min(len(recs), int(n * 1.3)))]
    import cv2
    cv2.setNumThreads(1)
    with Pool(12) as pool:
        made = [x["path"] for x in pool.imap_unordered(M.window, jobs, chunksize=4) if x]
    by = Counter(Path(p).name.split("_")[0] for p in made)
    extra = []
    for name, n in EXTRA.items():
        extra += [p for p in made if Path(p).name.startswith(name)][:n]
    cand = neg + extra
    # ③ soup_v7r2 로 거르기
    from ultralytics import YOLO
    m = YOLO(str(BASE / "runs_person/soup_v7r2/weights/best.pt")); keep, drop = [], Counter()
    for i in range(0, len(cand), 16):
        for p, r in zip(cand[i:i + 16], m.predict(cand[i:i + 16], imgsz=1280, conf=0.5, verbose=False, quantize="fp16")):
            if len(r.boxes):
                drop[Path(p).name.split("_")[0]] += 1
            else:
                keep.append(p)
    final = sorted(pos + keep)
    (LISTS / "train_v12.txt").write_text("\n".join(final) + "\n")
    st = {"양성": len(pos), "음성 (남김)": len(keep), "음성 비율": round(len(keep) / len(final), 3),
          "뺀 빈 프레임 음성 (NOMAD · WiSARD · HERIDAL · SARD)": sum(1 for p in L if p.endswith("_n.jpg") and Path(p).name.startswith(DROP_EMPTY)),
          "더 자른 음성 (라벨된 프레임의 빈 자리)": dict(by), "soup_v7r2 ≥ 0.5 로 버림": dict(drop),
          "출처별 음성": dict(Counter(Path(p).name.split("_")[0] for p in keep))}
    s = json.load(open(BASE / "metrics/v12_data_stats.json")); s["음성 다시 고르기 (make_v12_neg)"] = st
    s["학습 창 (최종)"] = len(final)
    (BASE / "metrics/v12_data_stats.json").write_text(json.dumps(s, ensure_ascii=False, indent=1))
    print(json.dumps(st, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
