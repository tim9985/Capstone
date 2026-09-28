"""
make_v9_data.py — v9 추가 데이터: NOMAD 가림 · 누움 · 숨음 부분집합 (2026-09-28)

  근거: 최종 모델의 남은 약점 — test_obl 가림 재현율 0.24 · 누움 0.36 · 앉음 0.52 (→ _학습 큐 「다음 계획」)
  NOMAD (미국 농장 · 배우 100명 · 5.4K · 고도 10~90 m) 는 배우·고도별로 **Walking · Hiding · Laying · Hiding (Laying)**
  구간이 프레임 단위로 있고 박스마다 가시도(visibility 10~100 %) 가 있다 → 약점만 골라 넣는다
  고르는 기준
    · 행동이 Hiding · Laying · Hiding (Laying) 이거나 · 가시도 ≤ 70 % 인 사람이 있는 프레임
    · 가시도 10 % 는 거의 안 보여 라벨이 불확실 → 그 박스만 있는 프레임은 뺀다 (박스는 남기고 크롭 대상에서만 제외)
  크롭: make_v6_data.crop_one (사람 긴 변 31~94 px 로그균등 · 1280×720) · 상한 3,000 (행동별 균등)
  v7 에서 NOMAD 전체를 뺐다 (농장 한 곳 쏠림) → 이번엔 **약점 부분만 · 상한** 으로 넣는다
출력  data/raw/NOMAD/yolo_v9/ · data/det_v6/nomad_v9/ · configs/lists/nomad_v9.txt · metrics/v9_data_stats.json
실행: python make_v9_data.py
"""
import json, random, re, zlib
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

from make_v6_data import crop_one

BASE = Path(__file__).resolve().parent
RAW = BASE / "data" / "raw" / "NOMAD"
STAGE = RAW / "yolo_v9"
OUT = BASE / "data" / "det_v6" / "nomad_v9" / "images" / "train"
LISTS = BASE / "configs" / "lists"
CAP = 3000
TARGET = ("Hiding", "Laying", "Hiding (Laying)")


def _num(x):
    m = re.search(r"\d+(\.\d+)?", str(x))                 # '1320}' · '1140.1598' 같은 원본 오타
    return float(m.group()) if m else None


def activity_of(acts, actor, dist, frame):
    for name, spans in acts.get(actor, {}).get(dist, {}).items():
        for sp in spans:
            if not sp:
                continue
            a, b = _num(sp[0]), _num(sp[-1])           # 시작만 있는 구간도 있다
            if a is not None and b is not None and a <= frame <= b:
                return name
    return None


def main():
    ann = json.load(open(RAW / "annotations.json"))
    acts = {a["id"]: a["labels"] for a in json.load(open(RAW / "activityLabels.json"))}
    STAGE.mkdir(parents=True, exist_ok=True)
    cand = defaultdict(list); seen = Counter()
    for r in ann:
        fn = r["file_name"]; actor_s, dist_s, f_s = fn[:-4].split("_")
        img = RAW / "images" / actor_s / f"{actor_s}_{dist_s}" / fn
        if not img.exists() or not r["annotations"]:
            continue
        actor, dist, frame = int(actor_s[5:]), dist_s[1:], int(f_s[1:])
        act = activity_of(acts, actor, dist, frame) or "?"
        vis = [int(b.get("visibility", 100)) for b in r["annotations"]]
        occl = any(20 <= v <= 70 for v in vis)
        if not (act in TARGET or occl) or all(v < 20 for v in vis):
            continue
        W, H = r["width"], r["height"]
        lab = STAGE / f"{fn[:-4]}.txt"
        lab.write_text("".join(f"0 {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f}\n"
                               for x, y, w, h in (b["bbox"] for b in r["annotations"])))
        key = act if act in TARGET else "가림(걷기 등)"
        cand[key].append((str(img), str(lab), None, str(OUT / f"nomad_{fn}"), zlib.crc32(fn.encode())))
        seen[(key, dist_s)] += 1
    rng = random.Random(9); per = CAP // max(1, len(cand)); jobs = []
    for k in sorted(cand):
        v = sorted(cand[k], key=lambda j: j[0]); rng.shuffle(v); jobs += v[:per]
    with Pool(6) as pool:
        crops = sorted(r for r in pool.imap_unordered(crop_one, jobs, chunksize=8) if r)
    (LISTS / "nomad_v9.txt").write_text("\n".join(c[0] for c in crops) + "\n")
    stats = {"후보_프레임(행동·고도)": {f"{a}|{d}": n for (a, d), n in sorted(seen.items())},
             "행동별_후보": {k: len(v) for k, v in cand.items()}, "행동별_상한": per,
             "크롭": len(crops), "크롭_사람px_중앙": round(sorted(c[1] for c in crops)[len(crops) // 2], 1) if crops else None}
    (BASE / "metrics" / "v9_data_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1))
    print(json.dumps({k: stats[k] for k in ("행동별_후보", "행동별_상한", "크롭", "크롭_사람px_중앙")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
