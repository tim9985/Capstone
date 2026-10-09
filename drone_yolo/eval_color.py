"""
eval_color.py — 상의 색상 판정 평가 (W1-5 · 2026-10-05) — NOMAD 배우 42명 (`metrics/nomad_actor_selection.csv` upper_color)

  NOMAD 원본 5472×3078 → **1920×1080 으로 줄여** 운용 카메라와 같은 화소에서 자른다 (정답 박스 · 가림 ≥50 %)
  배우 × 거리 (a10 · a30 · a50 · a70) × 자세 (걷기 → 서기 · 눕기 → 누움) 마다 최대 N 장 (고르게)
  색 이름 맞추기: blue · cyan → 파랑 (cyan 은 초록도 경계 정답) · gray · light gray → 회색 · dark gray → 회색 또는 검정
  지표: 한 장 1순위 · 1·2순위 정확도 · 판정 불가율 · 거리 · 자세 · 색별 · 혼동 행렬
        누적 (배우 × 거리 × 자세 묶음) 1순위 · 1·2순위
        **찾는 사람 순위** — 배우마다 자기 색을 조건으로 줬을 때 42명 중 몇 번째 (같은 색 배우는 함께 앞에 오는 게 정상 → "자기보다 앞선 다른 색 배우 수" 도 같이)
  크롭 (1.5배 여백 · 1080p) 과 프레임 통계를 한 번 저장 (--cache · git 밖) → 방식 (화이트밸런스 · 노출 · 문턱) 여러 개를 빠르게 비교
실행: python eval_color.py [--n 12] [--workers 6] [--cache 경로]
출력: metrics/eval_color.json (방식별) · 콘솔 표
"""
import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / "deploy" / "vision_worker"))
from app.vision_core.color import COLORS, ColorAccumulator, ColorProfiler, frame_stats, match  # noqa: E402
from make_posture_crops import NOMAD, _nomad_act  # noqa: E402

MAP = {"blue": {"blue"}, "cyan": {"blue", "green"}, "gray": {"gray"}, "light gray": {"gray", "white"},
       "dark gray": {"gray", "black"}, "white": {"white"}, "black": {"black"}, "green": {"green"},
       "purple": {"purple"}, "pink": {"pink"}, "yellow": {"yellow"}, "red": {"red"}}
PRIMARY = {"blue": "blue", "cyan": "blue", "gray": "gray", "light gray": "gray", "dark gray": "gray", "white": "white",
           "black": "black", "green": "green", "purple": "purple", "pink": "pink", "yellow": "yellow", "red": "red"}
POSE = {"Walking": "standing", "Laying": "lying", "Hiding (Laying)": "lying"}
VARIANTS = {   # 이름: ColorProfiler 인자
    "V0 회색가정": dict(wb="gray_world"),
    "V1 보정없음": dict(wb=None),
    "V2 무채색기준": dict(wb="achromatic"),
    "V3 보정없음+노출": dict(wb=None, v_target=110),
    "V4 무채색+노출": dict(wb="achromatic", v_target=110),
    "V5 무채색+노출+검정0.2": dict(wb="achromatic", v_target=110, v_black=0.20),
    "V6 V5+무채색S0.25": dict(wb="achromatic", v_target=110, v_black=0.20, s_achro=0.25),
    "V7 보정없음+노출+검정0.2+S0.25": dict(wb=None, v_target=110, v_black=0.20, s_achro=0.25),
    "V8 V4+검정표0.5": dict(wb="achromatic", v_target=110, w_black=0.5),
    "V9 V4+검정표0.3": dict(wb="achromatic", v_target=110, w_black=0.3),
    "V10 V4+검정표0.3+회색표0.7": dict(wb="achromatic", v_target=110, w_black=0.3, w_gray=0.7),
}


def extract(args):
    """5K 원본 → 1080p → 1.5배 여백 크롭 + 크롭 안 박스 + 프레임 통계"""
    path, box, pose = args
    img = cv2.imread(path)
    if img is None:
        return None
    s = 1080 / img.shape[0]
    fr = cv2.resize(img, (round(img.shape[1] * s), 1080), interpolation=cv2.INTER_AREA)
    x, y, w, h = [v * s for v in box]
    cx, cy = x + w / 2, y + h / 2
    X1, Y1 = int(max(0, cx - 0.75 * w)), int(max(0, cy - 0.75 * h))
    X2, Y2 = int(min(fr.shape[1], cx + 0.75 * w)), int(min(fr.shape[0], cy + 0.75 * h))
    return {"crop": fr[Y1:Y2, X1:X2].copy(), "box": (x - X1, y - Y1, x + w - X1, y + h - Y1), "pose": pose,
            "stats": frame_stats(fr), "long_px": round(max(w, h), 1)}


def run_variant(args):
    item, kw = args
    if item is None:
        return None
    return ColorProfiler(**kw).profile(item["crop"], item["box"], pose=item["pose"], stats=item["stats"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--cache", default="/tmp/eval_color_cache.pkl")
    a = ap.parse_args()
    truth = {r["actor"]: r["upper_color"].strip().lower() for r in csv.DictReader(open(BASE / "metrics" / "nomad_actor_selection.csv", encoding="utf-8-sig"))}
    acts = {x["id"]: x["labels"] for x in json.load(open(NOMAD / "activityLabels.json"))}
    groups = defaultdict(list)
    for r in json.load(open(NOMAD / "annotations.json")):
        actor_s, dist_s, f_s = r["file_name"][:-4].split("_")
        if actor_s not in truth or truth[actor_s] not in MAP:
            continue
        pose = POSE.get(_nomad_act(acts, int(actor_s[5:]), dist_s[1:], int(f_s[1:])))
        boxes = [b["bbox"] for b in r["annotations"] if int(b.get("visibility", 100)) >= 50]
        if pose is None or len(boxes) != 1:
            continue
        p = NOMAD / "images" / actor_s / f"{actor_s}_{dist_s}" / r["file_name"]
        groups[(actor_s, dist_s, pose)].append((str(p), boxes[0], pose))
    jobs, keys = [], []
    for k, v in sorted(groups.items()):
        idx = np.linspace(0, len(v) - 1, min(a.n, len(v))).round().astype(int)
        for i in sorted(set(idx)):
            jobs.append(v[i]); keys.append(k)
    print(f"배우 {len({k[0] for k in groups})} · 묶음 {len(groups)} · 장 {len(jobs)}", flush=True)
    import pickle
    cache = Path(a.cache)
    if cache.exists():
        items = pickle.load(open(cache, "rb"))
    else:
        with Pool(a.workers, initializer=cv2.setNumThreads, initargs=(1,)) as pool:
            items = pool.map(extract, jobs, chunksize=4)
        pickle.dump(items, open(cache, "wb"))
    allout = {}
    for vname, kw in VARIANTS.items():
        with Pool(a.workers, initializer=cv2.setNumThreads, initargs=(1,)) as pool:
            res = pool.map(run_variant, [(it, kw) for it in items], chunksize=16)
        allout[vname] = score(keys, res, truth)
        o = allout[vname]
        print(f"{vname:32s} 한 장 1순위 {o['한 장']['전체']['1순위']:.3f} · 1·2순위 {o['한 장']['전체']['1·2순위']:.3f} · 판정불가 {o['한 장']['전체']['판정불가율']:.3f}"
              f" · 누적 1순위 {o['누적 (배우·거리·자세 묶음)']['1순위']:.3f} · 순위 중앙 {o['찾는 사람 순위 (배우 42명 중)']['순위 중앙']}"
              f" · 3위 안 {o['찾는 사람 순위 (배우 42명 중)']['3위 안']}", flush=True)
    (BASE / "metrics" / "eval_color.json").write_text(json.dumps(allout, ensure_ascii=False, indent=1))
    best = max(allout, key=lambda k: (allout[k]["누적 (배우·거리·자세 묶음)"]["1순위"], -allout[k]["찾는 사람 순위 (배우 42명 중)"]["순위 평균"]))
    print(f"\n== 가장 나은 방식: {best}")
    for g, v in allout[best]["한 장"].items():
        print(f"  {g:14s} {v}")
    for t, c in allout[best]["혼동 (정답 대표색 → 1순위)"].items():
        print(f"  혼동 {t:7s} → {c}")


def score(keys, res, truth):

    tally = defaultdict(lambda: Counter())
    conf = defaultdict(Counter)
    acc = defaultdict(ColorAccumulator)
    actor_acc = defaultdict(ColorAccumulator)
    for k, pr in zip(keys, res):
        if pr is None:
            continue
        actor, dist, pose = k
        t = truth[actor]; ok_set = MAP[t]
        for g in ("전체", f"거리 {dist}", f"자세 {pose}", f"색 {t}"):
            c = tally[g]; c["장"] += 1
            if pr["status"] != "ok":
                c["판정불가"] += 1; continue
            top = [x[0] for x in pr["top"]]
            c["판정"] += 1; c["1순위"] += top[0] in ok_set; c["1·2순위"] += bool(set(top[:2]) & ok_set)
        if pr["status"] == "ok":
            conf[PRIMARY[t]][pr["top"][0][0]] += 1
        acc[k].add(pr); actor_acc[actor].add(pr)

    def rate(c):
        return {"장": c["장"], "판정불가율": round(c["판정불가"] / max(c["장"], 1), 3),
                "1순위": round(c["1순위"] / max(c["판정"], 1), 3), "1·2순위": round(c["1·2순위"] / max(c["판정"], 1), 3)}
    out = {"한 장": {g: rate(c) for g, c in sorted(tally.items())}}
    # 누적 — 배우 × 거리 × 자세 묶음
    ga = Counter()
    for k, A in acc.items():
        r = A.result(); ga["묶음"] += 1
        if r["status"] != "ok":
            ga["판정불가"] += 1; continue
        top = [x[0] for x in r["top"]]; s = MAP[truth[k[0]]]
        ga["1순위"] += top[0] in s; ga["1·2순위"] += bool(set(top) & s); ga["판정"] += 1
    out["누적 (배우·거리·자세 묶음)"] = {"묶음": ga["묶음"], "판정불가": ga["판정불가"],
                                 "1순위": round(ga["1순위"] / max(ga["판정"], 1), 3), "1·2순위": round(ga["1·2순위"] / max(ga["판정"], 1), 3)}
    # 찾는 사람 순위 — 배우별 전체 누적 · 조건 = 자기 대표색
    R = {act: A.result() for act, A in actor_acc.items()}
    ranks, ahead_other, matched = [], [], 0
    for act, r in R.items():
        q = PRIMARY[truth[act]]
        scores = {o: match(R[o], q)[1] for o in R}
        st, _ = match(r, q); matched += st == "일치"
        better = [o for o in R if o != act and scores[o] > scores[act]]
        ranks.append(1 + len(better))
        ahead_other.append(sum(1 for o in better if PRIMARY[truth[o]] != q))
    out["찾는 사람 순위 (배우 42명 중)"] = {"배우": len(R), "일치 판정": matched,
                                    "순위 중앙": float(np.median(ranks)), "순위 평균": round(float(np.mean(ranks)), 2),
                                    "3위 안": round(float(np.mean(np.array(ranks) <= 3)), 3),
                                    "앞선 다른 색 배우 수 평균": round(float(np.mean(ahead_other)), 2)}
    out["혼동 (정답 대표색 → 1순위)"] = {t: dict(c.most_common()) for t, c in sorted(conf.items())}
    return out


if __name__ == "__main__":
    main()
