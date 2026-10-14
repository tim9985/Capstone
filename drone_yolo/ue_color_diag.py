"""
ue_color_diag.py — UE 촬영의 정답 박스로 상의색 모듈 (worker color.py) 진단 · 변형 비교 (10-14 U · 16 m 로만 고른다)

  정답 박스 (boxes.jsonl · 마스크 ≥ --min-px · 잘리지 않음) → ColorProfiler 변형마다 → 장 단위 1순위 · 배우별 누적 1·2순위
  변형: base (worker 그대로) · noexp (노출 맞춤 끔) · nowb (화이트밸런스 끔) · noexp_nowb · expup (밝히기만 · 배율 1~2)
실행  python ue_color_diag.py S4_a16 S3_a16 S1_a16 S2_a16 [--step 3]
"""
import argparse, json, sys
from collections import Counter, defaultdict
from pathlib import Path
import cv2
import numpy as np

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / "deploy" / "vision_worker" / "app"))
from vision_core import color as C  # noqa: E402

VARIANTS = {"base": {}, "noexp": {"v_target": None}, "nowb": {"wb": None}, "noexp_nowb": {"v_target": None, "wb": None},
            "expup": {"exp_clip": (1.0, 2.0)}, "expup_nowb": {"exp_clip": (1.0, 2.0), "wb": None}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--step", type=int, default=3)
    ap.add_argument("--min-px", type=int, default=800)
    a = ap.parse_args()
    res = {}
    for run in a.runs:
        d = BASE / "ue_flight" / run
        actors = json.load(open(d / "actors.json"))
        rows = [json.loads(l) for l in open(d / "boxes.jsonl")][::a.step]
        prof = {k: C.ColorProfiler(**v) for k, v in VARIANTS.items()}
        acc = {k: defaultdict(C.ColorAccumulator) for k in VARIANTS}
        hit = {k: Counter() for k in VARIANTS}; tot = Counter(); conf = {k: defaultdict(Counter) for k in VARIANTS}
        for r in rows:
            img = cv2.imread(str(d / "frames" / f"{r['frame']}.jpg"))
            if img is None:
                continue
            st = C.frame_stats(img)
            for n, g in r["boxes"].items():
                tc = actors.get(n, {}).get("upper_color")
                if not tc or g["cut"] or g["px"] < a.min_px:
                    continue
                tot[n] += 1
                for k, p in prof.items():
                    o = p.profile(img, g["bb"], stats=st)
                    if o["status"] != "ok":
                        continue
                    acc[k][n].add(o); top = o["top"][0][0]
                    hit[k][n] += top == tc; conf[k][tc][top] += 1
        out = {}
        for k in VARIANTS:
            fin = {n: [c for c, _ in (acc[k][n].result().get("top") or [])] for n in acc[k]}
            t1 = [fin[n][:1] == [actors[n]["upper_color"]] for n in fin]
            t2 = [actors[n]["upper_color"] in fin[n][:2] for n in fin]
            out[k] = {"frame_top1": round(sum(hit[k].values()) / max(sum(tot.values()), 1), 3),
                      "actor_top1": f"{sum(t1)}/{len(t1)}", "actor_top2": f"{sum(t2)}/{len(t2)}",
                      "confusion": {tc: dict(c.most_common(3)) for tc, c in conf[k].items()}}
            print(f"{run:8s} {k:11s} 장 1순위 {out[k]['frame_top1']:.3f} · 배우 누적 1순위 {out[k]['actor_top1']} · 1·2순위 {out[k]['actor_top2']}")
        res[run] = out
    (BASE / "metrics" / "ue_color_diag.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
