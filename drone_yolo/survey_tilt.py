"""
survey_tilt.py — 데이터셋 전수조사: 수직 하향 vs 비스듬 (2026-09-22)

왜
  운용 마운트각을 60° 로 잡을지 판단하려면 **학습 데이터가 어떤 시점인지** 알아야 한다.
  CLAUDE.md §4-8 은 "대부분 비스듬" 이라 적고 있지만 표본으로는 반대로 보였다 → 전수 확인.

  EXIF·XMP 는 비어 있다 (영상에서 뽑은 프레임이라 메타데이터가 지워졌다).
  → 라벨로 추정한다.

원리
  사선거리 R(v) = h / sin( θ + atan((v−H/2)/f) )   ·  사람 px ∝ 1/R
  · 비스듬(θ<90°): 화면 **아래**가 가까워 사람이 크다 → px 와 y 가 **양의 상관**
  · 수직 하향(θ=90°): 위·아래가 대칭이라 중앙이 가장 크다 → 상관 ≈ 0 (이차형)

판정 (그룹당 박스 50개 이상)
  r ≥ 0.30  → 비스듬                  (아래가 크다)
  r ≤ −0.30 → 비스듬 (카메라가 뒤로)   (위가 크다)
  그 외      → 수직 하향에 가깝다
  보조 지표 : 아래 1/3 중앙값 ÷ 위 1/3 중앙값

실행: python survey_tilt.py
출력: metrics/survey_tilt.csv
"""
import csv, glob, json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

BASE = Path(__file__).resolve().parent
MIN_N = 50


def add(g, cy, px):
    g["cy"].append(cy); g["px"].append(px)


def nomad():
    out = defaultdict(lambda: {"cy": [], "px": []})
    seen = set()
    for d in sorted(BASE.glob("data/raw/NOMAD*")):
        ann = d / "annotations.json"
        if not ann.exists():
            continue
        for r in json.load(open(ann, encoding="utf-8")):
            fn = r["file_name"]
            if fn in seen:
                continue
            seen.add(fn)
            W, H = r["width"], r["height"]
            m = fn.split("_")
            key = f"NOMAD {m[0]}_{m[1]}" if len(m) > 1 else f"NOMAD {fn}"
            for b in r.get("annotations", []):
                x, y, w, h = b["bbox"]
                add(out[key], (y + h / 2) / H, max(w, h))
    return out


def wisard():
    out = defaultdict(lambda: {"cy": [], "px": []})
    for d in sorted((BASE / "data/raw/WiSARD").iterdir()):
        if not d.is_dir():
            continue
        wh = None
        for img in sorted(list(d.rglob("*.jpg")) + list(d.rglob("*.jpeg"))):
            t = img.with_suffix(".txt")
            if not t.exists():
                continue
            rows = [l.split() for l in t.read_text().splitlines() if l.strip()]
            if not rows:
                continue
            if wh is None:
                with Image.open(img) as im:
                    wh = im.size
            W, H = wh
            for r in rows:
                cy, w, h = float(r[2]), float(r[3]) * W, float(r[4]) * H
                add(out[f"WiSARD {d.name}"], cy, max(w, h))
    return out


def main():
    groups = {**nomad(), **wisard()}
    rows = []
    print(f"{'그룹':<46} {'박스':>7} {'상관 r':>8} {'아래/위':>8}  판정")
    print("─" * 88)
    for k, g in sorted(groups.items()):
        cy, px = np.array(g["cy"]), np.array(g["px"])
        if len(cy) < MIN_N:
            continue
        r = float(np.corrcoef(cy, px)[0, 1]) if cy.std() > 0 else 0.0
        lo, hi = np.percentile(cy, [33, 67])
        top, bot = px[cy <= lo], px[cy >= hi]
        ratio = float(np.median(bot) / np.median(top)) if len(top) and len(bot) else float("nan")
        verdict = "비스듬(아래가 큼)" if r >= 0.30 else ("비스듬(위가 큼)" if r <= -0.30 else "수직 하향에 가까움")
        print(f"{k:<46} {len(cy):>7,} {r:>8.3f} {ratio:>8.2f}  {verdict}")
        rows.append({"group": k, "n": len(cy), "corr_cy_px": round(r, 4),
                     "bottom_over_top": round(ratio, 3), "verdict": verdict,
                     "px_median": round(float(np.median(px)), 1)})
    out = BASE / "metrics" / "survey_tilt.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    ob = [r for r in rows if r["verdict"].startswith("비스듬")]
    print(f"\n그룹 {len(rows)} 중 비스듬 {len(ob)} · 박스 기준 "
          f"{sum(r['n'] for r in ob):,} / {sum(r['n'] for r in rows):,} "
          f"({sum(r['n'] for r in ob)/sum(r['n'] for r in rows):.1%})")
    print(f"→ {out}")


if __name__ == "__main__":
    main()
