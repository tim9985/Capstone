"""
survey_tilt2.py — 남은 원천 전수조사 (2026-09-22)
survey_tilt.py 가 NOMAD·WiSARD 만 봤다. AI-Hub·VisDrone·Okutama·SARD·Unicamp 를 추가한다.

AI-Hub 182 는 **라벨 파일명에 각도가 박혀 있다** (`…-맑음-90-35m-…`) → 추정 불필요, 확정 90°.
나머지는 박스 크기 vs y좌표 상관으로 추정한다 (프레임당 사람이 여럿이라 NOMAD 보다 신뢰도가 높다).
"""
import csv, glob, re
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image

BASE = Path(__file__).resolve().parent
MIN_N = 80
G = lambda: {"cy": [], "px": []}


def visdrone():
    out = defaultdict(G)
    for ann in glob.glob(str(BASE / "data/raw/VisDrone/VisDrone2019-DET-*/annotations/*.txt")):
        p = Path(ann)
        img = next((q for q in (p.parents[1] / "images" / (p.stem + ".jpg"),
                                BASE / "data/raw/VisDrone/images/train" / (p.stem + ".jpg"),
                                BASE / "data/raw/VisDrone/images/val" / (p.stem + ".jpg"),
                                BASE / "data/raw/VisDrone/images/test" / (p.stem + ".jpg")) if q.exists()), None)
        if img is None:
            continue
        with Image.open(img) as im:
            W, H = im.size
        seq = p.stem.split("_")[0]
        for ln in open(ann, encoding="utf-8", errors="replace"):
            f = ln.strip().rstrip(",").split(",")
            if len(f) < 8 or f[5] not in ("1", "2"):      # 사람만
                continue
            x, y, w, h = (int(v) for v in f[:4])
            out[f"VisDrone seq{seq}"]["cy"].append((y + h / 2) / H)
            out[f"VisDrone seq{seq}"]["px"].append(max(w, h))
    return out


def okutama():
    out = defaultdict(G)
    for lp in glob.glob(str(BASE / "data/raw/okutama/**/Labels/SingleActionLabels/3840x2160/*.txt"), recursive=True):
        seq = Path(lp).stem
        for ln in open(lp, encoding="utf-8", errors="replace"):
            t = ln.split()
            if len(t) < 6:
                continue
            try:
                x1, y1, x2, y2 = (int(t[i]) for i in (1, 2, 3, 4))
            except ValueError:
                continue
            out[f"Okutama {seq}"]["cy"].append((y1 + y2) / 2 / 2160)
            out[f"Okutama {seq}"]["px"].append(max(x2 - x1, y2 - y1) * (720 / 2160))  # 1280x720 환산
    return out


def yolo_dirs(tag, pairs):
    out = defaultdict(G)
    for name, idir in pairs:
        for img in sorted(glob.glob(idir + "/*.jpg"))[:4000]:
            t = Path(img.replace("/images/", "/labels/")).with_suffix(".txt")
            if not t.exists():
                continue
            rows = [l.split() for l in t.read_text().splitlines() if l.strip()]
            if not rows:
                continue
            with Image.open(img) as im:
                W, H = im.size
            for r in rows:
                out[f"{tag} {name}"]["cy"].append(float(r[2]))
                out[f"{tag} {name}"]["px"].append(max(float(r[3]) * W, float(r[4]) * H))
    return out


def main():
    groups = {}
    groups.update(visdrone()); print("VisDrone 수집 완료", flush=True)
    groups.update(okutama()); print("Okutama 수집 완료", flush=True)
    groups.update(yolo_dirs("SARD", [(s, str(BASE / f"data/raw/sard2/search-and-rescue-2/{s}/images"))
                                     for s in ("train", "valid", "test")]))
    groups.update(yolo_dirs("Unicamp", [("sample", str(BASE / "data/raw/unicamp_uav_repo/Test Images - Sample"))]))
    print("SARD·Unicamp 수집 완료\n", flush=True)

    rows = []
    print(f"{'그룹':<28} {'박스':>8} {'상관 r':>8} {'아래/위':>8}  {'px중앙':>7}  판정")
    print("─" * 84)
    for k, g in sorted(groups.items()):
        cy, px = np.array(g["cy"]), np.array(g["px"])
        if len(cy) < MIN_N:
            continue
        r = float(np.corrcoef(cy, px)[0, 1]) if cy.std() > 0 else 0.0
        lo, hi = np.percentile(cy, [33, 67])
        top, bot = px[cy <= lo], px[cy >= hi]
        ratio = float(np.median(bot) / np.median(top)) if len(top) and len(bot) else float("nan")
        v = "비스듬(아래가 큼)" if r >= 0.30 else ("비스듬(위가 큼)" if r <= -0.30 else "수직 하향에 가까움")
        print(f"{k:<28} {len(cy):>8,} {r:>8.3f} {ratio:>8.2f}  {np.median(px):>7.1f}  {v}")
        rows.append({"group": k, "n": len(cy), "corr_cy_px": round(r, 4),
                     "bottom_over_top": round(ratio, 3), "px_median": round(float(np.median(px)), 1), "verdict": v})
    out = BASE / "metrics" / "survey_tilt2.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    ob = [r for r in rows if r["verdict"].startswith("비스듬")]
    print(f"\n그룹 {len(rows)} 중 비스듬 {len(ob)} · 박스 {sum(r['n'] for r in ob):,}/{sum(r['n'] for r in rows):,}"
          f" ({sum(r['n'] for r in ob)/sum(r['n'] for r in rows):.1%})")
    print(f"→ {out}")


if __name__ == "__main__":
    main()
