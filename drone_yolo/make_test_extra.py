"""
make_test_extra.py — 학습에 안 쓴 외부 데이터 → 평가 전용 세트 (2026-10-04)

  data/test_<이름>/images (원본으로 가는 심볼릭 링크) + labels (YOLO · 사람 1클래스)
  auair  AU-AIR (덴마크 오르후스 · Parrot Bebop 2 · 1920×1080 · 고도 10~30 m · 45~90°) — 'Human' 만
         연속 프레임 — 사람 있는 장은 프레임 번호 5 간격 · 없는 장은 30 간격(≈1초) (오탐 확인용) · 판정 블록 = 영상 8편
         ⚠ 자전거·오토바이 탄 사람은 'Human' 이 아니라 탈것으로만 라벨 → 그 위 예측은 오탐으로 잡힌다
  cy     Small Object Aerial Person (키프로스 캠퍼스 · 민방위 훈련 · Zenodo 7740081 · CC BY 4.0 · 1920×1080 수직) — 세 분할 전부
         (학습에 안 씀) · ⚠ 사람이 작다 — 긴 변 중앙 15 px (운용 하한 24 px 아래가 대부분)
  ladd   LADD (러시아 숲 수색 · Dataset Ninja 사본 · Supervisely 형식 · 4000×3000 등) — 전부 (분할 없음)
         4:3 → 가운데 16:9 로 자르고 1920×1080 으로 줄여 **새 jpg** 로 쓴다 (운용 카메라와 같은 화면) · 절반 넘게 잘린 사람은 뺀다

실행: python make_test_extra.py auair [cy ladd]
"""
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent
DATA = (BASE / "data").resolve()
RAW = DATA / "raw"


def write(name, items):
    """items: [(원본 경로, 폭, 높이, [(x0, y0, w, h) px …])]"""
    out = DATA / f"test_{name}"
    (out / "images").mkdir(parents=True, exist_ok=True); (out / "labels").mkdir(exist_ok=True)
    n = 0
    for src, W, H, boxes in items:
        stem = Path(src).stem
        dst = out / "images" / f"{stem}{Path(src).suffix.lower()}"
        if not dst.exists():
            os.symlink(src, dst)
        rows = [f"0 {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f}" for x, y, w, h in boxes if w > 1 and h > 1]
        (out / "labels" / f"{stem}.txt").write_text("\n".join(rows) + ("\n" if rows else ""))
        n += len(rows)
    print(f"test_{name}: {len(items):,}장 · 사람 {n:,} · 사람 없는 장 {sum(not b for *_, b in items):,} → {out}")


def auair():
    a = json.load(open(RAW / "auair" / "annotations.json"))["annotations"]
    seen, items = set(), []
    for x in a:
        v = x["image_name"].split("_")[1]; i = int(x["image_name"].split("_")[-1][:-4])
        boxes = [(b["left"], b["top"], b["width"], b["height"]) for b in x["bbox"] if b["class"] == 0]
        if i % (5 if boxes else 30) or (v, i) in seen:
            continue
        seen.add((v, i))
        items.append((str(RAW / "auair" / "images" / x["image_name"]), 1920, 1080, boxes))
    write("auair", sorted(items))


def cy():
    root = RAW / "cyprus_sop"; items = []
    lab = {t.stem: t for t in (root / "Annotations" / "Yolo").rglob("*.txt")}
    for img in sorted((root / "Images").rglob("*.jpg")):
        t = lab.get(img.stem)
        if t is None:
            continue
        rows = [l.split() for l in t.read_text().splitlines() if l.strip()]
        boxes = [((float(x) - float(w) / 2) * 1920, (float(y) - float(h) / 2) * 1080, float(w) * 1920, float(h) * 1080) for _, x, y, w, h in rows]
        items.append((str(img), 1920, 1080, boxes))
    write("cy", items)


def ladd():
    import cv2
    out = DATA / "test_ladd" / "_jpg"; out.mkdir(parents=True, exist_ok=True)
    items = []
    for ann in sorted((RAW / "ladd").rglob("ann/*.json")):
        img = ann.parent.parent / "img" / ann.name[:-5]
        if not img.exists():
            continue
        a = json.load(open(ann)); W, H = a["size"]["width"], a["size"]["height"]
        cw, ch = (W, round(W * 9 / 16)) if W / H < 16 / 9 else (round(H * 16 / 9), H)
        ox, oy = (W - cw) // 2, (H - ch) // 2; sc = 1920 / cw
        dst = out / f"{img.stem}.jpg"
        if not dst.exists():
            im = cv2.imread(str(img))[oy:oy + ch, ox:ox + cw]
            cv2.imwrite(str(dst), cv2.resize(im, (1920, 1080), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 95])
        boxes = []
        for o in a["objects"]:
            (x0, y0), (x1, y1) = o["points"]["exterior"][:2]
            x0, x1 = sorted((x0, x1)); y0, y1 = sorted((y0, y1)); area = (x1 - x0) * (y1 - y0)
            cx0, cy0, cx1, cy1 = max(x0, ox), max(y0, oy), min(x1, ox + cw), min(y1, oy + ch)
            if area <= 0 or cx1 <= cx0 or cy1 <= cy0 or (cx1 - cx0) * (cy1 - cy0) < 0.5 * area:
                continue
            boxes.append(((cx0 - ox) * sc, (cy0 - oy) * sc, (cx1 - cx0) * sc, (cy1 - cy0) * sc))
        items.append((str(dst), 1920, 1080, boxes))
    write("ladd", items)


if __name__ == "__main__":
    for n in sys.argv[1:] or ["auair"]:
        globals()[n]()
