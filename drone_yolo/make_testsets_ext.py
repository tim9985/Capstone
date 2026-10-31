"""
make_testsets_ext.py — 데이터셋별 비교용 평가셋 (11-01 · bench_all.py 입력 · 평가 전용 · git 밖 data/)

  test_visdrone  VisDrone2019-DET val 548장 · pedestrian + people → 사람 (v12 학습에 안 씀 ✓)
  test_nomad     NOMAD 배우 096~100 (있는 것 097 · 100) 전 고도 · annotations.json 박스 전부 (가려진 부분 포함 전신) (v12 학습에 안 씀 ✓)
  test_sard      SARD (Roboflow 재배포 v1) test 198장 · 자세 클래스 → 사람 (⚠ v12 학습에 test 까지 씀)
  test_heridal   HERIDAL ImageSets test 101장 (⚠ v12 학습에 전부 씀)
  이름 = <셋>_<그룹>_<번호 4자리> (eval 블록 = 그룹 안 연속 20장) · 라벨 = YOLO (클래스 0)
실행  python make_testsets_ext.py
"""
import json, shutil
from collections import defaultdict
from pathlib import Path

import cv2

BASE = Path(__file__).resolve().parent
RAW = BASE.parent / "data" / "raw"


def write(name, items):
    out = BASE / "data" / name
    if out.exists():
        shutil.rmtree(out)
    (out / "images").mkdir(parents=True); (out / "labels").mkdir()
    n = 0
    for stem, img, boxes in items:            # boxes = [(cx, cy, w, h) 정규화]
        shutil.copy2(img, out / "images" / f"{stem}.jpg")
        (out / "labels" / f"{stem}.txt").write_text("".join(f"0 {a:.6f} {b:.6f} {c:.6f} {d:.6f}\n" for a, b, c, d in boxes))
        n += len(boxes)
    (out / "manifest.json").write_text(json.dumps({"장수": len(items), "사람": n}, ensure_ascii=False))
    print(name, len(items), "장 · 사람", n)


def visdrone():
    items = []
    for k, p in enumerate(sorted((RAW / "VisDrone/images/val").glob("*.jpg"))):
        h, w = cv2.imread(str(p)).shape[:2]; bx = []
        for l in open(RAW / "VisDrone/VisDrone2019-DET-val/annotations" / f"{p.stem}.txt"):
            v = l.strip().split(",")
            if len(v) >= 6 and v[5] in ("1", "2"):
                x, y, bw, bh = map(float, v[:4]); bx.append(((x + bw / 2) / w, (y + bh / 2) / h, bw / w, bh / h))
        items.append((f"visdrone_{p.stem.split('_')[0]}_{k:04d}", p, bx))
    return items


def nomad():
    ann = json.load(open(RAW / "NOMAD/annotations.json")); items = []
    for r in ann:
        actor, dist, _ = r["file_name"][:-4].split("_")
        if int(actor[5:]) < 96:
            continue
        p = RAW / "NOMAD/images" / actor / f"{actor}_{dist}" / r["file_name"]
        if not p.exists():
            continue
        h, w = cv2.imread(str(p)).shape[:2]
        bx = [((b["bbox"][0] + b["bbox"][2] / 2) / w, (b["bbox"][1] + b["bbox"][3] / 2) / h, b["bbox"][2] / w, b["bbox"][3] / h) for b in r["annotations"]]
        items.append((f"nomad_{actor}{dist}_{len(items):04d}", p, bx))
    return items


def yolo_set(name, imgs, labdir):
    items = []
    for k, p in enumerate(imgs):
        lp = labdir(p); bx = []
        if lp.exists():
            for l in open(lp):
                v = l.split()
                if len(v) >= 5:
                    bx.append(tuple(map(float, v[1:5])))
        items.append((f"{name}_g{k // 20}_{k:04d}", p, bx))
    return items


if __name__ == "__main__":
    write("test_visdrone", visdrone())
    write("test_nomad", nomad())
    S = RAW / "sard2/search-and-rescue-2/test"
    write("test_sard", yolo_set("sard", sorted((S / "images").glob("*")), lambda p: S / "labels" / f"{p.stem}.txt"))
    H = RAW / "HERIDAL/heridal_keras_retinanet_voc"
    hs = [H / "JPEGImages" / f"{s.strip()}.jpg" for s in open(H / "ImageSets/Main/test.txt") if s.strip()]
    write("test_heridal", yolo_set("heridal", [p for p in hs if p.exists()], lambda p: RAW / "HERIDAL/yolo_labels" / f"{p.stem}.txt"))
