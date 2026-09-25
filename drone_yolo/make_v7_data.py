"""
make_v7_data.py — v7 데이터 (2026-09-26)

  v6_1 (v6 + SARD + 비스듬 음성 · val_v6b) 에서
  + HERIDAL 전부 (크로아티아·보스니아 야생 지형 · 수직 · 4000×3000 · CC BY 3.0) → 사람 31~94 px 크롭
  − NOMAD (농장 한 곳) — 수직 쪽 장소 쏠림을 줄이고 비스듬 비율을 유지하려고 HERIDAL 로 바꾼다
  근거: v6_obl 진단 — test_v2(숲) 아예 못 봄 21 % · 누운 0.32 · 앉은 0.50

출력  data/raw/HERIDAL/yolo_labels/ · data/det_v6/heridal/ · configs/lists/train_v7.txt · configs/data_v7.yaml
      metrics/v7_data_stats.json
"""
import json, zlib, random
import xml.etree.ElementTree as ET
from multiprocessing import Pool
from pathlib import Path

from make_v6_data import crop_one

BASE = Path(__file__).resolve().parent
D = BASE / "data"
H = D / "raw" / "HERIDAL" / "heridal_keras_retinanet_voc"
YL = D / "raw" / "HERIDAL" / "yolo_labels"
OUT = D / "det_v6" / "heridal" / "images" / "train"
LISTS = BASE / "configs" / "lists"


def voc_to_yolo(xml_path):
    r = ET.parse(xml_path).getroot()
    W, Hh = 4000, 3000
    sz = r.find("size")
    if sz is not None and sz.find("width") is not None:
        W, Hh = int(float(sz.find("width").text)), int(float(sz.find("height").text))
    rows = []
    for o in r.findall("object"):
        if o.find("name").text.strip().lower() != "person":
            continue
        b = o.find("bndbox")
        x1, x2 = float(b.find("xmin").text), float(b.find("xmax").text)
        y1, y2 = float(b.find("ymin").text), float(b.find("ymax").text)
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        rows.append(f"0 {(x1+x2)/2/W:.6f} {(y1+y2)/2/Hh:.6f} {(x2-x1)/W:.6f} {(y2-y1)/Hh:.6f}")
    return rows


def main():
    YL.mkdir(parents=True, exist_ok=True)
    jobs, n_xml, n_box = [], 0, 0
    for img in sorted((H / "JPEGImages").glob("*.jpg")):
        x = H / "Annotations" / f"{img.stem}.xml"
        if not x.exists():
            continue
        rows = voc_to_yolo(x); n_xml += 1; n_box += len(rows)
        if not rows:
            continue
        lab = YL / f"{img.stem}.txt"; lab.write_text("\n".join(rows) + "\n")
        jobs.append((str(img), str(lab), None, str(OUT / f"{img.stem}.jpg"), zlib.crc32(img.name.encode())))
    with Pool(6) as pool:
        her = [r[0] for r in pool.imap_unordered(crop_one, jobs, chunksize=8) if r]
    v61 = [l.strip() for l in open(LISTS / "train_v6_1.txt") if l.strip()]
    nomad = [p for p in v61 if "/nomad_" in p]
    train = [p for p in v61 if "/nomad_" not in p] + her
    random.Random(7).shuffle(train)
    (LISTS / "train_v7.txt").write_text("\n".join(train) + "\n")
    (BASE / "configs" / "data_v7.yaml").write_text(
        "# v7 (2026-09-26) — v6_1 + HERIDAL(야생 지형) − NOMAD · val = val_v6b\n"
        f"train: {LISTS / 'train_v7.txt'}\nval: {LISTS / 'val_v6b.txt'}\nnc: 1\nnames: ['person']\n")
    obl = sum(1 for p in train if "/unicamp/" in p or "/visdrone/" in p or ("(1-" in p and "2022" in p))
    neg = sum(1 for p in train if "/det_neg/" in p or "/uc_neg/" in p)
    stats = {"HERIDAL_라벨": n_xml, "HERIDAL_사람": n_box, "HERIDAL_크롭": len(her), "NOMAD_뺌": len(nomad),
             "train": len(train), "음성": neg, "비스듬_비율(양성)": round(obl / (len(train) - neg), 3)}
    (BASE / "metrics" / "v7_data_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1))
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
