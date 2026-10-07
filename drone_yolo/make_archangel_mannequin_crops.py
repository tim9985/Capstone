"""
make_archangel_mannequin_crops.py — Archangel-Mannequin 자세 크롭 (10-07 · K2 다음)

  원본  data/raw/archangel/Archangel_mannequin_*.zip (11개 · 598 조각) — 조각마다 mp4 (1920×1080 · 30 fps · 10 초 = 300 프레임) + 같은 이름 JSON
        JSON = 프레임마다 박스 목록 [{id, category, visibility, annotated_by, x, y, width, height}] (300개)
        category: mannequin - standing · kneeling · lying down · vehicle (person - … 도 있으면 같이)
        이름 DTRA_Trial-<n>_CIR_VIS_<고도>m_<각>deg_cam_10sec-<시작 프레임>
  규칙  visibility = visible 만 · 1 초 (30 프레임) 간격 · 가장자리에 걸린 박스 뺌 · 크롭 = make_posture_crops.crop (1080p 그대로)
        자세 standing → standing · lying down → lying · kneeling → kneeling
  출력  data/pose_eval/archangel_mannequin/<자세>/*.jpg + crops.csv (file · source · place=Trial · seg · frame · alt_tag · cam_deg · pose · track · by · w1080 · h1080 · y_bottom)
  ⚠ 파생물 — git 에 안 올림

실행: /home/se/miniconda3/envs/drone/bin/python make_archangel_mannequin_crops.py
"""
import csv
import json
import re
import subprocess
from collections import Counter
from pathlib import Path

import cv2

from make_posture_crops import OUT_EVAL, crop

RAW = Path("/home/se/JupyterLAB/Capstone/data/raw/archangel")
VID = RAW / "mannequin"
OUT = OUT_EVAL / "archangel_mannequin"
STEP = 30
POSE = {"standing": "standing", "lying down": "lying", "kneeling": "kneeling"}
NAME = re.compile(r"DTRA_Trial-(\d+)_CIR_VIS_([\d-]+)m_(\d+)deg_cam_10sec-(\d+)")


def main():
    for z in sorted(RAW.glob("Archangel_mannequin_*.zip")):
        d = VID / z.stem
        if not d.exists():
            subprocess.run(["unzip", "-n", "-j", "-q", str(z), "-d", str(d)], check=False)
    rows, cnt = [], Counter()
    for js in sorted(VID.glob("*/*.json")):
        mp4 = js.with_suffix(".mp4")
        m = NAME.search(js.stem)
        if not mp4.exists() or not m:
            cnt["짝 없음"] += 1; continue
        frames = json.load(open(js))
        cap = cv2.VideoCapture(str(mp4)); W, H = cap.get(3), cap.get(4)
        for fi in range(len(frames)):
            ok, img = cap.read()
            if not ok:
                break
            if fi % STEP:
                continue
            for b in frames[fi]:
                cat = b["category"].split(" - ")[-1].strip()
                if b["visibility"] != "visible" or cat not in POSE:
                    continue
                x1, y1, x2, y2 = b["x"], b["y"], b["x"] + b["width"], b["y"] + b["height"]
                if b["width"] < 3 or b["height"] < 3 or x1 <= 2 or y1 <= 2 or x2 >= W - 2 or y2 >= H - 2:
                    cnt["edge"] += 1; continue
                pose = POSE[cat]
                name = f"{js.stem}_{fi:03d}_{b['id'][:8]}.jpg"
                (OUT / pose).mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(OUT / pose / name), crop(img, (x1, y1, x2, y2), 1080 / H), [cv2.IMWRITE_JPEG_QUALITY, 95])
                rows.append({"file": f"{pose}/{name}", "source": "archangel_mannequin", "place": f"Trial-{m.group(1)}", "seg": js.stem,
                             "frame": int(m.group(4)) + fi, "alt_tag": m.group(2), "cam_deg": m.group(3), "pose": pose, "track": b["id"][:8],
                             "by": b["annotated_by"], "w1080": round(b["width"] * 1080 / H, 1), "h1080": round(b["height"] * 1080 / H, 1),
                             "y_bottom": round(y2 / H, 4)})
                cnt[pose] += 1
        cap.release()
    with open(OUT / "crops.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"끝 — 크롭 {len(rows):,} · {dict(cnt)}")


if __name__ == "__main__":
    main()
