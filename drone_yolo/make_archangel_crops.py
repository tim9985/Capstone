"""
make_archangel_crops.py — Archangel-Real (실제 사람 · 고도 15~50 m · 원 궤도) 자세 크롭 (A1 · 2026-10-07)

  원본  data/raw/archangel/Archangel_{15..50}m.zip · Archangel_moving.zip (영상 1304×978 · 10 fps) · Archangel_annotations.zip (Anno.Ai JSON)
  라벨  objectOccurrences[].labels[0].name — person- standing · walking · kneeling · lying down · crawling (vehicle 제외)
        tracks[].annotations[] = position (클립 안 프레임 · 0부터) + xMin·yMin·xMax·yMax (영상 px)
        클립 이름 <영상>-<a>-<b>.mp4 → 영상 프레임 = a − 1 + position
  영상 이름 AA_BP_<회차>_<X>_<고도 m>_<반경 m>_<날짜>_<시각>_EO — X 는 22.5 · 45 (뜻 미확인 · 메타데이터로만 남김)
  크롭  make_posture_crops.crop 과 같은 규칙 (정사각 · 긴 변 × 1.5 · 128 px) · 1080p 환산 = 1080 / 978
        사람마다 0.5 초 (5 프레임) 간격 · 화면 가장자리에 잘린 박스 (변이 2 px 안) 뺌
  자세  standing · walking → standing · lying down → lying · kneeling → kneeling · crawling → crawling (3자세 평가는 lying vs standing 중심)
  출력  data/pose_eval/archangel_real/<자세>/*.jpg + crops.csv
        (file · source · place=영상 · view=oblique · alt · radius · x_code · pose · label · track · frame · frame_n · w1080 · h1080 · x_c · y_bottom)
  ⚠ 원본 파생물 — git 에 올리지 않는다 (data/ 는 gitignore)

실행: /home/se/miniconda3/envs/drone/bin/python make_archangel_crops.py [--limit=영상 수]
"""
import csv
import json
import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import cv2

from make_posture_crops import OUT_EVAL, crop

RAW = Path("/home/se/JupyterLAB/Capstone/data/raw/archangel")
VID = RAW / "real"
OUT = OUT_EVAL / "archangel_real"
STEP = 5
H_SRC = 978
TO1080 = 1080 / H_SRC
POSE = {"person- standing": ("standing", "standing"), "person- walking": ("standing", "walking"),
        "person- lying down": ("lying", "lying"), "person- kneeling": ("kneeling", "kneeling"),
        "person- crawling": ("crawling", "crawling")}
NAME = re.compile(r"AA_BP_(\d+)_([\d-]+)_(\d+)_(\d+)_(?:(moving)_)?\d+_\d+_EO")   # 움직이는 대상 영상은 _moving_ 이 붙는다


def load_ann():
    jobs = defaultdict(list)            # 영상 이름 → [(영상 프레임, 박스, 자세, 라벨, 추적)]
    with zipfile.ZipFile(RAW / "Archangel_annotations.zip") as z:
        for n in z.namelist():
            if not n.endswith(".json"):
                continue
            d = json.loads(z.read(n))
            for v in d["data"]:
                m = re.match(r"(.+_EO)-(\d+)-(\d+)\.mp4$", v["name"])
                vid, a = m.group(1), int(m.group(2))
                for o in v["objectOccurrences"]:
                    lab = o["labels"][0]["name"] if o["labels"] else o["name"]
                    if lab not in POSE:
                        continue
                    for t in o["tracks"]:
                        for k, an in enumerate(sorted(t["annotations"], key=lambda x: x["position"])):
                            if k % STEP:
                                continue
                            c = an["coords"]
                            jobs[vid].append((a - 1 + an["position"], (c["xMin"], c["yMin"], c["xMax"], c["yMax"]),
                                              POSE[lab][0], POSE[lab][1], o["id"][:8]))
    return jobs


def unzip_videos(names):
    """시스템 unzip 으로 푼다 — Archangel_30m.zip 은 Deflate64 라 파이썬 zipfile 이 못 푼다"""
    import subprocess
    VID.mkdir(parents=True, exist_ok=True)
    for z in sorted(RAW.glob("Archangel_*m.zip")) + [RAW / "Archangel_moving.zip"]:
        if "mannequin" in z.name:
            continue
        subprocess.run(["unzip", "-n", "-j", "-q", str(z), "*.mp4", "-d", str(VID)], check=False)


def main():
    limit = next((int(a.split("=")[1]) for a in sys.argv[1:] if a.startswith("--limit=")), None)
    jobs = load_ann()
    names = sorted(jobs)[:limit] if limit else sorted(jobs)
    unzip_videos(names)
    rows, cnt, miss = [], Counter(), []
    for vid in names:
        p = VID / f"{vid}.mp4"
        if not p.exists():
            miss.append(vid); continue
        m = NAME.match(vid)
        x_code, alt, radius = m.group(2).replace("-", "."), int(m.group(3)), int(m.group(4))
        by_f = defaultdict(list)
        for j in jobs[vid]:
            by_f[j[0]].append(j)
        cap = cv2.VideoCapture(str(p)); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); W, H = cap.get(3), cap.get(4)
        fi, last = 0, max(by_f)
        while fi <= last:
            ok, img = cap.read()
            if not ok:
                break
            for (_, (x1, y1, x2, y2), pose, lab, tr) in by_f.get(fi, []):
                if x2 - x1 < 3 or y2 - y1 < 3 or x1 <= 2 or y1 <= 2 or x2 >= W - 2 or y2 >= H - 2:
                    cnt["edge"] += 1; continue
                name = f"{vid}_{fi:05d}_{tr}.jpg"
                (OUT / pose).mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(OUT / pose / name), crop(img, (x1, y1, x2, y2), TO1080), [cv2.IMWRITE_JPEG_QUALITY, 95])
                rows.append({"file": f"{pose}/{name}", "source": "archangel", "place": vid, "view": "oblique", "alt": alt,
                             "radius": radius, "x_code": x_code, "moving": int(bool(m.group(5))), "pose": pose, "label": lab, "track": tr, "frame": fi,
                             "frame_n": n, "w1080": round((x2 - x1) * TO1080, 1), "h1080": round((y2 - y1) * TO1080, 1),
                             "x_c": round((x1 + x2) / 2 / W, 4), "y_bottom": round(y2 / H, 4)})
                cnt[pose] += 1
            fi += 1
        cap.release()
        print(f"{vid}: 누적 {dict(cnt)}", flush=True)
    with open(OUT / "crops.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"끝 — 크롭 {len(rows):,} · {dict(cnt)} · 영상 {len(names) - len(miss)} (없음 {len(miss)})")


if __name__ == "__main__":
    main()
