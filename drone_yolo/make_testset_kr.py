"""
make_testset_kr.py — test_kr: 한국 장소 분리 평가셋 (2026-09-26)

  AI-Hub 182 산악5 (화성26 · 경기 화성 비봉면 · 2021-01-04 오후 · 겨울 산 비탈 · 수직 90° · 고도 10~40 m)
  · 학습(저수지2·평지(흙)3·산악6 · v8 추가 산악2·3·4·8·평지(수풀)1)·val(저수지1) 어디에도 없는 장소
  · 원본 4K → 1920×1080 축소 (배포 입력과 같게) · 고도별 균등 표집 최대 1,000장 · 자세 메타 보존
  평가: python eval_test_v2.py --data data/test_kr --tag test_kr --weights …   (타일 4장 · test_v2 와 같은 방식)
  ⚠ AI-Hub 는 가림이 적어 쉬운 편 — 한국 지형 확인용이지 NFR-V03 단독 판정용이 아니다
"""
import json, random
from collections import defaultdict
from pathlib import Path
import cv2

from aihub_prep import boxes_from_json

BASE = Path(__file__).resolve().parent
RAW = Path("/home/se/JupyterLAB/Capstone/data/raw/AIHub182")
LAB = RAW / "064.드론_이동체_인지_영상(도로_고정)/01.데이터/2.Validation/라벨링데이터/03_survivor/조난자_화성26"
OUT = BASE / "data" / "test_kr"
N_MAX = 1000


def main():
    labs = {p.stem: p for p in LAB.rglob("*.json")}
    imgs = [p for p in (RAW / "조난자_화성26").rglob("*.jpg") if p.stem in labs and "산악5" in str(p)]
    by_alt = defaultdict(list)
    for p in imgs:
        a = next((t for t in p.stem.split("-") if t.endswith("m") and t[:-1].isdigit()), "?")
        by_alt[a].append(p)
    rng = random.Random(5); per = max(1, N_MAX // max(1, len(by_alt)))
    pick = []
    for a, v in sorted(by_alt.items()):
        v.sort(); rng.shuffle(v); pick += v[:per]
    for d in ("images", "labels", "meta"):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    npers, poses = 0, defaultdict(int)
    for p in pick:
        img = cv2.imread(str(p))
        if img is None:
            continue
        H0, W0 = img.shape[:2]
        boxes, meta = boxes_from_json(labs[p.stem])
        small = cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(OUT / "images" / f"{p.stem}.jpg"), small, [cv2.IMWRITE_JPEG_QUALITY, 95])
        rows, pl = [], []
        for b in boxes:
            x, y, w, h = b["bbox"]
            if w < 2 or h < 2:
                continue
            rows.append(f"0 {(x + w / 2) / W0:.6f} {(y + h / 2) / H0:.6f} {w / W0:.6f} {h / H0:.6f}")
            pl.append(b["pose"]); poses[b["pose"]] += 1
        (OUT / "labels" / f"{p.stem}.txt").write_text("\n".join(rows) + ("\n" if rows else ""))
        (OUT / "meta" / f"{p.stem}.json").write_text(json.dumps({"alt": meta["alt"], "poses": pl}, ensure_ascii=False))
        npers += len(rows)
    man = {"목적": "한국 장소 분리 평가 (AI-Hub 182 산악5 · 학습 미사용)", "장수": len(pick), "사람": npers,
           "고도별": {a: min(len(v), per) for a, v in sorted(by_alt.items())}, "자세": dict(poses)}
    (OUT / "manifest.json").write_text(json.dumps(man, ensure_ascii=False, indent=1))
    print(json.dumps(man, ensure_ascii=False))


if __name__ == "__main__":
    main()
