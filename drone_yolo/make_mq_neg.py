"""
make_mq_neg.py — MN (11-02) M 의 도로 오탐을 줄이려 같은 촬영장의 빈 창을 더함 · 판정 = _학습 큐 「11-02 MN」

  원본  Archangel-Mannequin 학습 회차 (1 · 10 · 3 · 4 · 5 — M 과 같음 · 평가 회차 2 · 11 · 6 은 안 씀)
  빈 창 1초 표본 프레임에서 1280×720 창을 무작위로 골라 **어떤 마네킹 박스 (보임 · 가려짐 모두) 와도 40 px 여백 안에서 안 겹치는 것**만
        → 차량 · 말뚝 · 원뿔 · 풀밭 · 기체 다리 (사람처럼 작은 물체) · 라벨 빈 파일 · 영상당 ≤ 4 · 개수 = M 창과 같게 (811)
  출력  data/det_v12/{images,labels}/train_mqneg/ · configs/lists/train_v12mqn.txt (= train_v12mq + 빈 창) · train_v12mqn92.txt · data yaml 2개
실행  python make_mq_neg.py [--check 그림경로]
"""
import argparse, glob, json, random, re
from pathlib import Path
import cv2, numpy as np

BASE = Path(__file__).resolve().parent
RAW = BASE.parent / "data" / "raw" / "archangel" / "mannequin"
OUT = BASE / "data" / "det_v12"
TRAIN_TRIALS = {"1", "10", "3", "4", "5"}
CW, CH, PAD = 1280, 720, 40


def main(check=None, seed=0):
    rng = random.Random(seed)
    n = len(list((OUT / "images" / "train_mq").glob("*.jpg")))
    vids = [v for v in sorted(glob.glob(f"{RAW}/*/*.mp4")) if re.search(r"Trial-(\d+)_", v).group(1) in TRAIN_TRIALS and "m15-20" not in v]
    rng.shuffle(vids)
    (OUT / "images" / "train_mqneg").mkdir(parents=True, exist_ok=True); (OUT / "labels" / "train_mqneg").mkdir(parents=True, exist_ok=True)
    made, samples = [], []
    for v in vids:
        if len(made) >= n:
            break
        ann = json.load(open(v[:-4] + ".json")); cap = cv2.VideoCapture(v); k = -1; got = 0
        while got < 4 and len(made) < n:
            ok, img = cap.read(); k += 1
            if not ok or k >= len(ann):
                break
            if k % 30 or rng.random() < 0.5:
                continue
            boxes = [(o["x"], o["y"], o["x"] + o["width"], o["y"] + o["height"]) for o in (ann[k] or []) if "mannequin" in o.get("category", "")]
            for _ in range(30):
                x0, y0 = rng.randint(0, 1920 - CW), rng.randint(0, 1080 - CH)
                if all(b[2] < x0 - PAD or b[0] > x0 + CW + PAD or b[3] < y0 - PAD or b[1] > y0 + CH + PAD for b in boxes):
                    stem = f"mqneg_{Path(v).stem.replace('-', '_')}_{k:03d}"
                    crop = img[y0:y0 + CH, x0:x0 + CW]
                    cv2.imwrite(str(OUT / "images" / "train_mqneg" / f"{stem}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    (OUT / "labels" / "train_mqneg" / f"{stem}.txt").write_text("")
                    made.append(str(OUT / "images" / "train_mqneg" / f"{stem}.jpg")); got += 1
                    if len(samples) < 8:
                        samples.append(cv2.resize(crop, (640, 360)))
                    break
        cap.release()
    L = BASE / "configs" / "lists"
    for base, tag in (("train_v12mq", "v12mqn"), ("train_v12mq92", "v12mqn92")):
        src = [l.strip() for l in open(L / f"{base}.txt") if l.strip()]
        (L / f"train_{tag}.txt").write_text("\n".join(src + made) + "\n")
        (BASE / "configs" / f"data_{tag}.yaml").write_text(f"# {tag} (11-02 MN) — {base} (마네킹 창 포함) + 같은 촬영장 빈 창 {len(made)} · val = val_v12\n"
                                                         f"train: {L}/train_{tag}.txt\nval: {L}/val_v12.txt\nnc: 1\nnames: ['person']\n")
    if check and samples:
        while len(samples) % 4:
            samples.append(np.zeros_like(samples[0]))
        cv2.imwrite(check, np.vstack([np.hstack(samples[i:i + 4]) for i in range(0, len(samples), 4)]))
    print("빈 창", len(made), "/ 목표", n)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--check"); a = ap.parse_args()
    main(a.check)
