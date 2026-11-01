"""
make_mq_data.py — M (11-01) 누운 마네킹 실영상 창을 v12 학습에 더하기 · 판정 = _학습 큐 「11-01 M」

  원본  data/raw/archangel/mannequin (Archangel-Mannequin · 실제 드론 · 45° · 원 궤도 · 1920×1080 · v12 에 안 씀)
  나누기 (회차 = 장소) 학습: Trial 1 · 10 (15 m) · 3 (15~20 m) · 4 (20 m) · 5 (25~30 m)
                     평가 전용: Trial 2 · 11 (20~25 m) · 6 (30~35 m)   ·   40~50 m (7 · 8 · 9) 는 안 씀 (운용 밖)
  창    영상마다 1초 (30 프레임) 표본 중 「보이는 누운 마네킹」 이 있는 프레임 · 그 누움 하나를 고르고 ±240 px 흔들어 1280×720 창 (확대 없음)
        라벨 = 창 안 **보이는 (visibility=visible) 마네킹 전부** (누움 · 무릎 · 서기 → 사람) · 창에 60 % 넘게 걸친 것만 · 긴 변 < 8 px 뺌
        가려진 (hidden) 마네킹은 화면에 안 보이니 라벨 안 함 · 차량은 사람 아님
  예산  창 1,500 · 영상당 ≤ 4
  출력  data/det_v12/{images,labels}/train_mq/ · configs/lists/train_v12mq.txt (= train_v12_f91 + 창) · train_v12mq92.txt (= train_v12_f92 + 같은 창) · data yaml 2개
실행  python make_mq_data.py [--n 1500] [--check 그림경로]
"""
import argparse, glob, json, random, re
from pathlib import Path
import cv2, numpy as np

BASE = Path(__file__).resolve().parent
RAW = BASE.parent / "data" / "raw" / "archangel" / "mannequin"
OUT = BASE / "data" / "det_v12"
TRAIN_TRIALS = {"1", "10", "3", "4", "5"}
CW, CH = 1280, 720


def main(n=1500, check=None, seed=0):
    rng = random.Random(seed)
    vids = [v for v in sorted(glob.glob(f"{RAW}/*/*.mp4")) if re.search(r"Trial-(\d+)_", v).group(1) in TRAIN_TRIALS and "m15-20" not in v]
    rng.shuffle(vids)
    per = max(1, -(-n // len(vids)))                        # 영상당 몫 (올림)
    per = min(per, 4)
    (OUT / "images" / "train_mq").mkdir(parents=True, exist_ok=True); (OUT / "labels" / "train_mq").mkdir(parents=True, exist_ok=True)
    made, samples = [], []
    for v in vids:
        if len(made) >= n:
            break
        ann = json.load(open(v[:-4] + ".json")); cap = cv2.VideoCapture(v); k = -1; cand = []
        while True:
            ok, img = cap.read(); k += 1
            if not ok or k >= len(ann):
                break
            if k % 30:
                continue
            obs = [o for o in (ann[k] or []) if "mannequin" in o.get("category", "") and o.get("visibility") == "visible"]
            ly = [o for o in obs if "lying" in o["category"]]
            if ly:
                cand.append((k, img, obs, ly))
        cap.release()
        rng.shuffle(cand)
        for k, img, obs, ly in cand[:per]:
            o = rng.choice(ly); cx = o["x"] + o["width"] / 2 + rng.uniform(-240, 240); cy = o["y"] + o["height"] / 2 + rng.uniform(-240, 240)
            x0 = int(min(max(cx - CW / 2, 0), 1920 - CW)); y0 = int(min(max(cy - CH / 2, 0), 1080 - CH))
            rows = []
            for b in obs:
                x1, y1, x2, y2 = b["x"], b["y"], b["x"] + b["width"], b["y"] + b["height"]
                ix1, iy1, ix2, iy2 = max(x1, x0), max(y1, y0), min(x2, x0 + CW), min(y2, y0 + CH)
                if ix2 <= ix1 or iy2 <= iy1 or (ix2 - ix1) * (iy2 - iy1) < 0.6 * (x2 - x1) * (y2 - y1) or max(ix2 - ix1, iy2 - iy1) < 8:
                    continue
                rows.append(f"0 {((ix1 + ix2) / 2 - x0) / CW:.6f} {((iy1 + iy2) / 2 - y0) / CH:.6f} {(ix2 - ix1) / CW:.6f} {(iy2 - iy1) / CH:.6f}")
            if not rows:
                continue
            stem = f"mq_{Path(v).stem.replace('-', '_')}_{k:03d}"
            crop = img[y0:y0 + CH, x0:x0 + CW]
            cv2.imwrite(str(OUT / "images" / "train_mq" / f"{stem}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            (OUT / "labels" / "train_mq" / f"{stem}.txt").write_text("\n".join(rows) + "\n")
            made.append(str(OUT / "images" / "train_mq" / f"{stem}.jpg"))
            if len(samples) < 8:
                vis = crop.copy()
                for r in rows:
                    _, a, b, c, d = map(float, r.split()); cv2.rectangle(vis, (int((a - c / 2) * CW), int((b - d / 2) * CH)), (int((a + c / 2) * CW), int((b + d / 2) * CH)), (0, 0, 255), 2)
                samples.append(cv2.resize(vis, (640, 360)))
            if len(made) >= n:
                break
    L = BASE / "configs" / "lists"
    for base, tag in (("train_v12_f91", "v12mq"), ("train_v12_f92", "v12mq92")):
        src = [l.strip() for l in open(L / f"{base}.txt") if l.strip()]
        (L / f"train_{tag}.txt").write_text("\n".join(src + made) + "\n")
        (BASE / "configs" / f"data_{tag}.yaml").write_text(f"# {tag} (11-01 M) — {base} + 누운 마네킹 실영상 창 {len(made)} (Archangel-Mannequin 학습 회차) · val = val_v12\n"
                                                         f"train: {L}/train_{tag}.txt\nval: {L}/val_v12.txt\nnc: 1\nnames: ['person']\n")
    if check and samples:
        while len(samples) % 4:
            samples.append(np.zeros_like(samples[0]))
        cv2.imwrite(check, np.vstack([np.hstack(samples[i:i + 4]) for i in range(0, len(samples), 4)]))
    print("창", len(made), "· 영상", len(vids), "· 영상당 ≤", per)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=1500); ap.add_argument("--check"); a = ap.parse_args()
    main(a.n, a.check)
