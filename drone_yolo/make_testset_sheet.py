"""
make_testset_sheet.py — 화각 시험셋 눈으로 확인용 모음판 (2026-09-16)

칸 하나에서 이미지를 골라 정답 박스를 그려 붙인다.
  초록 = 채점 대상 (원본 칸 · 이음새 아님 · 기준 크기 ±30 %)   · 박스 위 숫자는 NOMAD 가시도
  회색 = 채점 제외 (거울 복제 칸 · 이음새 · 기준 크기 밖)
"서 있음 / 누움" 은 자세 라벨이 아니라 **0.5 m / 1.7 m 물체의 크기 가정**이라는 것을 눈으로 확인하는 용도.

실행: python make_testset_sheet.py --cells s20_38 l20_128 --out /tmp/sheets
"""
import argparse, json, random
from pathlib import Path
import cv2, numpy as np

BASE = Path(__file__).resolve().parent


def tile(p, w=640, h=360):
    img = cv2.imread(str(p))
    H, W = img.shape[:2]
    meta = json.loads(Path(str(p).replace("/images/", "/meta/")).with_suffix(".json").read_text())
    lines = Path(str(p).replace("/images/", "/labels/")).with_suffix(".txt").read_text().splitlines()
    for ln, m in zip(lines, meta):
        _, cx, cy, bw, bh = map(float, ln.split())
        x1, y1 = int((cx - bw / 2) * W), int((cy - bh / 2) * H)
        x2, y2 = int((cx + bw / 2) * W), int((cy + bh / 2) * H)
        scored = m["orig"] and not m["seam"] and m["near_ref"]
        cv2.rectangle(img, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), (0, 255, 0) if scored else (150, 150, 150), 3 if scored else 1)
        if scored and m["vis"] >= 0:
            cv2.putText(img, str(m["vis"]), (x1 - 3, max(y1 - 8, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    t = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    cv2.putText(t, p.stem[:52], (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(BASE / "data" / "det_fov_test"))
    ap.add_argument("--cells", nargs="+", default=["s20_38", "l20_128"])
    ap.add_argument("--out", default="/tmp/sheets")
    ap.add_argument("--per-kind", type=int, default=4)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rnd = random.Random(0)
    for cell in args.cells:
        d = Path(args.root) / cell / "images"
        picks = []
        for kind in ("nomad", "wisard"):
            fs = sorted(d.glob(f"{kind}_*.jpg"))
            picks += rnd.sample(fs, min(args.per_kind, len(fs)))
        rows = [np.hstack([tile(p) for p in picks[i:i + 4]]) for i in range(0, len(picks), 4)]
        sheet = np.vstack(rows)
        cv2.putText(sheet, f"{cell}  (green=scored, gray=excluded, number=NOMAD visibility)", (10, sheet.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imwrite(str(out / f"{cell}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
        print("→", out / f"{cell}.jpg")


if __name__ == "__main__":
    main()
