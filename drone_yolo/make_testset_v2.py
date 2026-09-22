"""
make_testset_v2.py — NFR-V03 용 **장소 분리** 시험셋 (2026-09-22)

왜
  NFR-V03 은 "지정한 평가셋(학습과 장소 분리)에서 사람 AP50 ≥ 0.80" 인데
  **지금 그 평가셋이 없다.** 지금 쓰는 숫자는 셋마다 0.21~0.995 로 흔들린다.
  · det_fov_test_budget : 어렵다(가림) · 장소 분리 ✗ (NOMAD val 배우 + WiSARD val 비행)
  · det_aihub val       : 장소 분리 ✓ · 하지만 **너무 쉽다** (AP50 0.9950 · 재현율 1.0000)
  → 둘(어렵다 + 장소 분리)을 동시에 만족하는 셋을 만든다 → obsidian/11 서버 학습 계획/7 평가 프로토콜

어떻게
  WiSARD 를 **장소 단위**로 가른다. Carnation · Karen 을 **평가 전용**으로 빼고,
  학습에서는 그 장소 크롭을 목록(.txt)에서 제외한다 — **크롭 재생성이 필요 없다.**

  장소 선정 근거 (실측 · 사람 긴 변 px 5/50/95)
    Carnation 16 / 46 /  97   · 1920×1080 네이티브 · 2,054 프레임
    Karen     19 / 47 / 200   · 2704×1520, 3840×2160 → 1920×1080 으로 축소
  둘 다 우리 운용 구간(18~50 px)에 걸친다. 나머지 장소는 사람이 너무 크거나(SuddenValley 180)
  음성이 과반(FHL 65 %)이라 평가셋으로 부적합하다.

출력
  data/test_v2/{images,labels,meta}/   — 1920×1080 전체 프레임 (배포 파이프라인과 같은 입력)
  configs/data_v3_place.yaml           — 두 장소를 뺀 학습 목록
"""
import argparse, json, re
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

BASE = Path(__file__).resolve().parent
OUT_W, OUT_H = 1920, 1080
EVAL_PLACES = ("Carnation", "Karen")       # 평가 전용 — 학습에서 뺀다
MIN_BOX_PX = 4


def place_of(name: str) -> str:
    p = name.split("_")
    return p[1] if len(p) > 1 else name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(BASE / "data" / "raw" / "WiSARD"))
    ap.add_argument("--out", default=str(BASE / "data" / "test_v2"))
    ap.add_argument("--neg-keep", type=float, default=0.15,
                    help="음성 프레임을 이 비율만큼 남긴다 (NFR-V06 오탐 측정용)")
    ap.add_argument("--step", type=int, default=1, help="N 프레임마다 1장 (연속 프레임 상관 줄이기)")
    args = ap.parse_args()

    raw, out = Path(args.raw), Path(args.out)
    for sub in ("images", "labels", "meta"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    frames = []
    for d in sorted(raw.iterdir()):
        if not d.is_dir() or place_of(d.name) not in EVAL_PLACES:
            continue
        imgs = sorted(list(d.rglob("*.jpg")) + list(d.rglob("*.jpeg")))
        frames += [(place_of(d.name), d.name, p) for p in imgs]
    print(f"평가 전용 장소 {EVAL_PLACES} — 원본 프레임 {len(frames):,}", flush=True)

    st = Counter(); px_all = []; per_place = defaultdict(Counter)
    neg_seen = 0
    for i, (place, flight, p) in enumerate(frames):
        if i % args.step:
            continue
        t = p.with_suffix(".txt")
        rows = [l.split() for l in t.read_text().splitlines() if l.strip()] if t.exists() else []
        if not rows:
            neg_seen += 1
            if (neg_seen % max(1, int(1 / args.neg_keep))):    # 음성은 일부만
                st["음성건너뜀"] += 1; continue
        img = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            st["읽기실패"] += 1; continue
        H, W = img.shape[:2]
        # 16:9 로 가정하고 곧장 축소 — YOLO 정규화 좌표는 해상도에 무관하므로 라벨은 그대로다
        rs = cv2.resize(img, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)

        lines, meta = [], []
        for r in rows:
            cx, cy, w, h = (float(v) for v in r[1:5])
            bw, bh = w * OUT_W, h * OUT_H
            if bw < MIN_BOX_PX or bh < MIN_BOX_PX:
                st["작은박스제외"] += 1; continue
            lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            meta.append({"px": round(max(bw, bh), 1)})
            px_all.append(max(bw, bh))
        stem = re.sub(r"[^\w.-]", "_", p.stem)
        cv2.imencode(".jpg", rs, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tofile(str(out / "images" / f"{stem}.jpg"))
        (out / "labels" / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        (out / "meta" / f"{stem}.json").write_text(json.dumps(
            {"place": place, "flight": flight, "src_wh": [W, H], "boxes": meta}, ensure_ascii=False))
        st["장" if lines else "음성장"] += 1
        per_place[place]["장" if lines else "음성장"] += 1
        per_place[place]["박스"] += len(lines)

    a = np.array(px_all)
    print(f"\n=== test_v2 완성 → {out}")
    for k, v in sorted(st.items()):
        print(f"  {k:>12} {v:,}")
    for p, c in per_place.items():
        print(f"  {p:>12}: 양성 {c['장']:,}장 · 음성 {c['음성장']:,}장 · 사람 {c['박스']:,}")
    if len(a):
        q = np.percentile(a, [5, 25, 50, 75, 95]).round(1)
        print(f"  사람 px 분위수 5/25/50/75/95: {q}")
        print(f"  운용 구간(18~50 px) 비율: {((a >= 18) & (a <= 50)).mean():.1%}")
    (out / "manifest.json").write_text(json.dumps({
        "목적": "NFR-V03 장소 분리 평가셋",
        "평가_전용_장소": list(EVAL_PLACES),
        "입력": f"{OUT_W}x{OUT_H} 전체 프레임 (배포 파이프라인과 동일)",
        "통계": {k: v for k, v in st.items()},
        "사람_px_분위수": q.tolist() if len(a) else [],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
