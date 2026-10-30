"""
make_testset_sar.py — 외부 평가셋 UAV-SAR (10-31 · E · 평가 전용 · 학습 안 씀)

  원본  data/raw/uav_sar (download_uavsar.py · Roboflow neptunet-bewas/uav-sar-human-detection-dataset v1 · CC BY 4.0)
        DJI Air 3S · 1920×1080 · 15 · 30 m × 45° · 90° · 조건마다 75장 · 사람 663 (라벨 대부분 다각형 → 감싸는 박스)
        의자 · 가방은 라벨 없음 (오탐 시험) · 자세 라벨 없음
  출력  data/test_sar   (300장 · 조건 4개)      — 이름 uavsar_<조건>_<프레임 4자리> (eval_test_v2 블록 = 조건 안 연속 20장)
        data/test_sar45 (150장 · 45° 만 · 운용 각도)
  ⚠ 원본 · 파생물은 git 에 올리지 않음 (data/ 무시) · 출처 표기 필요 (CC BY 4.0)
실행  python make_testset_sar.py
"""
import json, re, shutil
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parent
RAW = BASE.parent / "data" / "raw" / "uav_sar" / "test"
CITE = "UAV-SAR-Human-Detection-Dataset v1 (Roboflow Universe · neptunet-bewas · CC BY 4.0)"


def box(vals):
    if len(vals) == 4:
        cx, cy, w, h = vals
        x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
    else:
        xs, ys = vals[0::2], vals[1::2]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    x1, y1, x2, y2 = (min(max(v, 0.0), 1.0) for v in (x1, y1, x2, y2))
    return (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1


def build(name, keep):
    out = BASE / "data" / name
    if out.exists():
        shutil.rmtree(out)
    (out / "images").mkdir(parents=True); (out / "labels").mkdir()
    cnt, ppl = Counter(), Counter()
    for img in sorted(RAW.glob("images/*.jpg")):
        m = re.match(r"(\d+m\d+d)_frame(\d+)_jpg", img.name)
        cond, fr = m.group(1), int(m.group(2))
        if not keep(cond):
            continue
        stem = f"uavsar_{cond}_{fr:04d}"
        shutil.copy2(img, out / "images" / f"{stem}.jpg")
        rows = []
        for l in open(RAW / "labels" / (img.stem + ".txt")):
            v = [float(x) for x in l.split()[1:]]
            if v:
                cx, cy, w, h = box(v)
                if w > 0 and h > 0:
                    rows.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        (out / "labels" / f"{stem}.txt").write_text("\n".join(rows) + ("\n" if rows else ""))
        cnt[cond] += 1; ppl[cond] += len(rows)
    man = {"목적": "외부 평가셋 (우리 조건과 가장 비슷한 실사진 · 학습 미사용)", "출처": CITE, "장수": sum(cnt.values()), "사람": sum(ppl.values()),
           "조건별 장": dict(cnt), "조건별 사람": dict(ppl), "메모": "의자 · 가방 라벨 없음 (오탐 시험) · 자세 라벨 없음 · 다각형 → 감싸는 박스"}
    (out / "manifest.json").write_text(json.dumps(man, ensure_ascii=False, indent=1))
    print(name, man["장수"], "장 · 사람", man["사람"], dict(cnt))


if __name__ == "__main__":
    build("test_sar", lambda c: True)
    build("test_sar45", lambda c: c.endswith("45d"))
