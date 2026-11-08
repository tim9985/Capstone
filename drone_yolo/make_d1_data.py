"""
make_d1_data.py — D1 (11-08 준비 · 실기체 데이터로 처음부터 다시 학습) 데이터 · 판정 = _학습 큐 「11-08 D1」

  입력  비행 목록 json (기본 configs/d1_flights.json) — 비행마다 라벨 고친 프레임 폴더 · 장소
          {"F03": {"dir": "data/real_flight/1110/F03/labeled", "place": "A", "empty_ok": false}, ...}
          dir = images/*.jpg (1920×1080 원본 · 녹화 frames) + labels/*.txt (YOLO · 사람 0 · 사람이 고친 것)
          place  A · B = 학습 · **C = 평가 전용** (학습에 절대 안 들어감)
          empty_ok  true = 그 비행의 빈 라벨 프레임은 사람이 「사람 없음」 확인함 (T2 빈 장면) → 음성 창으로 씀 · false 면 빈 라벨 프레임 버림
  학습 창 (v12 규칙 그대로)  프레임 → 운용 4조각과 같은 1280×720 창 4개 (확대 없음) · 창 안 보이는 넓이 < 25 % 조각은 가림 (창 평균색) · 25 % 이상은 잘린 박스로 라벨
        프레임 간격 --gap 장 (녹화 5 fps → 기본 5 = 1초) · 비행마다 창 상한 --cap · 음성 창 ≤ 양성의 10 %
        몫 --frac (실기체 창 / 최종 목록 · 기본 0.20) — 모자라면 같은 창을 최대 --rep 번 반복해 목록에 넣음 (실제 몫 출력)
  평가셋  C 비행 프레임 → data/test_real_c/{images,labels} (1920 원본 · 이름 real_<비행>_<프레임> → 비행마다 블록 부트스트랩) · eval_test_v2.py --data data/test_real_c --tag test_real_c --merge edge
  출력  data/det_<tag>/{images,labels}/train_real · configs/lists/train_<tag>_f91.txt (= train_v12_f91 + 실기체) · _f92 · configs/data_<tag>_f91.yaml · _f92 · metrics/<tag>_data_stats.json
  ⚠ 실기체 원본 · 파생물 (얼굴) 은 git 밖 (data/) · 평가 뒤 지움
실행  python make_d1_data.py [--flights configs/d1_flights.json] [--tag d1] [--frac 0.2] [--check 그림.jpg]
"""
import argparse, json, random
from collections import Counter
from pathlib import Path

import cv2, numpy as np

BASE = Path(__file__).resolve().parent
L = BASE / "configs" / "lists"
CW, CH = 1280, 720


def read_yolo(p, W, H):
    out = []
    if p.exists():
        for l in p.read_text().split("\n"):
            v = l.split()
            if len(v) >= 5 and v[0] == "0":
                cx, cy, w, h = (float(x) for x in v[1:5]); out.append((W * (cx - w / 2), H * (cy - h / 2), W * (cx + w / 2), H * (cy + h / 2)))
    return out


def tiles(W, H):
    xs = sorted({0, max(0, W - CW)} | set(range(0, max(1, W - CW + 1), CW // 2))); ys = sorted({0, max(0, H - CH)} | set(range(0, max(1, H - CH + 1), CH // 2)))
    return [(x, y) for y in ys for x in xs]


def window(img, boxes, x0, y0):
    """창 하나 · 보이는 넓이 25 % 미만 조각은 가리고 나머지는 잘린 박스로 라벨 (make_v12_data ③)"""
    crop = img[y0:y0 + CH, x0:x0 + CW].copy(); fillc = crop.reshape(-1, 3).mean(0); rows = []
    for x1, y1, x2, y2 in boxes:
        ix1, iy1, ix2, iy2 = max(x1, x0), max(y1, y0), min(x2, x0 + CW), min(y2, y0 + CH)
        if ix2 <= ix1 or iy2 <= iy1:
            continue
        vis = (ix2 - ix1) * (iy2 - iy1) / max(1e-6, (x2 - x1) * (y2 - y1))
        if vis < 0.25 or max(ix2 - ix1, iy2 - iy1) < 6:
            crop[int(iy1 - y0):int(np.ceil(iy2 - y0)), int(ix1 - x0):int(np.ceil(ix2 - x0))] = fillc; continue
        rows.append(f"0 {((ix1 + ix2) / 2 - x0) / CW:.6f} {((iy1 + iy2) / 2 - y0) / CH:.6f} {(ix2 - ix1) / CW:.6f} {(iy2 - iy1) / CH:.6f}")
    return crop, rows


def main(a):
    rng = random.Random(0); FL = json.load(open(a.flights)); tag = a.tag
    OUT = BASE / "data" / f"det_{tag}"; TE = BASE / "data" / (a.test_dir or "test_real_c")
    for d in (OUT / "images" / "train_real", OUT / "labels" / "train_real", TE / "images", TE / "labels"):
        d.mkdir(parents=True, exist_ok=True)
    pos, neg, st, samples = [], [], Counter(), []
    for fid, f in sorted(FL.items()):
        assert "_" not in fid, f"비행 이름 {fid} 에 _ 금지 (블록 = 비행)"
        D = BASE / f["dir"] if not Path(f["dir"]).is_absolute() else Path(f["dir"]); place = f["place"].upper()
        ims = sorted((D / "images").glob("*.jpg"))[::a.gap]
        if place == "C":                                                     # 평가 전용 — 원본 그대로
            for p in ims:
                lp = D / "labels" / (p.stem + ".txt")
                if not lp.exists() or (not lp.read_text().strip() and not f.get("empty_ok")):
                    continue
                stem = f"real_{fid}_{p.stem.replace('_', '')}"
                (TE / "images" / f"{stem}.jpg").write_bytes(p.read_bytes()); (TE / "labels" / f"{stem}.txt").write_text(lp.read_text())
                st["C_frames"] += 1; st["C_people"] += len(lp.read_text().split("\n")) - 1 if lp.read_text().strip() else 0
            continue
        assert place in ("A", "B"), f"{fid} 장소 {place} (A · B · C)"
        fp, fn = [], []
        for p in ims:
            lp = D / "labels" / (p.stem + ".txt")
            if not lp.exists():
                continue
            img = cv2.imread(str(p)); H, W = img.shape[:2]; bx = read_yolo(lp, W, H)
            if not bx and not f.get("empty_ok"):
                continue                                                     # 사람이 확인 안 한 빈 프레임 = 라벨 구멍일 수 있음
            for x0, y0 in tiles(W, H):
                crop, rows = window(img, bx, x0, y0); stem = f"real_{fid}_{p.stem.replace('_', '')}_{x0}x{y0}"
                (fp if rows else fn).append((stem, crop, rows))
        rng.shuffle(fp); rng.shuffle(fn); fp = fp[:a.cap]; fn = fn[:max(1, len(fp) // 10)] if fp else fn[:a.cap // 10]
        for stem, crop, rows in fp + fn:
            ip = OUT / "images" / "train_real" / f"{stem}.jpg"; cv2.imwrite(str(ip), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            (OUT / "labels" / "train_real" / f"{stem}.txt").write_text("\n".join(rows) + ("\n" if rows else ""))
            (pos if rows else neg).append(str(ip))
            if rows and len(samples) < 8 and rng.random() < 0.3:
                vis = crop.copy()
                for r in rows:
                    _, cx, cy, w, h = map(float, r.split()); cv2.rectangle(vis, (int((cx - w / 2) * CW), int((cy - h / 2) * CH)), (int((cx + w / 2) * CW), int((cy + h / 2) * CH)), (0, 0, 255), 2)
                samples.append(cv2.resize(vis, (640, 360)))
        st[f"{place}_pos"] += len(fp); st[f"{place}_neg"] += len(fn); st[f"{place}_people"] += sum(len(r) for _, _, r in fp)
    real = pos + neg; assert real, "학습 창 0 — A · B 비행 · 라벨 확인"
    for base, sfx in (("train_v12_f91", "f91"), ("train_v12_f92", "f92")):
        src = [l.strip() for l in open(L / f"{base}.txt") if l.strip()]
        want = round(a.frac / (1 - a.frac) * len(src)); rep = min(a.rep, max(1, -(-want // len(real))))
        add = (real * rep)[:max(len(real), want)]
        (L / f"train_{tag}_{sfx}.txt").write_text("\n".join(src + add) + "\n")
        (BASE / "configs" / f"data_{tag}_{sfx}.yaml").write_text(
            f"# {tag}_{sfx} (11-08 D1) — {base} + 실기체 창 {len(real)} × ≤{rep} (A · B 장소 · C 는 평가 전용) · val = val_v12\n"
            f"train: {L}/train_{tag}_{sfx}.txt\nval: {L}/val_v12.txt\nnc: 1\nnames: ['person']\n")
        st[f"{sfx}_share"] = round(len(add) / (len(src) + len(add)), 3); st[f"{sfx}_rep"] = rep
    (TE / "manifest.json").write_text(json.dumps({"목적": "D1 판정 — 실기체 C 장소 (학습 미사용)", "입력": "1920 원본 · 4조각", "장수": st["C_frames"], "사람": st["C_people"]}, ensure_ascii=False, indent=1))
    json.dump(dict(st), open(BASE / "metrics" / f"{tag}_data_stats.json", "w"), ensure_ascii=False, indent=1)
    if a.check and samples:
        while len(samples) % 4:
            samples.append(np.zeros_like(samples[0]))
        cv2.imwrite(a.check, np.vstack([np.hstack(samples[i:i + 4]) for i in range(0, len(samples), 4)]))
    print(dict(st))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--flights", default=str(BASE / "configs" / "d1_flights.json")); ap.add_argument("--tag", default="d1")
    ap.add_argument("--test-dir", default=None, help="평가셋 폴더 이름 (기본 test_real_c)")
    ap.add_argument("--frac", type=float, default=0.20); ap.add_argument("--rep", type=int, default=3)
    ap.add_argument("--gap", type=int, default=5); ap.add_argument("--cap", type=int, default=1500); ap.add_argument("--check")
    main(ap.parse_args())
