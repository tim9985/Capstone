"""
make_v13_occ.py — v12 학습 창에 「나무 잎 가림」 합성 (10-14 O · 판정 기준 = _학습 큐 「10-14 O」)

  ① 잎 조각 은행: v12 창 중 자연 출처의 사람 박스 (여백 24 px) 와 안 겹치는 영역에서 64~192 px 정사각 조각
     조건 = 초록 · 갈색 계열 (H 15~95 · S ≥ 0.18) 화소 ≥ 55 % · 질감 (라플라시안 분산 ≥ 60) — 잔디처럼 밋밋한 면은 뺌
  ② 자연 출처 (NOMAD · WiSARD · HERIDAL · SARD · Archangel · LADD · AI-Hub) 양성 창의 30 % (이름 해시로 결정 → 반복 목록끼리 같은 창 · 같은 가림) · 키 16 px 이상 박스 1~3개
     → 박스 크기의 1.1~1.8 배 조각 (모양 비율 ≤ 0.7) · 불규칙 모양 (흐린 잡음 문턱 · 가장자리 부드럽게) · 박스 넓이의 20~60 % 를 덮도록 문턱을 맞춤
     → 밝기를 박스 주변 평균에 맞춤 (±) · 라벨 그대로
  ③ 출력 data/det_v13occ/{images,labels}/train — 바뀐 창만 새 jpg · 나머지는 심볼릭 링크 · 목록 train_v13o_f91/f92 (v12 와 같은 95 % 뽑기)
실행  python make_v13_occ.py [--frac 0.3] [--preview 24]
"""
import argparse, hashlib, json, os, random
from pathlib import Path
import cv2
import numpy as np

BASE = Path(__file__).resolve().parent
SRC = BASE / "data" / "det_v12"
OUT = BASE / "data" / "det_v13occ"
NATURE = ("nomad", "wisard", "heridal", "sard", "archangel", "ladd")      # 잎 조각을 오리는 곳
TARGET = NATURE + ("aihub",)                                            # 가림을 넣는 곳 — 도시 (VisDrone · Unicamp · NII-CU) 는 뺌 (보도 위 잎은 부자연)


def lab_path(img):
    return Path(str(img).replace("/images/", "/labels/")).with_suffix(".txt")


def read_boxes(img, w, h):
    p = lab_path(img)
    out = []
    if p.exists():
        for line in p.read_text().split("\n"):
            v = line.split()
            if len(v) == 5:
                cx, cy, bw, bh = (float(x) for x in v[1:])
                out.append(((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h))
    return out


def h01(name, salt):
    return int(hashlib.md5(f"{salt}:{name}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def bank(paths, n_target=3000, seed=0):
    rng = random.Random(seed)
    nat = [p for p in paths if p.name.split("_")[0] in NATURE]
    rng.shuffle(nat)
    out = []
    for p in nat:
        if len(out) >= n_target:
            break
        img = cv2.imread(str(p))
        if img is None:
            continue
        H, W = img.shape[:2]
        occ = np.zeros((H, W), np.uint8)
        for x1, y1, x2, y2 in read_boxes(p, W, H):
            cv2.rectangle(occ, (int(x1) - 24, int(y1) - 24), (int(x2) + 24, int(y2) + 24), 1, -1)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        veg = ((hsv[..., 0] >= 8) & (hsv[..., 0] <= 48) & (hsv[..., 1] >= 46)).astype(np.uint8)
        for _ in range(4):
            s = rng.choice((64, 96, 128, 192))
            if W <= s or H <= s:
                continue
            x, y = rng.randrange(0, W - s), rng.randrange(0, H - s)
            if occ[y:y + s, x:x + s].any() or veg[y:y + s, x:x + s].mean() < 0.55:
                continue
            patch = img[y:y + s, x:x + s]
            if cv2.Laplacian(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var() < 60:
                continue
            out.append(patch.copy())
    return out


def blob_mask(h, w, cover, rng):
    """불규칙 모양 0~1 — 흐린 잡음을 문턱 → 넓이 비율 cover 에 맞춤 · 가장자리 부드럽게"""
    noise = rng.random((max(4, h // 6), max(4, w // 6))).astype(np.float32)
    noise = cv2.resize(noise, (w, h), interpolation=cv2.INTER_CUBIC)
    noise = cv2.GaussianBlur(noise, (0, 0), max(1.0, min(h, w) / 12))
    noise -= noise.min(); noise /= max(float(noise.max()), 1e-6)
    wy, wx = np.sin(np.linspace(0, np.pi, h)) ** 0.6, np.sin(np.linspace(0, np.pi, w)) ** 0.6
    noise *= np.outer(wy, wx).astype(np.float32)                       # 가장자리로 갈수록 0 → 조각 테두리에서 직선으로 잘리지 않음
    thr = np.quantile(noise, 1 - cover)
    m = (noise >= thr).astype(np.float32)
    return cv2.GaussianBlur(m, (0, 0), 1.2)


def occlude(img, boxes, patches, rng):
    H, W = img.shape[:2]
    cand = [b for b in boxes if b[3] - b[1] >= 16]
    rng.shuffle(cand)
    done = []
    for x1, y1, x2, y2 in cand[:rng.integers(1, 4)]:
        bw, bh = x2 - x1, y2 - y1
        k = rng.uniform(1.1, 1.8)                                    # 박스보다 큰 조각 → 모양이 항상 불규칙 (같은 크기면 네모 덩어리가 됨)
        pw, ph = max(8, int(bw * k)), max(8, int(bh * k))
        px = int(x1 + bw / 2 - pw / 2 + rng.uniform(-0.3, 0.3) * bw)
        py = int(y1 + bh / 2 - ph / 2 + rng.uniform(-0.3, 0.3) * bh)
        px0, py0, px1, py1 = max(0, px), max(0, py), min(W, px + pw), min(H, py + ph)
        if px1 - px0 < 4 or py1 - py0 < 4:
            continue
        patch = cv2.resize(patches[rng.integers(len(patches))], (pw, ph), interpolation=cv2.INTER_AREA)[py0 - py:py1 - py, px0 - px:px1 - px]
        target = rng.uniform(0.2, 0.6)
        # 박스 안을 target 만큼 덮도록 — 조각 영역에서의 넓이 비율로 근사 후 박스 안 실제 비율 확인
        a = blob_mask(py1 - py0, px1 - px0, min(0.7, target * bw * bh / max((px1 - px0) * (py1 - py0), 1) * 1.3), rng)
        reg = img[py0:py1, px0:px1].astype(np.float32)
        ring = np.ones((py1 - py0, px1 - px0), bool)                   # 밝기 기준 = 사람 박스 바깥 배경 (밝은 옷에 맞추면 노란 덩어리가 됨)
        ring[max(0, int(y1) - py0):max(0, int(y2) - py0), max(0, int(x1) - px0):max(0, int(x2) - px0)] = False
        bg = reg[ring].mean() if ring.sum() > 16 else reg.mean()
        p = patch.astype(np.float32)
        p *= float(np.clip((bg + 1) / (p.mean() + 1), 0.7, 1.3)) * rng.uniform(0.9, 1.05)
        img[py0:py1, px0:px1] = np.clip(reg * (1 - a[..., None]) + p * a[..., None], 0, 255).astype(np.uint8)
        bx0, by0, bx1, by1 = int(max(x1, px0)), int(max(y1, py0)), int(min(x2, px1)), int(min(y2, py1))
        cov = float(a[by0 - py0:by1 - py0, bx0 - px0:bx1 - px0].sum()) / max(bw * bh, 1) if bx1 > bx0 and by1 > by0 else 0.0
        done.append(round(cov, 3))
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frac", type=float, default=0.3)
    ap.add_argument("--preview", type=int, default=24)
    a = ap.parse_args()
    paths = [Path(l.strip()) for l in open(BASE / "configs/lists/train_v12.txt") if l.strip()]
    patches = bank(paths)
    print(f"잎 조각 {len(patches)}")
    (OUT / "images/train").mkdir(parents=True, exist_ok=True); (OUT / "labels/train").mkdir(parents=True, exist_ok=True)
    stats = {"windows": len(paths), "occluded_windows": 0, "occluded_boxes": 0, "cover": [], "bank": len(patches)}
    prev = []
    for p in paths:
        dst, ldst = OUT / "images/train" / p.name, OUT / "labels/train" / (p.stem + ".txt")
        if not ldst.exists() and lab_path(p).exists():
            os.symlink(lab_path(p), ldst)
        if dst.exists() or dst.is_symlink():
            continue
        lab = lab_path(p)
        pos = lab.exists() and lab.stat().st_size > 0
        if pos and p.name.split("_")[0] in TARGET and h01(p.name, "occ") < a.frac:
            img = cv2.imread(str(p)); H, W = img.shape[:2]
            rng = np.random.default_rng(int(h01(p.name, "seed") * 2**31))
            cov = occlude(img, read_boxes(p, W, H), patches, rng)
            if cov:
                cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                stats["occluded_windows"] += 1; stats["occluded_boxes"] += len(cov); stats["cover"] += cov
                if len(prev) < a.preview:
                    prev.append(dst)
                continue
        os.symlink(p, dst)
    c = np.array(stats.pop("cover"))
    stats["cover_p10_p50_p90"] = [round(float(np.percentile(c, q)), 3) for q in (10, 50, 90)] if len(c) else None
    for s in (91, 92):
        src = BASE / f"configs/lists/train_v12_f{s}.txt"
        (BASE / f"configs/lists/train_v13o_f{s}.txt").write_text("".join(str(OUT / "images/train" / Path(l.strip()).name) + "\n" for l in open(src) if l.strip()))
        (BASE / f"configs/data_v13o_f{s}.yaml").write_text(
            f"# v13o_f{s} (10-14 O) — v12_f{s} 와 같은 95 % 목록 · 양성 창 30 % 잎 가림 합성 · val = val_v12\n"
            f"train: {BASE}/configs/lists/train_v13o_f{s}.txt\nval: {BASE}/configs/lists/val_v12.txt\nnc: 1\nnames: ['person']\n")
    (BASE / "metrics/v13occ_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1))
    print(json.dumps(stats, ensure_ascii=False))
    sheet = []
    for d in prev:
        im = cv2.imread(str(d)); sheet.append(cv2.resize(im, (640, 360)))
    if sheet:
        rows = [np.hstack(sheet[i:i + 4]) for i in range(0, len(sheet) - len(sheet) % 4, 4)]
        cv2.imwrite(str(BASE / "logs" / "v13occ_preview.jpg"), np.vstack(rows))


if __name__ == "__main__":
    main()
