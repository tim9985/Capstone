"""
cp_lying.py — 누운 사람 복사-붙여넣기 증강 (10-30 CP · 근거: 비스듬 드론 사람 복사-붙여넣기 (Drones 2026) · 합성 자세 다양성 (Shen 2024) · 회전 박스 (Kalra ICCV 2021))

  재료  data/lying_bank (lying_bank.py · SAM 마스크 · Archangel lying down · crawling + NOMAD 배우 1~95 Laying)
        v2 (10-30 · 1판 눈 확인 뒤): 품질 기준 — Archangel 긴 변 ≥ 28 px · SAM ≥ 0.85 · 마스크 비율 0.25~0.85 / NOMAD SAM ≥ 0.80 · 0.20~0.90
        · **원본보다 1.3 배 넘게 키우지 않음** (작은 원본을 키우면 번진 검은 얼룩) · 색 맞춤 0~20 %
  대상  v12_f91 학습 창 중 **비스듬** 출처 (visdrone_obl · unicamp · niicu · nomad · sard · archangel) — 수직 창엔 안 붙임 (시점 안 맞음)
  붙이기 창마다 1~3명
    크기   창 안 사람 박스 (긴 변) 를 y 에 대해 직선 맞춤 (≥ 2명) → 그 자리 크기 × U(0.8, 1.3) · 사람 없으면 긴 변 log-U(14, 110) px × 원근 (위쪽 작게)
    방향   회전 U(−180°, 180°) (70 %) · 아니면 ±15° · 좌우 뒤집기 50 % — **마스크를 같이 돌려 박스를 마스크에서 다시 잼** (가장 큰 박스 문제 없음)
    자리   위 12 % 빼고 · 기존 사람 · 붙인 사람과 안 겹침 (여백 4 px)
    섞기   마스크 가장자리 흐림 (σ 0.6~1.2) · 밝기 맞춤 (둘레 평균 쪽으로 0~35 %) · 밝기 ×U(0.85, 1.15) · 흐림 σ U(0, 0.8)
  출력  data/det_v12/{images,labels}/train_cp/<원래 이름>_cp.jpg · configs/lists/train_v12cp.txt (= train_v12_f91 + 붙인 창) · configs/data_v12cp.yaml
실행  python cp_lying.py [--n 3000] [--seed 0]
"""
import argparse, json, math, random
from pathlib import Path

import cv2
import numpy as np

BASE = Path(__file__).resolve().parent
BANK = BASE / "data" / "lying_bank"
D = BASE / "data" / "det_v12"
OBL = ("visdrone_obl", "unicamp", "niicu", "nomad", "sard", "archangel")


def src_of(p):
    n = Path(p).name
    return "visdrone_obl" if n.startswith("visdrone_obl") else "visdrone_nadir" if n.startswith("visdrone_nadir") else n.split("_")[0]


def read_labels(p, W, H):
    lab = Path(str(p).replace("/images/", "/labels/")).with_suffix(".txt")
    out = []
    if lab.exists():
        for l in open(lab):
            t = l.split()
            if len(t) >= 5:
                cx, cy, w, h = (float(v) for v in t[1:5])
                out.append([(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H])
    return out


def size_model(boxes, H, rng):
    """y (박스 아래) → 긴 변 px — 사람 ≥ 2 면 직선 맞춤 · 아니면 무작위 기준 × 원근"""
    if len(boxes) >= 2:
        ys = np.array([b[3] for b in boxes]); ls = np.array([max(b[2] - b[0], b[3] - b[1]) for b in boxes])
        if np.ptp(ys) > 30:
            a, c = np.polyfit(ys, ls, 1)
            return lambda y: max(10.0, a * y + c)
        m = float(np.median(ls)); return lambda y: m
    if len(boxes) == 1:
        m = max(boxes[0][2] - boxes[0][0], boxes[0][3] - boxes[0][1]); return lambda y: m
    base = math.exp(rng.uniform(math.log(14), math.log(110)))
    return lambda y: base * (0.7 + 0.6 * y / H)                    # 위쪽 (먼 곳) 작게


def rotate_rgba(inst, ang, flip):
    if flip:
        inst = inst[:, ::-1]
    h, w = inst.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
    c, s = abs(M[0, 0]), abs(M[0, 1]); nw, nh = int(h * s + w * c) + 2, int(h * c + w * s) + 2
    M[0, 2] += nw / 2 - w / 2; M[1, 2] += nh / 2 - h / 2
    out = cv2.warpAffine(inst, M, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))
    ys, xs = np.where(out[..., 3] > 127)
    if len(xs) < 4:
        return None
    return out[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def paste(img, inst, x, y, rng):
    h, w = inst.shape[:2]
    rgb = inst[..., :3].astype(np.float32); a = inst[..., 3].astype(np.float32) / 255
    a = cv2.GaussianBlur(a, (0, 0), rng.uniform(0.6, 1.2)) if min(h, w) >= 6 else a
    roi = img[y:y + h, x:x + w].astype(np.float32)
    ring = img[max(0, y - h // 2):y + h + h // 2, max(0, x - w // 2):x + w + w // 2].reshape(-1, 3).astype(np.float32)
    m_in = (rgb * a[..., None]).sum((0, 1)) / max(a.sum(), 1e-3); m_bg = ring.mean(0)
    rgb = rgb + rng.uniform(0, 0.20) * (m_bg - m_in)                  # 밝기 · 색 맞춤 (일부만 · v2 0.35 → 0.20)
    rgb = rgb * rng.uniform(0.85, 1.15)
    s = rng.uniform(0, 0.8)
    if s > 0.2:
        rgb = cv2.GaussianBlur(rgb, (0, 0), s)
    img[y:y + h, x:x + w] = np.clip(roi * (1 - a[..., None]) + rgb * a[..., None], 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=3000); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed); nrng = np.random.default_rng(a.seed)
    bank = [json.loads(l) for l in open(BANK / "meta.jsonl")]
    insts = [(m, cv2.imread(str(BANK / m["file"]), cv2.IMREAD_UNCHANGED)) for m in bank]
    def good(m):
        if m["src"] == "archangel":
            return max(m["w"], m["h"]) >= 28 and m["sam"] >= 0.85 and 0.25 <= m["mask_ratio"] <= 0.85
        return m["sam"] >= 0.80 and 0.20 <= m["mask_ratio"] <= 0.90
    insts = [(m, x) for m, x in insts if x is not None and x.shape[2] == 4 and good(m)]
    base = [l.strip() for l in open(BASE / "configs/lists/train_v12_f91.txt") if l.strip()]
    cand = [p for p in base if src_of(p) in OBL]
    pick = rng.sample(cand, min(a.n, len(cand)))
    (D / "images/train_cp").mkdir(parents=True, exist_ok=True); (D / "labels/train_cp").mkdir(parents=True, exist_ok=True)
    out, n_p, sizes = [], 0, []
    for p in pick:
        img = cv2.imread(p)
        if img is None:
            continue
        H, W = img.shape[:2]
        boxes = read_labels(p, W, H); occ = [list(b) for b in boxes]
        L = size_model(boxes, H, rng); new = []
        for _ in range(rng.choice((1, 1, 2, 2, 3))):
            for _try in range(20):
                m, inst = insts[rng.randrange(len(insts))]
                y = rng.uniform(0.12 * H, 0.98 * H)
                tgt = L(y) * rng.uniform(0.8, 1.3)
                ang = rng.uniform(-180, 180) if rng.random() < 0.7 else rng.uniform(-15, 15)
                r = rotate_rgba(inst, ang, rng.random() < 0.5)
                if r is None:
                    continue
                k = tgt / max(r.shape[:2])
                if k <= 0 or max(r.shape[:2]) * k < 8 or k > 1.3:            # v2: 1.3 배 넘게 키우지 않음
                    continue
                r = cv2.resize(r, (max(2, round(r.shape[1] * k)), max(2, round(r.shape[0] * k))), interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_LINEAR)
                h, w = r.shape[:2]; x0 = rng.uniform(0, W - w); y0 = y - h
                if x0 < 0 or y0 < 0.12 * H or y0 + h > H or x0 + w > W:
                    continue
                bb = [x0, y0, x0 + w, y0 + h]
                if any(not (bb[2] + 4 < o[0] or bb[0] - 4 > o[2] or bb[3] + 4 < o[1] or bb[1] - 4 > o[3]) for o in occ):
                    continue
                paste(img, r, int(x0), int(y0), nrng)
                ys, xs = np.where(r[..., 3] > 127)
                tb = [int(x0) + xs.min(), int(y0) + ys.min(), int(x0) + xs.max() + 1, int(y0) + ys.max() + 1]
                occ.append(tb); new.append(tb); sizes.append(max(tb[2] - tb[0], tb[3] - tb[1])); n_p += 1
                break
        if not new:
            continue
        stem = Path(p).stem + "_cp"
        cv2.imwrite(str(D / "images/train_cp" / f"{stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        lab_src = Path(p.replace("/images/", "/labels/")).with_suffix(".txt")
        lines = open(lab_src).read().strip().splitlines() if lab_src.exists() else []
        lines += [f"0 {(b[0] + b[2]) / 2 / W:.6f} {(b[1] + b[3]) / 2 / H:.6f} {(b[2] - b[0]) / W:.6f} {(b[3] - b[1]) / H:.6f}" for b in new]
        (D / "labels/train_cp" / f"{stem}.txt").write_text("\n".join(l for l in lines if l.strip()) + "\n")
        out.append(str(D / "images/train_cp" / f"{stem}.jpg"))
    (BASE / "configs/lists/train_v12cp.txt").write_text("\n".join(base + out) + "\n")
    (BASE / "configs/data_v12cp.yaml").write_text(f"# v12cp (10-30 CP) — train_v12_f91 + 누운 사람 복사-붙여넣기 창 {len(out)} · val = val_v12\ntrain: {BASE}/configs/lists/train_v12cp.txt\nval: {BASE}/configs/lists/val_v12.txt\nnc: 1\nnames: ['person']\n")
    st = {"재료": len(insts), "붙인 창": len(out), "붙인 사람": n_p, "긴 변 px 분위 (10 · 50 · 90)": [round(float(np.percentile(sizes, q)), 1) for q in (10, 50, 90)]}
    json.dump(st, open(BASE / "metrics/cp_lying_stats.json", "w"), ensure_ascii=False, indent=1); print(st)


if __name__ == "__main__":
    main()
