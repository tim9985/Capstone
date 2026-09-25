"""
eval_test_v2.py — NFR-V03 측정 (2026-09-23)

왜 전용 도구인가
  NFR-V03 은 "지정한 평가셋(학습과 **장소 분리**)에서 사람 **AP50 ≥0.80**" 이다.
  · eval_fullframe.py 는 s38/l128/neg 하위 폴더 구조를 전제해 test_v2(평면)에 못 쓴다
  · 기존 도구들은 대부분 **재현율@0.15** 만 낸다. NFR-V03 은 **AP50**(임계값 무관)이다

무엇을
  배포 파이프라인 그대로 — 1920×1080 전체 프레임을 1280×720 타일 4장(겹침 50 %)으로 나눠
  추론하고 NMS(IoU 0.6)로 합친 뒤, conf 전 구간 PR 곡선으로 AP50 을 낸다.

실행: python eval_test_v2.py --weights runs_person/v3_place/weights/best.pt
"""
import argparse, csv, json
from pathlib import Path
import numpy as np

BASE = Path(__file__).resolve().parent
W, H, TW, TH = 1920, 1080, 1280, 720
TILES = [(x, y) for y in (0, H - TH) for x in (0, W - TW)]


def load_gt(p, W=W, H=H):
    t = Path(str(p).replace("/images/", "/labels/")).with_suffix(".txt")
    if not t.exists():
        return np.zeros((0, 4), np.float32)
    rows = [l.split() for l in t.read_text().splitlines() if l.strip()]
    if not rows:
        return np.zeros((0, 4), np.float32)
    a = np.array([[float(v) for v in r[1:5]] for r in rows], np.float32)
    return np.c_[(a[:, 0] - a[:, 2] / 2) * W, (a[:, 1] - a[:, 3] / 2) * H,
                 (a[:, 0] + a[:, 2] / 2) * W, (a[:, 1] + a[:, 3] / 2) * H]


BLOCK = 20   # 부트스트랩 단위: 비행·시퀀스 안 연속 20장 (연속 프레임은 서로 닮았다)


def ap101(s, t, n_gt):
    """점수·정답여부 → 101점 보간 AP50"""
    if not len(s) or n_gt == 0:
        return 0.0
    o = np.argsort(-s); t = t[o]
    tp = np.cumsum(t); fp = np.cumsum(~t)
    rec = tp / n_gt; prec = tp / np.maximum(tp + fp, 1)
    return float(np.mean([prec[rec >= r].max() if (rec >= r).any() else 0 for r in np.linspace(0, 1, 101)]))


def blocks_of(names):
    """이미지 이름 → 블록 번호 (그룹 = 이름에서 끝 프레임 번호를 뺀 것 · 그룹 안 정렬 후 BLOCK 장씩)"""
    grp = {}
    for i, n in enumerate(names):
        grp.setdefault(n.rsplit("_", 1)[0], []).append(i)
    b = np.zeros(len(names), int); k = 0
    for g in sorted(grp):
        idx = sorted(grp[g], key=lambda i: names[i])
        for j in range(0, len(idx), BLOCK):
            b[idx[j:j + BLOCK]] = k; k += 1
    return b, k, len(grp)


def boot_ap(s, t, img, n_gt_img, blk, nblk, B=1000, seed=0):
    """블록 부트스트랩 AP50 표본 (같은 seed 면 모델 간 짝 비교가 된다)"""
    rng = np.random.default_rng(seed)
    pred_blk = blk[img]
    by_blk = [np.where(pred_blk == k)[0] for k in range(nblk)]
    gt_blk = np.bincount(blk, weights=n_gt_img, minlength=nblk)
    out = []
    for _ in range(B):
        pick = rng.integers(0, nblk, nblk)
        idx = np.concatenate([by_blk[k] for k in pick]) if nblk else np.array([], int)
        out.append(ap101(s[idx], t[idx], gt_blk[pick].sum()))
    return np.array(out)


def iou_mat(g, p):
    if not len(g) or not len(p):
        return np.zeros((len(g), len(p)), np.float32)
    x1 = np.maximum(g[:, None, 0], p[None, :, 0]); y1 = np.maximum(g[:, None, 1], p[None, :, 1])
    x2 = np.minimum(g[:, None, 2], p[None, :, 2]); y2 = np.minimum(g[:, None, 3], p[None, :, 3])
    it = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    ar = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])
    return it / (ar(g)[:, None] + ar(p)[None, :] - it + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="runs_person/v3_place/weights/best.pt")
    ap.add_argument("--data", default=str(BASE / "data" / "test_v2"))
    ap.add_argument("--conf-min", type=float, default=0.01, help="PR 곡선용 하한")
    ap.add_argument("--op-conf", type=float, default=0.15, help="운용 임계값")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--mode", choices=("tile", "single"), default="tile",
                    help="tile: 1920×1080 → 1280×720 4장 (test_v2) · single: 원본 그대로 imgsz 1280 (test_obl 1280×720)")
    ap.add_argument("--tag", default="test_v2", help="결과 파일 접두사 → metrics/<tag>_<모델>.csv")
    ap.add_argument("--imgsz", type=int, default=1280, help="모델 입력 크기 (single 모드에서 1920 이면 원본 그대로 한 장)")
    ap.add_argument("--boot", type=int, default=1000, help="블록 부트스트랩 횟수 (0 이면 끔)")
    args = ap.parse_args()

    import cv2
    from ultralytics import YOLO
    model = YOLO(args.weights)
    name = Path(args.weights).parent.parent.name

    imgs = sorted((Path(args.data) / "images").glob("*.jpg"))
    print(f"{name} · {args.tag} · {len(imgs):,}장 · {'타일 4×(1280×720)' if args.mode == 'tile' else '단일 추론'} · conf ≥{args.conf_min}", flush=True)
    BINS = [(0, 24), (24, 36), (36, 50), (50, 80), (80, 1e9)]
    bin_gt = [0] * len(BINS); bin_hit = [0] * len(BINS)

    scores, tps, n_gt, fp_neg, n_neg = [], [], 0, 0, 0
    pimg, ngt_img, pname = [], [], []
    for i, p in enumerate(imgs):
        if i % 200 == 0:
            print(f"  {i:,}/{len(imgs):,}", flush=True)
        img = cv2.imread(str(p))
        if img is None:
            continue
        ih, iw = img.shape[:2]
        gt = load_gt(p, iw, ih); n_gt += len(gt); ngt_img.append(len(gt)); pname.append(p.stem); ii = len(ngt_img) - 1
        if args.mode == "tile":
            offs = TILES
            tiles = [img[y:y + TH, x:x + TW] for x, y in TILES]
        else:
            offs, tiles = [(0, 0)], [img]
        res = model.predict(tiles, imgsz=args.imgsz, conf=args.conf_min, batch=len(tiles),
                            quantize="fp16", verbose=False)
        box, cf = [], []
        for (ox, oy), r in zip(offs, res):
            b = r.boxes
            if not len(b):
                continue
            box.append(b.xyxy.cpu().numpy() + np.array([ox, oy, ox, oy], np.float32))
            cf.append(b.conf.cpu().numpy())
        if box:
            box = np.concatenate(box); cf = np.concatenate(cf)
            keep = cv2.dnn.NMSBoxes(
                [[float(x1), float(y1), float(x2 - x1), float(y2 - y1)] for x1, y1, x2, y2 in box],
                cf.tolist(), args.conf_min, 0.6)
            keep = np.array(keep).ravel().astype(int) if len(keep) else np.array([], int)
            box, cf = box[keep], cf[keep]
        else:
            box, cf = np.zeros((0, 4), np.float32), np.zeros(0, np.float32)

        if not len(gt):                      # 음성 프레임
            n_neg += 1
            fp_neg += int((cf >= args.op_conf).sum())
        order = np.argsort(-cf)
        box, cf = box[order], cf[order]
        M = iou_mat(gt, box); used = np.zeros(len(gt), bool)
        for j in range(len(box)):
            k = -1
            if len(gt):
                cand = np.where((M[:, j] >= 0.5) & ~used)[0]
                if len(cand):
                    k = cand[np.argmax(M[cand, j])]
            scores.append(float(cf[j])); tps.append(k >= 0); pimg.append(ii)
            if k >= 0:
                used[k] = True
        # 크기 구간별 재현율@운용 임계값 (층화 — 어디서 못 찾나)
        if len(gt):
            Mo = iou_mat(gt, box[cf >= args.op_conf]); hit = np.zeros(len(gt), bool); u = set()
            for gi, pj in zip(*np.unravel_index(np.argsort(-Mo, axis=None), Mo.shape)) if Mo.size else []:
                if Mo[gi, pj] < 0.5:
                    break
                if hit[gi] or pj in u:
                    continue
                hit[gi] = True; u.add(pj)
            L = np.maximum(gt[:, 2] - gt[:, 0], gt[:, 3] - gt[:, 1])
            for bi, (lo, hi) in enumerate(BINS):
                m = (L >= lo) & (L < hi); bin_gt[bi] += int(m.sum()); bin_hit[bi] += int(hit[m].sum())

    s = np.array(scores); t = np.array(tps, bool)
    o = np.argsort(-s); t = t[o]; s = s[o]
    tp = np.cumsum(t); fp = np.cumsum(~t)
    rec = tp / max(n_gt, 1); prec = tp / np.maximum(tp + fp, 1)
    # 101점 보간 AP
    ap50 = float(np.mean([prec[rec >= r].max() if (rec >= r).any() else 0
                          for r in np.linspace(0, 1, 101)]))
    m = s >= args.op_conf
    r_op = float(tp[m][-1] / max(n_gt, 1)) if m.any() else 0.0
    p_op = float(prec[m][-1]) if m.any() else 0.0

    print(f"\n=== NFR-V03 — 장소 분리 평가셋 {args.tag}")
    print(f"  정답 {n_gt:,} · 예측 {len(s):,} · 음성 프레임 {n_neg}")
    print(f"  **AP50            {ap50:.4f}**   (목표 ≥0.80 → {'✅ 통과' if ap50>=0.80 else '❌ 미달'})")
    print(f"  재현율@{args.op_conf:g}      {r_op:.4f}")
    print(f"  정밀도@{args.op_conf:g}      {p_op:.4f}")
    print(f"  음성 프레임 오탐   {fp_neg/max(n_neg,1):.3f} 건/프레임")

    out = BASE / "metrics" / f"{args.tag}_{name}.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model", "n_gt", "AP50", f"recall@{args.op_conf:g}",
                                          f"precision@{args.op_conf:g}", "fp_per_neg_frame"])
        w.writeheader()
        w.writerow({"model": name, "n_gt": n_gt, "AP50": round(ap50, 4),
                    f"recall@{args.op_conf:g}": round(r_op, 4),
                    f"precision@{args.op_conf:g}": round(p_op, 4),
                    "fp_per_neg_frame": round(fp_neg / max(n_neg, 1), 3)})
    print(f"→ {out}")
    ob = BASE / "metrics" / f"{args.tag}_{name}_bins.csv"
    with open(ob, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["px_bin", "n_gt", f"recall@{args.op_conf:g}"])
        for (lo, hi), g, h in zip(BINS, bin_gt, bin_hit):
            lab = f"{int(lo)}-{int(hi)}" if hi < 1e8 else f"{int(lo)}+"
            w.writerow([lab, g, round(h / max(g, 1), 4)]); print(f"  {lab:>7} px  n={g:>6,}  재현율 {h/max(g,1):.3f}")
    print(f"→ {ob}")
    if args.boot:
        names = pname
        blk, nblk, ngrp = blocks_of(names)
        S, T, I = np.array(scores), np.array(tps, bool), np.array(pimg, int)
        G = np.array(ngt_img, float)
        bs = boot_ap(S, T, I, G, blk, nblk, args.boot)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        print(f"  AP50 95 % 구간  {lo:.4f} ~ {hi:.4f}  (블록 {nblk}개 · 그룹 {ngrp}개 · {args.boot}회)")
        oc = BASE / "metrics" / f"{args.tag}_{name}_ci.csv"
        with open(oc, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["model", "AP50", "ci_lo", "ci_hi", "blocks", "groups", "B"])
            w.writerow([name, round(ap50, 4), round(lo, 4), round(hi, 4), nblk, ngrp, args.boot])
        dump = Path(args.weights).parent.parent / f"eval_{args.tag}.npz"
        np.savez_compressed(dump, s=S, t=T, img=I, ngt=G, names=np.array(names))
        print(f"→ {oc} · 짝 비교용 {dump}")


if __name__ == "__main__":
    main()
