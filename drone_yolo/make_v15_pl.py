"""
make_v15_pl.py — 선생 모델 의사 라벨 (10-15 C · 판정 기준 = _학습 큐 「10-15 C」)

  선생 = v12_f91_swa · f92_swa · f93_swa (각각 imgsz 1280 · conf 0.3 · FP16) → v12 학습 창 전부 (train_v12.txt)
  추가 라벨 = 선생 2개 이상이 IoU ≥ 0.5 로 같이 잡고 평균 확신 ≥ 0.5 · 기존 라벨과 IoU < 0.3 그리고 기존 박스 안 50 % 미만
             가림 사각형 (박스 안 화소 표준편차 < 4) 위 · 긴 변 < 10 px 은 뺌
  출력 data/det_v15pl/{images (원본 링크), labels (기존 + 추가)}/train · 목록 train_v15pl_f91/f92 (v12 와 같은 95 %)
       metrics/v15pl_stats.json · logs/v15pl_preview.jpg (추가 라벨 표본 · 눈으로 확인)
실행  python make_v15_pl.py [--conf 0.5] [--agree 2]
"""
import argparse, json, os
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

BASE = Path(__file__).resolve().parent
OUT = BASE / "data" / "det_v15pl"
TEACHERS = ("v12_f91_swa", "v12_f92_swa", "v12_f93_swa")


def iou_mat(a, b):
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    it = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return it / np.maximum(aa[:, None] + bb[None, :] - it, 1e-6), it / np.maximum(aa[:, None], 1e-6)


def lab_path(p):
    return Path(str(p).replace("/images/", "/labels/")).with_suffix(".txt")


def read_gt(p, W, H):
    out = []
    lp = lab_path(p)
    if lp.exists():
        for ln in lp.read_text().split("\n"):
            v = ln.split()
            if len(v) == 5:
                cx, cy, w, h = map(float, v[1:])
                out.append(((cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H))
    return np.array(out, np.float32).reshape(-1, 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--agree", type=int, default=2)
    a = ap.parse_args()
    from ultralytics import YOLO
    paths = [l.strip() for l in open(BASE / "configs/lists/train_v12.txt") if l.strip()]
    cache = BASE / "runs_state" / "v15pl_teacher.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True); preds = z["preds"].item()
    else:
        preds = {t: [] for t in TEACHERS}
        for t in TEACHERS:
            m = YOLO(str(BASE / "runs_person" / t / "weights" / "best.pt"))
            for i in range(0, len(paths), 16):
                for r in m.predict(paths[i:i + 16], imgsz=1280, conf=0.3, verbose=False, half=True):
                    preds[t].append(np.concatenate([r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()[:, None]], 1) if len(r.boxes) else np.zeros((0, 5), np.float32))
            print(t, "끝")
        np.savez_compressed(cache, preds=np.array(preds, dtype=object))
    (OUT / "images/train").mkdir(parents=True, exist_ok=True); (OUT / "labels/train").mkdir(parents=True, exist_ok=True)
    st = Counter(); by_src = Counter(); prev = []
    for k, p in enumerate(paths):
        name = Path(p).name; src = name.split("_")[0]
        dst = OUT / "images/train" / name
        if not dst.exists():
            os.symlink(p, dst)
        img = None
        allb = [preds[t][k] for t in TEACHERS]
        base = allb[0]
        add = []
        if any(len(b) for b in allb):
            img = cv2.imread(p); H, W = img.shape[:2]
            gt = read_gt(p, W, H)
            pool = np.concatenate([np.c_[b, np.full(len(b), j)] for j, b in enumerate(allb) if len(b)], 0)
            pool = pool[np.argsort(-pool[:, 4])]; used = np.zeros(len(pool), bool)
            for i in range(len(pool)):
                if used[i]:
                    continue
                io, _ = iou_mat(pool[i:i + 1, :4], pool[:, :4])
                grp = np.where((io[0] >= 0.5) & ~used)[0]
                grp = grp[np.unique(pool[grp, 5], return_index=True)[1]]       # 선생마다 하나
                used[np.where(io[0] >= 0.5)[0]] = True
                if len(grp) < a.agree or pool[grp, 4].mean() < a.conf:
                    continue
                bx = pool[grp, :4].mean(0)
                if max(bx[2] - bx[0], bx[3] - bx[1]) < 10:
                    st["작음"] += 1; continue
                if len(gt):
                    i1, ios = iou_mat(bx[None], gt)
                    if i1.max() >= 0.3 or ios.max() >= 0.5:
                        continue
                x1, y1, x2, y2 = [int(round(v)) for v in bx]
                patch = img[max(0, y1):y2, max(0, x1):x2]
                if patch.size == 0 or patch.reshape(-1, 3).std(0).mean() < 4:
                    st["가림 사각형 위"] += 1; continue
                add.append((bx, float(pool[grp, 4].mean())))
        lp = OUT / "labels/train" / (Path(p).stem + ".txt")
        old = lab_path(p).read_text() if lab_path(p).exists() else ""
        if add:
            H, W = img.shape[:2]
            lines = [f"0 {(b[0] + b[2]) / 2 / W:.6f} {(b[1] + b[3]) / 2 / H:.6f} {(b[2] - b[0]) / W:.6f} {(b[3] - b[1]) / H:.6f}" for b, _ in add]
            lp.write_text(old.rstrip("\n") + ("\n" if old.strip() else "") + "\n".join(lines) + "\n")
            st["추가 박스"] += len(add); st["바뀐 창"] += 1; by_src[src] += len(add)
            st["음성 창 → 양성"] += int(not old.strip())
            prev += [(p, b, c) for b, c in add]
        else:
            if lp.exists() or lp.is_symlink():
                lp.unlink()
            os.symlink(lab_path(p), lp) if lab_path(p).exists() else lp.write_text("")
    for s in (91, 92):
        src_list = [l.strip() for l in open(BASE / f"configs/lists/train_v12_f{s}.txt") if l.strip()]
        (BASE / f"configs/lists/train_v15pl_f{s}.txt").write_text("".join(str(OUT / "images/train" / Path(x).name) + "\n" for x in src_list))
        (BASE / f"configs/data_v15pl_f{s}.yaml").write_text(
            f"# v15pl_f{s} (10-15 C) — v12_f{s} 와 같은 95 % 목록 · 선생 3개 의사 라벨 추가 · val = val_v12\n"
            f"train: {BASE}/configs/lists/train_v15pl_f{s}.txt\nval: {BASE}/configs/lists/val_v12.txt\nnc: 1\nnames: ['person']\n")
    out = {"창": len(paths), **st, "출처별 추가": dict(by_src), "규칙": f"선생 {a.agree}개 이상 · 평균 확신 ≥ {a.conf}"}
    (BASE / "metrics/v15pl_stats.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps(out, ensure_ascii=False))
    tiles = []
    for p, b, c in prev[::max(1, len(prev) // 32)][:32]:                  # 출처 고르게 (이름순 목록에서 일정 간격)
        im = cv2.imread(p); x1, y1, x2, y2 = b.astype(int); m = int(max(x2 - x1, y2 - y1) * 1.5) + 16
        X1, Y1, X2, Y2 = max(0, x1 - m), max(0, y1 - m), min(im.shape[1], x2 + m), min(im.shape[0], y2 + m)
        t = im[Y1:Y2, X1:X2].copy(); cv2.rectangle(t, (x1 - X1, y1 - Y1), (x2 - X1, y2 - Y1), (0, 0, 255), 1)
        t = cv2.resize(t, (200, 200)); cv2.putText(t, f"{Path(p).name.split('_')[0]} {c:.2f}", (3, 14), 0, 0.45, (255, 255, 255), 1)
        tiles.append(t)
    rows = [np.hstack(tiles[i:i + 8]) for i in range(0, len(tiles) - len(tiles) % 8, 8)]
    if rows:
        cv2.imwrite(str(BASE / "logs/v15pl_preview.jpg"), np.vstack(rows))


if __name__ == "__main__":
    main()
