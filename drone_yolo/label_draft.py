"""
label_draft.py — 실기체 영상 라벨 초안 (11-08 · D1 라벨 작업 줄이기) · 사람이 고치는 초안이지 학습용 의사 라벨이 아님 (막다른 길 10-17 과 다름)

  초안  우리 v12 (운용 4조각 · 확신 ≥ --t-ours 0.20) — **기본은 우리만** (자가시험 11-08: GDINO 를 합치면 고칠 것이 늘거나 같음 · 61.2 → 62.4~81.5 / 사람 100명)
        --t-gd > 0 이면 Grounding DINO base (「person.」 · 4조각) 를 두 번째 의견으로 합침 · 겹치면 (IoU ≥ 0.5) 우리 박스 하나만
        박스마다 표시 — 둘 다 찾음 = 「확실」 · 하나만 = 「확인」 → review.csv 를 「확인」 많은 프레임 순으로 · 미리보기 (실선 = 확실 · 점선 = 확인)
  출력  <out>/images (원본 링크) · labels/*.txt (YOLO · 사람 0) · review.csv · preview/*.jpg → FiftyOne · CVAT 로 고친 뒤 make_d1_data.py 의 비행 폴더로
  자가시험 (--selftest · GPU 안 씀)  runs_fair 의 저장된 검출로 규칙별 P · R · 「고칠 것」 (헛 + 놓침 / 사람 100명) 비교 → 기본 문턱 고름
실행  /home/se/venvs/state/bin/python label_draft.py <프레임 폴더> --out <폴더> [--every 5]     (state 환경 = transformers + ultralytics 8.4.102)
      /home/se/venvs/state/bin/python label_draft.py --selftest
"""
import argparse, csv, json, os
from pathlib import Path

import cv2
import numpy as np

BASE = Path(__file__).resolve().parent
RUN = BASE / "runs_fair"
OURS_W = BASE / "runs_person" / "v12_f91_swa" / "weights" / "best.pt"
GD_ID = "IDEA-Research/grounding-dino-base"


def iou(a, b):
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    ar = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])
    return inter / (ar(a)[:, None] + ar(b)[None, :] - inter + 1e-9)


def merge(bo, bg):
    """우리 박스 전부 + 우리와 안 겹치는 GDINO 박스 · 표시 = 둘 다 (True) / 하나만 (False)"""
    M = iou(bo, bg); both_o = (M >= 0.5).any(1) if len(bg) else np.zeros(len(bo), bool)
    only_g = ~(M >= 0.5).any(0) if len(bo) else np.ones(len(bg), bool)
    box = np.concatenate([bo.reshape(-1, 4), bg[only_g].reshape(-1, 4)]); sure = np.concatenate([both_o, np.zeros(only_g.sum(), bool)])
    return box, sure


def tp_count(gt, box):
    M = iou(gt, box); used = np.zeros(len(gt), bool); tp = 0
    for j in range(len(box)):
        c = np.where((M[:, j] >= 0.5) & ~used)[0] if len(gt) else []
        if len(c):
            used[c[np.argmax(M[c, j])]] = True; tp += 1
    return tp


def selftest():
    sets = ["test_sar", "test_obl", "test_cy", "test_v2", "test_kr", "test_nomad", "test_ladd_h"]
    rules = [("우리만 0.20", 0.20, None), ("우리만 0.10", 0.10, None)] + [(f"우리 {to} ∪ GDINO {tg}", to, tg) for to in (0.20, 0.10) for tg in (0.25, 0.35, 0.45)]
    out = {}
    for s in sets:
        g = np.load(RUN / s / "gt.npz"); o = np.load(RUN / s / "v12_f91_swa@tile1280.npz"); d = np.load(RUN / s / "sota_gdino@tile.npz"); n = len(g["names"])
        for name, to, tg in rules:
            TP = FP = NG = SURE = ALL = 0
            for i in range(n):
                gt = g["gt"][g["gt_img"] == i]; bo = o["box"][(o["img"] == i) & (o["conf"] >= to)]
                bg = d["box"][(d["img"] == i) & (d["conf"] >= tg)] if tg else np.zeros((0, 4))
                box, sure = merge(bo, bg) if tg else (bo, np.zeros(len(bo), bool))
                tp = tp_count(gt, box); TP += tp; FP += len(box) - tp; NG += len(gt); SURE += sure.sum(); ALL += len(box)
            out.setdefault(name, {})[s] = {"P": round(TP / max(1, TP + FP), 3), "R": round(TP / max(1, NG), 3),
                                           "고칠것_100명": round(100 * (FP + NG - TP) / max(1, NG), 1), "확실_비율": round(SURE / max(1, ALL), 3) if tg else None}
    print(f"{'규칙':<22}" + "".join(f"{s[5:12]:>10}" for s in sets) + "   평균  (칸 = 사람 100명당 고칠 것 = 헛 + 놓침)")
    for name in out:
        v = [out[name][s]["고칠것_100명"] for s in sets]
        print(f"{name:<22}" + "".join(f"{x:>10.1f}" for x in v) + f"{np.mean(v):>8.1f}")
    json.dump(out, open(BASE / "metrics" / "label_draft_selftest.json", "w"), ensure_ascii=False, indent=1)


def tiles(W, H, tw=1280, th=720):
    xs = [0] if W <= tw else sorted(set(list(range(0, W - tw + 1, tw // 2)) + [W - tw]))
    ys = [0] if H <= th else sorted(set(list(range(0, H - th + 1, th // 2)) + [H - th]))
    return [(x, y) for y in ys for x in xs]


def nms(b, c, thr=0.6):
    if not len(b):
        return b, c
    k = cv2.dnn.NMSBoxes([[float(a), float(q), float(e - a), float(f - q)] for a, q, e, f in b], c.tolist(), 0.0, thr)
    k = np.array(k).ravel().astype(int) if len(k) else np.array([], int); return b[k], c[k]


def draft(a):
    import torch
    from PIL import Image
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    from ultralytics import YOLO
    yo = YOLO(str(OURS_W)); use_gd = a.t_gd > 0
    if use_gd:
        proc = AutoProcessor.from_pretrained(GD_ID); gd = AutoModelForZeroShotObjectDetection.from_pretrained(GD_ID, torch_dtype=torch.float16).cuda().eval()

    @torch.no_grad()
    def gdino(crops):
        ims = [Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for c in crops]
        inp = proc(images=ims, text=["person."] * len(ims), return_tensors="pt").to("cuda"); inp["pixel_values"] = inp["pixel_values"].half()
        res = proc.post_process_grounded_object_detection(gd(**inp), inp.input_ids, threshold=a.t_gd, text_threshold=a.t_gd, target_sizes=[c.shape[:2] for c in crops])
        return [(r["boxes"].float().cpu().numpy(), r["scores"].float().cpu().numpy()) for r in res]

    src = Path(a.frames); ims = sorted((src / "images" if (src / "images").is_dir() else src).glob("*.jpg"))[::a.every]
    out = Path(a.out); [(out / d).mkdir(parents=True, exist_ok=True) for d in ("images", "labels", "preview")]
    rows = []
    for k, p in enumerate(ims):
        img = cv2.imread(str(p)); H, W = img.shape[:2]; s = 1920 / W
        im = cv2.resize(img, (1920, round(H * s))) if s != 1 else img; T = tiles(1920, im.shape[0]); crops = [im[y:y + 720, x:x + 1280] for x, y in T]
        r = yo.predict(crops, imgsz=1280, conf=a.t_ours, half=True, verbose=False, batch=len(crops))
        bo = np.concatenate([q.boxes.xyxy.cpu().numpy() + np.array([x, y, x, y], np.float32) for (x, y), q in zip(T, r)]); co = np.concatenate([q.boxes.conf.cpu().numpy() for q in r])
        if not use_gd:
            bo, co = nms(bo, co); box = bo / s; sure = np.ones(len(box), bool)                         # 우리만 — 박스 전부 「확인 대상」 아님 · 확신 낮은 것은 review 열에
        else:
          g = gdino(crops); bg = np.concatenate([b + np.array([x, y, x, y], np.float32) for (x, y), (b, _) in zip(T, g)]); cg = np.concatenate([c for _, c in g])
          (bo, co), (bg, cg) = nms(bo, co), nms(bg, cg); box, sure = merge(bo / s, bg / s)
        dst = out / "images" / p.name
        if not dst.exists():
            os.symlink(p.resolve(), dst)
        (out / "labels" / f"{p.stem}.txt").write_text("".join(f"0 {(b[0] + b[2]) / 2 / W:.6f} {(b[1] + b[3]) / 2 / H:.6f} {(b[2] - b[0]) / W:.6f} {(b[3] - b[1]) / H:.6f}\n" for b in box))
        if not use_gd:
            sure = co >= 0.40                                                                         # 우리만일 때 「확인」 = 확신 0.20~0.40
        rows.append({"frame": p.name, "boxes": len(box), "확실": int(sure.sum()), "확인": int((~sure).sum()), "우리만": len(bo) - int(sure.sum()) if use_gd else len(bo), "GDINO만": len(box) - len(bo)})
        if k % a.preview_every == 0 or (~sure).sum():
            vis = img.copy()
            for b, sr in zip(box.astype(int), sure):
                if sr:
                    cv2.rectangle(vis, tuple(b[:2]), tuple(b[2:]), (0, 200, 0), 2)
                else:
                    for t in range(0, 4):
                        xa, ya, xb, yb = b; pts = [((xa, ya), (xb, ya)), ((xb, ya), (xb, yb)), ((xb, yb), (xa, yb)), ((xa, yb), (xa, ya))][t]
                        L = max(1, int(np.hypot(pts[1][0] - pts[0][0], pts[1][1] - pts[0][1]) // 8))
                        for u in range(0, L, 2):
                            p0 = np.array(pts[0]) + (np.array(pts[1]) - np.array(pts[0])) * u / L; p1 = np.array(pts[0]) + (np.array(pts[1]) - np.array(pts[0])) * min(L, u + 1) / L
                            cv2.line(vis, tuple(p0.astype(int)), tuple(p1.astype(int)), (0, 160, 255), 2)
            cv2.imwrite(str(out / "preview" / p.name), cv2.resize(vis, (W // 2, H // 2)), [cv2.IMWRITE_JPEG_QUALITY, 80])
    rows.sort(key=lambda r: -r["확인"])
    with open(out / "review.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    n = sum(r["boxes"] for r in rows); u = sum(r["확인"] for r in rows)
    print(f"프레임 {len(rows)} · 박스 {n} · 확인 필요 {u} ({u / max(1, n):.0%}) · 확인 있는 프레임 {sum(r['확인'] > 0 for r in rows)} → {out}/review.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("frames", nargs="?"); ap.add_argument("--out"); ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--every", type=int, default=5); ap.add_argument("--t-ours", type=float, default=0.20); ap.add_argument("--t-gd", type=float, default=0.0, help="> 0 이면 GDINO 같이 (예 0.45)")
    ap.add_argument("--preview-every", type=int, default=20)
    a = ap.parse_args()
    selftest() if a.selftest else draft(a)
