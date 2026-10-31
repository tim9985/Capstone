"""
bench_all.py — 데이터셋 × 모델 (우리 + 기성) 을 같은 방식으로 채점 (11-01 · 사용자 「기성 모델과 우리 모델 성능 차」)

  추론 (모두 같음)  그림을 가로 1920 으로 (비율 유지) → 1280×720 창 (겹침 50 % · 세로가 길면 창 더) → imgsz → 사람 클래스만
                   → 창 합쳐 NMS 0.6 · FP16 · 확신 ≥ 0.01   (우리 운용 추론과 같은 틀 · 경계 합치기 · 무시 영역은 안 씀 → 결정 노트 수치와 조금 다름)
  모델  ours = runs_person/<이름>/weights/best.pt (imgsz 1280)
        ext  = data/raw/ext_det/… (클래스 person · pedestrian · people) · 640 학습이라 imgsz 1280 · 640 둘 다 → 표엔 좋은 쪽
  채점  IoU 0.5 · 101점 AP50 · 재현율 · 정밀도 @0.20 · 사람 없는 장 오탐 @0.20 (eval_external_det 함수 그대로)
  셋    장수가 MAXN 넘으면 고르게 뽑음
  출력  metrics/bench_all.csv (줄마다 덧붙임 · 이미 한 (모델, imgsz, 셋) 은 건너뜀 → 끊겨도 다시 실행)
실행  python bench_all.py [--models a,b] [--sets x,y]
"""
import argparse, csv, time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from eval_external_det import ap101, iou_mat, load_gt

BASE = Path(__file__).resolve().parent
EXT = BASE.parent / "data" / "raw" / "ext_det"
OUT = BASE / "metrics" / "bench_all.csv"
MAXN = 500
SETS = ["test_sar", "test_obl", "test_v2", "test_kr", "test_ladd_h", "test_visdrone", "test_nomad", "test_nii", "test_auair",
        "test_sard", "test_heridal", "test_cy"]
OURS = ["v12_f91_swa", "v12_f92_swa", "v12_f93_swa", "soup_v7r2", "v12ly_swa", "v12cp_swa", "v12el_swa", "v13o_f91_swa", "v15pl_f91_swa",
        "soup_v9x2", "v9_s31", "v10_f61", "v11s_f71", "v8", "v7", "v6_obl", "v3_place", "y26_s31", "vd_11m", "l1_11l_1280", "m2_11m_1280_sard"]
EXTS = {f"ap_{d}_26{s}": EXT / "aerialperson" / f"{d}_yolo26{s}.pt" for d in ("heridal", "ladd", "nomad", "sard") for s in ("l", "m")}
EXTS |= {"ap_mix_26l": EXT / "aerialperson" / "mix_yolo26l.pt", "aeronova": EXT / "siddh10" / "aeronova.pt",
         "visdrone_yolov9e": EXT / "visdrone-yolov9e" / "best.pt", "visdrone_yolo11x": EXT / "visdrone-yolo11x" / "best.pt",
         "visdrone_yolo26x": EXT / "visdrone-yolo26x" / "best.pt"}


def tiles(W, H, tw=1280, th=720):
    xs = [0] if W <= tw else sorted(set(list(range(0, W - tw + 1, tw // 2)) + [W - tw]))
    ys = [0] if H <= th else sorted(set(list(range(0, H - th + 1, th // 2)) + [H - th]))
    return [(x, y) for y in ys for x in xs]


class Model:
    def __init__(self, path, imgsz):
        self.m = YOLO(str(path)); self.imgsz = imgsz
        names = {v.lower(): k for k, v in self.m.names.items()}
        self.cls = [names[c] for c in ("person", "pedestrian", "people") if c in names]
        assert self.cls, f"사람 클래스 없음: {self.m.names}"

    def __call__(self, img):
        h, w = img.shape[:2]; s = 1920 / w
        im = cv2.resize(img, (1920, max(1, round(h * s))), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        H = im.shape[0]; T = tiles(1920, H)
        crops = [im[y:y + 720, x:x + 1280] for x, y in T]
        res = self.m.predict(crops, imgsz=self.imgsz, conf=0.01, classes=self.cls, half=True, verbose=False, batch=len(crops))
        B, C = [], []
        for (x, y), r in zip(T, res):
            b = r.boxes.xyxy.cpu().numpy(); B.append(b + np.array([x, y, x, y], np.float32)); C.append(r.boxes.conf.cpu().numpy())
        b, c = np.concatenate(B), np.concatenate(C)
        if len(b):
            keep = cv2.dnn.NMSBoxes([[float(a), float(q), float(e - a), float(d - q)] for a, q, e, d in b], c.tolist(), 0.01, 0.6)
            keep = np.array(keep).ravel().astype(int) if len(keep) else np.array([], int); b, c = b[keep], c[keep]
        return b / s, c                                     # 원래 그림 좌표


def score(model, data):
    imgs = sorted((data / "images").glob("*.jpg"))
    if len(imgs) > MAXN:
        imgs = [imgs[i] for i in np.linspace(0, len(imgs) - 1, MAXN).astype(int)]
    S, T = [], []; n_gt = n_neg = fp_neg = 0; ms = []
    for p in imgs:
        img = cv2.imread(str(p)); ih, iw = img.shape[:2]; gt = load_gt(p, iw, ih); n_gt += len(gt)
        t0 = time.perf_counter(); box, cf = model(img); ms.append((time.perf_counter() - t0) * 1000)
        if not len(gt):
            n_neg += 1; fp_neg += int((cf >= 0.2).sum())
        o = np.argsort(-cf); box, cf = box[o], cf[o]; M = iou_mat(gt, box); used = np.zeros(len(gt), bool)
        for j in range(len(box)):
            k = -1
            if len(gt):
                cand = np.where((M[:, j] >= 0.5) & ~used)[0]
                if len(cand):
                    k = cand[np.argmax(M[cand, j])]; used[k] = True
            S.append(float(cf[j])); T.append(k >= 0)
    S, T = np.array(S), np.array(T, bool); sel = S >= 0.2
    return {"imgs": len(imgs), "n_gt": n_gt, "AP50": round(ap101(S, T, n_gt), 4) if n_gt else "",
            "R@0.2": round(float(T[sel].sum() / n_gt), 4) if n_gt else "", "P@0.2": round(float(T[sel].mean()), 4) if sel.any() else "",
            "neg_imgs": n_neg, "FP/neg@0.2": round(fp_neg / n_neg, 3) if n_neg else "", "ms": round(float(np.median(ms)), 1)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--models"); ap.add_argument("--sets"); a = ap.parse_args()
    sets = a.sets.split(",") if a.sets else SETS
    jobs = [(m, "ours", BASE / "runs_person" / m / "weights" / "best.pt", 1280) for m in OURS] + \
           [(m, "ext", p, z) for m, p in EXTS.items() for z in (1280, 640)]
    if a.models:
        jobs = [j for j in jobs if j[0] in a.models.split(",")]
    done = set()
    if OUT.exists():
        done = {(r["model"], int(r["imgsz"]), r["set"]) for r in csv.DictReader(open(OUT))}
    new = not OUT.exists()
    f = open(OUT, "a", newline=""); w = csv.DictWriter(f, fieldnames=["model", "kind", "imgsz", "set", "imgs", "n_gt", "AP50", "R@0.2", "P@0.2", "neg_imgs", "FP/neg@0.2", "ms"])
    if new:
        w.writeheader()
    for name, kind, path, z in jobs:
        todo = [s for s in sets if (name, z, s) not in done and (BASE / "data" / s / "images").exists()]
        if not todo or not path.exists():
            continue
        m = Model(path, z)
        for s in todo:
            r = {"model": name, "kind": kind, "imgsz": z, "set": s} | score(m, BASE / "data" / s)
            w.writerow(r); f.flush(); print(name, z, s, r["AP50"], r["R@0.2"], flush=True)
        del m


if __name__ == "__main__":
    main()
