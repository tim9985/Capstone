"""
kp_flypose.py — K1b: 항공 관절점 모델 (FlyPose) 이 누운 몸통을 더 잘 찾는가 (10-10 · 판정 기준은 _학습 큐 「K1b 기준」)

  데이터  FlyPose-104 (실제 항공 · COCO 17 관절 정답 · 104장 193명 · data/raw/flypose · git 밖 · 사용자 폼 신청)
          그림을 높이 1080 으로 줄여 worker 크기로 · 정답 박스로 크롭 (탐지기 안 씀)
  자세    모델을 돌리기 전에 크롭을 눈으로 봐서 붙인 표시 (POSTURE · _학습 큐 와 같음)
  모델    FlyPose-S · FlyPose-H — 공식 inference.py 전처리 그대로 (패딩 1.1 · 아핀 256×192 · RGB 정규화 · 히트맵 최댓값 + 0.25 보정)
          yolo11s-pose · yolo11m-pose — K1 방식 (긴 변 0.5 여백 크롭 → 긴 변 256 으로 키움 → imgsz 640 · IoU 최대 사람)
  지표    몸통 4점 (어깨 · 골반) PCK@0.1 (긴 변 · 정답 v>0 점만) — 주: 확신 문턱 없이 · 보조: 모델 문턱 (FlyPose 0.2 · yolo 0.5)
          보조 (첫 결과를 본 뒤 더함 · 10-10): COCO OKS 관절별 ≥ 0.5 (어깨 σ 0.079 · 골반 0.107 · s² = 박스 넓이) — 사람 정답과 모델의
          관절 위치 습관 차 (두꺼운 옷 · 배낭 10~20 px) 를 PCK@0.1 이 모두 틀림으로 쳐서 절대값이 낮다 · 비교 (차) 는 둘 다 본다
          크기 (1080 긴 변) · 자세 묶음 · 그림 블록 부트스트랩 1,000회 (S − m)
실행: /home/se/miniconda3/envs/drone/bin/python kp_flypose.py   → metrics/kp_flypose.json (수치만)
"""
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO

BASE = Path(__file__).resolve().parent
FP = BASE.parent / "data" / "raw" / "flypose"
DATA = FP / "FlyPose-104" / "FlyPose-104"
TORSO = (5, 6, 11, 12)
SIGMA = {5: 0.079, 6: 0.079, 11: 0.107, 12: 0.107}          # COCO 관절별 σ
LYING = {10, 17, 23, 24, 79, 111, 112, 113, 170, 171, 172, 173, 177}
WATER = {15, 16, 18, 19, 35, 36, 48, 51, 52, 53, 54, 84, 85, 101, 102, 103, 149, 150}
UNCLEAR = {2, 5, 11, 12, 13, 14, 22, 25, 30, 31, 47, 49, 56, 57, 59, 60, 62, 65, 66, 69, 82, 83, 89, 98, 104, 115, 116, 117,
           141, 142, 145, 146, 148, 152, 153, 154, 158, 191, 192}
BINS = [(0, 45), (45, 61), (61, 81), (81, 1e9)]
POSE_IN = (192, 256)                       # (w, h) — 공식 default_config
MEAN, STD = np.array([123.675, 116.28, 103.53], np.float32), np.array([58.395, 57.12, 57.375], np.float32)


def group(k):
    return "lying" if k in LYING else "water" if k in WATER else "unclear" if k in UNCLEAR else "upright"


# ---- FlyPose 공식 inference.py 그대로 (preprocess_pose_crop · decode_pose · refine_peak) ----
def fp_preprocess(frame, box, padding=1.1):
    x1, y1, x2, y2 = box
    c = np.array([(x1 + x2) / 2, (y1 + y2) / 2], np.float32)
    wh = np.array([max((x2 - x1) * padding, 1.0), max((y2 - y1) * padding, 1.0)], np.float32)
    iw, ih = POSE_IN
    s = min(iw / wh[0], ih / wh[1])
    M = np.array([[s, 0, iw / 2 - s * c[0]], [0, s, ih / 2 - s * c[1]]], np.float32)
    warped = cv2.warpAffine(frame, M, (iw, ih), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    t = (warped.astype(np.float32)[:, :, ::-1] - MEAN) / STD
    return np.transpose(t, (2, 0, 1))[None].copy(), cv2.invertAffineTransform(M)


def fp_decode(hm, inv):
    k, hh, hw = hm.shape
    iw, ih = POSE_IN
    kp, sc = np.empty((k, 2), np.float32), np.empty(k, np.float32)
    for i in range(k):
        m = hm[i]; f = int(m.argmax()); y, x = f // hw, f % hw
        x, y = float(x), float(y)
        if 1 <= int(x) < hw - 1 and 1 <= int(y) < hh - 1:
            x += 0.25 * np.sign(m[int(y), int(x) + 1] - m[int(y), int(x) - 1])
            y += 0.25 * np.sign(m[int(y) + 1, int(x)] - m[int(y) - 1, int(x)])
        px, py = x * iw / hw, y * ih / hh
        kp[i] = (inv[0, 0] * px + inv[0, 1] * py + inv[0, 2], inv[1, 0] * px + inv[1, 1] * py + inv[1, 2])
        sc[i] = float(m[int(np.clip(round(y), 0, hh - 1)), int(np.clip(round(x), 0, hw - 1))])
    return kp, sc


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u if u > 0 else 0.0


def yolo_kp(model, frame, box):
    """K1 방식 — 여백 크롭 → 긴 변 256 으로 → IoU 최대 사람 → 그림 좌표"""
    x1, y1, x2, y2 = box; L = max(x2 - x1, y2 - y1); m = 0.5 * L
    X1, Y1 = int(max(0, x1 - m)), int(max(0, y1 - m)); X2, Y2 = int(min(frame.shape[1], x2 + m)), int(min(frame.shape[0], y2 + m))
    crop = frame[Y1:Y2, X1:X2]; f = 256 / L
    up = cv2.resize(crop, (max(1, round(crop.shape[1] * f)), max(1, round(crop.shape[0] * f))), interpolation=cv2.INTER_LINEAR)
    k = up.shape[1] / crop.shape[1]
    r = model.predict(up, imgsz=640, conf=0.1, verbose=False, half=True)[0]
    if r.keypoints is None or r.boxes is None or len(r.boxes) == 0:
        return None
    gb = np.array([x1 - X1, y1 - Y1, x2 - X1, y2 - Y1]) * k
    b = r.boxes.xyxy.cpu().numpy(); j = max(range(len(b)), key=lambda i: iou(b[i], gb))
    if iou(b[j], gb) < 0.3:
        return None
    xy = r.keypoints.xy.cpu().numpy()[j] / k + np.array([X1, Y1]); c = r.keypoints.conf.cpu().numpy()[j]
    return xy, c


def main():
    ann = json.load(open(DATA / "annotations.json"))
    imgs = {i["id"]: i for i in ann["images"]}
    prov = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    fps = {n: ort.InferenceSession(str(FP / "checkpoints" / "pose" / d / "end2end.onnx"), providers=prov) for n, d in (("flypose_s", "flypose_s"), ("flypose_h", "flypose_h"))}
    print("onnx providers", {n: s.get_providers()[0] for n, s in fps.items()})
    yolos = {n: YOLO(str(FP.parent / "pose_weights" / f"{n}.pt")) for n in ("yolo11s-pose", "yolo11m-pose")}
    thr = {"flypose_s": 0.2, "flypose_h": 0.2, "yolo11s-pose": 0.5, "yolo11m-pose": 0.5}
    rows = []                                           # (사람, 그림, 자세, 긴 변, 모델, 맞음[문턱 없음], 맞음[문턱], 점 수, 맞음[OKS ≥ 0.5])
    frames = {}
    for k, a in enumerate(ann["annotations"]):
        im = imgs[a["image_id"]]
        if im["id"] not in frames:
            frames.clear(); img = cv2.imread(str(DATA / "frames" / im["file_name"])); s = 1080 / img.shape[0]
            frames[im["id"]] = (cv2.resize(img, (round(img.shape[1] * s), 1080), interpolation=cv2.INTER_AREA) if s < 1 else img, min(s, 1.0))
        frame, s = frames[im["id"]]
        x, y, w, h = [v * s for v in a["bbox"]]; box = (x, y, x + w, y + h); L = max(w, h)
        kp = np.array(a["keypoints"], np.float32).reshape(17, 3); gxy = kp[:, :2] * s; vis = kp[:, 2]
        pts = [i for i in TORSO if vis[i] > 0]
        if not pts:
            continue
        preds = {}
        for n, sess in fps.items():
            t, inv = fp_preprocess(frame, box)
            preds[n] = fp_decode(sess.run(None, {"input": t})[0][0], inv)
        for n, m in yolos.items():
            preds[n] = yolo_kp(m, frame, box)
        for n, p in preds.items():
            if p is None:
                rows.append((k, im["id"], group(k), L, n, 0, 0, len(pts), 0)); continue
            d = [np.linalg.norm(p[0][i] - gxy[i]) <= 0.1 * L for i in pts]
            g = [ok and p[1][i] >= thr[n] for ok, i in zip(d, pts)]
            area = w * h
            o = [np.exp(-np.sum((p[0][i] - gxy[i]) ** 2) / (2 * area * (2 * SIGMA[i]) ** 2)) >= 0.5 for i in pts]
            rows.append((k, im["id"], group(k), L, n, sum(d), sum(g), len(pts), sum(o)))
    models = list(fps) + list(yolos)

    def pck(sel, col=5):
        r = [x for x in rows if sel(x)]
        return round(sum(x[col] for x in r) / max(sum(x[7] for x in r), 1), 3), len({x[0] for x in r})
    res = {"persons": len({x[0] for x in rows}), "groups": {g: len({x[0] for x in rows if x[2] == g}) for g in ("upright", "lying", "water", "unclear")},
           "table": {}, "table_threshold": {}, "table_oks": {}}
    sels = {"all": lambda x: True, "upright": lambda x: x[2] == "upright", "horizontal": lambda x: x[2] in ("lying", "water"),
            "lying": lambda x: x[2] == "lying", "water": lambda x: x[2] == "water"}
    for n in models:
        for gname, gs in sels.items():
            res["table"][f"{n} {gname}"] = {f"{a}-{b if b < 1e9 else '∞'}": pck(lambda x, a=a, b=b: x[4] == n and gs(x) and a <= x[3] < b) for a, b in BINS} | {"전체": pck(lambda x: x[4] == n and gs(x))}
            res["table_threshold"][f"{n} {gname}"] = pck(lambda x: x[4] == n and gs(x), col=6)
            res["table_oks"][f"{n} {gname}"] = {f"{a}-{b if b < 1e9 else '∞'}": pck(lambda x, a=a, b=b: x[4] == n and gs(x) and a <= x[3] < b, col=8) for a, b in BINS} | {"전체": pck(lambda x: x[4] == n and gs(x), col=8)}
    # 그림 블록 부트스트랩 — (FlyPose-S − yolo11m) · (FlyPose-H − yolo11m)
    by_img = defaultdict(list)
    for x in rows:
        by_img[x[1]].append(x)
    ids = sorted(by_img); rng = np.random.default_rng(0); diffs = defaultdict(list)
    for _ in range(1000):
        sample = [x for i in rng.choice(ids, len(ids)) for x in by_img[i]]
        for gname, gs in sels.items():
            for col, tag in ((5, ""), (8, " [OKS]")):
                acc = {}
                for n in ("flypose_s", "flypose_h", "yolo11m-pose"):
                    r = [x for x in sample if x[4] == n and gs(x)]; tot = sum(x[7] for x in r)
                    acc[n] = sum(x[col] for x in r) / tot if tot else None
                for n in ("flypose_s", "flypose_h"):
                    if acc[n] is not None and acc["yolo11m-pose"] is not None:
                        diffs[f"{n} − yolo11m {gname}{tag}"].append(acc[n] - acc["yolo11m-pose"])
    res["diff_95"] = {k: [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)] for k, v in diffs.items()}
    hz = res["diff_95"]["flypose_s − yolo11m horizontal"]; al = res["diff_95"]["flypose_s − yolo11m all"]
    res["K1b 판정"] = {"① 수평 몸 (S − m) > 0 → K2 를 FlyPose-S 로": bool(hz[0] > 0), "② 전체 (S − m) > 0": bool(al[0] > 0)}
    (BASE / "metrics" / "kp_flypose.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
