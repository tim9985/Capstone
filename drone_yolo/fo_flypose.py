"""
fo_flypose.py — FlyPose-104 (항공 관절점 정답 · 104장 193명) 를 FiftyOne `raw/flypose104` 로 (10-10)

  필드  ground_truth  정답 박스 (자세 표시 posture · 1080 긴 변 long_px · 보이는 관절 수)
        gt_keypoints  정답 17관절 (보이지 않는 점은 빔) · 뼈대 연결은 데이터셋 기본 skeleton
        pred_flypose_h · pred_flypose_s · pred_yolo11m  관절 예측 (정답 박스로 크롭 · kp_flypose.py 와 같은 방식)
                      사람마다 torso_ok = 몸통 4점 중 OKS ≥ 0.5 인 수 · torso_n = 정답이 있는 몸통 점 수
  자세  모델을 돌리기 전에 눈으로 붙인 표시 (kp_flypose.py 의 LYING · WATER · UNCLEAR) — lying · water · upright · unclear
  보기  lying_ground · water · small_45_61 (운용 크기) · yolo_torso_miss · h_torso_miss (몸통을 다 못 맞힌 사람)
실행: /home/se/miniconda3/envs/drone/bin/python fo_flypose.py   (CPU onnx · 2~3분)
"""
import json

import cv2
import fiftyone as fo
import numpy as np
import onnxruntime as ort
from fiftyone import ViewField as F
from ultralytics import YOLO

import kp_flypose as K

NAMES = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
         "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]
EDGES = [[0, 1], [0, 2], [1, 3], [2, 4], [5, 6], [5, 7], [7, 9], [6, 8], [8, 10], [5, 11], [6, 12], [11, 12], [11, 13], [13, 15], [12, 14], [14, 16]]


def kp_label(xy, W, H, conf=None, vis=None, **attrs):
    pts = [(float(x) / W, float(y) / H) if (vis is None or vis[i] > 0) else (float("nan"), float("nan")) for i, (x, y) in enumerate(xy)]
    kw = {"confidence": [float(c) for c in conf]} if conf is not None else {}
    return fo.Keypoint(label="person", points=pts, **kw, **attrs)


def torso_ok(pred, gxy, vis, area):
    pts = [i for i in K.TORSO if vis[i] > 0]
    ok = sum(np.exp(-np.sum((pred[i] - gxy[i]) ** 2) / (2 * area * (2 * K.SIGMA[i]) ** 2)) >= 0.5 for i in pts)
    return int(ok), len(pts)


def main():
    name = "raw/flypose104"
    ann = json.load(open(K.DATA / "annotations.json")); imgs = {i["id"]: i for i in ann["images"]}
    by_img = {}
    for k, a in enumerate(ann["annotations"]):
        by_img.setdefault(a["image_id"], []).append((k, a))
    fps = {n: ort.InferenceSession(str(K.FP / "checkpoints" / "pose" / n / "end2end.onnx"), providers=["CPUExecutionProvider"]) for n in ("flypose_h", "flypose_s")}
    ym = YOLO(str(K.FP.parent / "pose_weights" / "yolo11m-pose.pt"))
    if name in fo.list_datasets():
        fo.delete_dataset(name)
    ds = fo.Dataset(name, persistent=True)
    ds.default_skeleton = fo.KeypointSkeleton(labels=NAMES, edges=EDGES)
    samples = []
    for iid, im in sorted(imgs.items()):
        img = cv2.imread(str(K.DATA / "frames" / im["file_name"])); s = min(1.0, 1080 / img.shape[0])
        fr = cv2.resize(img, (round(img.shape[1] * s), round(img.shape[0] * s)), interpolation=cv2.INTER_AREA) if s < 1 else img
        W, H = fr.shape[1], fr.shape[0]
        dets, gts, preds = [], [], {"pred_flypose_h": [], "pred_flypose_s": [], "pred_yolo11m": []}
        groups = set()
        for k, a in by_img.get(iid, []):
            x, y, w, h = [v * s for v in a["bbox"]]; box = (x, y, x + w, y + h)
            kp = np.array(a["keypoints"], np.float32).reshape(17, 3); gxy = kp[:, :2] * s; vis = kp[:, 2]
            g = K.group(k); groups.add(g)
            dets.append(fo.Detection(label="person", bounding_box=[x / W, y / H, w / W, h / H], posture=g, person_idx=k,
                                     long_px=round(max(w, h), 1), visible_kp=int((vis > 0).sum())))
            gts.append(kp_label(gxy, W, H, vis=vis, posture=g, person_idx=k))
            out = {}
            for n, sess in fps.items():
                t, inv = K.fp_preprocess(fr, box); out[f"pred_{n}"] = K.fp_decode(sess.run(None, {"input": t})[0][0], inv)
            out["pred_yolo11m"] = K.yolo_kp(ym, fr, box)
            for f, p in out.items():
                if p is None:
                    continue
                ok, n_ = torso_ok(p[0], gxy, vis, w * h)
                preds[f].append(kp_label(p[0], W, H, conf=p[1], posture=g, person_idx=k, torso_ok=ok, torso_n=n_, long_px=round(max(w, h), 1)))
        smp = fo.Sample(filepath=str(K.DATA / "frames" / im["file_name"]), tags=sorted(groups),
                        ground_truth=fo.Detections(detections=dets), gt_keypoints=fo.Keypoints(keypoints=gts),
                        **{f: fo.Keypoints(keypoints=v) for f, v in preds.items()})
        samples.append(smp)
    ds.add_samples(samples)
    ds.compute_metadata()
    ds.info = {"출처": "FlyPose-104 (Farooq 외 · WACV 2026 · CC BY 4.0 · 연구용 · 폼 신청)", "자세": "모델 돌리기 전 눈으로 붙인 표시",
               "결과": "_학습 큐 K1b — 몸통 OKS: H 0.55 · yolo11m 0.25 · S 0.16 (땅 누움 H 0.67)"}
    ds.add_dynamic_sample_fields()
    ds.save_view("lying_ground", ds.filter_labels("ground_truth", F("posture") == "lying"), overwrite=True)
    ds.save_view("water", ds.filter_labels("ground_truth", F("posture") == "water"), overwrite=True)
    ds.save_view("small_45_61", ds.filter_labels("ground_truth", (F("long_px") >= 45) & (F("long_px") < 61)), overwrite=True)
    ds.save_view("yolo_torso_miss", ds.filter_labels("pred_yolo11m", F("torso_ok") < F("torso_n")), overwrite=True)
    ds.save_view("h_torso_miss", ds.filter_labels("pred_flypose_h", F("torso_ok") < F("torso_n")), overwrite=True)
    ds.save()
    print(name, len(ds), "장 ·", sum(len(s.ground_truth.detections) for s in ds), "명 · 보기", ds.list_saved_views())


if __name__ == "__main__":
    main()
