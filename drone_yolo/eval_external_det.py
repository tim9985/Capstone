"""
eval_external_det.py — E1: 외부 항공 사람 검출기 (FlyPose 검출기 · RT-DETRv2-S) 를 우리 평가셋으로 (10-10 · 판정 기준은 _학습 큐 「K2b · E1」)

  검출기  data/raw/flypose/checkpoints/detector — TensorRT FP32 (3090) · 학습 데이터 VisDrone · Manipal · HIT-UAV · TinyPerson ·
          SeaDronesSee · HERIDAL · VTSaR · DroneRGBT · VTUAV · COCO 사람 → 우리 평가셋 (WiSARD · Okutama · AI-Hub) 과 겹치지 않음
  조건    official  공식 전처리 그대로 — 한 장 통째로 1280×1280 으로 늘림 · BGR · /255 · 점수 전 구간 (상위 300)
          rgb       같은데 RGB 로 (민감도)
          tile      우리 방식 — 1920×1080 → 1280×720 타일 4장 (겹침 50 %) · 타일마다 1280×1280 으로 늘림 · NMS 0.6 (test_obl 은 원본이 1280×720 이라 official 과 같음 → 생략)
  채점    eval_test_v2 함수 그대로 (정답 · IoU 0.5 · 101점 AP50 · 블록) · 짝 비교용 runs_person/ext_flypose_<조건>/eval_<tag>.npz (compare_ci 와 같은 형식)
실행: /home/se/miniconda3/envs/drone/bin/python eval_external_det.py [--sets test_v2,test_obl,test_kr] [--modes official,rgb,tile]
"""
import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np
import tensorrt as trt
import torch

from eval_test_v2 import TILES, TH, TW, ap101, blocks_of, boot_ap, iou_mat, load_gt

BASE = Path(__file__).resolve().parent
ENGINE = BASE.parent / "data" / "raw" / "flypose" / "checkpoints" / "detector" / "flyposeDetector_fp32_3090.engine"
SETS = {"test_v2": BASE / "data" / "test_v2", "test_obl": BASE / "data" / "test_obl", "test_kr": BASE / "data" / "test_kr"}


class Det:
    def __init__(self, path):
        log = trt.Logger(trt.Logger.WARNING)
        self.e = trt.Runtime(log).deserialize_cuda_engine(open(path, "rb").read()); self.ctx = self.e.create_execution_context()
        self.st = torch.cuda.Stream()

    def __call__(self, imgs, rgb=False):
        """imgs: BGR 그림 목록 (≤ 4) → [(박스 xyxy 원본 좌표, 점수)]"""
        n = len(imgs)
        x = np.stack([cv2.resize(im[:, :, ::-1] if rgb else im, (1280, 1280), interpolation=cv2.INTER_LINEAR) for im in imgs]).astype(np.float32) / 255.0
        x = torch.from_numpy(np.ascontiguousarray(x.transpose(0, 3, 1, 2))).cuda()
        sz = torch.tensor([[im.shape[1], im.shape[0]] for im in imgs], dtype=torch.int64, device="cuda")
        lab = torch.empty(n, 300, dtype=torch.int64, device="cuda"); box = torch.empty(n, 300, 4, device="cuda"); sc = torch.empty(n, 300, device="cuda")
        self.ctx.set_input_shape("images", tuple(x.shape)); self.ctx.set_input_shape("orig_target_sizes", tuple(sz.shape))
        for name, t in (("images", x), ("orig_target_sizes", sz), ("labels", lab), ("boxes", box), ("scores", sc)):
            self.ctx.set_tensor_address(name, t.data_ptr())
        self.ctx.execute_async_v3(self.st.cuda_stream); self.st.synchronize()
        out = []
        for i in range(n):
            m = (lab[i] == 0) & (sc[i] >= 0.01)
            out.append((box[i][m].cpu().numpy(), sc[i][m].cpu().numpy()))
        return out


def run(det, data, mode, tag, op_conf=(0.3, 0.5, 0.7)):
    imgs = sorted((data / "images").glob("*.jpg"))
    scores, tps, pimg, ngt_img, names = [], [], [], [], []
    n_gt = n_neg = 0; fp_neg = {c: 0 for c in op_conf}; ms = []
    for p in imgs:
        img = cv2.imread(str(p)); ih, iw = img.shape[:2]
        gt = load_gt(p, iw, ih); n_gt += len(gt); ngt_img.append(len(gt)); names.append(p.stem); ii = len(ngt_img) - 1
        t0 = time.perf_counter()
        if mode == "tile" and (iw, ih) == (1920, 1080):
            res = det([img[y:y + TH, x:x + TW] for x, y in TILES])
            box = np.concatenate([b + np.array([x, y, x, y], np.float32) for (x, y), (b, _) in zip(TILES, res)]); cf = np.concatenate([s for _, s in res])
            if len(box):
                keep = cv2.dnn.NMSBoxes([[float(a), float(b), float(c - a), float(d - b)] for a, b, c, d in box], cf.tolist(), 0.01, 0.6)
                keep = np.array(keep).ravel().astype(int) if len(keep) else np.array([], int); box, cf = box[keep], cf[keep]
        else:
            box, cf = det([img], rgb=(mode == "rgb"))[0]
        ms.append((time.perf_counter() - t0) * 1000)
        if not len(gt):
            n_neg += 1
            for c in op_conf:
                fp_neg[c] += int((cf >= c).sum())
        o = np.argsort(-cf); box, cf = box[o], cf[o]
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
    S, T = np.array(scores), np.array(tps, bool)
    ap = ap101(S, T, n_gt)
    o = np.argsort(-S); tp = np.cumsum(T[o]); prec = tp / np.arange(1, len(o) + 1)
    rp = {c: (round(float(tp[S[o] >= c][-1] / n_gt), 4) if (S >= c).any() else 0.0, round(float(prec[S[o] >= c][-1]), 4) if (S >= c).any() else 0.0) for c in op_conf}
    blk, nblk, ngrp = blocks_of(names); I, G = np.array(pimg, int), np.array(ngt_img, float)
    bs = boot_ap(S, T, I, G, blk, nblk, 1000); lo, hi = np.percentile(bs, [2.5, 97.5])
    out = BASE / "runs_person" / f"ext_flypose_{mode}"; out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"eval_{tag}.npz", s=S, t=T, img=I, ngt=G, names=np.array(names))
    return {"set": tag, "mode": mode, "n_gt": n_gt, "AP50": round(ap, 4), "ci": [round(float(lo), 4), round(float(hi), 4)],
            "recall/precision@conf": rp, "fp_per_neg_frame@conf": {c: round(v / max(n_neg, 1), 3) for c, v in fp_neg.items()},
            "neg_frames": n_neg, "ms_median": round(float(np.median(ms)), 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="test_v2,test_obl,test_kr")
    ap.add_argument("--modes", default="official,rgb,tile")
    a = ap.parse_args()
    det = Det(ENGINE); rows = []
    for tag in a.sets.split(","):
        for mode in a.modes.split(","):
            if mode == "tile" and tag == "test_obl":
                continue
            r = run(det, SETS[tag], mode, tag); rows.append(r); print(r, flush=True)
    with open(BASE / "metrics" / "eval_external_det.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["set", "mode", "n_gt", "AP50", "ci_lo", "ci_hi", "recall@0.3", "recall@0.7", "fp_neg@0.3", "ms"])
        for r in rows:
            w.writerow([r["set"], r["mode"], r["n_gt"], r["AP50"], *r["ci"], r["recall/precision@conf"][0.3][0], r["recall/precision@conf"][0.7][0],
                        r["fp_per_neg_frame@conf"][0.3], r["ms_median"]])


if __name__ == "__main__":
    main()
