"""
ignore_regions.py — test_v2 무시 영역 (10-12 · 사용자 결정 「라벨 구멍은 무시 영역으로」)

  파일  data/test_v2/ignore_v1.json (git 밖 · WiSARD 파생)
        roi   비행 → 평가 구역 [x1, y1, x2, y2] — Carnation_Inspire 는 x < 1605 만 라벨이 있다 (정답 x2 최대 1605 · 띠 안에 라벨 없는 사람)
        boxes 그림 → 무시 박스 — 세 모델의 확신 높은 「배경」 오탐을 트랙으로 묶어 눈으로 본 것 중 사람 14 · 애매 27 트랙 (물체 280 은 그대로 오탐)
  채점  ① 평가 구역이 있는 비행: 검출 박스를 구역으로 자른다 (정답도 그 경계에서 잘려 있다) · 남은 넓이 < 50 % 면 「구역 밖」
        ② 점수 순 탐욕 짝 (IoU ≥ 0.5) — 지금과 같음
        ③ 짝 없는 검출이 「구역 밖」 이거나 무시 박스에 50 % 이상 들어가면 **세지 않는다** (맞음도 오탐도 아님) · 정답 수는 그대로
"""
import json
from pathlib import Path

import numpy as np

DEFAULT = Path(__file__).resolve().parent / "data" / "test_v2" / "ignore_v1.json"


class Ignore:
    def __init__(self, path=DEFAULT):
        d = json.load(open(path, encoding="utf-8"))
        self.roi = {k: np.array(v, np.float32) for k, v in d.get("roi", {}).items()}
        self.boxes = {k: np.array(v, np.float32).reshape(-1, 4) for k, v in d.get("boxes", {}).items()}
        self.name = Path(path).stem

    def prepare(self, img_name, b):
        """검출 박스 b (N×4) → (구역으로 자른 박스, 구역 밖 여부 N)"""
        r = self.roi.get(img_name.rsplit("_", 1)[0])
        if r is None or not len(b):
            return b, np.zeros(len(b), bool)
        c = np.c_[np.maximum(b[:, 0], r[0]), np.maximum(b[:, 1], r[1]), np.minimum(b[:, 2], r[2]), np.minimum(b[:, 3], r[3])]
        a0 = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        a1 = np.clip(c[:, 2] - c[:, 0], 0, None) * np.clip(c[:, 3] - c[:, 1], 0, None)
        out = a1 < 0.5 * np.maximum(a0, 1e-6)
        c[out] = b[out]                                   # 구역 밖 박스는 자르지 않고 둔다 (어차피 세지 않음)
        return c.astype(b.dtype), out

    def skip(self, img_name, box, out):
        """짝 없는 검출을 세지 않을지"""
        if out:
            return True
        ib = self.boxes.get(img_name)
        if ib is None:
            return False
        x1 = np.maximum(box[0], ib[:, 0]); y1 = np.maximum(box[1], ib[:, 1]); x2 = np.minimum(box[2], ib[:, 2]); y2 = np.minimum(box[3], ib[:, 3])
        ioa = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None) / max(float((box[2] - box[0]) * (box[3] - box[1])), 1e-6)
        return bool((ioa >= 0.5).any())


def ap_ign(dets, gts, names, ign, thr=0.5, sel=None):
    """analyze_dets.ap_at 과 같은 채점 + 무시 영역 (덤프로 지난 모델 다시 재기)"""
    from eval_test_v2 import ap101, iou_mat
    S, T, n = [], [], 0
    for i, gt in enumerate(gts):
        if sel is not None and not sel(i):
            continue
        b, s = dets[i]; n += len(gt)
        o = np.argsort(-s); b, s = b[o], s[o]
        b, out = ign.prepare(names[i], b)
        M = iou_mat(gt, b) if len(gt) else np.zeros((0, len(b))); used = np.zeros(len(gt), bool)
        for j in range(len(b)):
            k = -1
            if len(gt):
                c = np.where((M[:, j] >= thr) & ~used)[0]
                if len(c):
                    k = c[np.argmax(M[c, j])]; used[k] = True
            if k < 0 and ign.skip(names[i], b[j], out[j]):
                continue
            S.append(s[j]); T.append(k >= 0)
    return round(ap101(np.array(S), np.array(T, bool), n), 4) if n else None
