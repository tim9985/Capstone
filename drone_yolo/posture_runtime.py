"""
posture_runtime.py — 상태 파이프라인 안에서 쓰는 자세 판정기 (P5 · 2026-10-03)

  box    10-01 v1 — 비스듬 박스 모양 3자세 로지스틱 (SARD + NOMAD)
  p3     posture_v2.py p3 가 고른 비스듬 판정기 (weights/posture_v2_p3.pkl) — 박스 · 상대 키 · DINOv2 · SigLIP 2 조합
  ft_sN  posture_ft.py 미세조정 모델의 비스듬 머리 (runs_posture/ft_sN/model.pt)

  입력: 1280×720 영상 + 추적 박스 (같은 1초 표본의 모든 사람) → 자세 확률 (lying · sitting · standing)
  박스 크기는 1080p 환산 (×1.5) · 크롭은 make_posture_crops.crop 그대로 (평가 크롭과 같은 방식)
  상대 키: 이 영상에서 지금까지 본 다른 추적 박스 중 화면 높이 ±72 px 띠의 높이 · 긴 변 중앙값 (인과적 — 미래 안 봄)
"""
import csv
import os
import pickle

import cv2
import numpy as np
import torch
from PIL import Image

BASE = os.path.dirname(os.path.abspath(__file__))
TO1080 = 1080 / 720


def posture_model():
    """비스듬 박스 모양 3자세 판정 — SARD + NOMAD 크롭 (posture_fusion 과 같은 특징)."""
    from sklearn.linear_model import LogisticRegression
    X, y = [], []
    for part in ("sard", "nomad"):
        for r in csv.DictReader(open(os.path.join(BASE, "data", "pose_cls", part, "train", "crops.csv"))):
            w, h = max(float(r["w1080"]), 1), max(float(r["h1080"]), 1)
            X.append([np.log(h / w), np.log(max(w, h))]); y.append(("lying", "sitting", "standing").index(r["pose"]))
    return LogisticRegression(max_iter=2000, class_weight="balanced").fit(np.array(X), np.array(y))


class Runtime:
    def __init__(self, spec="box"):
        self.spec, self.hist = spec, []
        self.need_dino = self.need_sig = False
        if spec == "box":
            self.pm = posture_model()
        elif spec == "p3":
            m = pickle.load(open(os.path.join(BASE, "weights", "posture_v2_p3.pkl"), "rb"))["oblique"]
            self.kind, self.clf = m["kind"], m["clf"]
            self.need_dino, self.need_sig = "dino" in self.kind, "sig" in self.kind
        elif spec.startswith(("ft_s", "smoke_ft")):
            import timm
            import torchvision.transforms as T
            from posture_ft import Net
            ck = torch.load(os.path.join(BASE, "runs_posture", spec, "model.pt"), weights_only=False)
            self.net = Net().cuda(); self.net.load_state_dict({k: v.float() for k, v in ck["state"].items()}); self.net.eval()
            self.mu, self.sd = ck["mu"], ck["sd"]
            cfg = timm.data.resolve_data_config({}, model=self.net.b)
            self.tf = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(cfg["mean"], cfg["std"])])
        else:
            raise ValueError(spec)
        if self.need_dino:
            from posture_foundation import Dino
            self.dino = Dino()
        if self.need_sig:
            from posture_foundation import Siglip
            self.sig = Siglip(); self.T = self.sig.text()

    def reset(self):
        self.hist = []

    def _relh(self, t, b):
        cy, h, lng = (b[1] + b[3]) / 2, b[3] - b[1], max(b[2] - b[0], b[3] - b[1])
        ref = [(hh, ll) for (c, hh, ll, tt) in self.hist if tt != t and abs(c - cy) < 72]
        if not ref:
            return [0.0, 0.0, 1.0]
        r = np.array(ref)
        return [float(np.log(np.clip(h / np.median(r[:, 0]), 0.05, 20))), float(np.log(np.clip(lng / np.median(r[:, 1]), 0.05, 20))), 0.0]

    @torch.no_grad()
    def predict(self, img, cur):
        """cur = {추적 ID: ((x1, y1, x2, y2), 확신도)} (1280×720 px) → {추적 ID: 확률 3개}."""
        if not cur:
            return {}
        from make_posture_crops import crop
        tids = list(cur)
        boxes = [cur[t][0] for t in tids]
        wh = np.array([[max((b[2] - b[0]) * TO1080, 1), max((b[3] - b[1]) * TO1080, 1)] for b in boxes])
        boxf = np.c_[np.log(wh[:, 1] / wh[:, 0]), np.log(wh.max(1))]
        relh = np.array([self._relh(t, b) for t, b in zip(tids, boxes)])
        for t, b in zip(tids, boxes):                                         # 이번 표본도 이후 기준에 넣는다
            self.hist.append(((b[1] + b[3]) / 2, b[3] - b[1], max(b[2] - b[0], b[3] - b[1]), t))
        if self.spec == "box":
            return dict(zip(tids, self.pm.predict_proba(boxf)))
        ims = [Image.fromarray(cv2.cvtColor(crop(img, b, 1.0), cv2.COLOR_BGR2RGB)) for b in boxes] \
            if (self.need_dino or self.need_sig or self.spec.startswith(("ft_s", "smoke_ft"))) else []
        if self.spec.startswith(("ft_s", "smoke_ft")):
            x = torch.stack([self.tf(im) for im in ims]).cuda()
            ax = torch.tensor(((np.c_[boxf, relh] - self.mu) / self.sd).astype(np.float32)).cuda()
            with torch.autocast("cuda", dtype=torch.float16):
                p = torch.softmax(self.net(x, torch.ones(len(ims), dtype=torch.long).cuda(), ax).float(), 1)
            return dict(zip(tids, p.cpu().numpy()))
        cols = []
        for part in self.kind.split("+"):
            if part == "box":
                cols.append(boxf)
            elif part == "relh":
                cols.append(relh)
            elif part == "dino":
                x = torch.stack([self.dino.t(im) for im in ims]).cuda().half()
                cols.append(torch.nn.functional.normalize(self.dino.m(x).float(), dim=-1).cpu().numpy())
            elif part == "sig":
                x = self.sig.p(images=ims, return_tensors="pt")["pixel_values"].cuda().half()
                f = self.sig.m.get_image_features(pixel_values=x)
                f = f.pooler_output if hasattr(f, "pooler_output") else f
                z = 100 * (torch.nn.functional.normalize(f.float(), dim=-1).cpu().numpy() @ self.T.T)
                z = np.exp(z - z.max(1, keepdims=True)); cols.append(np.log(np.clip(z / z.sum(1, keepdims=True), 1e-4, 1)))
        p = self.clf.predict_proba(np.concatenate(cols, 1)); out = np.zeros((len(p), 3))
        for j, c in enumerate(self.clf.classes_):
            out[:, c] = p[:, j]
        return dict(zip(tids, out))
