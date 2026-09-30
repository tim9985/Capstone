"""
posture_foundation.py — 기반 모델로 자세 판별 (상태 인지 A2 비교 · 2026-09-30)

왜
  지난 9번은 작은 데이터로 CNN 을 학습해 장소가 바뀌면 무너졌다. 큰 데이터로 미리 학습된
  기반 모델의 특징은 장소 이동에 강한지 — YOLO11-cls (chain_s1) 와 같은 평가셋 · 같은 지표로 잰다.

방법
  siglip  SigLIP 2 제로샷 — 크롭과 문장 3개("누운 · 앉은 · 선 사람")의 유사도 → 학습 없음
  linear  DINOv2 · SigLIP 2 이미지 특징 + 로지스틱 회귀 — AI-Hub 학습 5곳으로 맞추고 산악8 로 C 고름

평가: eval_posture.py 와 같은 평가셋 · 지표 (aihub_test · okutama_obl · okutama_nadir)
실행 (state 환경 — drone 환경에는 transformers 가 없다):
  /home/se/venvs/state/bin/python posture_foundation.py siglip
  /home/se/venvs/state/bin/python posture_foundation.py linear
출력: metrics/posture_<방법>.json
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from eval_posture import CLASSES, EVAL, SETS, load, metrics

BASE = Path(__file__).resolve().parent
CLS = BASE / "data" / "pose_cls"
DEV = "cuda"
SIGLIP = "google/siglip2-base-patch16-224"
DINO = "vit_base_patch14_dinov2.lvd142m"
PROMPTS = {"lying": ["an aerial photo of a person lying on the ground", "a drone photo of a person lying down"],
           "sitting": ["an aerial photo of a person sitting", "a drone photo of a person sitting on the ground"],
           "standing": ["an aerial photo of a person standing", "a drone photo of a person walking"]}


def batches(files, bs=256):
    for i in range(0, len(files), bs):
        yield [Image.open(f).convert("RGB") for f in files[i:i + bs]]


class Siglip:
    def __init__(self):
        from transformers import AutoModel, AutoProcessor
        self.m = AutoModel.from_pretrained(SIGLIP, torch_dtype=torch.float16).to(DEV).eval()
        self.p = AutoProcessor.from_pretrained(SIGLIP)

    @torch.no_grad()
    def image(self, files):
        out = []
        for ims in batches(files):
            x = self.p(images=ims, return_tensors="pt")["pixel_values"].to(DEV, torch.float16)
            f = self.m.get_image_features(pixel_values=x)
            f = f.pooler_output if hasattr(f, "pooler_output") else f
            out.append(torch.nn.functional.normalize(f.float(), dim=-1).cpu())
        return torch.cat(out).numpy()

    @torch.no_grad()
    def text(self):
        emb = []
        for c in CLASSES:
            t = self.p(text=PROMPTS[c], padding="max_length", max_length=64, return_tensors="pt").to(DEV)
            f = self.m.get_text_features(**t)
            f = f.pooler_output if hasattr(f, "pooler_output") else f
            emb.append(torch.nn.functional.normalize(torch.nn.functional.normalize(f.float(), dim=-1).mean(0), dim=-1))
        return torch.stack(emb).cpu().numpy()


class Dino:
    def __init__(self):
        import timm
        self.m = timm.create_model(DINO, pretrained=True, num_classes=0, img_size=224).to(DEV).eval().half()
        cfg = timm.data.resolve_data_config({}, model=self.m)
        self.t = timm.data.create_transform(**{**cfg, "input_size": (3, 224, 224), "crop_pct": 1.0})

    @torch.no_grad()
    def image(self, files):
        out = []
        for ims in batches(files):
            x = torch.stack([self.t(im) for im in ims]).to(DEV).half()
            out.append(torch.nn.functional.normalize(self.m(x).float(), dim=-1).cpu())
        return torch.cat(out).numpy()


def train_files(split):
    rows = []
    for c in CLASSES:
        rows += [(str(f), CLASSES.index(c)) for f in sorted((CLS / "aihub" / split / c).glob("*.jpg"))]
    return [r[0] for r in rows], np.array([r[1] for r in rows])


def run_eval(name, predict_prob):
    res = {}
    for s in SETS:
        if not (EVAL / s / "crops.csv").exists():
            continue
        files, y, geo, block = load(s)
        prob = predict_prob(files)
        res[s] = r = metrics(y, prob, geo, block, np.random.default_rng(0))
        print(f"== {name} · {s}  n={r['n']}\n   재현율 {r['재현율']} · 매크로 F1 {r['매크로F1']}"
              f"\n   누움 AUROC 모델 {r['누움AUROC_분류기']} {r['95%구간']['누움AUROC_분류기']}"
              f" · 박스 모양 {r['누움AUROC_박스모양']} {r['95%구간']['누움AUROC_박스모양']}", flush=True)
    (BASE / "metrics" / f"posture_{name}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "siglip"
    if mode == "siglip":
        m = Siglip()
        T = m.text()
        run_eval("siglip2_zeroshot", lambda files: torch.softmax(torch.tensor(m.image(files) @ T.T) * 100, 1).numpy())
    elif mode == "linear":
        from sklearn.linear_model import LogisticRegression
        Xtr_f, ytr = train_files("train"); Xva_f, yva = train_files("val")
        for tag, m in (("dinov2", Dino()), ("siglip2", Siglip())):
            Xtr, Xva = m.image(Xtr_f), m.image(Xva_f)
            best = None
            for C in (0.1, 0.3, 1.0, 3.0):
                clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced").fit(Xtr, ytr)
                acc = float((clf.predict(Xva) == yva).mean())
                print(f"   {tag} C={C} 산악8 정확도 {acc:.3f}", flush=True)
                if best is None or acc > best[0]:
                    best = (acc, C, clf)
            print(f"   {tag} 고른 C={best[1]}", flush=True)
            clf = best[2]
            run_eval(f"{tag}_linear", lambda files: clf.predict_proba(m.image(files)))
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
