"""
adabn_probe.py — 라벨 없이 현장 영상으로 BN 통계만 다시 맞추기 (AdaBN · 11-08 성능 향상 검토)

  왜   실기체 첫 비행 영상 (라벨 없음) 만으로 할 수 있는 현장 적응 — 학습 · 라벨 0 · 탐지 시간 그대로 (BN 은 Conv 에 합쳐짐)
       Li 2016 「Revisiting Batch Normalization for Practical Domain Adaptation」 · 지금까지 BN 재계산은 **학습 데이터** 로만 해 봄 (soup --bn · 박스 넓어짐) — 현장 데이터로는 처음
  방법 v12_f91_swa 의 BN 통계를 대상 그림 (운용과 같은 타일 1280×720 · 레터박스 1280×736) 으로 다시 계산 → 원래 통계와 섞기 (α = 현장 몫 · 1.0 = 순수 AdaBN)
  판정 평가셋을 시퀀스 (장소) 반으로 — A 로 맞추고 B 로 잼 · B 로 맞추고 A 로 잼 → 두 쪽 합 AP (맞춘 그림으로 재지 않음)
  출력 runs_person/adabn_<셋>_<쪽>_a<α>/weights/best.pt (dump_dets.py 로 뽑아 diag 와 같은 채점)
실행  /home/se/miniconda3/envs/drone/bin/python adabn_probe.py --set test_obl --alpha 1.0,0.5
"""
import argparse, copy
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np
import torch

BASE = Path(__file__).resolve().parent
SRC_W = BASE / "runs_person" / "v12_f91_swa" / "weights" / "best.pt"


def halves(names):
    """시퀀스 (이름 앞부분) 를 번갈아 A · B 로"""
    seqs = sorted({n.rsplit("_", 1)[0] for n in names}); A = set(seqs[0::2])
    return [n for n in names if n.rsplit("_", 1)[0] in A], [n for n in names if n.rsplit("_", 1)[0] not in A]


def tiles_of(img):
    h, w = img.shape[:2]
    if w <= 1280:
        return [img]
    s = 1920 / w; im = cv2.resize(img, (1920, round(h * s))) if s != 1 else img
    return [im[y:y + 720, x:x + 1280] for y in (0, im.shape[0] - 720) for x in (0, 640)]


def prep(t):
    h, w = t.shape[:2]; r = 1280 / max(h, w); t = cv2.resize(t, (round(w * r), round(h * r))) if r != 1 else t
    H = int(np.ceil(t.shape[0] / 32) * 32); W = int(np.ceil(t.shape[1] / 32) * 32); out = np.full((H, W, 3), 114, np.uint8)
    y0, x0 = (H - t.shape[0]) // 2, (W - t.shape[1]) // 2; out[y0:y0 + t.shape[0], x0:x0 + t.shape[1]] = t
    return torch.from_numpy(out[:, :, ::-1].copy()).permute(2, 0, 1).float() / 255


def adapt(files, alpha, bs=8):
    ck = torch.load(SRC_W, map_location="cpu", weights_only=False); net = ck["model"].float().cuda().eval()
    bns = [m for m in net.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    src = [(m.running_mean.clone(), m.running_var.clone()) for m in bns]
    for m in bns:
        m.reset_running_stats(); m.momentum = None; m.train()                  # 누적 평균 = 대상 전체 통계
    batch = []
    with torch.no_grad():
        for f in files:
            batch += [prep(t) for t in tiles_of(cv2.imread(str(f)))]
            while len(batch) >= bs:
                net(torch.stack(batch[:bs]).cuda()); batch = batch[bs:]
        if batch:
            net(torch.stack(batch).cuda())
    for m, (mu, var) in zip(bns, src):
        m.running_mean.mul_(alpha).add_(mu.cuda() * (1 - alpha)); m.running_var.mul_(alpha).add_(var.cuda() * (1 - alpha)); m.eval()
    return ck, net


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--set", default="test_obl"); ap.add_argument("--alpha", default="1.0,0.5"); a = ap.parse_args()
    D = BASE / "data" / a.set / "images"; names = sorted(p.stem for p in D.glob("*.jpg")); A, B = halves(names)
    for al in [float(x) for x in a.alpha.split(",")]:
        for side, fit in (("A", A), ("B", B)):
            ck, net = adapt([D / f"{n}.jpg" for n in fit], al)
            ck = dict(ck, model=copy.deepcopy(net).half(), ema=None, optimizer=None, adabn=dict(set=a.set, side=side, alpha=al, n=len(fit)),
                      date=datetime.now(timezone(timedelta(hours=9))).isoformat(timespec="seconds"))
            name = f"adabn_{a.set}_{side}_a{al:g}"; d = BASE / "runs_person" / name / "weights"; d.mkdir(parents=True, exist_ok=True)
            torch.save(ck, d / "best.pt"); print(name, "맞춘 그림", len(fit), flush=True)
    (BASE / "runs_person" / f"adabn_{a.set}_split.txt").write_text("A\n" + "\n".join(A) + "\nB\n" + "\n".join(B) + "\n")


if __name__ == "__main__":
    main()
