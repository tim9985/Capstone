"""
posture_ft.py — 자세 판별 P4: DINOv2 일부 미세조정 + 시점별 머리 (2026-10-03)

계획 → obsidian 「11 서버 학습 계획/08 자세 판별 학습 계획」
  10-01 YOLO cls 전체 미세조정은 비스듬 앉음이 무너졌다 → 백본은 **마지막 블록 4개만** 풀고 머리를 시점별로 나눈다

  백본   DINOv2 ViT-B/14 (posture_foundation 과 같은 가중치) · 224 px · 마지막 블록 4개 + norm 학습
  머리   수직 · 비스듬 각각 Linear(768 + 보조 5 → 3) · 보조 = 박스 [log 세로/가로 · log 긴 변] + 상대 키 [log rel_h · log rel_long · 결측]
  데이터 수직 = AI-Hub 학습 5곳 (P3 에서 C2A 가 뽑히면 + C2A) · 비스듬 = P3 가 고른 출처 (SN 또는 SNC)
         `metrics/posture_v2_p3.json` 「고른 것」 을 그대로 따른다 — Okutama 는 안 본다
  표집   (머리 × 자세) 균형 · 에폭 = 30,000 장 · 8 에폭 고정 (val 로 고르지 않는다)
  증강   크기 0.7~1 자르기 · 좌우 뒤집기 · 밝기/대비 0.2 · 수직은 90° 회전 무작위
  최적화 AdamW · 백본 2e-5 · 머리 1e-3 · 가중치 감쇠 0.05 · 코사인 · FP16 자동 혼합
  반복   --seed 0 · 1 · 2 (순수 PyTorch 라 seed 가 실제로 바뀐다 — ultralytics 와 다름)
출력  runs_posture/ft_s<seed>/{model.pt, probs.npz} (git 밖) · metrics/posture_ft_s<seed>.json
실행  /home/se/venvs/state/bin/python posture_ft.py --seed=0 [--smoke]
"""
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

import posture_v2 as pv

BASE = Path(__file__).resolve().parent
SEED = next((int(a.split("=")[1]) for a in sys.argv[1:] if a.startswith("--seed=")), 0)
SMOKE = "--smoke" in sys.argv
EPOCHS, PER_EPOCH, BS = (1, 256, 64) if SMOKE else (8, 30000, 96)
DEV = "cuda"


def aux(D):
    w, h = np.maximum(D["w"], 1), np.maximum(D["h"], 1)
    return np.c_[np.log(h / w), np.log(np.maximum(w, h)), D["relh"]].astype(np.float32)


class Crops(torch.utils.data.Dataset):
    def __init__(self, files, y, head, ax, tf, train):
        self.f, self.y, self.head, self.ax, self.tf, self.train = files, y, head, ax, tf, train

    def __len__(self):
        return len(self.f)

    def __getitem__(self, i):
        im = Image.open(self.f[i]).convert("RGB")
        if self.train and self.head[i] == 0 and random.random() < 0.75:      # 수직: 90° 회전 무작위
            im = im.rotate(90 * random.randint(1, 3))
        return self.tf(im), self.y[i], self.head[i], self.ax[i]


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        import timm
        self.b = timm.create_model(pv_dino(), pretrained=True, num_classes=0, img_size=224)
        for p in self.b.parameters():
            p.requires_grad = False
        for m in list(self.b.blocks[-4:]) + [self.b.norm]:
            for p in m.parameters():
                p.requires_grad = True
        self.heads = nn.ModuleList([nn.Linear(768 + 5, 3), nn.Linear(768 + 5, 3)])

    def forward(self, x, head, ax):
        f = torch.cat([self.b(x), ax], 1)
        out = torch.stack([h(f) for h in self.heads], 1)                       # (n, 2, 3)
        return out[torch.arange(len(x)), head]


def pv_dino():
    from posture_foundation import DINO
    return DINO


def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    pv.SMOKE = SMOKE
    import timm
    chosen = json.loads((BASE / "metrics" / "posture_v2_p3.json").read_text())["고른 것"] if not SMOKE else \
        {"oblique": ["box", "SNC"], "nadir": ["dino", "A"]}
    obl_src, nad_src = pv.OBL_SRC[chosen["oblique"][1]], pv.NAD_SRC[chosen["nadir"][1]]
    tr = [(s, 0) for s in nad_src] + [(s, 1) for s in obl_src]
    sets = {s: pv.load_set(s) for s in {s for s, _ in tr} | set(pv.EVAL_SETS)}
    files = sum((sets[s]["files"] for s, _ in tr), [])
    y = np.concatenate([sets[s]["y"] for s, _ in tr]); head = np.concatenate([np.full(len(sets[s]["y"]), h) for s, h in tr])
    ax = np.concatenate([aux(sets[s]) for s, _ in tr]); mu, sd = ax.mean(0), ax.std(0) + 1e-6
    norm = lambda a: ((a - mu) / sd).astype(np.float32)
    print(f"seed {SEED} · 수직 {nad_src} · 비스듬 {obl_src} · 학습 크롭 {len(y):,}", flush=True)

    net = Net().to(DEV)
    cfg = timm.data.resolve_data_config({}, model=net.b)
    import torchvision.transforms as T
    tf_tr = T.Compose([T.RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(0.9, 1.1)), T.RandomHorizontalFlip(),
                       T.ColorJitter(0.2, 0.2), T.ToTensor(), T.Normalize(cfg["mean"], cfg["std"])])
    tf_ev = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(cfg["mean"], cfg["std"])])
    key = head * 3 + y; cnt = np.bincount(key, minlength=6)
    wts = 1.0 / np.maximum(cnt[key], 1)
    sampler = torch.utils.data.WeightedRandomSampler(torch.tensor(wts, dtype=torch.double), PER_EPOCH, replacement=True,
                                                     generator=torch.Generator().manual_seed(SEED))
    dl = torch.utils.data.DataLoader(Crops(files, y, head, norm(ax), tf_tr, True), batch_size=BS, sampler=sampler,
                                     num_workers=10, drop_last=True, persistent_workers=True)
    bb = [p for p in net.b.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": bb, "lr": 2e-5}, {"params": net.heads.parameters(), "lr": 1e-3}], weight_decay=0.05)
    steps = EPOCHS * len(dl); sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[2e-5, 1e-3], total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler()
    lossf = nn.CrossEntropyLoss()
    for ep in range(EPOCHS):
        net.train(); t0, tot, n = time.time(), 0.0, 0
        for x, yy, hh, aa in dl:
            x, yy, hh, aa = x.to(DEV, non_blocking=True), yy.to(DEV), hh.to(DEV), aa.to(DEV)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = lossf(net(x, hh, aa).float(), yy)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item() * len(yy); n += len(yy)
        print(f"  에폭 {ep + 1}/{EPOCHS} 손실 {tot / n:.4f} · {time.time() - t0:.0f} 초", flush=True)

    net.eval(); probs, res = {}, {}
    for s in pv.EVAL_SETS:
        D = sets[s]; h = 0 if pv.VIEW[s] == "nadir" else 1
        ds = Crops(D["files"], D["y"], np.full(len(D["y"]), h), norm(aux(D)), tf_ev, False)
        out = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            for x, _, hh, aa in torch.utils.data.DataLoader(ds, batch_size=256, num_workers=10):
                out.append(torch.softmax(net(x.to(DEV), hh.to(DEV), aa.to(DEV)).float(), 1).cpu().numpy())
        probs[s] = np.concatenate(out)
        res[s] = pv.full_metrics(D, probs[s])
        print(f"== {s}: 누움 AUROC {res[s]['누움AUROC_분류기']} · 매크로 F1 {res[s]['매크로F1']} · 재현율 {res[s]['재현율']}", flush=True)
    out = BASE / "runs_posture" / ("smoke_ft" if SMOKE else f"ft_s{SEED}"); out.mkdir(parents=True, exist_ok=True)
    if SMOKE:                                                  # 런타임 연기 실행용 모델만 (judge4 는 ft_s* 만 본다)
        torch.save({"state": {k: v.half() for k, v in net.state_dict().items()}, "mu": mu, "sd": sd, "chosen": chosen}, out / "model.pt")
        print("연기 실행 끝 (runs_posture/smoke_ft)"); return
    np.savez(out / "probs.npz", **probs)
    torch.save({"state": {k: v.half() for k, v in net.state_dict().items()}, "mu": mu, "sd": sd, "chosen": chosen}, out / "model.pt")
    (BASE / "metrics" / f"posture_ft_s{SEED}.json").write_text(json.dumps({"seed": SEED, "고른 출처": chosen, **res}, ensure_ascii=False, indent=1, default=float))


if __name__ == "__main__":
    main()
