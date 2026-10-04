"""
bn_dist.py — 수프 재료끼리 BN 통계가 얼마나 다른가 (R2 · 2026-10-05)

  무너진 수프 (v7+v7_r3 · v9_s33+s34 · v9 4개) 와 멀쩡한 수프 (v7+v7_r2 · v9_s31+s32 · y26) 를 같은 지표로 본다
  지표 (BN 층마다 → 층 평균 · 최댓값)
    mean_z  = |μa − μb| / sqrt((σa² + σb²)/2)    (running_mean 차를 표준편차로)
    logvar  = |log σa² − log σb²|
    w_cos   = 1 − cos(가중치 a, 가중치 b) (합성곱 · 층 평균) — 두 모델이 같은 골짜기에 있나
실행: python bn_dist.py v7:v7_r2 v7:v7_r3 v9_s31:v9_s32 v9_s33:v9_s34 …
출력: metrics/bn_dist.json · 콘솔
"""
import json
import sys
from pathlib import Path

import torch

BASE = Path(__file__).resolve().parent


def sd(name):
    ck = torch.load(BASE / "runs_person" / name / "weights" / "best.pt", map_location="cpu", weights_only=False)
    m = ck.get("ema") or ck["model"]
    return {k: v.float() for k, v in m.state_dict().items()}


def dist(a, b):
    mz, lv, wc = [], [], []
    for k in a:
        if k.endswith("running_mean") and k in b:
            v = k[:-len("running_mean")] + "running_var"
            s = ((a[v] + b[v]) / 2).clamp_min(1e-6).sqrt()
            mz.append(((a[k] - b[k]).abs() / s).mean().item())
            lv.append((a[v].clamp_min(1e-6).log() - b[v].clamp_min(1e-6).log()).abs().mean().item())
        elif k.endswith("conv.weight") and k in b and a[k].shape == b[k].shape:
            wc.append(1 - torch.nn.functional.cosine_similarity(a[k].flatten(), b[k].flatten(), dim=0).item())
    f = lambda x: {"평균": round(sum(x) / len(x), 4), "최대": round(max(x), 4)} if x else None
    return {"mean_z": f(mz), "logvar": f(lv), "w_cos": f(wc), "BN층": len(mz)}


def main():
    out_p = BASE / "metrics" / "bn_dist.json"
    out = json.loads(out_p.read_text()) if out_p.exists() else {}
    for pair in sys.argv[1:]:
        x, y = pair.split(":")
        try:
            d = dist(sd(x), sd(y))
        except FileNotFoundError as e:
            print(f"{pair}: 없음 ({e.filename})"); continue
        out[pair] = d
        print(f"{pair:24s} mean_z {d['mean_z']['평균']:.3f} (최대 {d['mean_z']['최대']:.2f}) · logvar {d['logvar']['평균']:.3f} · w_cos {d['w_cos']['평균']:.4f}")
    out_p.write_text(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
