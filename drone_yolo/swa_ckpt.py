"""
swa_ckpt.py — 한 학습 안의 마지막 에폭들 가중치 평균 (SWA 꼴 · 10-12 · v12)

  왜  반복 사이 수프는 BN 통계가 어긋나 자주 못 만든다 (BN 거리 mean_z ≥ 0.25 → v10 · S 수프 건너뜀)
      같은 학습 궤적의 마지막 에폭들은 BN 이 맞물려 평균해도 무너지지 않는다 (Izmailov 2018 SWA) · 학습 한 번 값으로 수프 효과
  입력 runs_person/<이름>/weights/epoch<N>.pt (학습 때 --set save_period=1) 의 ema
출력 runs_person/<이름>_swa/weights/best.pt (soup.py 와 같은 꼴 → eval_test_v2.py 그대로)
실행: python swa_ckpt.py v12_f91 [--from 26 --to 30]
"""
import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import torch

BASE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--from", dest="lo", type=int, default=26)
    ap.add_argument("--to", dest="hi", type=int, default=30)
    a = ap.parse_args()
    w = BASE / "runs_person" / a.name / "weights"
    eps = [e for e in range(a.lo, a.hi + 1) if (w / f"epoch{e}.pt").exists()]
    if len(eps) < 2 and (w / "last.pt").exists():
        raise SystemExit(f"에폭 체크포인트가 {len(eps)}개뿐 — save_period=1 로 학습했나?")
    cks = [torch.load(w / f"epoch{e}.pt", map_location="cpu", weights_only=False) for e in eps]
    sds = [(c.get("ema") or c["model"]).float().state_dict() for c in cks]
    avg = {k: sum(s[k] for s in sds) / len(sds) if sds[0][k].is_floating_point() else sds[0][k] for k in sds[0]}
    model = (cks[-1].get("ema") or cks[-1]["model"]).float(); model.load_state_dict(avg)
    ck = dict(cks[-1], model=model.half(), ema=None, optimizer=None, swa_epochs=eps,
              date=datetime.now(timezone(timedelta(hours=9))).isoformat(timespec="seconds"),
              train_args=dict(cks[-1]["train_args"], name=f"{a.name}_swa"))
    d = BASE / "runs_person" / f"{a.name}_swa" / "weights"; d.mkdir(parents=True, exist_ok=True)
    torch.save(ck, d / "best.pt")
    print(f"{a.name}_swa ← 에폭 {eps} 평균 · 텐서 {len(avg)}개 → {d / 'best.pt'}")


if __name__ == "__main__":
    main()
