"""
soup.py — 같은 구조 모델들의 가중치 평균 (model soup · 2026-09-26)

  같은 COCO 가중치에서 출발해 학습한 모델들은 가중치를 그대로 평균해도 동작하고, 실행마다의 흔들림이 줄어
  처음 보는 장소(OOD)에서 더 낫다는 보고가 있다 (Wortsman 2022 Model soups · Ramé 2022 DiWA). 추론 비용은 그대로.
  근거: v6_obl · v6_obl_r2 (같은 설정 · 순서만 다름) 가 test_kr 에서 3.4 %p 차이 (짝 구간 +1.0~+6.6 · 09-26)
  같은 구조여야 한다 (P2 머리 모델은 못 섞는다) · BN 통계도 같이 평균한다
출력: runs_person/<이름>/weights/best.pt  (평가는 다른 모델과 같게 eval_test_v2.py --weights …)
실행: python soup.py soup_v6x2 v6_obl v6_obl_r2
"""
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
import torch

BASE = Path(__file__).resolve().parent


def main():
    out, names = sys.argv[1], sys.argv[2:]
    assert len(names) >= 2, "재료 모델이 둘 이상 필요하다"
    cks = [torch.load(BASE / "runs_person" / n / "weights" / "best.pt", map_location="cpu", weights_only=False) for n in names]
    sds = [c["model"].float().state_dict() for c in cks]
    for n, s in zip(names, sds):
        assert s.keys() == sds[0].keys() and all(s[k].shape == sds[0][k].shape for k in s), f"{n}: 구조가 다르다"
    avg = {k: sum(s[k] for s in sds) / len(sds) if sds[0][k].is_floating_point() else sds[0][k] for k in sds[0]}
    model = cks[0]["model"]; model.load_state_dict(avg)
    ck = dict(cks[0], model=model.half(), ema=None, soup=names,
              date=datetime.now(timezone(timedelta(hours=9))).isoformat(timespec="seconds"),
              train_args=dict(cks[0]["train_args"], name=out))
    d = BASE / "runs_person" / out / "weights"; d.mkdir(parents=True, exist_ok=True)
    torch.save(ck, d / "best.pt")
    print(f"수프 {out} ← {' + '.join(names)} · 텐서 {len(avg)}개 → {d / 'best.pt'}")


if __name__ == "__main__":
    main()
