"""
soup.py — 같은 구조 모델들의 가중치 평균 (model soup · 2026-09-26)

  같은 COCO 가중치에서 출발해 학습한 모델들은 가중치를 그대로 평균해도 동작하고, 실행마다의 흔들림이 줄어
  처음 보는 장소(OOD)에서 더 낫다는 보고가 있다 (Wortsman 2022 Model soups · Ramé 2022 DiWA). 추론 비용은 그대로.
  근거: v6_obl · v6_obl_r2 (같은 설정 · 순서만 다름) 가 test_kr 에서 3.4 %p 차이 (짝 구간 +1.0~+6.6 · 09-26)
  같은 구조여야 한다 (P2 머리 모델은 못 섞는다)
  ⚠ 평균하면 BN 통계(running mean·var)가 실제 활성과 어긋나 출력이 무너질 수 있다 (REPAIR · Jordan 2022)
    → --bn <목록> : 평균 뒤 학습 크롭 N장(기본 512)을 흘려 BN 통계만 다시 잰다 (SWA 의 update_bn 과 같다)
    09-27: v7+v7_r3 평균이 test_kr 0.51 로 무너짐 — BN 재계산 전후 비교
출력: runs_person/<이름>/weights/best.pt  (평가는 다른 모델과 같게 eval_test_v2.py --weights …)
실행: python soup.py soup_v6x2 v6_obl v6_obl_r2 [--bn configs/lists/train_v6.txt] [--bn-n 512]
"""
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
import torch

BASE = Path(__file__).resolve().parent


def bn_reset(model, list_path, n):
    """평균한 모델의 BN 통계를 학습 크롭으로 다시 잰다 (가중치는 그대로)"""
    import random
    import cv2
    import numpy as np
    paths = [l.strip() for l in open(list_path) if l.strip()]
    random.Random(0).shuffle(paths); paths = paths[:n]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.float().to(dev).eval()
    bns = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    for m in bns:
        m.reset_running_stats(); m.momentum = None; m.train()      # 누적 평균
    with torch.no_grad():
        for i in range(0, len(paths), 8):
            xs = []
            for p in paths[i:i + 8]:
                im = cv2.imread(p)
                if im is None:
                    continue
                c = np.full((736, 1280, 3), 114, np.uint8); h, w = min(im.shape[0], 736), min(im.shape[1], 1280)
                c[:h, :w] = im[:h, :w]; xs.append(c[:, :, ::-1].transpose(2, 0, 1))
            if xs:
                model(torch.from_numpy(np.ascontiguousarray(np.stack(xs))).float().div(255).to(dev))
    return model.eval().cpu()


def main():
    args = sys.argv[1:]
    bn_list = bn_n = None
    if "--bn" in args:
        i = args.index("--bn"); bn_list = args[i + 1]; del args[i:i + 2]
    bn_n = 512
    if "--bn-n" in args:
        i = args.index("--bn-n"); bn_n = int(args[i + 1]); del args[i:i + 2]
    out, names = args[0], args[1:]
    assert len(names) >= 2, "재료 모델이 둘 이상 필요하다"
    cks = [torch.load(BASE / "runs_person" / n / "weights" / "best.pt", map_location="cpu", weights_only=False) for n in names]
    sds = [c["model"].float().state_dict() for c in cks]
    for n, s in zip(names, sds):
        assert s.keys() == sds[0].keys() and all(s[k].shape == sds[0][k].shape for k in s), f"{n}: 구조가 다르다"
    avg = {k: sum(s[k] for s in sds) / len(sds) if sds[0][k].is_floating_point() else sds[0][k] for k in sds[0]}
    model = cks[0]["model"]; model.load_state_dict(avg)
    if bn_list:
        model = bn_reset(model, BASE / bn_list, bn_n)
    ck = dict(cks[0], model=model.half(), ema=None, soup=names, soup_bn=bn_list,
              date=datetime.now(timezone(timedelta(hours=9))).isoformat(timespec="seconds"),
              train_args=dict(cks[0]["train_args"], name=out))
    d = BASE / "runs_person" / out / "weights"; d.mkdir(parents=True, exist_ok=True)
    torch.save(ck, d / "best.pt")
    print(f"수프 {out} ← {' + '.join(names)} · 텐서 {len(avg)}개{f' · BN 재계산 {bn_n}장' if bn_list else ''} → {d / 'best.pt'}")


if __name__ == "__main__":
    main()
