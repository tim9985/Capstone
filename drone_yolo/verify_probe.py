"""
verify_probe.py — 사람 모양 물체 오탐 줄이기: SigLIP 2 2차 확인 (11-08 · UAV-SAR 정밀도 0.50 지적 → 실험)

  왜    UAV-SAR 문턱 0.20 헛탐지 655 중 583 이 「배경」 · 확신 중앙 0.55 — 눈으로 보니 옷 걸친 흰 의자 · 배낭 · 옷 더미 (데이터셋이 일부러 둔 미끼 · 라벨 없음)
  방법  v12 (운용 4조각 · 문턱 0.20) 박스 → 긴 변 ≥ 40 px (1920 기준) 만 잘라 (사방 20 % 여유) SigLIP 2 base 제로샷 — 사람 문장 5 · 물체 문장 13 소프트맥스 → p(사람)
        p < τ 면 버림 · 40 px 미만은 그대로 (작은 사람은 확인기가 못 봄) · 학습 없음 · worker 1.3 에 이미 있는 SigLIP 2 base (사람당 약 3 ms)
  판정 (돌리기 전 · 11-08)  τ 는 UAV-SAR 30 m (개발) 에서 고름 = 재현율 손해 ≤ 1 %p 안에서 헛탐지 최소 → UAV-SAR 15 m (판정) 헛탐지 −50 % 이상
        그리고 어느 셋도 재현율 −2 %p 넘게 안 떨어짐 (NOMAD 가림 · 누움 · Okutama · 숲 · Cyprus · VisDrone · 한국 · LADD) 그리고 도로 헛탐지 안 늘어남
  출력  metrics/verify_probe.json
실행  /home/se/venvs/state/bin/python verify_probe.py
"""
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

BASE = Path(__file__).resolve().parent
RUN = BASE / "runs_fair"
TAG = "v12_f91_swa@tile1280"
SETS = ["test_sar", "test_nomad", "test_obl", "test_v2", "test_cy", "test_visdrone", "test_kr", "test_ladd_h", "test_auair"]
PERSON = ["an aerial photo of a person", "a person lying on the ground seen from above", "a person sitting on the ground",
          "a person standing or walking", "a person partly hidden by bushes or trees"]
OBJECT = ["a white plastic chair", "a chair with clothes on it", "a backpack", "a bag on the grass", "a pile of clothes",
          "a car", "a traffic cone", "a rock", "a tree", "grass", "a cardboard box", "a road sign", "a cart"]
L_MIN, OP = 40, 0.20


def iou(a, b):
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1]); x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None); ar = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])
    return inter / (ar(a)[:, None] + ar(b)[None, :] - inter + 1e-9)


@torch.no_grad()
def score_sets():
    proc = AutoProcessor.from_pretrained("google/siglip2-base-patch16-224"); m = AutoModel.from_pretrained("google/siglip2-base-patch16-224", dtype=torch.float16).cuda().eval()
    t = proc(text=PERSON + OBJECT, padding="max_length", max_length=64, return_tensors="pt").to("cuda")
    tf = m.get_text_features(input_ids=t["input_ids"]); tf = getattr(tf, "pooler_output", tf).float(); tf = tf / tf.norm(dim=-1, keepdim=True); scale = m.logit_scale.exp().float()
    out = {}
    for s in SETS:
        g = np.load(RUN / s / "gt.npz"); d = np.load(RUN / s / f"{TAG}.npz"); keep = d["conf"] >= OP
        img, box, conf = d["img"][keep], d["box"][keep], d["conf"][keep]; p = np.ones(len(conf), np.float32); L = np.zeros(len(conf), np.float32)
        crops, idx = [], []
        for i in np.unique(img):
            im = cv2.imread(str(BASE / "data" / s / "images" / str(g["names"][i]))); H, W = im.shape[:2]; k = 1920 / W
            for j in np.flatnonzero(img == i):
                x1, y1, x2, y2 = box[j]; L[j] = max(x2 - x1, y2 - y1) * k
                if L[j] < L_MIN:
                    continue
                mx, my = (x2 - x1) * 0.2, (y2 - y1) * 0.2
                c = im[int(max(0, y1 - my)):int(min(H, y2 + my)), int(max(0, x1 - mx)):int(min(W, x2 + mx))]
                if c.size:
                    crops.append(Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))); idx.append(j)
        for a in range(0, len(crops), 128):
            v = proc(images=crops[a:a + 128], return_tensors="pt").to("cuda"); f = m.get_image_features(pixel_values=v["pixel_values"].half()); f = getattr(f, "pooler_output", f).float()
            f = f / f.norm(dim=-1, keepdim=True); pr = torch.softmax((f @ tf.T) * scale, -1).float().cpu().numpy()
            p[idx[a:a + 128]] = pr[:, :len(PERSON)].sum(1)
        out[s] = dict(img=img, box=box, conf=conf, p=p, L=L, g=g); print(s, "박스", len(conf), "확인", len(idx), flush=True)
    return out


def stats(r, tau, sel_img=None):
    g = r["g"]; n = len(g["names"]); keep = r["p"] >= tau; tp = fp = ng = 0; nimg = 0
    for i in range(n):
        if sel_img is not None and not sel_img(i):
            continue
        nimg += 1; gt = g["gt"][g["gt_img"] == i]; ng += len(gt); m = (r["img"] == i) & keep; b, c = r["box"][m], r["conf"][m]
        o = np.argsort(-c); b = b[o]; used = np.zeros(len(gt), bool); M = iou(gt, b) if len(gt) and len(b) else np.zeros((len(gt), len(b)))
        for j in range(len(b)):
            cand = np.where((M[:, j] >= 0.5) & ~used)[0]
            if len(cand):
                used[cand[np.argmax(M[cand, j])]] = True; tp += 1
            else:
                fp += 1
    return dict(P=tp / max(tp + fp, 1), R=tp / max(ng, 1), FP=fp, FP_img=fp / max(nimg, 1))


def main():
    S = score_sets(); taus = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
    names = S["test_sar"]["g"]["names"]; dev = lambda i: str(names[i]).split("_")[1].startswith("30m"); test = lambda i: str(names[i]).split("_")[1].startswith("15m")
    base_dev = stats(S["test_sar"], 0.0, dev)
    ok = [t for t in taus if base_dev["R"] - stats(S["test_sar"], t, dev)["R"] <= 0.01]
    tau = min(ok, key=lambda t: stats(S["test_sar"], t, dev)["FP"]); print("고른 τ (UAV-SAR 30 m)", tau)
    res = {"τ": tau, "개발 30m": {t: stats(S["test_sar"], t, dev) for t in taus}, "참고 15m 곡선": {t: stats(S["test_sar"], t, test) for t in taus}, "판정": {}}
    for k in ("개발 30m", "참고 15m 곡선"):
        print(k, " · ".join(f"τ{t}: R {v['R']:.3f} 헛 {v['FP']}" for t, v in res[k].items()))
    # SigLIP 이 진짜 사람 · 헛탐지를 얼마나 가르나 (UAV-SAR · 확인한 박스만)
    r = S["test_sar"]; g = r["g"]; lab = np.zeros(len(r["conf"]), int) - 1
    for i in range(len(g["names"])):
        gt = g["gt"][g["gt_img"] == i]; js = np.flatnonzero(r["img"] == i)
        if len(js):
            M = iou(gt, r["box"][js]) if len(gt) else np.zeros((0, len(js)))
            for q, j in enumerate(js):
                lab[j] = int(len(gt) > 0 and M[:, q].max() >= 0.5)
    chk = r["L"] >= L_MIN; pos, neg = r["p"][chk & (lab == 1)], r["p"][chk & (lab == 0)]
    auc = float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())
    res["UAV-SAR 사람 vs 헛 AUROC"] = auc; print("UAV-SAR p(사람) — 사람 박스 중앙", round(float(np.median(pos)), 3), "· 헛 박스 중앙", round(float(np.median(neg)), 3), "· AUROC", round(auc, 3))
    for s in SETS:
        sel = test if s == "test_sar" else None
        b, a = stats(S[s], 0.0, sel), stats(S[s], tau, sel)
        res["판정"][s + (" (15 m)" if s == "test_sar" else "")] = {"전": b, "후": a}
        print(f"{s:14s} P {b['P']:.3f}→{a['P']:.3f} · R {b['R']:.3f}→{a['R']:.3f} ({(a['R'] - b['R']) * 100:+.1f}%p) · 헛 {b['FP_img']:.2f}→{a['FP_img']:.2f} 건/장", flush=True)
    sar = res["판정"]["test_sar (15 m)"]; worst = min((v["후"]["R"] - v["전"]["R"]) for k, v in res["판정"].items() if k != "test_auair")
    road = res["판정"]["test_auair"]
    res["합격"] = dict(헛탐지_감소=1 - sar["후"]["FP"] / max(sar["전"]["FP"], 1), 최악_재현율_손해=worst, 도로=[road["전"]["FP_img"], road["후"]["FP_img"]],
                     판정=bool(1 - sar["후"]["FP"] / max(sar["전"]["FP"], 1) >= 0.5 and worst >= -0.02 and road["후"]["FP_img"] <= road["전"]["FP_img"] + 1e-9))
    print("합격", res["합격"])
    json.dump(res, open(BASE / "metrics" / "verify_probe.json", "w"), ensure_ascii=False, indent=1, default=float)


if __name__ == "__main__":
    main()
