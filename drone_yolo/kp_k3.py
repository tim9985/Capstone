"""
kp_k3.py — K3: 관절점 모델 15종 비교 (10-10 · 판정 기준은 _학습 큐 「K3」)

  A  FlyPose-104 (실제 항공 정답 193명 · 높이 1080) — 몸통 4점 OKS ≥ 0.5 비율 (못 찾으면 틀림) · 전체 · 서기 · 땅 누움
  B  「박스가 서기처럼 보이는 곳」 (세로 ≥ 가로 · 긴 변 ≥ 45) — 특징 = 박스 2 + 물리 4 (몸통 · 다리 기울기 · 길이 · 크롭 긴 변 기준)
     못 찾으면 물리 4 = 비움 → 학습 평균으로 채움 (찾음 표시 없음 = 「못 찾음 = 누움」 지름길 차단)
     ① Archangel (누움 900 · 서기 3,000 무작위 · 회차 5겹 교차검증) ② Archangel 로 맞추고 UE 에서 (누움 2,246 · 서기 2,385)
     짝 부트스트랩 1,000회 — 박스만 대비 · FlyPose-H 대비 (UE 사진 블록 · Archangel 회차 블록)
  속도  한 사람 ms 중앙 (B 실행 중 · 3090 TensorRT FP32 · RTMO 는 CPU)
  관절은 모델마다 runs_state/k3/<모델>.npz 에 둔다 (git 밖 · 파생물)
실행: /home/se/miniconda3/envs/drone/bin/python kp_k3.py [--models a,b,…]   → metrics/kp_k3.json · .md
"""
import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import kp_flypose as KF
import kp_models as KM
from posture_v2 import auc

BASE = Path(__file__).resolve().parent
ARCH = BASE.parent / "data" / "pose_eval" / "archangel_real"
UE = BASE / "data" / "pose_cls" / "ue_level01"
OUT = BASE / "runs_state" / "k3"


def crops(src, csvp, sub, n_stand=None, seed=0):
    rows = []
    for r in csv.DictReader(open(csvp, encoding="utf-8")):
        w, h = float(r["w1080"]), float(r["h1080"])
        if r["pose"] in ("standing", "lying") and max(w, h) >= 45 and h >= w:
            rows.append(r | {"w": w, "h": h, "L": max(w, h), "path": str(src / sub / r["file"]) if sub else str(src / r["file"])})
    if n_stand:
        st = [r for r in rows if r["pose"] == "standing"]; ly = [r for r in rows if r["pose"] == "lying"]
        rows = ly + random.Random(seed).sample(st, min(n_stand, len(st)))
    return rows


def crop_box(img, r):
    side = img.shape[0] / 1.5; c = img.shape[0] / 2
    bw, bh = side * r["w"] / r["L"], side * r["h"] / r["L"]
    return (c - bw / 2, c - bh / 2, c + bw / 2, c + bh / 2), side, c


def geom(xy, side):
    sm, hm, am = (xy[5] + xy[6]) / 2, (xy[11] + xy[12]) / 2, (xy[15] + xy[16]) / 2
    v, u = hm - sm, am - hm
    return [abs(np.arctan2(v[0], v[1])), np.linalg.norm(v) / side, abs(np.arctan2(u[0], u[1])), np.linalg.norm(u) / side]


def run_model(name, A, arch, ue):
    f = OUT / f"{name}.npz"
    if f.exists():
        z = np.load(f, allow_pickle=True); return {k: z[k] for k in z.files}
    m = KM.REGISTRY[name](); res = {"A_ok": [], "A_n": [], "Bg_arch": [], "Bg_ue": [], "ms": []}
    for fr, box, g, area in A:
        p = m(fr, box); pts = [i for i in KF.TORSO if g[i, 2] > 0]
        ok = 0 if p is None else sum(np.exp(-np.sum((p[0][i] - g[i, :2]) ** 2) / (2 * area * (2 * KF.SIGMA[i]) ** 2)) >= 0.5 for i in pts)
        res["A_ok"].append(int(ok)); res["A_n"].append(len(pts))
    for key, rows in (("Bg_arch", arch), ("Bg_ue", ue)):
        for r in rows:
            img = cv2.imread(r["path"]); box, side, c = crop_box(img, r)
            p, ms = KM.timed(m, img, box); res["ms"].append(ms)
            res[key].append([np.nan] * 4 if p is None else geom(p[0], side))
    out = {k: np.array(v, float) for k, v in res.items()}
    OUT.mkdir(parents=True, exist_ok=True); np.savez(f, **out)
    return out


def boot_diff(pa, pb, y, blocks, B=1000, seed=0):
    ub = sorted(set(blocks)); idx = defaultdict(list)
    for i, b in enumerate(blocks):
        idx[b].append(i)
    rng = np.random.default_rng(seed); d = []
    for _ in range(B):
        s = np.concatenate([idx[b] for b in rng.choice(ub, len(ub))]); ys = y[s]
        if ys.any() and (~ys).any():
            d.append(auc(pb[s][ys], pb[s][~ys]) - auc(pa[s][ys], pa[s][~ys]))
    return [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(KM.REGISTRY))
    a = ap.parse_args()
    ann = json.load(open(KF.DATA / "annotations.json")); imgs = {i["id"]: i for i in ann["images"]}
    A, Agrp, cache = [], [], {}
    for k, an in enumerate(ann["annotations"]):
        im = imgs[an["image_id"]]
        if im["id"] not in cache:
            cache.clear(); img = cv2.imread(str(KF.DATA / "frames" / im["file_name"])); s = min(1.0, 1080 / img.shape[0])
            cache[im["id"]] = (cv2.resize(img, (round(img.shape[1] * s), round(img.shape[0] * s)), interpolation=cv2.INTER_AREA) if s < 1 else img, s)
        fr, s = cache[im["id"]]; x, y, w, h = [v * s for v in an["bbox"]]
        g = np.array(an["keypoints"], np.float32).reshape(17, 3); g[:, :2] *= s
        if any(g[i, 2] > 0 for i in KF.TORSO):
            A.append((fr, (x, y, x + w, y + h), g, w * h)); Agrp.append(KF.group(k))
    Agrp = np.array(Agrp)
    arch = crops(ARCH, ARCH / "crops.csv", None, n_stand=3000); ue = crops(UE, UE / "crops.csv", "crops")
    ya = np.array([r["pose"] == "lying" for r in arch]); yu = np.array([r["pose"] == "lying" for r in ue])
    ga = np.array(["_".join(r["place"].split("_")[:3]) for r in arch]); gu = np.array([r["file"].split("/")[-1].split("_Person")[0] for r in ue])
    box = lambda rows: np.c_[np.log(np.array([r["h"] / r["w"] for r in rows])), np.log(np.array([r["L"] for r in rows]))]
    Ba, Bu = box(arch), box(ue)
    pipe = lambda: make_pipeline(SimpleImputer(strategy="mean"), StandardScaler(), LogisticRegression(max_iter=3000))

    def score(Xa, Xu):
        p = np.zeros(len(ya))
        for tr, te in GroupKFold(5).split(Xa, ya, ga):
            p[te] = pipe().fit(Xa[tr], ya[tr]).predict_proba(Xa[te])[:, 1]
        return p, pipe().fit(Xa, ya).predict_proba(Xu)[:, 1]
    P = {"box": score(Ba, Bu)}; res = {"A 사람": len(A), "B Archangel 누움 · 서기": [int(ya.sum()), int((~ya).sum())],
                                       "B UE 누움 · 서기": [int(yu.sum()), int((~yu).sum())], "models": {}}
    res["models"]["box"] = {"B① Archangel AUROC": round(float(auc(P["box"][0][ya], P["box"][0][~ya])), 4),
                            "B② UE AUROC": round(float(auc(P["box"][1][yu], P["box"][1][~yu])), 4)}
    for name in a.models.split(","):
        r = run_model(name, A, arch, ue); print("관절", name, flush=True)
        Ga, Gu = r["Bg_arch"], r["Bg_ue"]
        P[name] = score(np.c_[Ba, Ga], np.c_[Bu, Gu])
        ok, n = r["A_ok"], r["A_n"]
        A_rate = lambda msk: round(float(ok[msk].sum() / max(n[msk].sum(), 1)), 3)
        res["models"][name] = {
            "A 몸통 OKS 전체 · 서기 · 땅 누움": [A_rate(np.ones(len(ok), bool)), A_rate(Agrp == "upright"), A_rate(Agrp == "lying")],
            "B 못 찾음 (Archangel · UE)": [round(float(np.isnan(Ga[:, 0]).mean()), 3), round(float(np.isnan(Gu[:, 0]).mean()), 3)],
            "B① Archangel AUROC": round(float(auc(P[name][0][ya], P[name][0][~ya])), 4),
            "B② UE AUROC": round(float(auc(P[name][1][yu], P[name][1][~yu])), 4),
            "B② − 박스 95%": boot_diff(P["box"][1], P[name][1], yu, gu), "B① − 박스 95%": boot_diff(P["box"][0], P[name][0], ya, ga),
            "ms 중앙": round(float(np.median(r["ms"])), 1)}
        (BASE / "metrics" / "kp_k3.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    if "flypose_h" in P:
        for name in a.models.split(","):
            if name != "flypose_h":
                res["models"][name]["B② − FlyPose-H 95%"] = boot_diff(P["flypose_h"][1], P[name][1], yu, gu)
    ranked = sorted([n for n in res["models"] if n != "box"], key=lambda n: -res["models"][n]["B② UE AUROC"])
    top = ranked[0]; near = [n for n in ranked if res["models"][top]["B② UE AUROC"] - res["models"][n]["B② UE AUROC"] <= 0.02]
    res["순위 (B② UE)"] = ranked; res["K2c 후보"] = {"1위": top, "0.02 안에서 가장 빠름": min(near, key=lambda n: res["models"][n]["ms 중앙"])}
    (BASE / "metrics" / "kp_k3.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    lines = ["| 모델 | A 몸통 OKS 전체 · 서기 · 땅 누움 | B 못 찾음 | B① Archangel | B② UE | B② − 박스 | ms |", "|---|---|---|---|---|---|---|",
             f"| 박스만 | — | — | {res['models']['box']['B① Archangel AUROC']} | {res['models']['box']['B② UE AUROC']} | — | — |"]
    for n in ranked:
        v = res["models"][n]
        lines.append(f"| {n} | {' · '.join(map(str, v['A 몸통 OKS 전체 · 서기 · 땅 누움']))} | {v['B 못 찾음 (Archangel · UE)']} | {v['B① Archangel AUROC']} | {v['B② UE AUROC']} | {v['B② − 박스 95%']} | {v['ms 중앙']} |")
    (BASE / "metrics" / "kp_k3.md").write_text("\n".join(lines) + f"\n\nK2c 후보: {res['K2c 후보']}\n")
    print("\n".join(lines)); print(res["K2c 후보"])


if __name__ == "__main__":
    main()
