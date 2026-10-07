"""
export_posture_models.py — worker 용 자세 판별 계수 내보내기 (V4 · 2026-10-07)

  worker 컨테이너에는 학습 데이터가 없다 → 박스 2특징 로지스틱의 계수만 JSON 으로 (numpy 로 그대로 계산)
    score  B0 = SARD + NOMAD (posture_runtime.posture_model 과 같음) — 상태 점수 · 등급의 누움 · 앉음 항 (Q1 에서 검증한 것)
    display V3 = Archangel 실제 (무릎 → 앉음) + UE level01 (자세마다 3,000 · 잘린 박스 제외) · 표준화 + 로지스틱
           (posture_mannequin_eval 의 "V3 실제+UE" 와 같은 뽑기 · K2 · M 에서 채택한 표시용 판정기)
  특징  [log(h/w), log(max(w, h))] · w · h = 1080p 환산 박스 px
  확인  sklearn predict_proba 와 numpy 계산 차이 · V3 를 Okutama 비스듬 정답 박스로 다시 재서 K2 기록 (앉음 F1 0.4219 · 누움 AUROC 0.9611) 과 비교
  출력  <out>/posture.json — 모델 묶음 (/models) 에 같이 넣는다 · ⚠ 연구용 데이터 파생물 → git 에 안 올림

실행: /home/se/miniconda3/envs/drone/bin/python export_posture_models.py --out=<모델 묶음 폴더>
"""
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from posture_fusion import box_feat
from posture_v2 import auc

BASE = Path(__file__).resolve().parent
CLASSES = ["lying", "sitting", "standing"]
MAP = {"lying": 0, "kneeling": 1, "sitting": 1, "standing": 2}


def table(path, keep=lambda r: True):
    R = [r for r in csv.DictReader(open(path)) if r["pose"] in MAP and keep(r)]
    return box_feat(np.array([[float(r["w1080"]), float(r["h1080"])] for r in R])), np.array([MAP[r["pose"]] for r in R])


def fit_b0():
    X, y = [], []
    for part in ("sard", "nomad"):
        for r in csv.DictReader(open(BASE / "data" / "pose_cls" / part / "train" / "crops.csv")):
            w, h = max(float(r["w1080"]), 1), max(float(r["h1080"]), 1)
            X.append([np.log(h / w), np.log(max(w, h))]); y.append(CLASSES.index(r["pose"]))
    X, y = np.array(X), np.array(y)
    return LogisticRegression(max_iter=2000, class_weight="balanced").fit(X, y), X, {c: int((y == i).sum()) for i, c in enumerate(CLASSES)}


def fit_v3():
    rng = np.random.default_rng(0)                         # posture_mannequin_eval 과 같은 순서로 뽑는다
    Xr, yr = table(BASE / "data/pose_eval/archangel_real/crops.csv")
    Xu, yu = table(BASE / "data/pose_cls/ue_level01/crops.csv", lambda r: r["cut"] == "0")
    i = np.concatenate([rng.permutation(np.where(yu == k)[0])[:3000] for k in range(3)])
    X, y = np.r_[Xr, Xu[i]], np.r_[yr, yu[i]]
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced")).fit(X, y)
    return m, X, {"archangel_real": int(len(yr)), "ue_level01": int(len(i)), **{c: int((y == k).sum()) for k, c in enumerate(CLASSES)}}


def lr_json(clf, scaler=None):
    assert list(clf.classes_) == [0, 1, 2]
    return {"mean": None if scaler is None else scaler.mean_.round(10).tolist(),
            "scale": None if scaler is None else scaler.scale_.round(10).tolist(),
            "coef": clf.coef_.round(10).tolist(), "intercept": clf.intercept_.round(10).tolist()}


def np_proba(m, X):
    z = X if m["mean"] is None else (X - np.array(m["mean"])) / np.array(m["scale"])
    s = z @ np.array(m["coef"]).T + np.array(m["intercept"])
    s = np.exp(s - s.max(1, keepdims=True))
    return s / s.sum(1, keepdims=True)


def okutama_check(m):
    O = list(csv.DictReader(open(BASE / "data/pose_eval/okutama_obl/crops.csv")))
    X = box_feat(np.array([[float(r["w1080"]) * 1.5, float(r["h1080"]) * 1.5] for r in O]))
    y = np.array([MAP[r["pose"]] for r in O]); P = np_proba(m, X); pr = P.argmax(1)
    tp = ((pr == 1) & (y == 1)).sum(); rec, prec = tp / max((y == 1).sum(), 1), tp / max((pr == 1).sum(), 1)
    return {"앉음F1": round(float(2 * rec * prec / max(rec + prec, 1e-9)), 4), "누움AUROC": round(float(auc(P[y == 0, 0], P[y != 0, 0])), 4)}


def main():
    out = Path(next(a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--out=")))
    b0, Xb, nb = fit_b0()
    v3, Xv, nv = fit_v3()
    js = {"schema": "posture-box-lr/1", "features": "[log(h/w), log(max(w,h))] · 1080p px", "classes": CLASSES,
          "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
          "score": lr_json(b0) | {"name": "B0 SARD+NOMAD", "n": nb},
          "display": lr_json(v3[-1], v3[0]) | {"name": "V3 Archangel 실제 (무릎→앉음) + UE level01", "n": nv}}
    d_b0 = float(np.abs(np_proba(js["score"], Xb) - b0.predict_proba(Xb)).max())
    d_v3 = float(np.abs(np_proba(js["display"], Xv) - v3.predict_proba(Xv)).max())
    js["check"] = {"sklearn 대비 최대 차 (score · display)": [d_b0, d_v3], "V3 Okutama 비스듬 정답 박스": okutama_check(js["display"]),
                   "K2 기록 (V3)": {"앉음F1": 0.4219, "누움AUROC": 0.9611}}
    out.mkdir(parents=True, exist_ok=True)
    (out / "posture.json").write_text(json.dumps(js, ensure_ascii=False, indent=1))
    print(json.dumps(js["check"], ensure_ascii=False), "→", out / "posture.json")
    assert d_b0 < 1e-6 and d_v3 < 1e-6, "numpy 계산이 sklearn 과 다르다"


if __name__ == "__main__":
    main()
