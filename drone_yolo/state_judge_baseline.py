"""state_judge_baseline.py — state_judge_eval.py 와 같은 시험 문항으로 우리 기준선 (2026-10-06)
  비스듬 (Okutama) = 박스 모양 로지스틱 (SARD + NOMAD) · 수직 (AI-Hub) = DINOv2 + 로지스틱 (AI-Hub 학습 · 산악8 로 C)
  ⚠ Okutama w1080 · h1080 은 720p px 그대로 → ×1.5
실행: /home/se/venvs/state/bin/python state_judge_baseline.py
"""
import csv, json, sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from state_judge_eval import t1_items, auc, CLASSES, EVAL, BASE
from posture_fusion import rows_of, box_feat, CLS
from posture_foundation import Dino, train_files

items = t1_items()
meta = {}
for s in ("okutama_obl", "aihub_test"):
    for r in csv.DictReader(open(EVAL / s / "crops.csv")):
        meta[str(EVAL / s / r["file"])] = (float(r["w1080"]), float(r["h1080"]))
parts = [rows_of(CLS / p / "train") for p in ("sard", "nomad")]
Xb = box_feat(np.concatenate([p[2] for p in parts])); yb = np.concatenate([p[1] for p in parts])
box = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xb, yb)
dino = Dino()
Ftr_f, ytr = train_files("train"); Fva_f, yva = train_files("val")
Ftr, Fva = dino.image(Ftr_f), dino.image(Fva_f)
dl = max((LogisticRegression(C=C, max_iter=2000, class_weight="balanced").fit(Ftr, ytr) for C in (0.1, 0.3, 1.0, 3.0)),
         key=lambda c: (c.predict(Fva) == yva).mean())
res, out_items = {}, []
for s, name in (("okutama_obl", "박스 모양 로지스틱"), ("aihub_test", "DINOv2 + 로지스틱")):
    R = [it for it in items if it["set"] == s]
    if s == "okutama_obl":
        wh = np.array([[meta[it["file"]][0] * 1.5, meta[it["file"]][1] * 1.5] for it in R]); P = box.predict_proba(box_feat(wh))
    else:
        P = dl.predict_proba(dino.image([it["file"] for it in R]))
    ly = [p[0] for p, it in zip(P, R) if it["pose"] == "lying"]; nl = [p[0] for p, it in zip(P, R) if it["pose"] != "lying"]
    pred = [CLASSES[int(np.argmax(p))] for p in P]
    out_items += [{"file": it["file"], "set": s, "pose": it["pose"], "help": round(float(p[0]), 4),
               "p": [round(float(v), 4) for v in p]} for p, it in zip(P, R)]
    res[s] = {"방법": name, "n": len(R), "P누움_AUROC": round(auc(ly, nl), 3),
              "3자세_정확도": round(float(np.mean([a == it["pose"] for a, it in zip(pred, R)])), 3),
              "재현율": {c: round(float(np.mean([a == c for a, it in zip(pred, R) if it["pose"] == c])), 3) for c in CLASSES}}
print(json.dumps(res, ensure_ascii=False, indent=1))
(BASE / "metrics" / "state_judge_baseline.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
(BASE / "metrics" / "state_judge_baseline_items.json").write_text(json.dumps(out_items, ensure_ascii=False))   # 도움 필요 = P(누움)
