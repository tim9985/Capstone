"""
make_robust_table.py — 판넬 그림: 강건성 종류별 학습 전 → 후 (11-08 · 사용자 「조건별로 딱딱 · 이전 · 이후 시각 표현 · 빠진 강건성도」)

  비교    같은 구조 YOLO11m — 학습 전 (COCO 공식 가중치 · 우리 출발점) vs 우리 학습 후 (v12_f91_swa · 운용 4조각)
  기준    한 가지로 통일 — **사람 없는 사진 헛탐지를 같게 맞춘 뒤** 그 조건의 사람 중 찾은 비율
          우리 = 운용 문턱 0.20 (AU-AIR 사람 없는 233장에서 0.515 건/장) · 학습 전 = 같은 헛탐지가 되는 문턱 (평가셋마다 가장 잘 나온 처리 방식)
  강건성  조도 · 가림 · 자세 · 시점 · 사람 크기 · 배경 (숲) · 지역 (한국 산지) · 헛탐지 — 종류마다 대표 조건 하나 (자세히는 후보 F)
  출력    발표자료/엔지니어링페어_비전/판넬/후보_G_학습전후_강건성.png (git 밖) · metrics/robust_beforeafter.json
실행  /home/se/miniconda3/envs/drone/bin/python make_robust_table.py
"""
import json

import cv2
import numpy as np
from PIL import Image, ImageDraw

from fair_eval_report import fp_neg, load
from make_panel_tables import ACC, BG, INK, INK2, OUT, RULE, F, BASE
from robust_eval import RUN, hits, nomad_meta, tag_of

OURS_T = 0.20


def eq_thr(tag):
    """학습 전 모델이 우리와 같은 도로 헛탐지 (≤ 0.515 건/장) 가 되는 가장 낮은 문턱"""
    road = load("test_auair", tag)[0]; ours = fp_neg(load("test_auair", "v12_f91_swa@tile1280")[0], OURS_T)
    ok = [t for t in np.round(np.arange(0.01, 0.995, 0.005), 3) if fp_neg(road, t) <= ours + 1e-9]
    return (min(ok) if ok else 1.0), ours


def recall(s, keep, cat):
    out = {}
    for who, mdl in (("before", "coco_yolo11m"), ("after", "v12_f91_swa")):
        tag = tag_of(mdl, s); t = OURS_T if who == "after" else eq_thr(tag)[0]
        H = hits(s, tag, t, keep, cat); n = sum(a for a, _ in H.values()); out[who] = sum(b for _, b in H.values()) / n; out["n"] = n
    return out


def main():
    allc = lambda i, gt: np.ones(len(gt), bool); every = lambda i: True
    g = np.load(RUN / "test_visdrone" / "gt.npz")
    br = np.array([cv2.imread(str(BASE / "data/test_visdrone/images" / n), cv2.IMREAD_GRAYSCALE).mean() for n in g["names"]])
    nm = nomad_meta()
    vis = lambda i, gt: np.array([50 <= v <= 90 for v in nm[i][0]][:len(gt)], bool) if nm[i] else np.zeros(len(gt), bool)
    lay = lambda i, gt: np.array([bool(nm[i]) and nm[i][1] in ("Laying", "Hiding (Laying)")] * len(gt), bool)
    sar = [n.split("_")[1] for n in np.load(RUN / "test_sar" / "gt.npz")["names"]]
    rows = [("조도", "밤 · 어두운 장면", "VisDrone", recall("test_visdrone", lambda i: br[i] < 80, allc)),
            ("가림", "몸이 일부 가려진 사람 (50~90 % 보임)", "NOMAD", recall("test_nomad", every, vis)),
            ("자세", "누워 있는 사람", "NOMAD", recall("test_nomad", every, lay)),
            ("시점", "바로 위에서 내려다봄 (90°)", "UAV-SAR", recall("test_sar", lambda i: sar[i].endswith("90d"), allc)),
            ("사람 크기", "아주 작게 보이는 사람 (약 15 px)", "Cyprus", recall("test_cy", every, allc)),
            ("배경", "숲 · 나무와 덤불 사이", "WiSARD", recall("test_v2", every, allc)),
            ("지역", "한국 산지 (조난자 수색)", "AI-Hub", recall("test_kr", every, allc))]
    # 헛탐지 — 사람을 가장 잘 찾는 문턱 (F1 최고) 에서 사람 없는 사진 헛탐지 · 8개 평가셋 평균
    M = json.load(open(BASE / "metrics" / "fair_metrics.json")); sets = ["test_sar", "test_obl", "test_cy", "test_v2", "test_kr", "test_visdrone", "test_nomad", "test_ladd_h"]
    fb = np.mean([M[s][tag_of("coco_yolo11m", s)]["road_fp_best"] for s in sets]); fa = np.mean([M[s]["v12_f91_swa@tile1280"]["road_fp_best"] for s in sets])
    rb = np.mean([M[s][tag_of("coco_yolo11m", s)]["R_best"] for s in sets]); ra = np.mean([M[s]["v12_f91_swa@tile1280"]["R_best"] for s in sets])
    json.dump({"rows": [(a, b, c, d) for a, b, c, d in rows], "헛탐지": {"before": fb, "after": fa, "재현율 before": rb, "재현율 after": ra}, "학습 전 같은 헛탐지 문턱": {s: eq_thr(tag_of("coco_yolo11m", s))[0] for s in ("test_visdrone", "test_nomad", "test_sar", "test_cy", "test_v2", "test_kr")}},
              open(BASE / "metrics" / "robust_beforeafter.json", "w"), ensure_ascii=False, indent=1)
    for r in rows:
        print(r[0], r[1], {k: round(v, 3) for k, v in r[3].items()})
    print("헛탐지", round(fb, 3), "→", round(fa, 3), "· 그때 재현율", round(rb, 3), "→", round(ra, 3))

    # ── 그림 ──
    W = 3000; X0, X1 = 1060, 2280; C_B, C_A, C_G = 2420, 2600, 2830; RH = 112; top = 330
    xs = lambda v: X0 + (X1 - X0) * v
    H = top + RH * (len(rows) + 1) + 560; im = Image.new("RGB", (W, H), BG); dr = ImageDraw.Draw(im)
    dr.text((0, 10), "학습 전 → 학습 후 : 강건성 종류별로 얼마나 좋아졌나", font=F(66, True), fill=INK)
    dr.text((0, 106), "같은 모델 구조 (YOLO11m) · 학습 전 = 공개 COCO 가중치 · 학습 후 = 우리 팀 · 값 = 그 조건의 사람 중 찾은 비율 (1 = 모두 찾음)", font=F(42), fill=INK2)
    dr.text((0, 166), "두 모델 모두 사람 없는 사진에서 잘못 찾는 수를 같게 (0.5 건/장) 맞춘 뒤 비교", font=F(42), fill=INK2)
    yh = top - 90; dr.line([0, yh - 10, W, yh - 10], fill=INK, width=4)
    dr.text((14, yh + 40), "강건성", font=F(42, True), fill=INK, anchor="lm"); dr.text((300, yh + 40), "조건 · 평가셋 · 사람 수", font=F(42, True), fill=INK, anchor="lm")
    for t in (0, 0.25, 0.5, 0.75, 1.0):
        dr.text((xs(t), yh + 40), f"{t:g}", font=F(34), fill=INK2, anchor="mm")
    for cx, t in ((C_B, "학습 전"), (C_A, "학습 후"), (C_G, "향상")):
        dr.text((cx, yh + 40), t, font=F(42, True), fill=INK, anchor="mm")
    dr.line([0, top - 4, W, top - 4], fill=INK, width=3)
    y = top
    for k, (kind, cond, ds, r) in enumerate(rows):
        cy = y + RH / 2
        if k % 2 == 0:
            dr.rectangle([0, y, W, y + RH], fill=(247, 247, 247))
        dr.text((14, cy), kind, font=F(48, True), fill=INK, anchor="lm")
        dr.text((300, cy - 20), cond, font=F(40), fill=INK, anchor="lm"); dr.text((300, cy + 26), f"{ds} · {r['n']:,}명", font=F(31), fill=INK2, anchor="lm")
        for t in (0, 0.25, 0.5, 0.75, 1.0):
            dr.line([xs(t), y + 14, xs(t), y + RH - 14], fill=RULE, width=2)
        b, a = r["before"], r["after"]
        dr.rectangle([xs(b), cy - 12, xs(a), cy + 12], fill=(178, 186, 196))                       # 오른 폭
        dr.ellipse([xs(b) - 17, cy - 17, xs(b) + 17, cy + 17], outline=(120, 120, 120), width=6, fill=BG)
        dr.ellipse([xs(a) - 21, cy - 21, xs(a) + 21, cy + 21], fill=ACC)
        dr.text((C_B, cy), f"{b:.2f}", font=F(46), fill=INK2, anchor="mm"); dr.text((C_A, cy), f"{a:.2f}", font=F(50, True), fill=INK, anchor="mm")
        dr.text((C_G, cy), f"+{(a - b) * 100:.0f}%p", font=F(50, True), fill=INK, anchor="mm")
        dr.line([0, y + RH, W, y + RH], fill=RULE, width=1); y += RH
    # 헛탐지 줄 — 낮을수록 좋음 (0~0.6 건/장 눈금)
    cy = y + RH / 2; dr.rectangle([0, y, W, y + RH], fill=(247, 247, 247) if len(rows) % 2 == 0 else BG)
    dr.text((14, cy), "헛탐지", font=F(48, True), fill=INK, anchor="lm")
    dr.text((300, cy - 20), "사람 없는 장면을 사람으로 착각 (건/장)", font=F(40), fill=INK, anchor="lm")
    dr.text((300, cy + 26), "AU-AIR 도로 233장 · 낮을수록 좋음 · 8개 평가셋 평균", font=F(31), fill=INK2, anchor="lm")
    fx = lambda v: X0 + (X1 - X0) * min(v / 0.6, 1)
    for t in (0, 0.2, 0.4, 0.6):
        dr.line([fx(t), y + 14, fx(t), y + RH - 14], fill=RULE, width=2); dr.text((fx(t), y + RH - 6), f"{t:g}", font=F(26), fill=INK2, anchor="mb")
    dr.rectangle([fx(fa), cy - 12, fx(fb), cy + 12], fill=(178, 186, 196))
    dr.ellipse([fx(fb) - 17, cy - 17, fx(fb) + 17, cy + 17], outline=(120, 120, 120), width=6, fill=BG); dr.ellipse([fx(fa) - 21, cy - 21, fx(fa) + 21, cy + 21], fill=ACC)
    dr.text((C_B, cy), f"{fb:.2f}", font=F(46), fill=INK2, anchor="mm"); dr.text((C_A, cy), f"{fa:.2f}", font=F(50, True), fill=INK, anchor="mm")
    dr.text((C_G, cy), f"-{(1 - fa / fb) * 100:.0f}%", font=F(50, True), fill=INK, anchor="mm")
    y += RH; dr.line([0, y + 4, W, y + 4], fill=INK, width=4)
    ly = y + 70
    dr.ellipse([14, ly - 17, 48, ly + 17], outline=(120, 120, 120), width=6, fill=BG); dr.text((66, ly), "학습 전 (YOLO11m · COCO 공식 가중치)", font=F(38), fill=INK, anchor="lm")
    dr.ellipse([800, ly - 21, 842, ly + 21], fill=ACC); dr.text((862, ly), "학습 후 (우리 팀 · 실제 운용 방식)", font=F(38), fill=INK, anchor="lm")
    dr.rectangle([1540, ly - 12, 1640, ly + 12], fill=(178, 186, 196)); dr.text((1660, ly), "학습으로 좋아진 폭", font=F(38), fill=INK, anchor="lm")
    notes = ["찾음 = 박스가 사람과 절반 이상 겹침 · 평가 사진은 모두 학습에 안 씀 (UAV-SAR · Cyprus 는 처음 보는 데이터셋 · 나머지는 같은 데이터셋의 다른 사진 · 다른 장소)",
             f"헛탐지 줄 = 두 모델이 각자 사람을 가장 잘 찾는 문턱 (F1 최고) 에서 · 그때 찾은 비율 평균 {rb:.2f} → {ra:.2f} — 더 많이 찾으면서 덜 착각함",
             "아직 약한 조건: 영상 압축이 심할 때 · 몸이 60 % 넘게 가려졌을 때 · 기체가 흔들려 번질 때 (실기체 시험에서 확인 예정)"]
    for k, t in enumerate(notes):
        dr.text((0, ly + 80 + k * 56), t, font=F(33), fill=INK2)
    im = im.crop((0, 0, W, ly + 80 + len(notes) * 56 + 30)); p = OUT / "후보_G_학습전후_강건성.png"; im.save(p, dpi=(450, 450)); print(p, im.size)


if __name__ == "__main__":
    main()
