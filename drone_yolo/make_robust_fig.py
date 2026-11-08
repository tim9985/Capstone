"""
make_robust_fig.py — 판넬 그림: 조도 · 가림 · 자세 · 시점별 강건성 (11-08 · 사용자 「기성 모델과 비교해 강건성 확보 · 신경 많이」)

  원자료  metrics/robust_conditions.json (robust_eval.py · 같은 사진 · 같은 채점 · 재현율 @ 모델마다 F1 최고 문턱 · 블록 부트스트랩 판정)
  그림    조건 16줄 × 점그림 (우리 · 학습 전 YOLO11m · 실시간 기성 최고 · Grounding DINO) + 수치 · 판정 칸 · 판넬 B · E 와 같은 회색조
  출력    발표자료/엔지니어링페어_비전/판넬/후보_F_조건별_강건성.png (git 밖)
실행  /home/se/miniconda3/envs/drone/bin/python make_robust_fig.py
"""
import json

from PIL import Image, ImageDraw

from make_panel_tables import ACC, BG, INK, INK2, MUTE, OUT, RULE, F, BASE

R = json.load(open(BASE / "metrics" / "robust_conditions.json"))["조건"]
ROWS = [("조도", [("밤 · 어두운 장면", "밤 · 어두운 장면", "VisDrone"), ("낮 · 밝은 장면", "낮 · 밝은 장면", "VisDrone"),
                ("아침 (낮은 해 · 긴 그림자)", "아침 · 낮은 해 · 긴 그림자", "Okutama"), ("한낮 (강한 햇빛)", "한낮 · 강한 햇빛", "Okutama")]),
        ("가림", [("일부 가림 (50~90 %)", "일부 가림 · 몸 50~90 % 보임", "NOMAD"), ("많이 가림 (40 % 이하)", "많이 가림 · 몸 40 % 이하 보임", "NOMAD")]),
        ("자세", [("서 있음", "서 있음", "Okutama"), ("걷기 · 뛰기", "걷기 · 뛰기", "Okutama"), ("앉음", "앉음", "Okutama"),
                ("누움", "누움", "Okutama"), ("숨기 (웅크림 · 엎드림)", "웅크려 숨음", "NOMAD"), ("누움 (NOMAD)", "누움", "NOMAD")]),
        ("시점", [("비스듬 45°", "비스듬히 내려다봄 · 45°", "UAV-SAR"), ("수직 90°", "바로 위에서 · 90°", "UAV-SAR"),
                ("고도 15 m", "고도 15 m", "UAV-SAR"), ("고도 30 m (사람 작음)", "고도 30 m · 사람 작게 보임", "UAV-SAR")])]
RT_NAME = {"coco": "YOLO11m 학습 전", "aeronova": "AeroNova", "v9e": "VisDrone v9e", "yworld": "YOLO-World", "y11x": "YOLO11x", "rtdetr": "RT-DETR-x"}
W = 3000
C_LAB, C_PLOT = 0, 760                     # 조건 칸 · 점그림 시작
PX0, PX1 = C_PLOT + 30, C_PLOT + 1130       # 점그림 0 · 1 위치
C_OURS, C_RT, C_V1, C_GD, C_V2 = 1960, 2150, 2500, 2660, 2840
RH, GH = 86, 64


def xs(v):
    return PX0 + (PX1 - PX0) * v


def mark(dr, x, y, kind, r=15):
    if kind == "ours":
        dr.ellipse([x - r - 2, y - r - 2, x + r + 2, y + r + 2], fill=ACC)
    elif kind == "coco":
        dr.ellipse([x - r, y - r, x + r, y + r], outline=(150, 150, 150), width=5, fill=BG)
    elif kind == "rt":
        dr.polygon([(x, y - r - 3), (x - r - 2, y + r - 1), (x + r + 2, y + r - 1)], outline=(95, 95, 95), width=5, fill=BG)
    else:
        dr.polygon([(x, y - r - 4), (x - r - 4, y), (x, y + r + 4), (x + r + 4, y)], outline=(95, 95, 95), width=4, fill=(225, 225, 225))


def badge(dr, cx, cy, v):
    w, h = 118, 50; box = [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]
    if v == "앞섬":
        dr.rounded_rectangle(box, 10, fill=ACC); dr.text((cx, cy), v, font=F(36, True), fill=BG, anchor="mm")
    elif v == "비슷":
        dr.rounded_rectangle(box, 10, outline=(150, 150, 150), width=3); dr.text((cx, cy), v, font=F(36), fill=INK2, anchor="mm")
    else:
        dr.rounded_rectangle(box, 10, fill=(222, 222, 222)); dr.text((cx, cy), v, font=F(36, True), fill=INK, anchor="mm")


def main():
    rows = [(g, key, lab, ds, R[g][key]) for g, items in ROWS for key, lab, ds in items]
    n_rt = {v: sum(r["판정 (실시간)"] == v for *_, r in rows) for v in ("앞섬", "비슷", "뒤짐")}
    n_gd = {v: sum(r["판정 (GDINO)"] == v for *_, r in rows) for v in ("앞섬", "비슷", "뒤짐")}
    gain = sum(r["ours"] - r["coco"] for *_, r in rows) / len(rows)
    top = 560; H = top + len(rows) * RH + len(ROWS) * GH + 820
    im = Image.new("RGB", (W, H), BG); dr = ImageDraw.Draw(im)
    dr.text((0, 10), "촬영 조건이 바뀌어도 사람을 놓치지 않는가 — 조도 · 가림 · 자세 · 시점별 비교", font=F(62, True), fill=INK)
    dr.text((0, 100), "같은 사진 · 같은 정답 · 같은 채점으로 공개 모델과 비교 · 값 = 그 조건의 사람 중 찾은 비율 (재현율 · 오른쪽일수록 좋음)", font=F(42), fill=INK2)
    # 요약 칸 셋
    bw = (W - 80) / 3; y0 = 190
    summ = [("실시간 공개 모델 중 최고와 비교", f"앞섬 {n_rt['앞섬']}  ·  비슷 {n_rt['비슷']}  ·  뒤짐 {n_rt['뒤짐']}", f"조건 {len(rows)}개 · 같은 속도대 모델 6종"),
            ("대형 모델 Grounding DINO 와 비교", f"앞섬 {n_gd['앞섬']}  ·  비슷 {n_gd['비슷']}  ·  뒤짐 {n_gd['뒤짐']}", "그쪽은 1080p 한 장 355 ms · 우리 22 ms"),
            ("학습 전 YOLO11m (출발점) 대비", f"평균 +{gain * 100:.0f}%p", "같은 구조 · 학습 데이터만 우리 목적에 맞춤")]
    for k, (h, v, sub) in enumerate(summ):
        X = k * (bw + 40); dr.rectangle([X, y0, X + bw, y0 + 220], outline=INK2, width=2)
        dr.text((X + 30, y0 + 44), h, font=F(38, True), fill=INK2, anchor="lm")
        dr.text((X + 30, y0 + 118), v, font=F(56, True), fill=INK, anchor="lm")
        dr.text((X + 30, y0 + 182), sub, font=F(34), fill=INK2, anchor="lm")
    # 머리줄
    yh = top - 120; dr.line([0, yh - 14, W, yh - 14], fill=INK, width=4)
    dr.text((10, yh + 34), "조건", font=F(42, True), fill=INK, anchor="lm"); dr.text((10, yh + 82), "평가셋 · 사람 수", font=F(32), fill=INK2, anchor="lm")
    for t in (0, 0.2, 0.4, 0.6, 0.8, 1.0):
        dr.text((xs(t), yh + 60), f"{t:.1f}", font=F(34), fill=INK2, anchor="mm")
    for cx, t1, t2 in ((C_OURS + 70, "우리 팀", ""), (C_RT + 150, "실시간 공개", "모델 최고"), (C_V1 + 80, "판정", ""), (C_GD + 60, "Grounding", "DINO"), (C_V2 + 80, "판정", "")):
        if t2:
            dr.text((cx, yh + 36), t1, font=F(36, True), fill=INK, anchor="mm"); dr.text((cx, yh + 80), t2, font=F(36, True), fill=INK, anchor="mm")
        else:
            dr.text((cx, yh + 60), t1, font=F(36, True), fill=INK, anchor="mm")
    dr.line([0, top - 4, W, top - 4], fill=INK, width=3)
    y = top
    for g, items in ROWS:
        dr.rectangle([0, y + 6, W, y + GH - 6], fill=(244, 244, 244)); dr.text((14, y + GH / 2), g, font=F(42, True), fill=INK, anchor="lm")
        y += GH; y_sec = y
        for key, lab, ds in items:
            r = R[g][key]; cy = y + RH / 2
            dr.text((14, cy - 17), lab, font=F(38, True), fill=INK, anchor="lm")
            dr.text((14, cy + 23), f"{ds} · {r['n']:,}명", font=F(30), fill=INK2, anchor="lm")
            for t in (0, 0.2, 0.4, 0.6, 0.8, 1.0):
                dr.line([xs(t), y, xs(t), y + RH], fill=RULE, width=2)
            best = r["실시간 기성 최고"]
            dr.line([xs(r["coco"]), cy, xs(r["ours"]), cy], fill=(185, 185, 185), width=6)          # 학습 전 → 우리 (학습으로 오른 폭)
            mark(dr, xs(r["coco"]), cy, "coco")
            if best != "coco":
                mark(dr, xs(r[best]), cy, "rt")
            mark(dr, xs(r["gdino"]), cy, "gd"); mark(dr, xs(r["ours"]), cy, "ours")
            dr.text((C_OURS + 70, cy), f"{r['ours']:.2f}", font=F(44, True), fill=INK, anchor="mm")
            dr.text((C_RT + 150, cy - 16), f"{r[best]:.2f}", font=F(40), fill=INK, anchor="mm")
            dr.text((C_RT + 150, cy + 24), RT_NAME[best], font=F(28), fill=INK2, anchor="mm")
            badge(dr, C_V1 + 80, cy, r["판정 (실시간)"])
            dr.text((C_GD + 70, cy), f"{r['gdino']:.2f}", font=F(40), fill=INK, anchor="mm")
            badge(dr, C_V2 + 80, cy, r["판정 (GDINO)"])
            dr.line([0, y + RH, W, y + RH], fill=RULE, width=1); y += RH
    dr.line([0, y + 4, W, y + 4], fill=INK, width=4)
    # 범례
    ly = y + 60; lx = 10
    for kind, t in (("ours", "우리 팀 (YOLO11m · 학습 후)"), ("coco", "YOLO11m 학습 전 (COCO · 출발점)"), ("rt", "실시간 공개 모델 중 최고"), ("gd", "Grounding DINO (대형 · 실시간 불가)")):
        mark(dr, lx + 18, ly, kind); dr.text((lx + 48, ly), t, font=F(34), fill=INK, anchor="lm"); lx += dr.textlength(t, font=F(34)) + 130
    dr.line([10, ly + 62, 80, ly + 62], fill=(185, 185, 185), width=6); dr.text((96, ly + 62), "회색 막대 = 학습으로 오른 폭 (학습 전 → 우리)", font=F(34), fill=INK, anchor="lm")
    badge(dr, 1250, ly + 62, "앞섬"); dr.text((1322, ly + 62), "차이가 확실히 0 보다 큼", font=F(32), fill=INK, anchor="lm")
    badge(dr, 1840, ly + 62, "비슷"); dr.text((1912, ly + 62), "차이를 가를 수 없음", font=F(32), fill=INK, anchor="lm")
    badge(dr, 2380, ly + 62, "뒤짐"); dr.text((2452, ly + 62), "상대가 확실히 높음", font=F(32), fill=INK, anchor="lm")
    notes = ["찾음 = 박스가 사람과 절반 이상 겹침 · 모델마다 그 데이터셋에서 F1 이 가장 높은 문턱 · 판정 = 블록 부트스트랩 5,000회 95 % 구간 (연속 20장 묶음)",
             "평가 사진은 모두 우리 학습에 안 씀 — Okutama (일본 · 아침/한낮 · 행동 라벨) · UAV-SAR (드론 수색 · 15/30 m · 45°/90°) 는 처음 보는 데이터셋",
             "VisDrone (중국 도시 · 밤 장면 포함) · NOMAD (미국 · 몸이 보이는 비율 · 숨기 라벨) 는 같은 데이터셋의 학습에 안 쓴 사진",
             "실시간 공개 모델 = YOLO11m 학습 전 · AeroNova · VisDrone YOLOv9e · YOLO-World v2 · YOLO11x · RT-DETR-x 중 그 조건 최고 (크기 · 처리 방식도 가장 잘 나온 것) · 우리 팀 = 실제 운용 방식 (1080p 4조각)",
             "많이 가림 (몸 40 % 이하) 은 모든 모델이 0.13 이하 · Okutama 누움은 정답 박스가 넓게 그려져 겹침 기준에서 모든 모델이 낮음 (가장 어려운 두 조건)"]
    for k, t in enumerate(notes):
        dr.text((0, ly + 150 + k * 56), t, font=F(32), fill=INK2)
    im = im.crop((0, 0, W, ly + 150 + len(notes) * 56 + 30)); OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "후보_F_조건별_강건성.png"; im.save(p, dpi=(450, 450)); print(p, im.size, n_rt, n_gd, round(gain, 3))


if __name__ == "__main__":
    main()
