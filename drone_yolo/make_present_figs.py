"""
make_present_figs.py — 최종발표 · 엔지니어링 페어 발표용 비전 그림 (11-08 · 사용자 「데이터셋 샘플 9개씩 · 임무와 동작 방식 · 또 뭐가 있을지」)

  양식   사용자가 다듬은 판넬과 맞춤 — Pretendard · 흰 바탕 · 검은 굵은 선 · 남색 점 · 연회색 막대 · 가로 5400 px
  그림   datasets   데이터셋마다 한 장 · 샘플 9개 (3×3) · 노란 박스 = 정답 사람 (박스 없는 판도) + 14종 한 장 요약
         flow       비전 임무와 동작 방식 — 쉬운 판 (페어) · 자세한 판 (캡스톤) · UE S3 장면에 배포 worker 그대로 (탐지 → 추적 → 자세 · 멈춘 시간 → 등급 → 상의 색)
         beforeafter 같은 사진에서 학습 전 YOLO11m (COCO) vs 우리 — 찾은 사람 · 놓친 사람 (같은 헛탐지 기준 문턱)
         journey    (캡스톤) 장소 분리 평가 AP50 이 어떻게 올라왔나 + 해 봤지만 안 된 것
         failures   (캡스톤) 아직 약한 곳 — 미끼 물체 오탐 · 많이 가린 사람 · 넓게 그린 정답 박스
  출력   발표자료/비전_공통그림/ (둘 다 쓰는 것) · 발표자료/최종발표/비전_추가/ · 발표자료/엔지니어링 페어 발표/비전_추가/  (모두 git 밖)
  ⚠ 데이터셋 원본 · 파생 이미지 (NOMAD · WiSARD · Okutama · AI-Hub 등) 는 git 에 올리지 않음 · 발표 화면 · 인쇄에만
실행  /home/se/miniconda3/envs/drone/bin/python make_present_figs.py datasets|flow|beforeafter|journey|failures|all
"""
import json, random, sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

BASE = Path(__file__).resolve().parent
PRES = BASE.parent / "발표자료"
FONT = str(PRES / "_도구" / "Pretendard.woff2")
OUT_COM, OUT_CAP, OUT_FAIR = PRES / "비전_공통그림", PRES / "최종발표" / "비전_추가", PRES / "엔지니어링 페어 발표" / "비전_추가"
INK, INK2, MUTE, RULE, BG = (17, 24, 39), (100, 107, 120), (156, 163, 175), (226, 228, 232), (255, 255, 255)
NAVY, BAR, YEL, RED = (31, 41, 59), (190, 198, 208), (255, 196, 0), (226, 68, 72)
W = 5400
_FC = {}


def F(n, w=400):
    k = (n, w)
    if k not in _FC:
        f = ImageFont.truetype(FONT, n); f.set_variation_by_axes([w]); _FC[k] = f
    return _FC[k]


def yolo_boxes(p, W_, H_):
    out = []
    if Path(p).exists():
        for l in open(p):
            v = l.split()
            if len(v) >= 5:
                cx, cy, w, h = map(float, v[1:5]); out.append((W_ * (cx - w / 2), H_ * (cy - h / 2), W_ * (cx + w / 2), H_ * (cy + h / 2)))
    return out


def fit169(img, boxes):
    """가운데 16:9 로 자르기 · 박스도 옮김"""
    h, w = img.shape[:2]; tw = min(w, round(h * 16 / 9)); th = round(tw * 9 / 16); x0 = (w - tw) // 2; y0 = (h - th) // 2
    return img[y0:y0 + th, x0:x0 + tw], [(a - x0, b - y0, c - x0, d - y0) for a, b, c, d in boxes]


def tile(img_path, lab_path, tw, th, draw_boxes=True, lw=6):
    img = cv2.imread(str(img_path)); H_, W_ = img.shape[:2]; bx = yolo_boxes(lab_path, W_, H_) if lab_path else []
    img, bx = fit169(img, bx); s = tw / img.shape[1]
    im = Image.fromarray(cv2.cvtColor(cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)); d = ImageDraw.Draw(im)
    if draw_boxes:
        for a, b, c, e in bx:
            a, b, c, e = a * s, b * s, c * s, e * s
            if c < 0 or e < 0 or a > tw or b > th:
                continue
            pad = 3; d.rectangle([a - pad, b - pad, c + pad, e + pad], outline=YEL, width=lw)
    return im, len(bx)


# ─────────────────────────── 1. 데이터셋 ───────────────────────────
TRAIN = [("visdrone", "VisDrone", "중국 14개 도시 · 드론 영상 · 비스듬 · 수직", "도시 거리 · 사람이 작고 많음 · 낮과 밤", "학습 3,669창 (비스듬 2,469 · 수직 1,200) · 다른 사진 500장은 평가"),
         ("nomad", "NOMAD", "미국 농장 · 실종자 수색 연출 (배우 100명)", "숨기 · 눕기 · 나무와 수풀에 가림 · 비스듬", "학습 2,849창 · 배우 097·100 은 평가"),
         ("wisard", "WiSARD", "미국 서부 · 야생 수색 훈련", "숲 · 산지 · 황무지 · 위에서 내려다봄", "학습 2,800창 · 다른 장소 (Carnation · Karen) 는 평가"),
         ("aihub", "AI-Hub 조난자 수색", "한국 · 산악 · 저수지 등 · 4K", "바로 위에서 (90°) · 국내 지형 · 계절", "학습 2,500창 · 다른 장소 (산악5) 는 평가"),
         ("heridal", "HERIDAL", "크로아티아 · 지중해 산지", "바위 · 덤불 · 위에서 내려다봄 · 사람 작음", "학습 1,600창"),
         ("sard", "SARD", "크로아티아 · 수색 연출", "비스듬 · 서기 · 앉기 · 눕기 등 자세 6가지", "학습 1,500창"),
         ("archangel", "Archangel", "미국 · 원을 그리며 비행 · 고도 15~50 m", "비스듬 · 서기 · 무릎 꿇기 · 엎드리기", "학습 1,500창 (실사진만)"),
         ("ladd", "LADD (Lacmus)", "러시아 · 숲 수색 봉사단", "숲 · 늪 · 위에서 내려다봄 · 사람 아주 작음", "학습 900창 · 뒤쪽 30 % 는 평가"),
         ("niicu", "NII-CU", "일본 · 45° · 고도 20~50 m · 4K", "운동장 · 공원 · 비스듬", "학습 384창"),
         ("unicamp", "Unicamp-UAV", "브라질 대학 캠퍼스 · 4K", "정면 비스듬 · 걷는 사람", "학습 300창")]
EVAL = [("test_sar", "UAV-SAR", "드론 수색 실사진 · 고도 15 · 30 m × 45° · 90°", "옷 걸친 의자 · 가방 같은 미끼 물체도 놓여 있음", "평가 전용 300장 · 어떤 모델도 학습에 안 씀"),
        ("test_obl1080", "Okutama-Action", "일본 오쿠타마 · 4K · 비스듬 · 고도 약 16 m", "걷기 · 뛰기 · 앉기 · 눕기 등 행동 12가지", "평가 전용 · 어떤 모델도 학습에 안 씀"),
        ("test_cy", "Cyprus", "키프로스 · 대학 캠퍼스 민방위 훈련", "바로 위에서 · 사람 약 15 px (아주 작음)", "평가 전용 · 어떤 모델도 학습에 안 씀"),
        ("test_auair", "AU-AIR (사람 없는 장면)", "덴마크 · 도로 · 교차로 · 차량", "사람이 없는 사진에서 잘못 찾는 수를 잼", "평가 전용 · 사람 없는 233장")]


def pick_train(src, n=9, seed=0):
    D = BASE / "data" / "det_v12"; ims = sorted((D / "images" / "train").glob(f"{src}_*.jpg")); rng = random.Random(seed); rng.shuffle(ims)
    chunks = [ims[k::n] for k in range(n)]; out = []
    for ch in chunks:
        for p in ch:
            lab = D / "labels" / "train" / f"{p.stem}.txt"; bx = yolo_boxes(lab, 1280, 720)
            if bx and max(max(c - a, e - b) for a, b, c, e in bx) >= 16:
                out.append((p, lab)); break
    return out


def pick_eval(s, n=9, seed=0):
    D = BASE / "data" / s; ims = sorted((D / "images").glob("*.jpg")); rng = random.Random(seed)
    if s == "test_auair":                                          # 사람 없는 장
        pos = [p for p in ims if not yolo_boxes(D / "labels" / f"{p.stem}.txt", 1, 1)]
    else:
        pos = [p for p in ims if yolo_boxes(D / "labels" / f"{p.stem}.txt", 1, 1)]
    groups = {}
    for p in pos:
        key = p.stem.split("_")[1] if s in ("test_sar", "test_obl1080") else p.stem.rsplit("_", 1)[0]
        groups.setdefault(key, []).append(p)
    keys = sorted(groups); rng.shuffle(keys); out = []
    while len(out) < n and any(groups.values()):
        for k in keys:
            if groups[k] and len(out) < n:
                out.append(rng.choice(groups[k])); groups[k].remove(out[-1])
    return [(p, D / "labels" / f"{p.stem}.txt") for p in out]


def dataset_sheet(name, title, l1, l2, role, items, out, boxes=True, train=True):
    G = 36; tw = (W - 2 * G) // 3; th = round(tw * 9 / 16); top = 430; H = top + 3 * th + 2 * G + 190
    im = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(im)
    d.text((0, 10), title, font=F(132, 700), fill=INK)
    tag = "학습 데이터" if train else "평가 전용"; tf = F(56, 600); tw_ = d.textlength(tag, font=tf)
    d.rounded_rectangle([W - tw_ - 80, 34, W, 134], 16, fill=NAVY if train else BG, outline=NAVY, width=4)
    d.text((W - tw_ / 2 - 40, 84), tag, font=tf, fill=BG if train else NAVY, anchor="mm")
    d.text((0, 190), l1, font=F(66, 500), fill=INK); d.text((0, 282), l2 + "   ·   " + role, font=F(54, 400), fill=INK2)
    d.line([0, top - 34, W, top - 34], fill=INK, width=7)
    n_people = 0
    for k, (ip, lp) in enumerate(items[:9]):
        r, c = divmod(k, 3); t, n = tile(ip, lp if boxes else None, tw, th, draw_boxes=boxes); n_people += n
        im.paste(t, (c * (tw + G), top + r * (th + G)))
    yb = top + 3 * th + 2 * G + 40
    if boxes and name != "test_auair":
        d.rectangle([0, yb + 10, 70, yb + 66], outline=YEL, width=8); d.text((100, yb + 38), "노란 박스 = 사람 위치 (정답 라벨)", font=F(50, 500), fill=INK, anchor="lm")
        d.text((1300, yb + 38), "학습 사진은 모델이 실제로 본 1280×720 창 그대로 · 평가 사진은 1920×1080 을 16:9 로", font=F(44), fill=INK2, anchor="lm")
    else:
        d.text((0, yb + 38), "학습 사진은 모델이 실제로 본 1280×720 창 그대로 · 평가 사진은 1920×1080 을 16:9 로" if name != "test_auair" else "사람이 없는 도로 사진 — 여기서 「사람」 으로 잘못 찾은 수가 헛탐지", font=F(44), fill=INK2, anchor="lm")
    out.mkdir(parents=True, exist_ok=True); im.save(out, dpi=(300, 300)) if out.suffix else None
    return im


def datasets():
    out = OUT_COM / "데이터셋_샘플"; out.mkdir(parents=True, exist_ok=True); cells = []
    for k, (src, title, l1, l2, role) in enumerate(TRAIN, 1):
        items = pick_train(src)
        for boxes, sfx in ((True, ""), (False, "_박스없음")):
            im = dataset_sheet(src, title, l1, l2, role, items, out, boxes=boxes, train=True)
            im.save(out / f"학습_{k:02d}_{title.split(' ')[0]}{sfx}.png", dpi=(300, 300))
        cells.append((items[0], title, l1.split(" · ")[0], True)); print("학습", title, len(items), flush=True)
    for k, (s, title, l1, l2, role) in enumerate(EVAL, 1):
        items = pick_eval(s)
        for boxes, sfx in ((True, ""), (False, "_박스없음")):
            if s == "test_auair" and not boxes:
                continue
            im = dataset_sheet(s, title, l1, l2, role, items, out, boxes=boxes and s != "test_auair", train=False)
            im.save(out / f"평가_{k:02d}_{title.split(' ')[0]}{sfx}.png", dpi=(300, 300))
        cells.append((items[0], title, l1.split(" · ")[0], False)); print("평가", title, len(items), flush=True)
    # 14종 한 장
    C, G = 5, 36; cw = (W - (C - 1) * G) // C; ch = round(cw * 9 / 16); cap = 170; top = 330
    H = top + 3 * (ch + cap + G) + 120; im = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(im)
    d.text((0, 10), "어떤 사진으로 배우고, 어떤 사진으로 시험했나", font=F(112, 700), fill=INK)
    d.text((0, 160), "공개 드론 사진 데이터 14종 — 학습 10종 (11개 출처 · 약 18,600장) · 시험 4종 (학습에 한 장도 안 씀)", font=F(56, 400), fill=INK2)
    d.line([0, top - 34, W, top - 34], fill=INK, width=7)
    for k, ((ip, lp), title, where, tr) in enumerate(cells):
        r, c = divmod(k, C); x0, y0 = c * (cw + G), top + r * (ch + cap + G)
        t, _ = tile(ip, lp if "AU-AIR" not in title else None, cw, ch, draw_boxes="AU-AIR" not in title, lw=4); im.paste(t, (x0, y0))
        d.rectangle([x0, y0 + ch, x0 + 14, y0 + ch + 120], fill=NAVY if tr else MUTE)
        d.text((x0 + 34, y0 + ch + 44), title, font=F(50, 700), fill=INK, anchor="lm")
        d.text((x0 + 34, y0 + ch + 106), ("학습 · " if tr else "시험 · ") + where, font=F(40), fill=INK2, anchor="lm")
    x0, y0 = 4 * (cw + G), top + 2 * (ch + cap + G)                  # 범례 칸
    d.rectangle([x0, y0, x0 + cw, y0 + ch + 120], outline=RULE, width=4)
    lg = [("■", NAVY, "학습에 쓴 데이터"), ("■", MUTE, "시험에만 쓴 데이터"), ("□", YEL, "노란 박스 = 사람 위치")]
    for j, (sym, col, t) in enumerate(lg):
        yy = y0 + 110 + j * 150
        if sym == "■":
            d.rectangle([x0 + 50, yy - 30, x0 + 110, yy + 30], fill=col)
        else:
            d.rectangle([x0 + 50, yy - 30, x0 + 110, yy + 30], outline=col, width=8)
        d.text((x0 + 140, yy), t, font=F(48, 500), fill=INK, anchor="lm")
    im.save(out / "00_데이터셋_14종_한눈에.png", dpi=(300, 300)); print("한눈에 저장")



# ─────────────────────────── 2. 임무 · 동작 방식 ───────────────────────────
GRADE_KO = {"URGENT": "긴급", "HIGH": "높음", "CHECK": "확인", "NORMAL": "보통"}
GRADE_COL = {"URGENT": (210, 45, 45), "HIGH": (232, 120, 30), "CHECK": (222, 180, 20), "NORMAL": (60, 150, 90)}
POSE_KO = {"lying": "누움", "sitting": "앉음", "standing": "서기"}
SW = {"red": (220, 40, 40), "orange": (245, 140, 30), "yellow": (240, 210, 40), "green": (40, 160, 70), "blue": (40, 100, 220), "navy": (30, 40, 110),
      "purple": (130, 60, 170), "pink": (240, 140, 180), "white": (245, 245, 245), "gray": (140, 140, 140), "black": (20, 20, 20), "brown": (120, 75, 40)}
FLOW_RUN, FLOW_FRAME, FLOW_FPS = "S3_a16", 373, 10.0


def worker_outputs():
    """UE S3_a16 프레임 0~373 에 배포 worker 모듈 그대로 (탐지 TensorRT · 추적 · 상태 · 상의 색 SigLIP 1.3) → 373 장 시점 출력 · 캐시"""
    cache = BASE / "metrics" / f"present_flow_{FLOW_RUN}_{FLOW_FRAME}.json"
    if cache.exists():
        return json.load(open(cache))
    sys.path.insert(0, str(BASE / "deploy" / "vision_worker"))
    from app.vision_core.detector import PersonDetector
    from app.vision_core.state import StateTracker
    from app.vision_core.color import ColorAccumulator, frame_stats, make_colorer, KO
    pack = BASE / "runs_person" / "_pack_v12_sim"
    det = PersonDetector(pack / "v12_f91_swa" / "best.pt", pack / "v12_f91_swa" / "best_trt11.3.0.99_RTX3090.engine", conf=0.20)
    trk = StateTracker(str(pack / "posture.json")); col = make_colorer(pack=str(pack / "color_siglip2")); acc = {}; ms_all = []
    frames = sorted((BASE / "ue_flight" / FLOW_RUN / "frames").glob("*.jpg"))
    for i, p in enumerate(frames[:FLOW_FRAME + 1]):
        fr = cv2.imread(str(p)); boxes, confs, ms = det.detect(fr); ms_all.append(ms)
        st = trk.update(i / FLOW_FPS, fr, [tuple(map(float, b)) for b in boxes], [float(c) for c in confs]) if len(boxes) else []
        stats = frame_stats(fr) if len(boxes) and i % 3 == 0 else None
        for b, (tid, s) in zip(boxes, st or []):
            if tid and stats is not None:
                acc.setdefault(tid, ColorAccumulator()).add(col.profile(fr, tuple(map(float, b)), stats=stats))
        last = (boxes, confs, st)
    boxes, confs, st = last; people = []
    for b, c, (tid, s) in zip(boxes, confs, st or [(None, None)] * len(boxes)):
        r = acc[tid].result() if tid in acc else None
        people.append(dict(box=[float(v) for v in b], conf=float(c), tid=str(tid) if tid else None,
                           grade=(s or {}).get("grade") if s and s.get("tracked") and s.get("observed_s", 0) >= 3 else "CHECK",
                           observed_s=(s or {}).get("observed_s"), posture=(s or {}).get("posture", {}).get("label"),
                           p=(s or {}).get("posture", {}).get("p"), still_s=(s or {}).get("motion", {}).get("still_s"),
                           moving=(s or {}).get("motion", {}).get("moving"),
                           color=(r["top"][0][0] if r and r["status"] == "ok" else None), color_ko=(KO[r["top"][0][0]] if r and r["status"] == "ok" else None)))
    out = dict(run=FLOW_RUN, frame=FLOW_FRAME, fps=FLOW_FPS, detector="v12_f91_swa TensorRT FP16 · 0.20", color=getattr(col, "name", "?"),
               detect_ms_median=float(np.median(ms_all)), people=people)
    json.dump(out, open(cache, "w"), ensure_ascii=False, indent=1); return out


def _crop(src, box, size, pad=1.6):
    x1, y1, x2, y2 = box; cx, cy = (x1 + x2) / 2, (y1 + y2) / 2; s = max(x2 - x1, y2 - y1) * pad / 2 + 20
    return src.crop((int(cx - s), int(cy - s), int(cx + s), int(cy + s))).resize((size, size), Image.LANCZOS)


def _arrow(d, x, y, w=70, col=MUTE):
    d.line([x, y, x + w, y], fill=col, width=8); d.polygon([(x + w + 22, y), (x + w - 6, y - 22), (x + w - 6, y + 22)], fill=col)


def _people_sorted(W_):
    rank = {"URGENT": 0, "HIGH": 1, "CHECK": 2, "NORMAL": 3}
    return sorted([p for p in W_["people"] if p["tid"]], key=lambda p: (rank.get(p["grade"], 4), -(p["still_s"] or 0)))


def flow():
    Wo = worker_outputs(); src = Image.open(BASE / "ue_flight" / FLOW_RUN / "frames" / f"{FLOW_FRAME:05d}.jpg").convert("RGB"); Wd, Hd = src.size
    ppl = _people_sorted(Wo); print("worker 출력", [(p["tid"], p["grade"], p["posture"], p["still_s"], p["color_ko"]) for p in ppl])
    # ── 쉬운 판 (페어) ──
    G, cw = 110, (W - 4 * 110) // 5; ih = round(cw * 9 / 16); top = 360
    H = top + 140 + ih + 470 + 230; im = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(im)
    d.text((0, 10), "비전이 하는 일", font=F(132, 700), fill=INK)
    d.text((0, 190), "드론 영상에서 사람을 찾아, 누가 먼저 도움이 필요한지 순서를 매겨 관제사에게 알립니다", font=F(64, 500), fill=INK2)
    d.line([0, top - 40, W, top - 40], fill=INK, width=7)
    s = cw / Wd
    def scaled(img):
        return img.resize((cw, ih), Image.LANCZOS)
    # 1 원본
    p1 = scaled(src)
    # 2 네 조각
    p2 = scaled(src).convert("RGBA"); ov = Image.new("RGBA", p2.size, (0, 0, 0, 0)); do = ImageDraw.Draw(ov)
    tiles = [(0, 0), (640, 0), (0, 360), (640, 360)]; tcol = [(255, 255, 255, 255), (255, 214, 0, 255), (120, 200, 255, 255), (255, 140, 140, 255)]
    for k, (x, y) in enumerate(tiles):
        r = [x * s + 6 + k * 3, y * s + 6 + k * 3, (x + 1280) * s - 6 - k * 3, (y + 720) * s - 6 - k * 3]
        do.rectangle(r, outline=tcol[k], width=6); do.text((r[0] + 14, r[1] + 6), "①②③④"[k], font=F(54, 700), fill=tcol[k])
    p2 = Image.alpha_composite(p2, ov).convert("RGB")
    # 3 탐지 박스
    p3 = scaled(src); d3 = ImageDraw.Draw(p3)
    for p in Wo["people"]:
        x1, y1, x2, y2 = [v * s for v in p["box"]]; d3.rectangle([x1 - 6, y1 - 6, x2 + 6, y2 + 6], outline=YEL, width=6)
    # 4 위급 순서 카드 (2×2)
    p4 = Image.new("RGB", (cw, ih), BG); d4 = ImageDraw.Draw(p4); cs = ih // 2 - 16
    for k, p in enumerate(ppl[:4]):
        r, c = divmod(k, 2); x0, y0 = c * (cw // 2), r * (ih // 2); g = p["grade"]
        p4.paste(_crop(src, p["box"], cs), (x0 + 8, y0 + 8)); d4.rectangle([x0 + 8, y0 + 8, x0 + 8 + cs, y0 + 8 + cs], outline=GRADE_COL[g], width=8)
        d4.text((x0 + cs + 26, y0 + 34), GRADE_KO[g], font=F(56, 800), fill=GRADE_COL[g])
        d4.text((x0 + cs + 26, y0 + 122), POSE_KO.get(p["posture"], ""), font=F(40, 600), fill=INK)
        d4.text((x0 + cs + 26, y0 + 182), (f"{p['still_s']}초 멈춤" if p["still_s"] else ("움직임" if p["moving"] else "방금 멈춤")), font=F(40, 500), fill=INK2)
    # 5 관제 화면 후보 목록
    p5 = Image.new("RGB", (cw, ih), (246, 247, 249)); d5 = ImageDraw.Draw(p5); rh = (ih - 70) // 4
    d5.text((24, 14), "관제 화면 · 후보 목록", font=F(40, 700), fill=INK)
    for k, p in enumerate(ppl[:4]):
        y0 = 70 + k * rh; g = p["grade"]; th_ = rh - 16
        p5.paste(_crop(src, p["box"], th_), (20, y0 + 6)); d5.rectangle([20, y0 + 6, 20 + th_, y0 + 6 + th_], outline=GRADE_COL[g], width=5)
        d5.rounded_rectangle([th_ + 44, y0 + 18, th_ + 184, y0 + 74], 10, fill=GRADE_COL[g]); d5.text((th_ + 114, y0 + 46), GRADE_KO[g], font=F(36, 700), fill=BG, anchor="mm")
        if p["color"]:
            d5.rectangle([th_ + 210, y0 + 24, th_ + 256, y0 + 70], fill=SW[p["color"]], outline=(120, 120, 120), width=2)
            d5.text((th_ + 272, y0 + 46), f"상의 {p['color_ko']}", font=F(34, 500), fill=INK, anchor="lm")
        d5.text((th_ + 44, y0 + th_ - 16), "지도에 위치 표시", font=F(30), fill=INK2, anchor="lm")
    steps = [("드론이 찍는다", p1, ["카메라가 1920×1080 영상을", "실시간으로 서버에 보냄", "고도 16~20 m · 45° 아래"]),
             ("4조각으로 크게 본다", p2, ["화면을 겹치게 4조각으로 나눠", "멀리 있는 작은 사람도", "크게 보고 찾음"]),
             ("사람을 찾는다", p3, ["직접 학습시킨 AI (YOLO11m) 가", "사람 위치를 박스로 표시", f"한 장 {22.4:.0f} ms"]),
             ("위급한 순서로", p4, ["자세와 멈춘 시간을 봄", "누워서 10초 넘게 안 움직이면", "「긴급」 · 순서만 정하고 거르지 않음"]),
             ("관제사에게 알린다", p5, ["후보마다 사진 · 위급도 · 위치", "상의 색 (12가지) 으로", "찾는 사람과 비교 · 판단은 관제사"])]
    for k, (t, pic, lines) in enumerate(steps):
        x0 = k * (cw + G); y = top + 10
        d.ellipse([x0, y, x0 + 96, y + 96], fill=NAVY); d.text((x0 + 48, y + 48), str(k + 1), font=F(60, 700), fill=BG, anchor="mm")
        d.text((x0 + 124, y + 48), t, font=F(64, 700), fill=INK, anchor="lm")
        im.paste(pic, (x0, top + 140)); d.rectangle([x0, top + 140, x0 + cw, top + 140 + ih], outline=RULE, width=3)
        for j, ln in enumerate(lines):
            d.text((x0, top + 140 + ih + 50 + j * 74), ln, font=F(48, 600 if j == 0 else 400), fill=INK if j == 0 else INK2)
        if k < 4:
            _arrow(d, x0 + cw + 10, top + 140 + ih // 2, w=G - 50)
    yb = top + 140 + ih + 300
    d.rectangle([0, yb, W, yb + 150], fill=(246, 247, 249))
    d.text((40, yb + 75), "한 장 처리 약 40 ms (1초에 약 25장)  ·  사람 찾기 22 ms (기준 30 ms 이하)  ·  사람 없는 장면 헛탐지 0.52 건/장 (기준 1 이하)  ·  최종 판단은 관제사",
           font=F(48, 500), fill=INK, anchor="lm")
    d.text((0, yb + 190), f"그림 = 시뮬레이션 (UE · 고도 16 m · 45°) {FLOW_FRAME}번째 장면에 실제 배포 프로그램을 그대로 돌린 결과 · 등급 · 멈춘 시간 · 상의 색 모두 프로그램 출력 (손으로 바꾼 값 없음)",
           font=F(40), fill=INK2)
    for o in (OUT_FAIR, OUT_CAP):
        o.mkdir(parents=True, exist_ok=True)
    im = im.crop((0, 0, W, yb + 260)); im.save(OUT_FAIR / "비전_임무와_동작_쉬운판.png", dpi=(300, 300)); print("쉬운 판", im.size)

    # ── 자세한 판 (캡스톤) ──
    PW = json.load(open(BASE / "metrics" / "profile_worker.json")); S = PW["stages"]
    st = [("입력", ["RTSP · SRT 1920×1080 H.264", "기체 자세 · 시각 (MAVLink sidecar)", "실비행 녹화 (1.3 · 선택)"], "—"),
          ("사람 탐지", ["YOLO11m · v12_f91_swa", "TensorRT FP16 · 1280×720 × 4조각", "NMS 0.6 + 경계 잘린 박스 정리", "문턱 0.20"], f"{S['탐지']['median_ms']:.1f} ms"),
          ("추적", ["BoT-SORT · 추적 문턱 0.15", "ID 튐 끊기 (reset-jump)", "카메라 움직임 보정"], ""),
          ("상태", ["자세 = 박스 모양 확률", "  (누움 · 앉음 · 서기)", "움직임 = 3초 중앙 속도", "멈춘 시간 누적"], f"{S['추적 + 상태']['median_ms']:.1f} ms"),
          ("위급 등급", ["긴급  누움 ≥ 0.7 · 멈춤 ≥ 10 s", "높음  누움+앉음 ≥ 0.7 · 멈춤 ≥ 10 s", "확인  관측 < 3 s · 누움 0.3~0.7", "보통  그 밖 (거르지 않음)"], "(상태에 포함)"),
          ("상의 색", ["SigLIP 2 base 제로샷 12색", "잔디 문장 · 남색 문턱", "추적마다 누적 (3장마다)"], "3.3 ms/명"),
          ("좌표 · 후보", ["발끝 광선 × 지형 (DEM)", "후보 기억 (같은 자리 묶기)", "재관측 제안 · 백엔드 전송"], f"{S['좌표 (사람 전부)']['median_ms']:.2f} ms")]
    n = len(st); G2 = 70; cw2 = (W - (n - 1) * G2) // n; top = 330; bh = 640
    tw4_ = (W - 3 * 40) // 4; H2 = top + bh + 90 + round(tw4_ * 9 / 16) + 120 + 110 + 6 * 92 + 120; im = Image.new("RGB", (W, H2), BG); d = ImageDraw.Draw(im)
    d.text((0, 10), "비전 처리 흐름 — 배포 프로그램 (worker 1.3)", font=F(120, 700), fill=INK)
    d.text((0, 180), "입력 영상 한 장마다 7단계 · 중앙 서버 GPU 1장 · 탐지는 거르지 않고 순위만 매김 (최종 판단은 관제사)", font=F(58, 400), fill=INK2)
    d.line([0, top - 40, W, top - 40], fill=INK, width=7)
    for k, (t, lines, ms) in enumerate(st):
        x0 = k * (cw2 + G2)
        d.rectangle([x0, top, x0 + cw2, top + bh], outline=INK, width=4); d.rectangle([x0, top, x0 + cw2, top + 110], fill=NAVY)
        d.text((x0 + 30, top + 55), f"{k + 1}  {t}", font=F(56, 700), fill=BG, anchor="lm")
        for j, ln in enumerate(lines):
            d.text((x0 + 30, top + 160 + j * 78), ln, font=F(40, 500 if j == 0 else 400), fill=INK if j == 0 else INK2)
        if ms:
            d.text((x0 + cw2 / 2, top + bh - 60), ms, font=F(50, 700), fill=INK, anchor="mm")
        if k < n - 1:
            _arrow(d, x0 + cw2 + 6, top + bh / 2, w=G2 - 36, col=INK2)
    # 아래: 실제 화면 4장 (작게)
    yy = top + bh + 90; tw4 = (W - 3 * 40) // 4; th4 = round(tw4 * 9 / 16)
    for k, (pic, cap) in enumerate(((p1, "입력 장면 (UE · 16 m · 45°)"), (p2, "4조각 (겹침 50 %)"), (p3, "탐지 결과 (문턱 0.20)"), (p4, "등급 · 자세 · 멈춘 시간 (worker 출력)"))):
        im.paste(pic.resize((tw4, th4), Image.LANCZOS), (k * (tw4 + 40), yy)); d.text((k * (tw4 + 40), yy + th4 + 20), cap, font=F(40, 500), fill=INK2)
    # 요구사항 충족표
    ty = yy + th4 + 120; d.line([0, ty, W, ty], fill=INK, width=5)
    cols = [("요구 (명세서 v1.7)", 0), ("기준", 1500), ("지금", 2500), ("판정", 4300)]
    for t, x in cols:
        d.text((x + 10, ty + 50), t, font=F(46, 700), fill=INK, anchor="lm")
    rows = [("NFR-V01 1080p 한 장 처리", "≤ 30 ms", f"탐지 {S['탐지']['median_ms']:.1f} ms (TensorRT FP16 · RTX 3090)", "충족"),
            ("NFR-V02 처리 속도", "≥ 5 장/초", "약 25 장/초 (자세 · 등급 · 색 포함 약 40 ms)", "충족"),
            ("NFR-V06 사람 없는 영상 헛탐지", "≤ 1 건/장", "0.52 건/장 (AU-AIR 도로 233장 · 문턱 0.20)", "충족"),
            ("NFR-V04 위치 오차", "≤ 10~15 m", "시뮬레이션 중앙 0.34~0.61 m (실기체 미측정)", "시뮬 충족"),
            ("NFR-V03 사람 AP50 (장소 분리)", "≥ 0.80", "한국 0.98 · UAV-SAR 0.95 · 숲 0.66 · 비스듬 0.62", "일부 미달"),
            ("누운 사람 재현율 (실기체)", "≥ 0.80", "공개 영상 0.69 · 실기체 11-09~ 측정", "측정 전")]
    for j, row in enumerate(rows):
        y = ty + 110 + j * 92
        if j % 2 == 0:
            d.rectangle([0, y - 8, W, y + 76], fill=(247, 247, 249))
        for (t, x), v in zip(cols, row):
            d.text((x + 10, y + 34), v, font=F(42, 700 if t == "판정" else 400), fill=INK, anchor="lm")
    im = im.crop((0, 0, W, ty + 110 + len(rows) * 92 + 60)); im.save(OUT_CAP / "비전_처리흐름_상세판.png", dpi=(300, 300)); print("상세 판", im.size)



# ─────────────────────────── 3. 같은 사진 · 학습 전 vs 후 ───────────────────────────
def _dashed(d, box, col, w=6, dash=18):
    x1, y1, x2, y2 = box
    for (a, b), (c, e) in (((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)), ((x2, y2), (x1, y2)), ((x1, y2), (x1, y1))):
        L = max(1.0, ((c - a) ** 2 + (e - b) ** 2) ** 0.5); n = int(L // dash)
        for k in range(0, n, 2):
            t0, t1 = k / max(n, 1), min(1, (k + 1) / max(n, 1)); d.line([a + (c - a) * t0, b + (e - b) * t0, a + (c - a) * t1, b + (e - b) * t1], fill=col, width=w)


def _match(gt, b):
    from eval_external_det import iou_mat
    used = np.zeros(len(gt), bool); tp = np.zeros(len(b), bool)
    if len(gt) and len(b):
        M = iou_mat(gt, b)
        for j in range(len(b)):
            c = np.where((M[:, j] >= 0.5) & ~used)[0]
            if len(c):
                used[c[np.argmax(M[c, j])]] = True; tp[j] = True
    return used, tp


def beforeafter():
    from make_robust_table import eq_thr
    from robust_eval import tag_of, nomad_meta
    RUN = BASE / "runs_fair"; G = json.load(open(BASE / "metrics" / "robust_beforeafter.json"))
    gstat = {r[0]: r[3] for r in G["rows"]}
    nm = nomad_meta()
    vd = np.load(RUN / "test_visdrone" / "gt.npz")["names"]
    br = {i: cv2.imread(str(BASE / "data/test_visdrone/images" / n), cv2.IMREAD_GRAYSCALE).mean() for i, n in enumerate(vd)}
    conds = [("시점", "바로 위에서 내려다봄 (90°)", "test_sar", lambda i, n: "90d" in n),
             ("가림", "몸이 일부 가려진 사람", "test_nomad", lambda i, n: bool(nm.get(i)) and any(50 <= v <= 90 for v in nm[i][0])),
             ("자세", "누워 있는 사람", "test_nomad", lambda i, n: bool(nm.get(i)) and nm[i][1] in ("Laying", "Hiding (Laying)")),
             ("조도", "밤 · 어두운 장면", "test_visdrone", lambda i, n: br[i] < 55),
             ("배경", "야생 지형 · 작게 보이는 사람", "test_v2", lambda i, n: True),
             ("지역", "한국 산지 (조난자 수색)", "test_kr", lambda i, n: True)]
    rows = []
    for kind, label, s, sel in conds:
        g = np.load(RUN / s / "gt.npz"); names = [str(x) for x in g["names"]]
        tb = tag_of("coco_yolo11m", s); ta = "v12_f91_swa@tile1280"; thb = eq_thr(tb)[0]
        db, da = np.load(RUN / s / f"{tb}.npz"), np.load(RUN / s / f"{ta}.npz"); best = None
        for i, n in enumerate(names):
            if not sel(i, n):
                continue
            gt = g["gt"][g["gt_img"] == i]
            if not len(gt):
                continue
            bb = db["box"][(db["img"] == i) & (db["conf"] >= thb)]; ba = da["box"][(da["img"] == i) & (da["conf"] >= 0.20)]
            ub, tpb = _match(gt, bb); ua, tpa = _match(gt, ba)
            fp_a = int((~tpa).sum()); score = (ua.sum() - ub.sum()) - 0.5 * fp_a
            if ua.sum() >= max(1, 0.8 * len(gt)) and (best is None or score > best[0]):
                best = (score, i, n, gt, bb, ba, ub, ua, tpb, tpa)
        rows.append((kind, label, s, best, thb)); print(kind, s, best[2] if best else None, f"우리 {best[7].sum()}/{len(best[3])} · 학습 전 {best[6].sum()}/{len(best[3])}" if best else "")

    def panel(img, gt, boxes, used, tp, crop, tw, th):
        x0, y0, x1, y1 = crop; c = img.crop(crop).resize((tw, th), Image.LANCZOS); d = ImageDraw.Draw(c); sx, sy = tw / (x1 - x0), th / (y1 - y0)
        T = lambda b: [(b[0] - x0) * sx, (b[1] - y0) * sy, (b[2] - x0) * sx, (b[3] - y0) * sy]
        for b, ok in zip(boxes, tp):
            r = T(b)
            if ok:
                d.rectangle([r[0] - 5, r[1] - 5, r[2] + 5, r[3] + 5], outline=YEL, width=8)
            else:
                d.rectangle([r[0] - 4, r[1] - 4, r[2] + 4, r[3] + 4], outline=(255, 255, 255), width=3)
        for b, u in zip(gt, used):
            if not u:
                r = T(b); _dashed(d, [r[0] - 10, r[1] - 10, r[2] + 10, r[3] + 10], RED, w=7)
        return c

    def crop_of(gt, W_, H_):
        x1, y1 = gt[:, 0].min(), gt[:, 1].min(); x2, y2 = gt[:, 2].max(), gt[:, 3].max()
        w = max(x2 - x1 + 160, 720); h = max(y2 - y1 + 120, 405)
        if w / h > 16 / 9:
            h = w * 9 / 16
        else:
            w = h * 16 / 9
        w, h = min(w, W_), min(h, H_); cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        a = min(max(cx - w / 2, 0), W_ - w); b = min(max(cy - h / 2, 0), H_ - h); return (int(a), int(b), int(a + w), int(b + h))

    LBL, GAP = 900, 50; tw = (W - LBL - 3 * GAP - 520) // 2; th = round(tw * 9 / 16); top = 420
    keep = [r for r in rows if r[3]][:6]
    def draw_rows(sel_rows, fname, title, sub):
        H = top + len(sel_rows) * (th + GAP) + 260; im = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(im)
        d.text((0, 10), title, font=F(120, 700), fill=INK); d.text((0, 180), sub, font=F(56, 400), fill=INK2)
        d.line([0, top - 120, W, top - 120], fill=INK, width=7)
        d.text((LBL + GAP + tw / 2, top - 60), "학습 전 · 일반 AI (YOLO11m · COCO)", font=F(52, 700), fill=INK2, anchor="mm")
        d.text((LBL + 2 * GAP + tw + tw / 2, top - 60), "학습 후 · 우리 팀", font=F(52, 700), fill=INK, anchor="mm")
        d.text((W - 260, top - 60), "이 조건 전체", font=F(46, 700), fill=INK, anchor="mm")
        for k, (kind, label, s, b, thb) in enumerate(sel_rows):
            _, i, n, gt, bb, ba, ub, ua, tpb, tpa = b; y = top + k * (th + GAP)
            img = Image.open(BASE / "data" / s / "images" / n).convert("RGB"); cr = crop_of(gt, *img.size)
            d.text((0, y + 60), kind, font=F(72, 800), fill=INK); d.text((0, y + 170), label, font=F(44, 500), fill=INK)
            ds = {"test_sar": "UAV-SAR", "test_nomad": "NOMAD", "test_visdrone": "VisDrone", "test_v2": "WiSARD", "test_kr": "AI-Hub"}[s]
            d.text((0, y + 240), ds, font=F(40), fill=INK2)
            d.text((0, y + th - 150), f"이 사진  {int(ub.sum())} / {len(gt)}명", font=F(48, 500), fill=INK2)
            d.text((0, y + th - 80), f"     →  {int(ua.sum())} / {len(gt)}명", font=F(48, 800), fill=INK)
            im.paste(panel(img, gt, bb, ub, tpb, cr, tw, th), (LBL + GAP, y)); im.paste(panel(img, gt, ba, ua, tpa, cr, tw, th), (LBL + 2 * GAP + tw, y))
            st = gstat.get(kind)
            if st:
                d.text((W - 260, y + th / 2 - 60), f"{st['before']:.2f}", font=F(56, 500), fill=INK2, anchor="mm")
                d.text((W - 260, y + th / 2 + 10), "↓", font=F(56, 500), fill=MUTE, anchor="mm")
                d.text((W - 260, y + th / 2 + 90), f"{st['after']:.2f}", font=F(72, 800), fill=INK, anchor="mm")
        yl = top + len(sel_rows) * (th + GAP) + 40
        d.rectangle([0, yl, 80, yl + 56], outline=YEL, width=8); d.text((110, yl + 28), "찾은 사람", font=F(48, 500), fill=INK, anchor="lm")
        _dashed(d, [480, yl, 560, yl + 56], RED, w=7); d.text((590, yl + 28), "놓친 사람", font=F(48, 500), fill=INK, anchor="lm")
        d.rectangle([960, yl + 4, 1040, yl + 52], outline=(160, 160, 160), width=3); d.text((1070, yl + 28), "사람이 아닌데 찾은 것", font=F(48, 500), fill=INK, anchor="lm")
        d.text((0, yl + 120), "두 모델 모두 사람 없는 사진의 헛탐지를 같게 (0.5 건/장) 맞춘 문턱 · 「이 조건 전체」 = 그 조건 사람 중 찾은 비율 · 사진은 조건마다 차이가 잘 보이는 것을 고름 (우리가 대부분 찾은 사진 중)",
               font=F(38), fill=INK2)
        im.save(fname, dpi=(300, 300)); print(fname.name, im.size)
    OUT_COM.mkdir(parents=True, exist_ok=True)
    draw_rows(keep[:3], OUT_COM / "같은사진_학습전후_1.png", "같은 사진, 다른 결과 — 학습 전과 후", "같은 모델 구조 (YOLO11m) · 학습 전 = 공개 COCO 가중치 · 학습 후 = 우리가 드론 사진 11종으로 학습")
    draw_rows(keep[3:6], OUT_COM / "같은사진_학습전후_2.png", "같은 사진, 다른 결과 — 학습 전과 후 (계속)", "같은 모델 구조 (YOLO11m) · 학습 전 = 공개 COCO 가중치 · 학습 후 = 우리가 드론 사진 11종으로 학습")



# ─────────────────────────── 4. (캡스톤) 성능이 올라온 과정 ───────────────────────────
def journey():
    # 수치 원본: obsidian _실험 색인 · 결정 노트 (장소 분리 평가 · 단일 모델 AP50) — 10-11 이후는 경계 합치기 · 10-12 이후 숲은 무시 영역 (측정 방식 바뀜 표시)
    steps = [("09-23", "장소 분리 평가\n첫 측정", 0.223, 0.193), ("09-25", "60° 비스듬\n데이터로 교체", 0.529, 0.619), ("09-26", "야생 지형\n(SARD · HERIDAL)", 0.578, 0.631),
             ("09-27", "같은 설정 2개\n가중치 평균", 0.593, 0.633), ("10-11", "타일 경계\n박스 정리 *", 0.601, 0.633), ("10-12", "숲 라벨 구멍\n무시 영역 *", 0.611, 0.633),
             ("10-13", "데이터 원본부터\n다시 (v12)", 0.659, 0.616)]
    dead = [("모델 키우기 (11m → 11l)", "−3 ~ −8 %"), ("드론 사진으로 먼저 학습 (VisDrone)", "−2.3 %p"), ("작은 사람 손실 · P2 머리", "−2.4 %p · ±0"), ("다른 구조 (YOLO26m)", "맞바꿈"),
            ("더 오래 학습 (60 에폭 · 이어서)", "−4.0 · −2.4 %p"), ("사람 없는 사진 15.8 %", "−7.3 %p"), ("모델이 만든 라벨 · 잎 가림 합성", "구분 안 됨 · 헛탐지 ↑"),
            ("누운 사람 3배 · 붙여넣기 · 타원 박스", "누움 개선 없음"), ("화질 열화 증강 · 추론 전 흐림", "헛탐지 1.1~1.9 건/장"), ("공개 가중치 13종 채택", "12셋 중 11셋 우리 우위"),
            ("현장 영상으로 BN 통계 맞추기 (11-08)", "숲 −7.7 ~ −18 %p"), ("SigLIP 2차 확인으로 오탐 거르기 (11-08)", "재현율 −3.6 %p")]
    H = 2900; im = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(im)
    d.text((0, 10), "성능이 올라온 과정 — 장소 분리 평가 AP50", font=F(120, 700), fill=INK)
    d.text((0, 180), "학습에 한 장도 안 쓴 장소로만 잼 (숲 = WiSARD · 비스듬 = Okutama) · 한 번에 하나씩 바꾸고 돌리기 전에 판정 기준을 적음", font=F(56), fill=INK2)
    d.line([0, 310, W, 310], fill=INK, width=7)
    X0, X1, Y0, Y1 = 260, 2950, 1500, 420; xs = lambda k: X0 + (X1 - X0) * k / (len(steps) - 1); ys = lambda v: Y0 - (Y0 - Y1) * v
    for v in (0, 0.2, 0.4, 0.6, 0.8, 1.0):
        d.line([X0 - 30, ys(v), X1 + 30, ys(v)], fill=RULE, width=2); d.text((X0 - 60, ys(v)), f"{v:.1f}", font=F(40), fill=INK2, anchor="rm")
    d.line([X0 - 30, ys(0.8), X1 + 30, ys(0.8)], fill=INK2, width=4); d.text((X1 + 50, ys(0.8)), "요구 0.80", font=F(42, 600), fill=INK2, anchor="lm")
    for key, idx, col, wd in (("숲 (WiSARD)", 2, NAVY, 10), ("비스듬 (Okutama)", 3, MUTE, 8)):
        pts = [(xs(k), ys(s_[idx])) for k, s_ in enumerate(steps)]; d.line(pts, fill=col, width=wd)
        for k, (x, y) in enumerate(pts):
            d.ellipse([x - 20, y - 20, x + 20, y + 20], fill=col if idx == 2 else BG, outline=col, width=6)
        x, y = pts[-1]; d.text((x + 50, y - 40 if idx == 2 else y + 40), f"{key}  {steps[-1][idx]:.2f}", font=F(44, 700 if idx == 2 else 500), fill=col if idx == 2 else INK2, anchor="lm")
    for k, (dt, t, v2, ob) in enumerate(steps):
        d.text((xs(k), Y0 + 60), dt, font=F(40, 600), fill=INK, anchor="mm")
        for j, ln in enumerate(t.split("\n")):
            d.text((xs(k), Y0 + 130 + j * 56), ln, font=F(38), fill=INK2, anchor="mm")
        d.text((xs(k), ys(v2) - 60), f"{v2:.2f}", font=F(40, 700), fill=NAVY, anchor="mm")
    d.text((X0, Y0 + 320), "* 같은 가중치로 측정 방식만 고친 단계 (타일 경계에서 잘린 박스 정리 · 라벨 없는 사람 무시) · 10-13 값은 배포 모델 (같은 설정 3회 평균 숲 0.64 · 비스듬 0.62)", font=F(36), fill=INK2)
    # 오른쪽: 해 봤지만 버린 것
    RX = 3700; d.text((RX, 400), "해 봤지만 효과가 없어 버린 것", font=F(60, 700), fill=INK); d.line([RX, 490, W, 490], fill=INK, width=4)
    for j, (a, b) in enumerate(dead):
        y = 540 + j * 92
        if j % 2 == 0:
            d.rectangle([RX, y - 10, W, y + 74], fill=(247, 247, 249))
        d.text((RX + 20, y + 32), a, font=F(38), fill=INK, anchor="lm"); d.text((W - 20, y + 32), b, font=F(38, 600), fill=INK2, anchor="rm")
    d.text((RX, 540 + len(dead) * 92 + 30), "판정 = 같은 설정 반복 2회 평균 · 짝 블록 부트스트랩 95 % 구간 · 세 평가셋 어디서도 손해 없음", font=F(34), fill=INK2)
    # 아래: 교훈 3줄
    ly = 1950; d.line([0, ly, W, ly], fill=INK, width=5)
    lessons = [("데이터가 구조보다 컸다", "60° 데이터 +31 %p · 야생 지형 +5 %p · 데이터 재구성 +5 %p ↔ 모델 키우기 · 구조 바꾸기는 손해이거나 반복 폭 안"),
               ("재는 법을 먼저 고쳤다", "장소가 섞인 평가는 0.86 (가짜) → 장소 분리 0.22 (진짜) 에서 다시 시작 · 운 좋은 1회 대신 반복 평균"),
               ("남은 0.80 은 우리 카메라 데이터로", "공개 데이터 조작은 다 막힘 · 다른 데이터로 학습한 모델과 합치면 +5.5~6.2 %p → 데이터 다양성 · 실기체 영상 (11-09~)")]
    for j, (a, b) in enumerate(lessons):
        y = ly + 60 + j * 200
        d.text((0, y), a, font=F(52, 700), fill=INK); d.text((0, y + 80), b, font=F(42), fill=INK2)
    OUT_CAP.mkdir(parents=True, exist_ok=True); im = im.crop((0, 0, W, ly + 60 + 3 * 200 + 20)); im.save(OUT_CAP / "성능_향상_과정.png", dpi=(300, 300)); print("journey", im.size)


# ─────────────────────────── 5. (캡스톤) 아직 약한 곳 ───────────────────────────
def failures():
    from eval_external_det import iou_mat
    RUN = BASE / "runs_fair"; panels = []
    # ① 미끼 물체 — UAV-SAR 배경 오탐 중 확신 최고
    g = np.load(RUN / "test_sar" / "gt.npz"); dd = np.load(RUN / "test_sar" / "v12_f91_swa@tile1280.npz"); best = None
    for i in range(len(g["names"])):
        gt = g["gt"][g["gt_img"] == i]; m = (dd["img"] == i) & (dd["conf"] >= 0.5); b, c = dd["box"][m], dd["conf"][m]
        M = iou_mat(gt, b) if len(gt) and len(b) else np.zeros((len(gt), len(b)))
        for j in range(len(b)):
            if (M[:, j].max() if len(gt) else 0) < 0.05 and (best is None or c[j] > best[0]):
                best = (float(c[j]), i, b[j], gt)
    c1, i1, b1, gt1 = best; panels.append(("test_sar", str(g["names"][i1]), [("pred", b1, f"우리 {c1:.2f}")], b1,
                                            "사람처럼 생긴 물체", ["옷을 걸친 의자를 사람으로 봄 (확신 0.88)", "UAV-SAR 헛탐지 대부분이 이런 미끼 물체", "→ 실기체 빈 장면에 미끼를 놓고 찍어 학습"]))
    # ② 많이 가린 사람 — NOMAD 가시도 ≤ 40 % · 우리도 놓침
    from robust_eval import nomad_meta
    nm = nomad_meta(); g = np.load(RUN / "test_nomad" / "gt.npz"); dd = np.load(RUN / "test_nomad" / "v12_f91_swa@tile1280.npz"); pick = None
    for i in range(len(g["names"])):
        if nm.get(i) and nm[i][0] and nm[i][0][0] <= 30:
            gt = g["gt"][g["gt_img"] == i]; m = (dd["img"] == i) & (dd["conf"] >= 0.20)
            if len(gt) and not (len(dd["box"][m]) and iou_mat(gt, dd["box"][m]).max() >= 0.5) and max(gt[0][2] - gt[0][0], gt[0][3] - gt[0][1]) > 80:
                pick = (i, gt[0]); break
    i2, gb = pick; panels.append(("test_nomad", str(g["names"][i2]), [("gt", gb, "놓침")], gb, "많이 가려진 사람", ["몸이 40 % 이하로 보이면 거의 못 찾음", "모든 모델 0.13 이하 (가장 어려운 조건)", "→ 재관측 (다른 각도 · 낮은 고도) 으로 보완"]))
    # ③ 정답 박스 습관 — WiSARD Karen 헐렁한 정답 vs 우리 딱 맞는 박스
    g = np.load(RUN / "test_v2" / "gt.npz"); dd = np.load(RUN / "test_v2" / "v12_f91_swa@tile1280.npz"); pick = None
    for i, n in enumerate(g["names"]):
        if "Karen" not in str(n):
            continue
        gt = g["gt"][g["gt_img"] == i]; m = (dd["img"] == i) & (dd["conf"] >= 0.3); b = dd["box"][m]
        if not len(gt) or not len(b):
            continue
        M = iou_mat(gt, b)
        for gi in range(len(gt)):
            j = int(np.argmax(M[gi])); A = lambda x: (x[2] - x[0]) * (x[3] - x[1])
            if 0.1 < M[gi, j] < 0.4 and A(b[j]) < 0.4 * A(gt[gi]) and gt[gi][0] <= b[j][0] and gt[gi][2] >= b[j][2]:
                pick = (i, gt[gi], b[j], float(M[gi, j])); break
        if pick:
            break
    i3, g3, b3, io = pick; panels.append(("test_v2", str(g["names"][i3]), [("gt", g3, "정답"), ("pred", b3, "우리")], g3, "정답 박스 그리는 습관", [f"정답 (빨강) 이 사람보다 2배 넘게 큼 · 겹침 {io:.2f}", "우리 박스는 사람에 딱 맞는데 「놓침」 으로 셈", "→ 박스 습관 빼면 숲 0.64 → 0.82 · 비스듬 0.62 → 0.86"]))
    pw = (W - 2 * 80) // 3; ph = pw; H = 330 + ph + 520; im = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(im)
    d.text((0, 10), "아직 약한 곳 — 숨기지 않고 대책과 함께", font=F(120, 700), fill=INK)
    d.text((0, 180), "공개 평가셋에서 직접 찾은 실패 사례 · 노랑 = 우리 모델 박스 · 빨간 점선 = 정답 (놓친 사람 · 정답 박스)", font=F(56), fill=INK2)
    d.line([0, 310, W, 310], fill=INK, width=7)
    for k, (s, n, marks, center, title, lines) in enumerate(panels):
        x0 = k * (pw + 80); img = Image.open(BASE / "data" / s / "images" / n).convert("RGB"); cx, cy = (center[0] + center[2]) / 2, (center[1] + center[3]) / 2
        half = max(center[2] - center[0], center[3] - center[1]) * 1.6 + 60; cr = (int(cx - half), int(cy - half), int(cx + half), int(cy + half))
        c = img.crop(cr).resize((pw, ph), Image.LANCZOS); dc = ImageDraw.Draw(c); sc = pw / (cr[2] - cr[0])
        for kind, b, lab in marks:
            r = [(b[0] - cr[0]) * sc, (b[1] - cr[1]) * sc, (b[2] - cr[0]) * sc, (b[3] - cr[1]) * sc]
            if kind == "pred":
                dc.rectangle([r[0] - 6, r[1] - 6, r[2] + 6, r[3] + 6], outline=YEL, width=10)
            else:
                _dashed(dc, [r[0] - 6, r[1] - 6, r[2] + 6, r[3] + 6], RED, w=9, dash=26)
        im.paste(c, (x0, 360)); d.text((x0, 360 + ph + 40), f"{k + 1}  {title}", font=F(64, 700), fill=INK)
        for j, ln in enumerate(lines):
            d.text((x0, 360 + ph + 150 + j * 70), ln, font=F(44, 600 if ln.startswith("→") else 400), fill=INK if ln.startswith("→") else INK2)
    im.save(OUT_CAP / "아직_약한_곳.png", dpi=(300, 300)); print("failures", im.size)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name in (["datasets", "flow", "beforeafter", "journey", "failures"] if what == "all" else [what]):
        globals()[name]()
