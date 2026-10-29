"""
color_lab.py — 상의 색 판정 방식 비교 (10-29 K · 판정 기준 = obsidian `_학습 큐` 「10-29 K」 · 계획 「19 상의 색 판별 개선 계획」)

  평가셋
    U  UAV-Human 속성 시험셋 (드론 · 사람 30명 · 학습과 겹침 0) · 05 여러색 뺌 · 10색 · 색마다 ≤ 120장 (사람마다 고르게 · 시드 0)
       원본 크롭 (키 중앙 410 px) → 사람 키 48 · 72 · 120 px 로 줄임 · 둘레 25 % 가장자리 복제로 「프레임」 을 만들어 worker 와 같은 입력
    N  NOMAD 42명 (eval_color.py 캐시 · 1080p 크롭 · V 판정기 고르는 데 이미 씀 → 참고)
  방식 = 단계 조합 (method 이름 → 설정) — 화이트밸런스 (frame · local · none) × 노출 × 사람 마스크 (grabcut · yoloseg · segformer) × 화소 분류 (hsv · w2c)
         + SigLIP 2 제로샷 (clip0) · SigLIP 2 특징 + 선형 (probe · UAV-Human 학습셋만)
  출력  metrics/color_lab.json (방식 × 셋 지표 · 기준 V10 대비 짝 블록 부트스트랩) · 캐시 (크롭 · 특징) 는 git 밖 data/raw/color/_lab/
  ⚠ UAV-Human · NOMAD 크롭 · 결과 이미지는 git 에 올리지 않는다
실행  /home/se/venvs/state/bin/python color_lab.py build            (평가셋 캐시)
      /home/se/venvs/state/bin/python color_lab.py run <방식 …|all>  (예측 · 지표)
"""
import json, pickle, random, re, sys, time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / "deploy" / "vision_worker"))
from app.vision_core.color import COLORS, ColorAccumulator, ColorProfiler, _classify_hsv, _person_mask, frame_stats, match  # noqa: E402

RAW = BASE.parent / "data" / "raw"
UAVH = RAW / "uav_human" / "AttributeRecognition" / "uavhuman"
LAB = RAW / "color" / "_lab"
W2C = RAW / "color" / "ColorNaming" / "w2c.txt"
SEG = RAW / "color" / "models" / "yolo11n-seg.pt"
NCACHE = Path("/tmp/eval_color_cache.pkl")
UC = {"01": "red", "02": "black", "03": "blue", "04": "green", "06": "gray", "07": "white", "08": "yellow", "09": "brown", "10": "purple", "11": "pink"}
HEIGHTS = (48, 72, 120)
CI = {c: i for i, c in enumerate(COLORS)}


# ───────────────────────── 평가셋 ─────────────────────────
def _frame_from_crop(img, h_target):
    """원본 사람 크롭 → 키 h_target px 로 줄이고 둘레 25 % 복제 → (프레임, 박스 xyxy)"""
    s = h_target / img.shape[0]
    im = cv2.resize(img, (max(4, round(img.shape[1] * s)), h_target), interpolation=cv2.INTER_AREA)
    ph, pw = round(0.25 * im.shape[0]), round(0.25 * im.shape[1])
    fr = cv2.copyMakeBorder(im, ph, ph, pw, pw, cv2.BORDER_REPLICATE)
    return fr, (pw, ph, pw + im.shape[1], ph + im.shape[0])


def uavh_list(split, per_class, seed=0, skip=()):
    by = defaultdict(lambda: defaultdict(list))
    for f in sorted((UAVH / split).glob("*.jpg")):
        m = re.search(r"UC(\d\d)", f.name)
        if m and m.group(1) in UC and f.name not in skip:
            by[UC[m.group(1)]][re.match(r"P(\d+)", f.name).group(1)].append(f)
    rng = random.Random(seed); out = []
    for c in sorted(by):
        persons = {p: rng.sample(v, len(v)) for p, v in sorted(by[c].items())}
        pick, k = [], 0
        while len(pick) < per_class and any(persons.values()):         # 사람마다 돌아가며 한 장씩 (고르게)
            for p in list(persons):
                if persons[p] and len(pick) < per_class:
                    pick.append((persons[p].pop(), c, p))
            k += 1
        out += pick
    return out


def build():
    LAB.mkdir(parents=True, exist_ok=True)
    U = []
    for f, c, p in uavh_list("test", 120):
        img = cv2.imread(str(f))
        for h in HEIGHTS:
            fr, box = _frame_from_crop(img, h)
            U.append({"frame": fr, "box": box, "stats": frame_stats(fr), "pose": None, "truth": {c}, "primary": c, "person": p, "h": h, "file": f.name})
    pickle.dump(U, open(LAB / "U.pkl", "wb"))
    print("U", len(U), Counter(u["primary"] for u in U if u["h"] == 72))
    # N — eval_color.py 의 묶음 · 순서를 그대로 다시 만든다 (캐시 순서와 같아야 함)
    import eval_color as E
    import csv
    truth = {r["actor"]: r["upper_color"].strip().lower() for r in csv.DictReader(open(BASE / "metrics" / "nomad_actor_selection.csv", encoding="utf-8-sig"))}
    acts = {x["id"]: x["labels"] for x in json.load(open(E.NOMAD / "activityLabels.json"))}
    groups = defaultdict(list)
    for r in json.load(open(E.NOMAD / "annotations.json")):
        actor_s, dist_s, f_s = r["file_name"][:-4].split("_")
        if actor_s not in truth or truth[actor_s] not in E.MAP:
            continue
        pose = E.POSE.get(E._nomad_act(acts, int(actor_s[5:]), dist_s[1:], int(f_s[1:])))
        boxes = [b["bbox"] for b in r["annotations"] if int(b.get("visibility", 100)) >= 50]
        if pose is None or len(boxes) != 1:
            continue
        groups[(actor_s, dist_s, pose)].append(1)
    keys = []
    for k, v in sorted(groups.items()):
        idx = np.linspace(0, len(v) - 1, min(12, len(v))).round().astype(int)
        keys += [k] * len(set(idx))
    items = pickle.load(open(NCACHE, "rb"))
    assert len(items) == len(keys), (len(items), len(keys))
    N = []
    for it, k in zip(items, keys):
        if it is None:
            continue
        t = truth[k[0]]
        N.append({"frame": it["crop"], "box": it["box"], "stats": it["stats"], "pose": it["pose"], "truth": E.MAP[t], "primary": E.PRIMARY[t],
                  "person": k[0], "group": k, "h": it["long_px"]})
    pickle.dump(N, open(LAB / "N.pkl", "wb"))
    print("N", len(N), "배우", len({n["person"] for n in N}))


# ───────────────────────── 단계 ─────────────────────────
_W2C = None


def w2c_table():
    global _W2C
    if _W2C is None:
        t = np.loadtxt(W2C)[:, 3:]                                     # 32768 × 11 (black blue brown grey green orange pink purple red white yellow)
        order = ["black", "blue", "brown", "gray", "green", "orange", "pink", "purple", "red", "white", "yellow"]
        T = np.zeros((32768, len(COLORS)), np.float32)
        for j, c in enumerate(order):
            T[:, CI[c]] = t[:, j]
        _W2C = T
    return _W2C


def _crop(it, margin=1.3):
    fr, (x1, y1, x2, y2) = it["frame"], it["box"]
    H, W = fr.shape[:2]; bw, bh = x2 - x1, y2 - y1; cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    X1, Y1 = int(max(0, cx - bw * margin / 2)), int(max(0, cy - bh * margin / 2))
    X2, Y2 = int(min(W, cx + bw * margin / 2)), int(min(H, cy + bh * margin / 2))
    rect = (max(int(round(x1 - X1)), 0), max(int(round(y1 - Y1)), 0), max(int(round(bw)), 1), max(int(round(bh)), 1))
    return fr[Y1:Y2, X1:X2].copy(), rect


def _achro_gain(px):
    hsv = cv2.cvtColor(px.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    a = (hsv[:, 1] < 40) & (hsv[:, 2] > 50) & (hsv[:, 2] < 245)
    if a.sum() < 30:
        return None
    m = px.reshape(-1, 3)[a].astype(np.float32).mean(0)
    return (m.mean() / np.maximum(m, 1e-3)).astype(np.float32)


_SEG = None


def seg_mask(crop, rect):
    """YOLO11n-seg 사람 마스크 (크롭을 키 ≥ 256 으로 키워 돌림) — 박스와 가장 많이 겹치는 사람 · 없으면 None"""
    global _SEG
    if _SEG is None:
        from ultralytics import YOLO
        _SEG = YOLO(str(SEG))
    k = max(1.0, 256 / crop.shape[0])
    big = cv2.resize(crop, (round(crop.shape[1] * k), round(crop.shape[0] * k)), interpolation=cv2.INTER_CUBIC)
    r = _SEG.predict(big, imgsz=320, classes=[0], conf=0.1, verbose=False, half=True)[0]
    if r.masks is None or len(r.masks) == 0:
        return None
    x, y, w, h = rect; best, bm = 0, None
    for m in r.masks.data.cpu().numpy():
        m = cv2.resize(m.astype(np.uint8), (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST)
        ov = m[y:y + h, x:x + w].sum()
        if ov > best:
            best, bm = ov, m
    return bm if bm is not None and bm.sum() >= 0.15 * w * h else None


_SF = None


def segformer_upper(crop, rect):
    """SegFormer 옷 분할 — 상의 (4) · 원피스 (7) · 목도리 (17) 화소 · 크롭을 키 ≥ 384 로 키움 · 적으면 None"""
    global _SF
    import torch
    if _SF is None:
        from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
        name = "mattmdjaga/segformer_b2_clothes"
        _SF = (AutoImageProcessor.from_pretrained(name), SegformerForSemanticSegmentation.from_pretrained(name).cuda().half().eval())
    proc, model = _SF
    k = max(1.0, 384 / crop.shape[0])
    big = cv2.resize(crop, (round(crop.shape[1] * k), round(crop.shape[0] * k)), interpolation=cv2.INTER_CUBIC)
    inp = proc(images=cv2.cvtColor(big, cv2.COLOR_BGR2RGB), return_tensors="pt")
    with torch.no_grad():
        lg = model(pixel_values=inp["pixel_values"].cuda().half()).logits
    lab = torch.nn.functional.interpolate(lg.float(), size=crop.shape[:2], mode="bilinear").argmax(1)[0].cpu().numpy()
    m = np.isin(lab, (4, 7, 17)).astype(np.uint8)
    x, y, w, h = rect
    box = np.zeros_like(m); box[y:y + h, x:x + w] = 1
    m &= box
    return m if m.sum() >= 15 else None


def classify_lab(L, a, b, c_achro=10.0, l_black=28.0, l_white=78.0):
    """CIE Lab (L 0~100 · a · b) 화소 → 색 번호 — 채도가 있으면 어두워도 검정으로 안 보냄 (진한 초록 · 보라 → 검정 오답 겨냥)
    무채색 (C < c_achro): L < l_black 검정 · > l_white 흰 · 사이 회색 / 유채색: 아주 어두움 (L < 12) 만 검정 · 색상각 h 로 나눔"""
    C = np.hypot(a, b); h = (np.degrees(np.arctan2(b, a)) + 360) % 360
    out = np.full(L.shape, -1, np.int16)
    ach = C < c_achro
    out[ach & (L < l_black)] = CI["black"]; out[ach & (L >= l_black) & (L <= l_white)] = CI["gray"]; out[ach & (L > l_white)] = CI["white"]
    ch = ~ach
    out[ch & (L < 12)] = CI["black"]
    ch = ch & (L >= 12)
    redh = (h >= 355) | (h < 45)                                        # 경계 = 실측 색상각 (10-29 · 빨강 34 · 분홍 351 · 보라 312~328 · 파랑 282 · 남색 291~297 · 하늘 250 · 청록 196)
    out[ch & redh & (L >= 62) & (C < 50)] = CI["pink"]
    out[ch & redh & ~((L >= 62) & (C < 50))] = CI["red"]
    out[ch & (h >= 45) & (h < 75) & (L >= 50)] = CI["orange"]
    out[ch & (h >= 45) & (h < 100) & (L < 50)] = CI["brown"]
    out[ch & (h >= 75) & (h < 105) & (L >= 50)] = CI["yellow"]
    out[ch & (h >= 105) & (h < 200)] = CI["green"]
    out[ch & (h >= 200) & (h < 303) & (L >= 30)] = CI["blue"]
    out[ch & (h >= 200) & (h < 303) & (L < 30)] = CI["navy"]
    out[ch & (h >= 303) & (h < 340)] = CI["purple"]
    out[ch & (h >= 340) & (h < 355)] = CI["pink"]
    return out


def rule(it, wb="frame", exp=(0.7, 2.0), mask="grabcut", pix="hsv", pre=None, glare_l=95.0, w_black=0.3, w_gray=0.7, region=(0.15, 0.60), min_px=40, navy_v=0.45, v_black=0.30):
    """규칙 계열 한 판 — worker ColorProfiler 와 같은 순서 · 단계만 바꿔 끼움 · 반환 12색 확률 또는 None (판정 불가)
    GrabCut 난수 시드를 매번 0 으로 (worker 도 같은 입력 두 번에 1 % 정도 1순위가 바뀜 · 시드 고정하면 worker 와 같음 10-29)"""
    cv2.setRNGSeed(0)
    crop, rect = _crop(it)
    if crop.size == 0 or rect[2] < 4 or rect[3] < 4:
        return None
    x, y, w, h = rect
    gain = None
    if wb == "frame":
        gain = it["stats"]["achromatic"]
    elif wb == "local":                                                 # 사람 둘레 (박스 좌우 1배 · 위아래 0.5배 · 박스 밖) 무채색 화소
        fr, (bx1, by1, bx2, by2) = it["frame"], it["box"]
        bw, bh = bx2 - bx1, by2 - by1
        X1, Y1, X2, Y2 = int(max(0, bx1 - bw)), int(max(0, by1 - bh * 0.5)), int(min(fr.shape[1], bx2 + bw)), int(min(fr.shape[0], by2 + bh * 0.5))
        ring = fr[Y1:Y2, X1:X2].copy(); keep = np.ones(ring.shape[:2], bool)
        keep[int(by1 - Y1):int(by2 - Y1), int(bx1 - X1):int(bx2 - X1)] = False
        gain = _achro_gain(ring[keep]) if keep.sum() > 30 else None
        if gain is None:
            gain = it["stats"]["achromatic"]
    if gain is not None:
        crop = np.clip(crop.astype(np.float32) * gain, 0, 255).astype(np.uint8)
    if pre == "clahe":                                                  # Lab 의 L 만 국소 대비 맞춤 (a · b = 색은 그대로)
        lab0 = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        t = max(1, min(4, crop.shape[0] // 24))
        lab0[..., 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(t, t)).apply(lab0[..., 0])
        crop = cv2.cvtColor(lab0, cv2.COLOR_LAB2BGR)
    reg_mask = None
    if mask == "segformer":
        reg_mask = segformer_upper(crop, rect)
        m0 = reg_mask if reg_mask is not None else _person_mask(crop, rect)
    else:
        m0 = seg_mask(crop, rect) if mask == "yoloseg" else None
        if m0 is None:
            m0 = _person_mask(crop, rect)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    ycc = cv2.cvtColor(crop, cv2.COLOR_BGR2YCrCb)                      # 피부는 노출 맞춤 전 (worker 와 같음)
    if exp is not None and it["stats"].get("v_median"):
        g = float(np.clip(110 / max(it["stats"]["v_median"], 1.0), *exp))
        hsv[..., 2] = np.clip(hsv[..., 2].astype(np.float32) * g, 0, 255).astype(np.uint8)
        if pix in ("w2c", "lab"):
            crop = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    if reg_mask is not None:
        region_m = reg_mask
    else:
        region_m = np.zeros(crop.shape[:2], np.uint8)
        if it["pose"] == "lying":
            region_m[y:y + h, x:x + w] = 1
        else:
            region_m[y + int(region[0] * h):y + int(region[1] * h), x:x + w] = 1
    skin = (ycc[..., 1] >= 135) & (ycc[..., 1] <= 180) & (ycc[..., 2] >= 85) & (ycc[..., 2] <= 135) & (hsv[..., 2] > 60)
    sel = (m0 > 0) & (region_m > 0) & ~skin & ~(hsv[..., 2] > 247)
    if int(sel.sum()) < min_px:
        return None
    vw = np.ones(len(COLORS)); vw[CI["black"]] = w_black; vw[CI["gray"]] = w_gray
    if pix == "hsv":
        lab = _classify_hsv(hsv[..., 0][sel], hsv[..., 1][sel], hsv[..., 2][sel], v_black)
        lab = lab[lab >= 0]
        p = np.bincount(lab, minlength=len(COLORS)).astype(np.float64) * vw + 0.5
    elif pix == "lab":
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)[sel].astype(np.float32)
        L, A, B = lab[:, 0] * 100 / 255, lab[:, 1] - 128, lab[:, 2] - 128
        keep = L <= glare_l                                             # 빛번짐 (아주 밝음) 화소 뺌
        if keep.sum() < min_px:
            return None
        k = classify_lab(L[keep], A[keep], B[keep])
        k = k[k >= 0]
        p = np.bincount(k, minlength=len(COLORS)).astype(np.float64) * vw + 0.5
    else:                                                               # w2c — 화소마다 11색 확률 (남색 = 어두운 파랑)
        px = crop[sel].astype(np.int32)                                 # BGR
        idx = px[:, 2] // 8 + 32 * (px[:, 1] // 8) + 1024 * (px[:, 0] // 8)
        P = w2c_table()[idx].copy()
        dark = hsv[..., 2][sel] < navy_v * 255
        P[dark, CI["navy"]] = P[dark, CI["blue"]]; P[dark, CI["blue"]] = 0
        p = P.sum(0).astype(np.float64) * vw + 0.5
    return p / p.sum()


# ───────────────────────── SigLIP 2 ─────────────────────────
SIGLIP = "google/siglip2-base-patch16-224"
_SL = {}
PROMPT = {"red": "red", "orange": "orange", "yellow": "yellow", "green": "green", "blue": "blue", "navy": "navy blue", "purple": "purple",
          "pink": "pink", "white": "white", "gray": "gray", "black": "black", "brown": "brown"}
SYN = {"red": ["red", "maroon", "burgundy", "dark red"], "orange": ["orange"], "yellow": ["yellow", "mustard"],                # C2 동의어
       "green": ["green", "olive green", "mint green", "khaki green"], "blue": ["blue", "light blue", "sky blue", "denim blue"],
       "navy": ["navy blue", "dark blue"], "purple": ["purple", "violet", "lavender"], "pink": ["pink", "light pink", "hot pink"],
       "white": ["white", "off-white"], "gray": ["gray", "light gray", "charcoal gray"], "black": ["black"], "brown": ["brown", "beige", "tan", "camel"]}
TMPL1 = ["a photo of a person wearing a {} shirt."]
TMPLS = ["a photo of a person wearing a {} shirt.", "a photo of a person wearing a {} top.", "a photo of a person in a {} jacket.",     # C1 문장 묶음
         "an aerial photo of a person wearing {} clothes.", "a low resolution photo of a person wearing a {} hoodie.", "a person dressed in {}."]


def siglip(name=SIGLIP):
    if name not in _SL:
        import torch
        from transformers import AutoModel, AutoProcessor
        _SL[name] = (AutoProcessor.from_pretrained(name), AutoModel.from_pretrained(name, dtype=torch.float16).cuda().eval())
    return _SL[name]


def person_rgb(it, part="person"):
    """part: person (박스) · upper (위 10~62 %) · upper_mask (사람 마스크 밖 회색 · C5) · upper_wb (프레임 무채색 화이트밸런스 + 노출 맞춤 · C6)"""
    fr, (x1, y1, x2, y2) = it["frame"], it["box"]
    if part == "bg":                                                     # E3 배경 — 박스 둘레 (가로 · 세로 1배씩) · 사람 칸은 회색
        bw, bh = x2 - x1, y2 - y1
        X1, Y1, X2, Y2 = int(max(0, x1 - bw)), int(max(0, y1 - bh)), int(min(fr.shape[1], x2 + bw)), int(min(fr.shape[0], y2 + bh))
        c = fr[Y1:Y2, X1:X2].copy(); c[int(y1 - Y1):int(y2 - Y1), int(x1 - X1):int(x2 - X1)] = 128
        return cv2.cvtColor(c, cv2.COLOR_BGR2RGB)
    if part == "upper_widemask":                                         # G3 — 가로 ≥ 세로 박스 (누운 · 기어가는 사람) 만 박스 전체 · GrabCut 바깥 회색
        if (x2 - x1) >= (y2 - y1):
            cv2.setRNGSeed(0)
            crop, rect = _crop(it)
            m = _person_mask(crop, rect)
            crop = crop.copy(); crop[m == 0] = 128
            x, y, w, h = rect
            c = crop[y:y + h, x:x + w]
            return cv2.cvtColor(c, cv2.COLOR_BGR2RGB) if c.size else np.zeros((8, 8, 3), np.uint8)
        part = "upper_nopose"
    if part == "upper_gtpose":                                           # E1 — 정답 자세 (누움 · 기어감 → 박스 전체)
        it = dict(it, pose=it.get("gt_pose")); part = "upper"
    if part == "upper_nopose":                                           # E0 — worker 지금 (자세 안 넘김)
        it = dict(it, pose=None); part = "upper"
    x1, y1, x2, y2 = int(max(0, x1)), int(max(0, y1)), int(min(fr.shape[1], x2)), int(min(fr.shape[0], y2))
    if part == "upper_mask":
        cv2.setRNGSeed(0)
        crop, rect = _crop(it)
        m = _person_mask(crop, rect)
        crop = crop.copy(); crop[m == 0] = 128
        x, y, w, h = rect
        c = crop[y + int(0.10 * h):y + int(0.62 * h), x:x + w] if it["pose"] != "lying" else crop[y:y + h, x:x + w]
        return cv2.cvtColor(c, cv2.COLOR_BGR2RGB) if c.size else np.zeros((8, 8, 3), np.uint8)
    if part in ("upper", "upper_wb") and it["pose"] != "lying":
        h = y2 - y1; y1, y2 = y1 + int(0.10 * h), y1 + int(0.62 * h)
    c = fr[y1:y2, x1:x2]
    if c.size and part == "upper_wb":
        st = it.get("stats") or frame_stats(fr)
        c = np.clip(c.astype(np.float32) * st["achromatic"], 0, 255)
        c = np.clip(c * float(np.clip(110 / max(st["v_median"], 1.0), 0.7, 2.0)), 0, 255).astype(np.uint8)
    return cv2.cvtColor(c, cv2.COLOR_BGR2RGB) if c.size else np.zeros((8, 8, 3), np.uint8)


_FEAT = {}


def siglip_feats(items, part="person", bs=64, model=SIGLIP):
    import torch
    key = (id(items), len(items), part, model)
    if key in _FEAT:
        return _FEAT[key]
    proc, m = siglip(model)
    out = []
    for i in range(0, len(items), bs):
        ims = [person_rgb(it, part) for it in items[i:i + bs]]
        inp = proc(images=ims, return_tensors="pt")
        with torch.no_grad():
            f = m.get_image_features(pixel_values=inp["pixel_values"].cuda().half())
        f = getattr(f, "pooler_output", f)
        out.append(torch.nn.functional.normalize(f.float(), dim=-1).cpu().numpy())
    _FEAT[key] = np.concatenate(out)
    return _FEAT[key]


def siglip_text(tmpls, syn=False, model=SIGLIP):
    """색마다 (문장 × 이름) 임베딩 → 문장끼리 평균 · 이름끼리는 따로 (점수 때 max) → [색][이름] 행렬 목록"""
    import torch
    proc, m = siglip(model)
    names = {c: (SYN[c] if syn else [PROMPT[c]]) for c in COLORS}
    T = {}
    for c in COLORS:
        rows = []
        for nm in names[c]:
            inp = proc(text=[t.format(nm) for t in tmpls], padding="max_length", max_length=64, return_tensors="pt")
            with torch.no_grad():
                t = m.get_text_features(input_ids=inp["input_ids"].cuda())
            t = torch.nn.functional.normalize(getattr(t, "pooler_output", t).float(), dim=-1).mean(0)
            rows.append(torch.nn.functional.normalize(t, dim=0).cpu().numpy())
        T[c] = np.stack(rows)
    return T, float(m.logit_scale.exp()), float(m.logit_bias)


def zs_logits(feats, tmpls=TMPL1, syn=False, model=SIGLIP):
    T, sc, b = siglip_text(tmpls, syn, model)
    return np.stack([(feats @ T[c].T).max(1) for c in COLORS], 1) * sc + b


def _softmax(lg):
    e = np.exp(lg - lg.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def navy_rule(lg, delta):
    """F1 — 1순위가 남색인데 (남색 − 검정) ≤ delta 이면 남색 · 검정 로짓을 맞바꿔 검정으로"""
    lg = lg.copy(); n, b = CI["navy"], CI["black"]
    sw = (lg.argmax(1) == n) & (lg[:, n] - lg[:, b] <= delta)
    lg[sw, n], lg[sw, b] = lg[sw, b], lg[sw, n]
    return lg


def navy_delta(part="upper", model=SIGLIP, target=0.05):
    """δ = UAV-Human 학습셋 검정 사진의 남색 판정률 ≤ target 이 되는 가장 작은 값 (확인 셋 안 봄)"""
    X, y, _ = train_feats(part, model)
    lg = zs_logits(X[y == CI["black"]], TMPL1, False, model)
    for d in np.arange(0, 10.01, 0.05):
        r = float(np.mean(navy_rule(lg, d).argmax(1) == CI["navy"]))
        if r <= target:
            print(f"  남색 문턱 δ={d:.2f} (학습셋 검정 {len(lg)}장 · 남색 판정 {np.mean(lg.argmax(1) == CI['navy']):.3f} → {r:.3f})", flush=True)
            return float(d)
    return 10.0


def grass_text(model=SIGLIP):
    import torch
    proc, m = siglip(model)
    inp = proc(text=["a photo of green grass."], padding="max_length", max_length=64, return_tensors="pt")
    with torch.no_grad():
        t = m.get_text_features(input_ids=inp["input_ids"].cuda())
    return torch.nn.functional.normalize(getattr(t, "pooler_output", t).float(), dim=-1).cpu().numpy()[0]


def zero_shot(items, part="person", tmpl=None, tmpls=None, syn=False, model=SIGLIP, bias=False, navy=False, bg=0.0, gbg=0.0, grass=0.0):
    tm = tmpls or ([tmpl] if tmpl else TMPL1)
    lg = zs_logits(siglip_feats(items, part, model=model), tm, syn, model)
    if bg:                                                              # E3 — 배경 로짓 (평균 뺀 것) × α 를 뺌
        lb = zs_logits(siglip_feats(items, "bg", model=model), tm, syn, model)
        lg = lg - bg * (lb - lb.mean(1, keepdims=True))
    G = CI["green"]
    if gbg:                                                             # G1 — 초록만 배경 빼기
        lb = zs_logits(siglip_feats(items, "bg", model=model), tm, syn, model)
        lg[:, G] = lg[:, G] - gbg * (lb[:, G] - lb.mean(1))
    if grass:                                                           # G2 — 잔디 문장 로짓 − 초록 셔츠 로짓 (같은 눈금) 만큼 초록 깎기
        f = siglip_feats(items, part, model=model); _, sc, b = siglip_text(tm, syn, model)
        g = f @ grass_text(model) * sc + b - lg[:, G]
        lg[:, G] = lg[:, G] - grass * np.maximum(0, g)
    if bias:
        lg = lg + zs_bias(part, tuple(tm), syn, model)
    if navy:
        lg = navy_rule(lg, navy_delta(part, model))
    return list(_softmax(lg))


_BIAS = {}


def zs_bias(part, tmpls, syn, model):
    """C4 — 제로샷 로짓에 색마다 편향 12개만 더함 · UAV-Human 학습셋 (89명) 으로 맞춤 (색 균형 교차 엔트로피 · 학습셋에 없는 주황 · 남색 = 0)"""
    key = (part, tmpls, syn, model)
    if key in _BIAS:
        return _BIAS[key]
    import torch
    X, y, g = train_feats(part, model)
    lg = torch.tensor(zs_logits(X, list(tmpls), syn, model), dtype=torch.float32)
    yt = torch.tensor(y)
    w = torch.zeros(len(COLORS)); cnt = torch.bincount(yt, minlength=len(COLORS)).float()
    w[cnt > 0] = 1 / cnt[cnt > 0]
    b = torch.zeros(len(COLORS), requires_grad=True)
    opt = torch.optim.LBFGS([b], max_iter=200)
    mask = (cnt > 0).float()

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(lg + b * mask, yt, weight=w) + 1e-3 * (b ** 2).sum()
        loss.backward(); return loss
    opt.step(closure)
    _BIAS[key] = (b * mask).detach().numpy()
    print("  편향", {c: round(float(v), 2) for c, v in zip(COLORS, _BIAS[key]) if v}, flush=True)
    return _BIAS[key]


# ───────────────────────── B2 — SigLIP 2 특징 + 선형 (UAV-Human 학습셋만) ─────────────────────────
_PROBE = {}


def train_feats(part, model=SIGLIP, per_class=600, seed=0, aug=False):
    """학습셋 89명 · 색마다 ≤ per_class 장 (사람마다 고르게) · 사람 키 40~140 px 무작위로 줄임 (운용 크기)
    aug (C9): 장마다 2벌 더 — 채널별 이득 0.85~1.15 · 밝기 0.7~1.3 (카메라 · 조명 흔들기)"""
    tag = "" if model == SIGLIP else "_" + model.split("/")[-1]
    fp = LAB / f"train_{part}{tag}{'_aug' if aug else ''}.npz"
    if fp.exists():
        z = np.load(fp, allow_pickle=True)
        return z["X"], z["y"], z["g"]
    rng = random.Random(seed); nrng = np.random.default_rng(seed); items = []
    for f, c, pid in uavh_list("train", per_class, seed):
        img = cv2.imread(str(f))
        for k in range(3 if aug else 1):
            im = img
            if k:
                im = np.clip(img.astype(np.float32) * nrng.uniform(0.85, 1.15, 3) * nrng.uniform(0.7, 1.3), 0, 255).astype(np.uint8)
            fr, box = _frame_from_crop(im, rng.randint(40, 140))
            items.append({"frame": fr, "box": box, "pose": None, "primary": c, "person": pid, "stats": frame_stats(fr)})
    X = siglip_feats(items, part, model=model); y = np.array([CI[it["primary"]] for it in items]); g = np.array([it["person"] for it in items])
    np.savez(fp, X=X, y=y, g=g)
    return X, y, g


def probe_train(part, model=SIGLIP, aug=False):
    """C 는 학습셋 안 사람 5겹 교차검증 (색 균형 정확도) 으로 고름 — 시험셋 U 는 안 봄"""
    key = (part, model, aug)
    if key in _PROBE:
        return _PROBE[key]
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    X, y, g = train_feats(part, model, aug=aug)
    best = None
    for C in (0.1, 0.3, 1, 3, 10, 30):
        accs = []
        for tr, va in GroupKFold(5).split(X, y, g):
            m = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(X[tr], y[tr])
            pr = m.predict(X[va]); accs.append(np.mean([np.mean(pr[y[va] == k] == k) for k in np.unique(y[va])]))
        a = float(np.mean(accs))
        if best is None or a > best[0]:
            best = (a, C)
    m = LogisticRegression(C=best[1], max_iter=3000, class_weight="balanced").fit(X, y)
    _PROBE[key] = (m, best)
    print(f"  probe {key} 고른 C={best[1]} (교차검증 {best[0]:.3f} · 학습 {len(y)}장 · {len(set(g))}명)", flush=True)
    return _PROBE[key]


def probe(items, part="person", model=SIGLIP, aug=False):
    m, _ = probe_train(part, model, aug)
    P = m.predict_proba(siglip_feats(items, part, model=model))
    out = np.zeros((len(items), len(COLORS)))
    out[:, m.classes_] = P
    return list(out)


# ───────────────────────── 확인 셋 (10-29 2차 · 고르는 데 안 씀) ─────────────────────────
PETA = RAW / "color" / "PETA" / "PETA dataset"
PETA_C = {"Black": "black", "Blue": "blue", "Brown": "brown", "Green": "green", "Grey": "gray", "Orange": "orange", "Pink": "pink",
          "Purple": "purple", "Red": "red", "White": "white", "Yellow": "yellow"}


def build_confirm(per_class=150, seed=0):
    """P = PETA 상의 색이 하나인 사람 · 색마다 ≤ per_class (사람마다 고르게) · 키 72 px / D = 디관 사진 배포 탐지 박스 + 잠정 라벨"""
    by = defaultdict(lambda: defaultdict(list))
    for sub in sorted(PETA.iterdir()):
        lab = {}
        for line in open(sub / "archive" / "Label.txt"):
            t = line.split()
            ups = [PETA_C[a[9:]] for a in t[1:] if a.startswith("upperBody") and a[9:] in PETA_C]
            if len(ups) == 1:
                lab[t[0]] = ups[0]
        for f in sorted((sub / "archive").iterdir()):
            if f.name == "Label.txt":
                continue
            pid = f.name if f.name in lab else f.name.split("_")[0]
            if pid in lab:
                by[lab[pid]][f"{sub.name}:{pid}"].append(f)
    rng = random.Random(seed); P = []
    for c in sorted(by):
        persons = {p: rng.sample(v, len(v)) for p, v in sorted(by[c].items())}
        order = list(persons); rng.shuffle(order); n = 0
        while n < per_class and any(persons.values()):
            for p in order:
                if persons[p] and n < per_class:
                    img = cv2.imread(str(persons[p].pop()))
                    if img is None:
                        continue
                    fr, box = _frame_from_crop(img, 72)
                    P.append({"frame": fr, "box": box, "stats": frame_stats(fr), "pose": None, "truth": {c}, "primary": c, "person": p, "h": 72}); n += 1
    pickle.dump(P, open(LAB / "P.pkl", "wb"))
    print("P", len(P), Counter(x["primary"] for x in P), "사람", len({x["person"] for x in P}))
    from make_present_dd import fhd, SRC
    res = json.load(open(BASE.parent / "발표자료" / "_도구" / "dd_work" / "results.json"))["images"]
    labs = json.load(open(LAB / "dgwan_labels.json"))["labels"]
    D, frames = [], {}
    for L in labs:
        if not L["truth"]:
            continue
        r = res[L["img"]]
        if L["img"] not in frames:
            frames = {L["img"]: fhd(cv2.imread(str(SRC / r["file"])))}
        fr = frames[L["img"]]; st = frame_stats(fr)
        x1, y1, x2, y2 = r["dets"][L["det"]]["box"]; w, h = x2 - x1, y2 - y1; cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        X1, Y1 = int(max(0, cx - 0.75 * w)), int(max(0, cy - 0.75 * h)); X2, Y2 = int(min(1920, cx + 0.75 * w)), int(min(1080, cy + 0.75 * h))
        D.append({"frame": fr[Y1:Y2, X1:X2].copy(), "box": (x1 - X1, y1 - Y1, x2 - X1, y2 - Y1), "stats": st, "pose": None,
                  "truth": set(L["truth"]), "primary": L["truth"][0], "person": f"img{L['img']}", "h": round(h, 1)})
    pickle.dump(D, open(LAB / "D.pkl", "wb"))
    print("D", len(D), Counter(x["primary"] for x in D), "사진 묶음", len({x["person"] for x in D}))


def build_ue(every=4):
    """E = UE S1~S4 (16 · 20 m) 정답 박스 · 잘림 없음 · 키 ≥ 20 px · every 장마다 · 크롭 = 박스 3배 (배경 빼기용 둘레 포함)"""
    E = []
    for run in sorted((BASE / "ue_flight").glob("S[1-4]_a[12][06]")):
        A = json.load(open(run / "actors.json"))
        for k, l in enumerate(open(run / "boxes.jsonl")):
            r = json.loads(l)
            if not r["boxes"] or k % every:
                continue
            fr = None
            for a, b in r["boxes"].items():
                x1, y1, x2, y2 = b["bb"]
                if b.get("cut") or y2 - y1 < 20 or a not in A:
                    continue
                if fr is None:
                    fr = cv2.imread(str(run / "frames" / f"{r['frame']}.jpg")); st = frame_stats(fr)
                w, h = x2 - x1, y2 - y1
                X1, Y1, X2, Y2 = int(max(0, x1 - w)), int(max(0, y1 - h)), int(min(fr.shape[1], x2 + w)), int(min(fr.shape[0], y2 + h))
                c = A[a]["upper_color"]
                E.append({"frame": fr[Y1:Y2, X1:X2].copy(), "box": (x1 - X1, y1 - Y1, x2 - X1, y2 - Y1), "stats": st, "pose": None,
                          "gt_pose": "lying" if A[a]["pose_raw"] in ("lying", "crawling") else None, "truth": {c}, "primary": c,
                          "person": f"{run.name}:{a}", "run": run.name, "h": round(h, 1)})
    pickle.dump(E, open(LAB / "E.pkl", "wb"))
    print("E", len(E), Counter(e["primary"] for e in E), Counter(e["run"] for e in E))


def ue_query(items, preds):
    """질의 — 배우마다 자기 색으로 질의할 때 같은 장면 (run) 다른 색 배우 관측의 「일치」 비율 (오일치) · 자기 색 배우의 일치 (MATCH)"""
    by_run = defaultdict(set)
    for it in items:
        by_run[it["run"]].add(it["primary"])
    hit, fa = [], []
    for it, p in zip(items, preds):
        r = {"status": "ok", "top": [(COLORS[i], float(p[i])) for i in np.argsort(-p)[:2]], "probs": {c: float(p[i]) for i, c in enumerate(COLORS)}, "n_obs": 1} if p is not None else {"status": "undetermined"}
        for q in by_run[it["run"]]:
            m = match(r, q)[0] == "일치"
            (hit if q == it["primary"] else fa).append(m)
    return round(float(np.mean(hit)), 3), round(float(np.mean(fa)), 3)


# ───────────────────────── 지표 ─────────────────────────
def metrics(items, preds, set_name):
    ok = [p is not None for p in preds]
    top1 = np.array([p is not None and COLORS[int(np.argmax(p))] in it["truth"] for it, p in zip(items, preds)])
    top2 = np.array([p is not None and bool({COLORS[i] for i in np.argsort(-p)[:2]} & it["truth"]) for it, p in zip(items, preds)])
    prim = np.array([it["primary"] for it in items])
    cls = sorted(set(prim))
    o = {"장": len(items), "판정불가율": round(1 - float(np.mean(ok)), 3), "1순위": round(float(top1.mean()), 3), "1·2순위": round(float(top2.mean()), 3),
         "판정된 것 1순위": round(float(top1[np.array(ok)].mean()) if any(ok) else 0, 3),
         "색 균형 1순위": round(float(np.mean([top1[prim == c].mean() for c in cls])), 3),
         "색 균형 1·2순위": round(float(np.mean([top2[prim == c].mean() for c in cls])), 3),
         "색별 1순위": {c: round(float(top1[prim == c].mean()), 3) for c in cls}}
    conf = defaultdict(Counter)
    for it, p in zip(items, preds):
        conf[it["primary"]][COLORS[int(np.argmax(p))] if p is not None else "불가"] += 1
    o["혼동"] = {t: dict(c.most_common(4)) for t, c in sorted(conf.items())}
    if set_name == "N":                                                 # 누적 · 찾는 사람 순위 (eval_color.py 와 같은 규칙)
        acc, actor_acc = defaultdict(ColorAccumulator), defaultdict(ColorAccumulator)
        for it, p in zip(items, preds):
            pr = {"status": "ok", "probs": {c: float(p[i]) for i, c in enumerate(COLORS)}, "n_pixels": 100} if p is not None else {"status": "undetermined"}
            acc[it["group"]].add(pr); actor_acc[it["person"]].add(pr)
        g = [A.result() for A in acc.values()]
        tr = {it["group"]: it["truth"] for it in items}
        o["누적 1순위"] = round(float(np.mean([r["status"] == "ok" and r["top"][0][0] in tr[k] for k, r in zip(acc.keys(), g)])), 3)
        prim_a = {it["person"]: it["primary"] for it in items}
        R = {a: A.result() for a, A in actor_acc.items()}
        ranks = []
        for a, r in R.items():
            sc = {b: match(R[b], prim_a[a])[1] for b in R}
            ranks.append(1 + sum(1 for b in R if b != a and sc[b] > sc[a]))
        o["찾는 사람 순위 중앙"] = float(np.median(ranks)); o["찾는 사람 3위 안"] = round(float(np.mean(np.array(ranks) <= 3)), 3)
    return o, top1


def boot(items, a, b, set_name, n=2000, seed=0):
    """짝 차이 (b − a) 사람 단위 블록 부트스트랩 95 % — U 는 색 균형 1순위 · N 은 한 장 1순위"""
    persons = sorted({it["person"] for it in items})
    by = defaultdict(list)
    for i, it in enumerate(items):
        by[it["person"]].append(i)
    prim = np.array([it["primary"] for it in items])
    rng = np.random.default_rng(seed); d = []

    def stat(idx, t):
        if set_name == "U":
            cls = sorted(set(prim[idx]))
            return np.mean([t[idx][prim[idx] == c].mean() for c in cls])
        return t[idx].mean()
    for _ in range(n):
        idx = np.concatenate([by[persons[j]] for j in rng.integers(0, len(persons), len(persons))])
        d.append(stat(idx, b) - stat(idx, a))
    return [round(float(np.percentile(d, 2.5)), 3), round(float(np.percentile(d, 97.5)), 3)]


# ───────────────────────── 방식 목록 ─────────────────────────
METHODS = {
    "V10": dict(kind="rule"),                                                      # 지금 worker
    "V10-노출": dict(kind="rule", exp=None),
    "V10-화이트밸런스": dict(kind="rule", wb=None),
    "V10+밝히기만": dict(kind="rule", exp=(1.0, 2.0)),
    "A1 w2c": dict(kind="rule", pix="w2c"),
    "A1 w2c 표가중없음": dict(kind="rule", pix="w2c", w_black=1.0, w_gray=1.0),
    "A1 w2c+밝히기만": dict(kind="rule", pix="w2c", exp=(1.0, 2.0)),
    "A1 w2c-노출": dict(kind="rule", pix="w2c", exp=None),
    "A2 국소WB": dict(kind="rule", wb="local"),
    "A2 국소WB+w2c": dict(kind="rule", wb="local", pix="w2c"),
    "A3 yoloseg": dict(kind="rule", mask="yoloseg"),
    "A3 yoloseg+w2c": dict(kind="rule", mask="yoloseg", pix="w2c"),
    "A3 segformer": dict(kind="rule", mask="segformer"),
    "A3 segformer+w2c": dict(kind="rule", mask="segformer", pix="w2c"),
    "B1 SigLIP 사람": dict(kind="zs", part="person"),
    "B1 SigLIP 상의": dict(kind="zs", part="upper"),
    "B1 SigLIP 사람 문장2": dict(kind="zs", part="person", tmpl="an aerial photo of a person in a {} top."),
    "L1 CLAHE(L)+HSV": dict(kind="rule", pre="clahe", exp=None),
    "L2 Lab규칙": dict(kind="rule", pix="lab", exp=None),
    "L3 CLAHE(L)+Lab규칙": dict(kind="rule", pre="clahe", pix="lab", exp=None),
    "L4 Lab규칙 표가중없음": dict(kind="rule", pix="lab", exp=None, w_black=1.0, w_gray=1.0),
    "B2 probe 사람": dict(kind="probe", part="person"),
    "B2 probe 상의": dict(kind="probe", part="upper"),
    # ── C 계열 (10-29 2차 · U · N 으로만 고름) ──
    "C1 문장묶음 상의": dict(kind="zs", part="upper", tmpls=TMPLS),
    "C2 동의어 상의": dict(kind="zs", part="upper", syn=True),
    "C1+C2 상의": dict(kind="zs", part="upper", tmpls=TMPLS, syn=True),
    "C4 편향 상의": dict(kind="zs", part="upper", bias=True),
    "C5 배경회색 상의": dict(kind="zs", part="upper_mask"),
    "C6 색보정 상의": dict(kind="zs", part="upper_wb"),
    "C7 large 상의": dict(kind="zs", part="upper", model="google/siglip2-large-patch16-256"),
    "C7 so400m 상의": dict(kind="zs", part="upper", model="google/siglip2-so400m-patch16-256"),
    "C3 B1×V10 0.7": dict(kind="fuse", a="B1 SigLIP 상의", b="V10", w=0.7),
    "C3 B1×L2 0.7": dict(kind="fuse", a="B1 SigLIP 상의", b="L2 Lab규칙", w=0.7),
    "C3 B1×L2 0.5": dict(kind="fuse", a="B1 SigLIP 상의", b="L2 Lab규칙", w=0.5),
    "C8 B1+probe 0.5": dict(kind="fuse", a="B1 SigLIP 상의", b="B2 probe 상의", w=0.5, mode="lin"),
    "C9 probe 증강 상의": dict(kind="probe", part="upper", aug=True),
    "C4×V10 0.7": dict(kind="fuse", a="C4 편향 상의", b="V10", w=0.7),
    "C8×V10 0.7": dict(kind="fuse", a="C8 B1+probe 0.5", b="V10", w=0.7),
    "C4+probe 0.5": dict(kind="fuse", a="C4 편향 상의", b="B2 probe 상의", w=0.5, mode="lin"),
    "C7 large 편향 상의": dict(kind="zs", part="upper", model="google/siglip2-large-patch16-256", bias=True),
    "C7 so400m 편향 상의": dict(kind="zs", part="upper", model="google/siglip2-so400m-patch16-256", bias=True),
    "F1 B1+남색문턱": dict(kind="zs", part="upper", navy=True),
    # ── 4차 (UE · 배포 경로 문제 보완) ──
    "E0 F1 자세없음": dict(kind="zs", part="upper_nopose", navy=True),
    "E1 F1 정답자세": dict(kind="zs", part="upper_gtpose", navy=True),
    "E2 F1 사람박스": dict(kind="zs", part="person", navy=True),
    "E3 F1 자세없음+배경0.5": dict(kind="zs", part="upper_nopose", navy=True, bg=0.5),
    "E3 F1 자세없음+배경1.0": dict(kind="zs", part="upper_nopose", navy=True, bg=1.0),
    "E3 F1 정답자세+배경0.5": dict(kind="zs", part="upper_gtpose", navy=True, bg=0.5),
    "E3 F1 정답자세+배경1.0": dict(kind="zs", part="upper_gtpose", navy=True, bg=1.0),
    # ── 5차 (풀밭 초록 질의) · 기준 = E0 (= 배포 1.3) ──
    "G1 초록배경 0.5": dict(kind="zs", part="upper_nopose", navy=True, gbg=0.5),
    "G1 초록배경 1.0": dict(kind="zs", part="upper_nopose", navy=True, gbg=1.0),
    "G1 초록배경 2.0": dict(kind="zs", part="upper_nopose", navy=True, gbg=2.0),
    "G2 잔디문장 0.5": dict(kind="zs", part="upper_nopose", navy=True, grass=0.5),
    "G2 잔디문장 1.0": dict(kind="zs", part="upper_nopose", navy=True, grass=1.0),
    "G3 넓은박스 마스크": dict(kind="zs", part="upper_widemask", navy=True),
}


def get_preds(name, items, set_name):
    pp = LAB / "preds" / f"{name.replace(' ', '_')}_{set_name}.pkl"
    if pp.exists():
        return pickle.load(open(pp, "rb"))
    out = predict(name, items, set_name)
    pickle.dump(out, open(pp, "wb"))
    return out


def predict(name, items, set_name):
    m = dict(METHODS[name]); kind = m.pop("kind")
    if kind == "probe":
        probe_train(m["part"], m.get("model", SIGLIP), m.get("aug", False))   # 학습은 시간 재기 밖
    if kind == "zs" and m.get("bias"):
        zs_bias(m["part"], tuple(m.get("tmpls") or TMPL1), m.get("syn", False), m.get("model", SIGLIP))
    t0 = time.time()
    if kind == "rule":
        preds = [rule(it, **m) for it in items]
    elif kind == "zs":
        preds = zero_shot(items, **m)
    elif kind == "probe":
        preds = probe(items, **m)
    elif kind == "fuse":                                                # C3 · C8 — 두 방식 확률 섞기 (geo = 가중 기하 평균 · lin = 가중 산술 평균)
        pa, ma = get_preds(m["a"], items, set_name); pb, mb = get_preds(m["b"], items, set_name)
        preds = []
        for x, y in zip(pa, pb):
            if y is None:
                preds.append(x); continue
            if m.get("mode", "geo") == "geo":
                z = np.exp(m["w"] * np.log(np.asarray(x) + 1e-6) + (1 - m["w"]) * np.log(np.asarray(y) + 1e-6))
            else:
                z = m["w"] * np.asarray(x) + (1 - m["w"]) * np.asarray(y)
            preds.append(z / z.sum())
        return preds, round(ma + mb, 2)
    else:
        raise ValueError(kind)
    return preds, round((time.time() - t0) * 1000 / len(items), 2)


def run(names, set_names=("U", "N")):
    sets = {s: pickle.load(open(LAB / f"{s}.pkl", "rb")) for s in set_names}
    out_p = BASE / "metrics" / ("color_lab.json" if set_names == ("U", "N") else "color_lab_confirm.json")
    res = json.load(open(out_p)) if out_p.exists() else {}
    (LAB / "preds").mkdir(exist_ok=True)
    base_t = {}
    for s, items in sets.items():
        base_t[s] = metrics(items, get_preds("V10", items, s)[0], "N" if s == "N" else "U")[1]
    for name in names:
        res.setdefault(name, {})
        for s, items in sets.items():
            preds, ms = get_preds(name, items, s)
            if s == "U":
                for h in HEIGHTS:
                    sub = [i for i, it in enumerate(items) if it["h"] == h]
                    o, t = metrics([items[i] for i in sub], [preds[i] for i in sub], "U")
                    o["V10 대비 (짝 · 95 %)"] = boot([items[i] for i in sub], base_t["U"][sub], t, "U")
                    o["ms/장"] = ms
                    res[name][f"U{h}"] = o
            else:
                o, t = metrics(items, preds, "N" if s == "N" else "U")
                o["V10 대비 (짝 · 95 %)"] = boot(items, base_t[s], t, "N" if s == "N" else "U"); o["ms/장"] = ms
                res[name][s] = o
        if set_names == ("U", "N"):
            u, n = res[name]["U72"], res[name]["N"]
            print(f"{name:26s} U72 균형 {u['색 균형 1순위']:.3f} {u['V10 대비 (짝 · 95 %)']} | U48 {res[name]['U48']['색 균형 1순위']:.3f} · U120 {res[name]['U120']['색 균형 1순위']:.3f}"
                  f" | N 1순위 {n['1순위']:.3f} {n['V10 대비 (짝 · 95 %)']} · 누적 {n['누적 1순위']:.3f} · 순위 {n['찾는 사람 순위 중앙']} | 평균 {(u['색 균형 1순위'] + n['1순위']) / 2:.3f} | {u['ms/장']} ms", flush=True)
        else:
            print(f"{name:26s} " + " | ".join(f"{s} 균형 {res[name][s]['색 균형 1순위']:.3f} {res[name][s]['V10 대비 (짝 · 95 %)']} · 1순위 {res[name][s]['1순위']:.3f}" for s in sets), flush=True)
        out_p.write_text(json.dumps(res, ensure_ascii=False, indent=1))


def run_ue(names):
    E = pickle.load(open(LAB / "E.pkl", "rb"))
    out_p = BASE / "metrics" / "color_lab_ue.json"
    res = json.load(open(out_p)) if out_p.exists() else {}
    for name in names:
        preds, ms = get_preds(name, E, "E")
        res[name] = {}
        for tag in ("a16", "a20"):
            idx = [i for i, it in enumerate(E) if it["run"].endswith(tag)]
            its, ps = [E[i] for i in idx], [preds[i] for i in idx]
            o, _ = metrics(its, ps, "U")
            o["질의 MATCH"], o["질의 오일치"] = ue_query(its, ps)
            res[name][tag] = o
        a, b = res[name]["a16"], res[name]["a20"]
        print(f"{name:26s} a16 균형 {a['색 균형 1순위']:.3f} · 오일치 {a['질의 오일치']:.3f} · MATCH {a['질의 MATCH']:.3f} | a20 균형 {b['색 균형 1순위']:.3f} · 오일치 {b['질의 오일치']:.3f} · MATCH {b['질의 MATCH']:.3f}", flush=True)
        out_p.write_text(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build()
    elif sys.argv[1] == "build_confirm":
        build_confirm()
    elif sys.argv[1] == "run":
        run(list(METHODS) if sys.argv[2:] == ["all"] else sys.argv[2:])
    elif sys.argv[1] == "confirm":                                      # 확인 셋 P · D (한 번만)
        run(sys.argv[2:], ("P", "D"))
    elif sys.argv[1] == "build_ue":
        build_ue()
    elif sys.argv[1] == "ue":                                           # 4차 — UE a16 (고르기) · a20 (판정)
        run_ue(sys.argv[2:])
    elif sys.argv[1] == "confirm_v":                                    # 3차 확인 셋 V (디관 영상 · 한 번만)
        run(sys.argv[2:], ("V",))
