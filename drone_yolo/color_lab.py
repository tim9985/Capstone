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
_SL = None
PROMPT = {"red": "red", "orange": "orange", "yellow": "yellow", "green": "green", "blue": "blue", "navy": "navy blue", "purple": "purple",
          "pink": "pink", "white": "white", "gray": "gray", "black": "black", "brown": "brown"}


def siglip():
    global _SL
    if _SL is None:
        import torch
        from transformers import AutoModel, AutoProcessor
        _SL = (AutoProcessor.from_pretrained(SIGLIP), AutoModel.from_pretrained(SIGLIP, torch_dtype=torch.float16).cuda().eval())
    return _SL


def person_rgb(it, part="person"):
    fr, (x1, y1, x2, y2) = it["frame"], it["box"]
    x1, y1, x2, y2 = int(max(0, x1)), int(max(0, y1)), int(min(fr.shape[1], x2)), int(min(fr.shape[0], y2))
    if part == "upper" and it["pose"] != "lying":
        h = y2 - y1; y1, y2 = y1 + int(0.10 * h), y1 + int(0.62 * h)
    c = fr[y1:y2, x1:x2]
    return cv2.cvtColor(c, cv2.COLOR_BGR2RGB) if c.size else np.zeros((8, 8, 3), np.uint8)


def siglip_feats(items, part="person", bs=64):
    import torch
    proc, model = siglip()
    out = []
    for i in range(0, len(items), bs):
        ims = [person_rgb(it, part) for it in items[i:i + bs]]
        inp = proc(images=ims, return_tensors="pt")
        with torch.no_grad():
            f = model.get_image_features(pixel_values=inp["pixel_values"].cuda().half())
        f = getattr(f, "pooler_output", f)
        out.append(torch.nn.functional.normalize(f.float(), dim=-1).cpu().numpy())
    return np.concatenate(out)


def siglip_text(tmpl):
    import torch
    proc, model = siglip()
    inp = proc(text=[tmpl.format(PROMPT[c]) for c in COLORS], padding="max_length", max_length=64, return_tensors="pt")
    with torch.no_grad():
        t = model.get_text_features(input_ids=inp["input_ids"].cuda())
    t = getattr(t, "pooler_output", t)
    return torch.nn.functional.normalize(t.float(), dim=-1).cpu().numpy(), float(model.logit_scale.exp()), float(model.logit_bias)


def zero_shot(items, part="person", tmpl="a photo of a person wearing a {} shirt."):
    f = siglip_feats(items, part)
    t, sc, b = siglip_text(tmpl)
    lg = f @ t.T * sc + b
    e = np.exp(lg - lg.max(1, keepdims=True))
    return list(e / e.sum(1, keepdims=True))


# ───────────────────────── B2 — SigLIP 2 특징 + 선형 (UAV-Human 학습셋만) ─────────────────────────
_PROBE = {}


def probe_train(part, per_class=600, seed=0):
    """학습셋 89명 · 색마다 ≤ per_class 장 (사람마다 고르게) · 사람 키 40~140 px 무작위로 줄임 (운용 크기)
    C 는 학습셋 안 사람 5겹 교차검증 (색 균형 정확도) 으로 고름 — 시험셋 U 는 안 봄"""
    if part in _PROBE:
        return _PROBE[part]
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    fp = LAB / f"train_{part}.npz"
    if fp.exists():
        z = np.load(fp, allow_pickle=True); X, y, g = z["X"], z["y"], z["g"]
    else:
        rng = random.Random(seed); items = []
        for f, c, pid in uavh_list("train", per_class, seed):
            fr, box = _frame_from_crop(cv2.imread(str(f)), rng.randint(40, 140))
            items.append({"frame": fr, "box": box, "pose": None, "primary": c, "person": pid})
        X = siglip_feats(items, part); y = np.array([CI[it["primary"]] for it in items]); g = np.array([it["person"] for it in items])
        np.savez(fp, X=X, y=y, g=g)
    best = None
    for C in (0.1, 0.3, 1, 3, 10, 30):
        accs = []
        for tr, va in GroupKFold(5).split(X, y, g):
            m = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(X[tr], y[tr])
            pr = m.predict(X[va]); accs.append(np.mean([np.mean(pr[y[va] == k] == k) for k in np.unique(y[va])]))
        a = float(np.mean(accs))
        print(f"  probe {part} C={C} 교차검증 색 균형 {a:.3f}", flush=True)
        if best is None or a > best[0]:
            best = (a, C)
    m = LogisticRegression(C=best[1], max_iter=3000, class_weight="balanced").fit(X, y)
    _PROBE[part] = (m, best)
    print(f"  probe {part} 고른 C={best[1]} (교차검증 {best[0]:.3f} · 학습 {len(y)}장 · {len(set(g))}명)", flush=True)
    return _PROBE[part]


def probe(items, part="person"):
    m, _ = probe_train(part)
    P = m.predict_proba(siglip_feats(items, part))
    out = np.zeros((len(items), len(COLORS)))
    out[:, m.classes_] = P
    return list(out)


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
}


def predict(name, items):
    m = dict(METHODS[name]); kind = m.pop("kind")
    if kind == "probe":
        probe_train(m["part"])                                          # 학습은 시간 재기 밖
    t0 = time.time()
    if kind == "rule":
        preds = [rule(it, **m) for it in items]
    elif kind == "zs":
        preds = zero_shot(items, **m)
    elif kind == "probe":
        preds = probe(items, **m)
    else:
        raise ValueError(kind)
    return preds, round((time.time() - t0) * 1000 / len(items), 2)


def run(names):
    sets = {s: pickle.load(open(LAB / f"{s}.pkl", "rb")) for s in ("U", "N")}
    out_p = BASE / "metrics" / "color_lab.json"
    res = json.load(open(out_p)) if out_p.exists() else {}
    pred_dir = LAB / "preds"; pred_dir.mkdir(exist_ok=True)
    base_t = {}
    for s, items in sets.items():
        bp = pred_dir / f"V10_{s}.pkl"
        if not bp.exists():
            pickle.dump(predict("V10", items), open(bp, "wb"))
        base_t[s] = metrics(items, pickle.load(open(bp, "rb"))[0], s)[1]
    for name in names:
        res[name] = {}
        for s, items in sets.items():
            pp = pred_dir / f"{name.replace(' ', '_')}_{s}.pkl"
            if pp.exists():
                preds, ms = pickle.load(open(pp, "rb"))
            else:
                preds, ms = predict(name, items); pickle.dump((preds, ms), open(pp, "wb"))
            if s == "U":
                for h in HEIGHTS:
                    sub = [i for i, it in enumerate(items) if it["h"] == h]
                    o, t = metrics([items[i] for i in sub], [preds[i] for i in sub], "U")
                    o["V10 대비 (짝 · 95 %)"] = boot([items[i] for i in sub], base_t["U"][sub], t, "U")
                    o["ms/장"] = ms
                    res[name][f"U{h}"] = o
            else:
                o, t = metrics(items, preds, "N")
                o["V10 대비 (짝 · 95 %)"] = boot(items, base_t["N"], t, "N"); o["ms/장"] = ms
                res[name]["N"] = o
        u, n = res[name]["U72"], res[name]["N"]
        print(f"{name:24s} U72 균형 {u['색 균형 1순위']:.3f} {u['V10 대비 (짝 · 95 %)']} · 불가 {u['판정불가율']:.2f} | U48 {res[name]['U48']['색 균형 1순위']:.3f} · U120 {res[name]['U120']['색 균형 1순위']:.3f}"
              f" | N 1순위 {n['1순위']:.3f} {n['V10 대비 (짝 · 95 %)']} · 누적 {n['누적 1순위']:.3f} · 순위 {n['찾는 사람 순위 중앙']} | {u['ms/장']} ms", flush=True)
        out_p.write_text(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build()
    elif sys.argv[1] == "run":
        run(list(METHODS) if sys.argv[2:] == ["all"] else sys.argv[2:])
