"""
color.py — 상의 색상 12종 판정 · 누적 · 조건 비교 (UC-0707 · 설계명세서 C-0711 ColorProfiler · C-0705 AppearanceComparator · 2026-10-05)

  규칙 기반 v0 (obsidian 「04 상의 색상 비교」 2절 · 3절 그대로 — 학습 없음)
    ① 박스 → 1.3배 여백 크롭 (1920×1080 원본에서)
    ② 사람 영역 = GrabCut (박스 사각형으로 시작 · 3회) · 실패하면 박스 안 가운데 타원
    ③ 상의 영역 — 서 있음/모름: 박스 위 15~60 % (머리 · 하의 제외) · 누움: 박스 전체 (머리 쪽을 몰라 v0 은 전체)
       피부 (YCrCb) · 포화 (V > 0.97) 화소는 뺀다
    ④ 화이트밸런스 = 프레임의 무채색에 가까운 화소 기준 · 노출 = 프레임 밝기 중앙값을 110 으로 (frame_stats · 프레임마다 한 번)
    ⑤ 화소마다 HSV 규칙으로 12색 투표 (검정 0.3 · 회색 0.7 가중) → 확률 · 1·2순위 · 확신도
    ⑥ 판정 불가: 상의 화소 < MIN_PIXELS
  누적 (ColorAccumulator): 관측마다 확률을 sqrt(화소 수) 가중 평균 — 한 장 오판을 여러 장이 덮는다
  비교 (match): 지정 색이 1순위 · 또는 2순위이면서 확률 ≥ 0.3 → 일치 · **불일치도 숨기지 않고 정렬만 뒤로**

  1.3 (10-29 · obsidian `_학습 큐` 「10-29 K」) — **SiglipColorProfiler** (기본 · 꾸러미가 있으면)
    SigLIP 2 base (google/siglip2-base-patch16-224 · Apache 2.0) 제로샷 — 상의 크롭 (박스 위 10~62 %) ↔ 「a photo of a person wearing a {색} shirt.」 12문장
    + 잔디 문장 (10-30 · 상의 크롭이 「a photo of green grass.」 에 「초록 셔츠」 보다 가까운 만큼 초록 로짓을 깎음 · 풀밭 위 누운 사람이 초록으로 잡히는 것)
    + 남색 문턱 (1순위가 남색인데 남색 − 검정 ≤ 0.95 면 검정 · UAV-Human 학습셋으로 정함)
    이미지 인코더는 이 파일의 plain torch 구현 (transformers 없이 · torch 2.x) · 문장 임베딩은 꾸러미에 미리 계산 (`text.npy`)
    평가 (색 균형 1순위 · 규칙 V10 → 제로샷): UAV-Human 드론 0.29 → 0.51 · NOMAD 0.34 → 0.41 · PETA 0.27 → 0.56 · 디관 사진 0.31 → 0.74 · 1.5 ms/명 (3090 · 묶음)
    꾸러미 = VISION_COLOR_PACK (기본 $INGEST_MODEL_DIR/color_siglip2 = /models/color_siglip2 · `vision.pt` · `text.npy` · `color_model.json` 해시 선검사) · VISION_COLOR=rule 이면 규칙 (아래 ColorProfiler)
"""
import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np

COLORS = ("red", "orange", "yellow", "green", "blue", "navy", "purple", "pink", "white", "gray", "black", "brown")
KO = {"red": "빨강", "orange": "주황", "yellow": "노랑", "green": "초록", "blue": "파랑", "navy": "남색",
      "purple": "보라", "pink": "분홍", "white": "흰색", "gray": "회색", "black": "검정", "brown": "갈색"}
MIN_PIXELS = 40          # 초안 150 은 1080p 30 m 서 있는 사람 (상의 ~100 화소) 을 다 버린다 → 평가로 다시 정한다
MATCH_P2 = 0.3


def frame_stats(frame):
    """프레임마다 한 번 — 화이트밸런스 이득 2가지 · 밝기 중앙값"""
    small = cv2.resize(frame, (192, 108), interpolation=cv2.INTER_AREA)
    f = small.reshape(-1, 3).astype(np.float32)
    lum = f.mean(1)
    lo, hi = np.percentile(lum, [5, 95])
    k = (lum > lo) & (lum < hi)
    m = f[k].mean(0) if k.any() else f.mean(0)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    a = (hsv[:, 1] < 40) & (hsv[:, 2] > 50) & (hsv[:, 2] < 245)          # 무채색에 가까운 화소 (길 · 바위 · 흰 물체)
    ma = f[a].mean(0) if a.sum() > 50 else m
    return {"gray_world": (m.mean() / np.maximum(m, 1e-3)).astype(np.float32),
            "achromatic": (ma.mean() / np.maximum(ma, 1e-3)).astype(np.float32),
            "v_median": float(np.median(hsv[:, 2]))}


def gray_world_gain(frame):
    return frame_stats(frame)["gray_world"]


def _classify_hsv(h, s, v, v_black=0.30, s_achro=0.18):
    """OpenCV HSV (H 0~179 · S,V 0~255) 화소 배열 → 색 번호 배열 (규칙 표 · 04 상의 색상 비교 3절)"""
    H = h.astype(np.float32) * 2.0; S = s.astype(np.float32) / 255.0; V = v.astype(np.float32) / 255.0
    out = np.full(H.shape, -1, np.int16)
    idx = {c: i for i, c in enumerate(COLORS)}
    black = V <= v_black
    achro = (~black) & (S < s_achro)
    out[black] = idx["black"]
    out[achro & (V > 0.78)] = idx["white"]
    out[achro & (V <= 0.78)] = idx["gray"]
    ch = ~(black | achro)
    red_h = (H >= 345) | (H < 15)
    pink = ch & (((H >= 300) & (H < 345)) | (red_h & (S < 0.5) & (V > 0.7)))
    out[pink] = idx["pink"]
    rest = ch & ~pink
    out[rest & red_h] = idx["red"]
    out[rest & (H >= 15) & (H < 40) & (V >= 0.5)] = idx["orange"]
    out[rest & (H >= 10) & (H < 40) & (V < 0.5)] = idx["brown"]
    out[rest & (H >= 40) & (H < 70)] = idx["yellow"]
    out[rest & (H >= 70) & (H < 170)] = idx["green"]
    out[rest & (H >= 170) & (H < 250) & (V >= 0.45)] = idx["blue"]
    out[rest & (H >= 200) & (H < 250) & (V < 0.45)] = idx["navy"]
    out[rest & (H >= 170) & (H < 200) & (V < 0.45)] = idx["blue"]
    out[rest & (H >= 250) & (H < 300)] = idx["purple"]
    return out


def _person_mask(crop, rect):
    """GrabCut 사람 영역 (0/1) — 실패하면 박스 안 가운데 타원"""
    m = np.zeros(crop.shape[:2], np.uint8)
    x, y, w, h = rect
    try:
        if w >= 6 and h >= 6:
            bg, fg = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
            cv2.grabCut(crop, m, rect, bg, fg, 3, cv2.GC_INIT_WITH_RECT)
            out = ((m == cv2.GC_FGD) | (m == cv2.GC_PR_FGD)).astype(np.uint8)
            if out.sum() >= 0.15 * w * h:
                return out
    except cv2.error:
        pass
    out = np.zeros(crop.shape[:2], np.uint8)
    cv2.ellipse(out, (x + w // 2, y + h // 2), (max(1, int(w * 0.4)), max(1, int(h * 0.45))), 0, 0, 360, 1, -1)
    return out


class ColorProfiler:
    # 기본값 = NOMAD 42명 비교에서 고른 V10 (무채색 기준 화이트밸런스 + 노출 맞춤 + 검정 표 0.3 · 회색 표 0.7 · eval_color.py · 10-05)
    #   회색 가정 (V0) 은 풀밭 장면에서 회색 · 흰 옷을 보라로 만든다 (1순위 0.28 → 0.39)
    def __init__(self, min_pixels=MIN_PIXELS, margin=1.3, wb="achromatic", v_target=110, v_black=0.30, s_achro=0.18, w_black=0.3, w_gray=0.7, exp_clip=(0.7, 2.0)):
        """exp_clip: 노출 맞춤 배율 범위 (10-14 UE — 밝은 풀밭에서 0.7 배로 어둡히면 상의가 검정이 된다 · 하한 1.0 = 밝히기만)
        wb: 화이트밸런스 'gray_world' | 'achromatic' | None · v_target: 프레임 밝기 중앙값을 이 값 (0~255) 으로 맞춤 (None 이면 안 함)"""
        self.min_pixels, self.margin, self.exp_clip = min_pixels, margin, exp_clip
        self.wb, self.v_target, self.v_black, self.s_achro = wb, v_target, v_black, s_achro
        self.vote_w = np.ones(len(COLORS)); self.vote_w[COLORS.index("black")] = w_black; self.vote_w[COLORS.index("gray")] = w_gray

    def profile(self, frame, box, pose=None, gain=None, stats=None):
        """frame BGR · box xyxy (프레임 px) · pose 'lying' | 'standing' | None · stats = frame_stats(프레임) (없으면 gain 만)
        반환 {"status": "ok" | "undetermined", "probs": {색: p}, "top": [(색, p), (색, p)], "n_pixels": n, "reason"}"""
        if stats is not None:
            gain = stats.get(self.wb) if self.wb else None
        H, W = frame.shape[:2]
        x1, y1, x2, y2 = [float(v) for v in box]
        bw, bh = x2 - x1, y2 - y1
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        X1, Y1 = int(max(0, cx - bw * self.margin / 2)), int(max(0, cy - bh * self.margin / 2))
        X2, Y2 = int(min(W, cx + bw * self.margin / 2)), int(min(H, cy + bh * self.margin / 2))
        crop = frame[Y1:Y2, X1:X2]
        if crop.size == 0 or bw < 4 or bh < 4:
            return {"status": "undetermined", "probs": {}, "top": [], "n_pixels": 0, "reason": "박스가 너무 작음"}
        if gain is not None:
            crop = np.clip(crop.astype(np.float32) * gain, 0, 255).astype(np.uint8)
        rx, ry = int(round(x1 - X1)), int(round(y1 - Y1))
        rect = (max(rx, 0), max(ry, 0), max(int(round(bw)), 1), max(int(round(bh)), 1))
        mask = _person_mask(crop, rect)
        region = np.zeros_like(mask)
        if pose == "lying":
            region[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]] = 1
        else:                                     # 서 있음 · 모름 — 위 15~60 %
            a, b = rect[1] + int(0.15 * rect[3]), rect[1] + int(0.60 * rect[3])
            region[a:b, rect[0]:rect[0] + rect[2]] = 1
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        if self.v_target and stats is not None and stats.get("v_median"):
            g = float(np.clip(self.v_target / max(stats["v_median"], 1.0), *self.exp_clip))   # 노출 맞춤 (밝기만)
            hsv[..., 2] = np.clip(hsv[..., 2].astype(np.float32) * g, 0, 255).astype(np.uint8)
        ycc = cv2.cvtColor(crop, cv2.COLOR_BGR2YCrCb)
        skin = (ycc[..., 1] >= 135) & (ycc[..., 1] <= 180) & (ycc[..., 2] >= 85) & (ycc[..., 2] <= 135) & (hsv[..., 2] > 60)
        sat = hsv[..., 2] > 247
        sel = (mask > 0) & (region > 0) & ~skin & ~sat
        n = int(sel.sum())
        if n < self.min_pixels:
            return {"status": "undetermined", "probs": {}, "top": [], "n_pixels": n, "reason": f"상의 화소 {n} < {self.min_pixels}"}
        lab = _classify_hsv(hsv[..., 0][sel], hsv[..., 1][sel], hsv[..., 2][sel], self.v_black, self.s_achro)
        lab = lab[lab >= 0]
        cnt = np.bincount(lab, minlength=len(COLORS)).astype(np.float64) * self.vote_w + 0.5     # 표 가중 (그림자 · 주름의 어두운 화소가 이기지 않게) · 작은 평활
        p = cnt / cnt.sum()
        order = np.argsort(-p)
        probs = {COLORS[i]: round(float(p[i]), 4) for i in range(len(COLORS))}
        return {"status": "ok", "probs": probs, "top": [(COLORS[i], round(float(p[i]), 3)) for i in order[:2]],
                "n_pixels": n, "reason": None}


class ColorAccumulator:
    """같은 후보의 관측을 모은다 — sqrt(화소 수) 가중 평균"""
    def __init__(self):
        self.sum, self.w, self.n = np.zeros(len(COLORS)), 0.0, 0

    def add(self, prof):
        if prof["status"] != "ok":
            return
        wt = float(np.sqrt(prof["n_pixels"]))
        self.sum += wt * np.array([prof["probs"][c] for c in COLORS]); self.w += wt; self.n += 1

    def result(self):
        if self.n == 0:
            return {"status": "undetermined", "top": [], "n_obs": 0}
        p = self.sum / self.w
        o = np.argsort(-p)
        return {"status": "ok", "top": [(COLORS[i], round(float(p[i]), 3)) for i in o[:2]],
                "probs": {COLORS[i]: round(float(p[i]), 4) for i in range(len(COLORS))}, "n_obs": self.n}


def match(result, query):
    """조건 비교 — ('일치' | '불일치' | '판정 불가', 정렬 점수 = 지정 색 확률)"""
    if result.get("status") != "ok":
        return "판정 불가", 0.0
    top = result["top"]
    p = result.get("probs", {}).get(query, 0.0)
    if top and top[0][0] == query:
        return "일치", p
    if len(top) > 1 and top[1][0] == query and top[1][1] >= MATCH_P2:
        return "일치", p
    return "불일치", p


# ───────────────────────── 1.3 SigLIP 2 제로샷 (10-29) ─────────────────────────
def _siglip_vision(sd, cfg):
    """SigLIP 이미지 인코더 (patch16 · 표준 SiglipVisionModel 과 같은 계산) — plain torch · 가중치 키 = HF 의 vision_model.* 에서 앞머리 뺀 것"""
    import torch
    import torch.nn.functional as F
    D, L, Hh, eps = cfg["hidden"], cfg["layers"], cfg["heads"], cfg["eps"]
    W = {k: v for k, v in sd.items()}

    def ln(x, p):
        return F.layer_norm(x, (D,), W[p + ".weight"], W[p + ".bias"], eps)

    def lin(x, p):
        return F.linear(x, W[p + ".weight"], W[p + ".bias"])

    def attn(q, k, v):
        B, Tq, _ = q.shape; Tk = k.shape[1]
        q = q.view(B, Tq, Hh, D // Hh).transpose(1, 2); k = k.view(B, Tk, Hh, D // Hh).transpose(1, 2); v = v.view(B, Tk, Hh, D // Hh).transpose(1, 2)
        return F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, Tq, D)

    def forward(x):                                       # x: (B, 3, 224, 224) 정규화된 값
        h = F.conv2d(x, W["embeddings.patch_embedding.weight"], W["embeddings.patch_embedding.bias"], stride=cfg["patch"])
        h = h.flatten(2).transpose(1, 2) + W["embeddings.position_embedding.weight"][None]
        for i in range(L):
            p = f"encoder.layers.{i}."
            r = h; h = ln(h, p + "layer_norm1")
            h = r + lin(attn(lin(h, p + "self_attn.q_proj"), lin(h, p + "self_attn.k_proj"), lin(h, p + "self_attn.v_proj")), p + "self_attn.out_proj")
            r = h; h = ln(h, p + "layer_norm2")
            h = r + lin(F.gelu(lin(h, p + "mlp.fc1"), approximate="tanh"), p + "mlp.fc2")
        h = ln(h, "post_layernorm")
        wq, wk, wv = W["head.attention.in_proj_weight"].chunk(3); bq, bk, bv = W["head.attention.in_proj_bias"].chunk(3)   # 주의 풀링 머리
        probe = W["head.probe"].expand(h.shape[0], -1, -1)
        o = F.linear(attn(F.linear(probe, wq, bq), F.linear(h, wk, bk), F.linear(h, wv, bv)), W["head.attention.out_proj.weight"], W["head.attention.out_proj.bias"])
        o = o + lin(F.gelu(lin(ln(o, "head.layernorm"), "head.mlp.fc1"), approximate="tanh"), "head.mlp.fc2")
        return F.normalize(o[:, 0].float(), dim=-1)
    return forward


def _sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


class SiglipColorProfiler:
    """ColorProfiler 와 같은 profile() 반환 형식 — 누적 (ColorAccumulator) · 비교 (match) 그대로 씀"""
    name = "siglip2-zs"

    def __init__(self, pack, device="cuda"):
        import torch
        from PIL import Image
        self._torch, self._Image = torch, Image
        pack = Path(pack)
        self.meta = json.load(open(pack / "color_model.json"))
        for fn, h in self.meta["sha256"].items():                       # 해시 선검사 (탐지 모델과 같은 규칙)
            if _sha256(pack / fn) != h:
                raise RuntimeError(f"color pack hash mismatch: {fn}")
        if not torch.cuda.is_available() and device == "cuda":
            raise RuntimeError("CUDA 없음")
        self.dev = torch.device(device)
        sd = torch.load(pack / "vision.pt", map_location=self.dev, weights_only=True)
        self.dtype = torch.float16 if self.dev.type == "cuda" else torch.float32
        sd = {k: v.to(self.dtype) for k, v in sd.items()}
        self.enc = _siglip_vision(sd, self.meta["vision"])
        self.text = torch.tensor(np.load(pack / "text.npy"), dtype=torch.float32, device=self.dev)   # 12 × D (COLORS 순서 · 정규화)
        assert tuple(self.meta["colors"]) == COLORS
        self.scale, self.navy_delta = float(self.meta["logit_scale"]), float(self.meta["navy_delta"])
        self.grass_alpha = float(self.meta.get("grass_alpha", 0.0))      # 10-30 · 꾸러미에 grass.npy 가 있을 때만
        self.grass = torch.tensor(np.load(pack / "grass.npy"), dtype=torch.float32, device=self.dev) if self.grass_alpha else None
        self.size, self.region = int(self.meta["image_size"]), tuple(self.meta["upper_region"])
        self.min_w, self.min_h = 4, 6

    def _crop(self, frame, box, pose):
        H, W = frame.shape[:2]
        x1, y1, x2, y2 = int(max(0, box[0])), int(max(0, box[1])), int(min(W, box[2])), int(min(H, box[3]))
        if pose != "lying":
            h = y2 - y1; y1, y2 = y1 + int(self.region[0] * h), y1 + int(self.region[1] * h)
        return frame[y1:y2, x1:x2]

    def _prep(self, crop):
        im = self._Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).resize((self.size, self.size), self._Image.BILINEAR)
        return (np.asarray(im, np.float32) / 255.0 - 0.5) / 0.5

    def logits(self, crops):
        """BGR 상의 크롭 목록 → (N, 12) 로짓 (잔디 문장 적용 · 남색 문턱 적용 전)"""
        torch = self._torch
        x = torch.tensor(np.stack([self._prep(c) for c in crops]).transpose(0, 3, 1, 2), device=self.dev, dtype=self.dtype)
        with torch.no_grad():
            f = self.enc(x)
            lg = f @ self.text.T * self.scale
            if self.grass is not None:                                   # 잔디 문장 — (잔디 − 초록 셔츠) 로짓이 양수인 만큼 초록을 깎음
                g = COLORS.index("green")
                lg[:, g] -= self.grass_alpha * torch.clamp((f @ self.grass - f @ self.text[g]) * self.scale, min=0)
        return lg.cpu().numpy()

    def _probs(self, lg):
        n, b = COLORS.index("navy"), COLORS.index("black")
        lg = lg.copy()
        if lg.argmax() == n and lg[n] - lg[b] <= self.navy_delta:
            lg[n], lg[b] = lg[b], lg[n]
        e = np.exp(lg - lg.max())
        return e / e.sum()

    def profile_many(self, frame, boxes, poses=None, stats=None):
        poses = poses or [None] * len(boxes)
        crops = [self._crop(frame, b, p) for b, p in zip(boxes, poses)]
        ok = [i for i, c in enumerate(crops) if c.shape[0] >= self.min_h and c.shape[1] >= self.min_w]
        out = [{"status": "undetermined", "probs": {}, "top": [], "n_pixels": int(crops[i].shape[0] * crops[i].shape[1]),
                "reason": "박스가 너무 작음", "method": self.name} for i in range(len(boxes))]
        if ok:
            L = self.logits([crops[i] for i in ok])
            for j, i in enumerate(ok):
                p = self._probs(L[j]); order = np.argsort(-p)
                out[i] = {"status": "ok", "probs": {COLORS[k]: round(float(p[k]), 4) for k in range(len(COLORS))},
                          "top": [(COLORS[k], round(float(p[k]), 3)) for k in order[:2]],
                          "n_pixels": int(crops[i].shape[0] * crops[i].shape[1]), "reason": None, "method": self.name}
        return out

    def profile(self, frame, box, pose=None, gain=None, stats=None):
        return self.profile_many(frame, [box], [pose])[0]


def make_colorer(pack=None, log=None):
    """VISION_COLOR = auto (기본 · 꾸러미 있으면 SigLIP · 없거나 실패면 규칙) | siglip (실패하면 오류) | rule"""
    mode = os.environ.get("VISION_COLOR", "auto").lower()
    pack = pack or os.environ.get("VISION_COLOR_PACK", str(Path(os.environ.get("INGEST_MODEL_DIR", "/models")) / "color_siglip2"))
    if mode != "rule" and (Path(pack) / "color_model.json").exists():
        try:
            return SiglipColorProfiler(pack)
        except Exception as e:
            if mode == "siglip":
                raise
            if log:
                log(f"color: SigLIP 꾸러미 실패 → 규칙 V10 ({type(e).__name__}: {e})")
    elif mode == "siglip":
        raise FileNotFoundError(f"color pack 없음: {pack}")
    c = ColorProfiler(); c.name = "rule-v10"
    return c
