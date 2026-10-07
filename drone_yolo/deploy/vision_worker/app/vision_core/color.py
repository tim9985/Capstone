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
"""
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
    def __init__(self, min_pixels=MIN_PIXELS, margin=1.3, wb="achromatic", v_target=110, v_black=0.30, s_achro=0.18, w_black=0.3, w_gray=0.7):
        """wb: 화이트밸런스 'gray_world' | 'achromatic' | None · v_target: 프레임 밝기 중앙값을 이 값 (0~255) 으로 맞춤 (None 이면 안 함)"""
        self.min_pixels, self.margin = min_pixels, margin
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
            g = float(np.clip(self.v_target / max(stats["v_median"], 1.0), 0.7, 2.0))   # 노출 맞춤 (밝기만)
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
