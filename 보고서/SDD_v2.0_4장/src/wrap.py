"""wrap.py — 맑은 고딕 9.5pt 근사(나눔고딕 폭 × 0.95 × 1.02)로 칸 안 줄 수를 추정"""
from functools import lru_cache
from PIL import ImageFont
FONT = ImageFont.truetype("/tmp/claude-1001/-home-se-JupyterLAB/027ea9bd-abdf-476f-ad0d-25f3e2ea7ffc/scratchpad/NanumGothic.ttf", 1000)
K = 0.95 * 1.02


@lru_cache(maxsize=None)
def w(s):
    return FONT.getlength(s) * K


@lru_cache(maxsize=None)
def lines(text, avail):
    if not text: return 1
    n = 1; cur = 0.0; sp = w(" ")
    for word in text.split(" "):
        ww = w(word)
        if cur == 0:
            if ww <= avail: cur = ww
            else:                                   # 긴 낱말은 글자 단위로 나눈다
                for ch in word:
                    c = w(ch)
                    if cur + c > avail: n += 1; cur = c
                    else: cur += c
        elif cur + sp + ww <= avail:
            cur += sp + ww
        else:
            n += 1; cur = 0.0
            if ww <= avail: cur = ww
            else:
                for ch in word:
                    c = w(ch)
                    if cur + c > avail: n += 1; cur = c
                    else: cur += c
    return n
