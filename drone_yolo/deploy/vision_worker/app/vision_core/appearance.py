"""
appearance.py — 인상착의 비교 (설계명세서 C-0705 AppearanceComparator · SD-0707 · 2026-10-09)

  입력  운영자가 넣은 찾는 사람의 외형 조건 (임무 appearance_query) — {"upper": ["red"]} 또는 {"upper": ["navy", "black"]}
        후보의 상의 색 누적 결과 (ColorAccumulator.result — 12색 확률 · 1·2순위 · 관측 수)
  출력  {"query": [...], "verdict": MATCH | PARTIAL | MISMATCH | UNDETERMINED, "score": 0~1, "n_obs": …, "basis": …}
        score = 지정 색 확률 합 (+ similar × 닮은 색 확률 합 — 10-09 C1 에서 이득 없음 → 기본 0)
        MATCH   지정 색이 1순위 · 또는 2순위이면서 확률 ≥ 0.3 (color.match 와 같은 규칙)
        PARTIAL 지정 색 대신 닮은 색이 1순위 (예: 남색 지정 · 검정 1순위)
        UNDETERMINED 상의 화소 부족 · 관측 없음
  검증  10-09 C1 (NOMAD 42명 · 배우×거리×자세 묶음 누적) — 점수 AUROC 0.766 · 자기 색 MATCH 0.38 · MATCH+PARTIAL 0.56 · 다른 색 오일치 0.08
        → 순위 보조로는 쓸 만 · 한 번의 판정을 믿을 수준은 아님 (상의 색 판정기 자체의 한계 · 거리 a70 에서 더 약함)
  원칙  불일치여도 후보를 지우지 않는다 — 순위만 뒤로 (UC-0707 · 최종 판단은 운영자)
"""
from .color import COLORS

MATCH_P2 = 0.3
# 닮은 색 — eval_color 혼동 행렬 (NOMAD 42명 · 10-05) 에서 자주 섞인 쌍 + 상식적인 이웃
SIMILAR = {
    "navy": {"black", "blue"}, "black": {"navy", "gray"}, "blue": {"navy"}, "gray": {"white", "black"}, "white": {"gray"},
    "red": {"orange", "pink"}, "orange": {"red", "brown", "yellow"}, "pink": {"red", "purple"}, "purple": {"pink", "navy"},
    "brown": {"orange"}, "yellow": {"orange"}, "green": set(),
}
ALIAS = {"cyan": "blue", "sky": "blue", "light gray": "gray", "dark gray": "gray", "beige": "brown", "khaki": "brown",
         "빨강": "red", "주황": "orange", "노랑": "yellow", "초록": "green", "파랑": "blue", "남색": "navy", "보라": "purple",
         "분홍": "pink", "흰색": "white", "회색": "gray", "검정": "black", "갈색": "brown"}


def normalize_query(q):
    """여러 모양의 조건 → ["red", …] (12색 이름만) · 없으면 []"""
    if not q:
        return []
    up = q.get("upper", q.get("upper_color", q)) if isinstance(q, dict) else q
    if isinstance(up, dict):
        up = up.get("colors") or up.get("color") or []
    if isinstance(up, str):
        up = [up]
    out = []
    for c in up or []:
        c = ALIAS.get(str(c).strip().lower(), str(c).strip().lower())
        if c in COLORS and c not in out:
            out.append(c)
    return out[:2]


class AppearanceComparator:
    def __init__(self, similar=0.0):
        self.similar = similar

    def compare(self, color_result, query):
        q = normalize_query(query)
        if not q:
            return None
        base = {"query": q, "n_obs": color_result.get("n_obs", 0)}
        if color_result.get("status") != "ok":
            return base | {"verdict": "UNDETERMINED", "score": 0.0, "basis": "upper-body pixels insufficient"}
        probs, top = color_result.get("probs", {}), color_result.get("top", [])
        near = set().union(*(SIMILAR.get(c, set()) for c in q)) - set(q)
        score = sum(probs.get(c, 0.0) for c in q) + self.similar * sum(probs.get(c, 0.0) for c in near)
        t1 = top[0][0] if top else None
        if t1 in q or (len(top) > 1 and top[1][0] in q and top[1][1] >= MATCH_P2):
            verdict = "MATCH"
        elif t1 in near:
            verdict = "PARTIAL"
        else:
            verdict = "MISMATCH"
        return base | {"verdict": verdict, "score": round(min(score, 1.0), 4), "top": [c for c, _ in top[:2]],
                       "basis": "12-color HSV vote accumulated over observations · priority aid only"}
