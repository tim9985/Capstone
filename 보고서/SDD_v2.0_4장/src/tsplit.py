"""tsplit.py — 표를 쪽 경계에서 나누는 한글 규칙 (CELL: 행을 칸 안 줄 단위로 나눠 다음 쪽에 잇는다)"""
from reflow import q, BODY
SPLIT_MIN = 4000                                    # 이보다 남은 공간이 적으면 행을 통째로 넘긴다 (실측: 3788 통째 · 4188 나눔)


def row_model(tb, override=None):
    """행마다: 최소 높이(칸 높이), 칸별 줄 바닥 위치 목록(쪽에서 잘린 줄 위치는 이어 붙여 복원), 위·아래 여백"""
    im = tb.find(q("inMargin")); rows = []
    trs = tb.findall(q("tr"))
    for r, tr in enumerate(trs):
        cz = 0; cells = []; mt = mb = 0
        for tc in tr.findall(q("tc")):
            if tc.find(q("cellSpan")).get("rowSpan") != "1": continue
            m = tc.find(q("cellMargin")) if tc.get("hasMargin") == "1" else im
            mt = max(mt, int(m.get("top"))); mb = max(mb, int(m.get("bottom")))
            cz = max(cz, int(tc.find(q("cellSz")).get("height")))
            bottoms = []; off = 0; prev = None
            for cp in tc.find(q("subList")).findall(q("p")):
                for s in cp.findall(q("linesegarray") + "/" + q("lineseg")):
                    v = int(s.get("vertpos")); h = int(s.get("vertsize")); sp = int(s.get("spacing"))
                    if prev is not None and v + off < prev[0]:      # 쪽에서 잘려 0 으로 돌아간 줄 → 이어 붙인다
                        off = prev[0] + prev[1] - v
                    bottoms.append((v + off, v + off + h)); prev = (v + off, h + sp)
            cells.append(bottoms)
        if override and r in override: cz = override[r]
        content = max((b[-1][1] for b in cells if b), default=0) + mt + mb
        rows.append(dict(h=max(cz, content), cz=cz, cells=cells, mt=mt, mb=mb))
    rep = tb.get("repeatHeader") == "1" and trs[0].find(q("tc")).get("header") == "1"
    if not rep: return rows, 0
    r0 = rows[0]; ct = max((c[-1][1] for c in r0["cells"] if c), default=0) + r0["mt"] + r0["mb"]
    return rows, min(r0["h"], ct + 650)            # 다음 쪽에 되풀이되는 머리행 높이 (실측 규칙: 목차 제목 66/66 일치)


def place(tb, y0, override=None, body=BODY):
    """y0 에서 시작하는 표 → (시작 쪽 증가(표 전체가 넘어갔으면 1), 추가 쪽 수, 마지막 쪽 끝 위치)"""
    rows, hdr = row_model(tb, override)
    y = y0; extra = 0; moved = 0
    if y > 0 and y + rows[0]["h"] > body:          # 첫 행이 안 들어가면 표 전체가 다음 쪽
        moved = 1; y = 0
    for k, r in enumerate(rows):
        if y + r["h"] <= body:
            y += r["h"]; continue
        if y == 0 or (k == 0):
            y += r["h"]; continue
        room = body - y - r["mt"] - r["mb"]
        fit = [len([b for b in c if b[1] <= room]) for c in r["cells"]]
        cut = any(n < len(c) for c, n in zip(r["cells"], fit))
        if cut and body - y >= SPLIT_MIN:          # 글이 잘리고 남은 공간이 넉넉하면 줄 단위로 나눠 남은 줄은 다음 쪽에
            rest = 0
            for c, n in zip(r["cells"], fit):
                if n < len(c): rest = max(rest, c[-1][1] - c[n][0])
            extra += 1; y = hdr + rest + r["mt"] + r["mb"]
        else:
            extra += 1; y = hdr + r["h"]
        while y > body:                             # 한 행이 한 쪽보다 길면
            extra += 1; y -= body - hdr
    return moved, extra, y
