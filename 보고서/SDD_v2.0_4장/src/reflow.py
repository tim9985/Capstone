"""reflow.py — 한글 저장본의 줄 배치(linesegarray)와 표 행 높이로 쪽 나눔을 다시 흘려 계산한다
   (표 행 높이를 바꾼 뒤의 쪽 번호를 구하려고 — 한글의 쪽 넘김 규칙을 흉내 낸다)
   규칙: 쪽 나눔 문단은 새 쪽 · 줄/그림이 안 들어가면 새 쪽 · 표는 행 단위로 넘기고 머리행 반복
        표 첫 행이 안 들어가면 표 전체가 다음 쪽 · '다음 문단과 함께'는 다음 문단 첫 단위와 같은 쪽"""
import zipfile, collections
from lxml import etree
HP = "http://www.hancom.co.kr/hwpml/2011/paragraph"; HH = "http://www.hancom.co.kr/hwpml/2011/head"
q = lambda t: "{%s}%s" % (HP, t)
BODY = 84188 - 4000 - 4000 - 2000 - 2000
TBL_GAP = 600                                       # 표 아래 바깥 여백


def txt(e):
    return "".join(t.text or "" for t in e.iter(q("t")))


def row_heights(tb, override=None):
    """행 높이 = max(칸 높이, 칸 안 줄 배치로 잰 내용 높이 + 칸 여백) · override: {행: 높이}"""
    n = int(tb.get("rowCnt")); hs = [0] * n; spans = []
    im = tb.find(q("inMargin"))
    for tr in tb.findall(q("tr")):
        for tc in tr.findall(q("tc")):
            r = int(tc.find(q("cellAddr")).get("rowAddr")); rs = int(tc.find(q("cellSpan")).get("rowSpan"))
            cz = int(tc.find(q("cellSz")).get("height"))
            m = tc.find(q("cellMargin")) if tc.get("hasMargin") == "1" else im
            cb = None
            for cp in tc.find(q("subList")).findall(q("p")):
                for s in cp.findall(q("linesegarray") + "/" + q("lineseg")):
                    cb = max(cb or 0, int(s.get("vertpos")) + int(s.get("vertsize")))
            hh = max(cz, cb + int(m.get("top")) + int(m.get("bottom"))) if cb is not None else cz
            if rs == 1: hs[r] = max(hs[r], hh)
            else: spans.append((r, rs, hh))
    for r, rs, hh in spans:
        have = sum(hs[r:r + rs])
        if hh > have: hs[r + rs - 1] += hh - have
    if override:
        for r, v in override.items(): hs[r] = v
    hdr = hs[0] if (tb.get("repeatHeader") == "1" and tb.find(q("tr")).find(q("tc")).get("header") == "1") else 0
    return hs, hdr


class Doc:
    def __init__(self, path):
        z = zipfile.ZipFile(path)
        self.root = etree.fromstring(z.read("Contents/section0.xml"))
        hdr = etree.fromstring(z.read("Contents/header.xml"))
        self.keep = {}
        for e in hdr.iter("{%s}paraPr" % HH):
            b = e.find("{%s}breakSetting" % HH)
            self.keep[e.get("id")] = (b.get("keepWithNext") == "1", b.get("keepLines") == "1")
        self.top = list(self.root.iterfind(q("p")))
        self._advance()

    def _advance(self):
        """문단마다 높이 정보: 줄 목록 (위치, 높이) · 다음 문단까지 간격(같은 쪽일 때 실측)"""
        info = []
        for i, p in enumerate(self.top):
            ls = [(int(s.get("vertpos")), int(s.get("vertsize")), int(s.get("spacing"))) for s in p.findall(q("linesegarray") + "/" + q("lineseg"))]
            tb = p.find(q("run") + "/" + q("tbl"))
            pic = p.find(".//" + q("pic")) is not None and tb is None
            info.append(dict(i=i, ls=ls, tbl=tb, pic=pic, pb=p.get("pageBreak") == "1", kwn=self.keep.get(p.get("paraPrIDRef"), (False, False))[0],
                             text=txt(p).strip()))
        # 문단 끝 여백(다음 문단 첫 줄까지 남는 간격) 실측 — paraPr 별 대표값
        tail = collections.defaultdict(list)
        for a, b in zip(info, info[1:]):
            if not a["ls"] or not b["ls"] or a["tbl"] is not None or b["pb"]: continue
            la = a["ls"][-1]; v_next = b["ls"][0][0]
            if v_next > la[0] and all(y2 > y1 for (y1, _, _), (y2, _, _) in zip(a["ls"], a["ls"][1:])):
                tail[self.top[a["i"]].get("paraPrIDRef")].append(v_next - la[0] - la[1])     # 마지막 줄 상자 뒤 간격
        self.tail = {k: collections.Counter(v).most_common(1)[0][0] for k, v in tail.items()}
        self.info = info

    def line_steps(self, it):
        """문단의 줄별 (높이, 다음 줄까지 간격)"""
        ls = it["ls"]; out = []
        for k, (v, h, sp) in enumerate(ls):
            if k + 1 < len(ls) and ls[k + 1][0] > v: step = ls[k + 1][0] - v
            elif k + 1 < len(ls): step = h + sp
            else: step = h + self.tail.get(self.top[it["i"]].get("paraPrIDRef"), sp)
            out.append((h, step))
        return out

    def flow(self, overrides=None, breaks=()):
        """overrides: {문단 번호: {행: 높이}} — 바꾼 표 행 높이로 다시 흘린다 · breaks: 쪽 나눔을 더할 문단 · 반환: 문단마다 시작 쪽"""
        overrides = overrides or {}; breaks = set(breaks); self.tstart = {}
        page = 1; y = 0; start = {}; self.ystart = {}
        n = len(self.info)

        def first_unit(j):
            it = self.info[j]
            if it["ls"]: return it["ls"][0][1]          # 표 문단이면 기준 줄 높이(100) — 한글은 표 행이 아니라 기준 줄로 판단
            return 0

        def need(j, y_):
            """j 문단(과 '다음 문단과 함께' 사슬)이 y_ 에서 시작할 때 첫 단위까지 들어가는가"""
            it = self.info[j]; h_all = 0
            k = j
            while k < n and self.info[k]["kwn"] and k + 1 < n and not self.info[k + 1]["pb"]:
                st = self.line_steps(self.info[k]) if self.info[k]["tbl"] is None else None
                if st is None:
                    hs, _ = row_heights(self.info[k]["tbl"], overrides.get(k)); h_all += sum(hs) + TBL_GAP
                else:
                    h_all += sum(s for _, s in st)
                k += 1
            return y_ + h_all + first_unit(k) <= BODY

        for j, it in enumerate(self.info):
            if j > 0 and (it["pb"] or j in breaks):
                page += 1; y = 0
            elif y > 0 and it["kwn"] and not need(j, y):
                page += 1; y = 0
            start[j] = page; self.ystart[j] = y
            if it["tbl"] is not None:
                from tsplit import place
                moved, extra, y_end = place(it["tbl"], y, overrides.get(j))
                self.tstart[j] = page + moved                 # 표 첫 행이 놓인 쪽
                page += moved + extra; y = y_end + TBL_GAP
                continue
            for k, (h, step) in enumerate(self.line_steps(it)):
                if y + h > BODY and y > 0:
                    page += 1; y = 0
                    if k == 0: start[j] = page; self.ystart[j] = 0
                y += step
        return start, page


def hangul_pages(doc):
    """한글이 실제로 나눈 쪽: 줄 위치가 줄어든 곳(또는 쪽 나눔)을 세어 문단 시작 쪽을 낸다 (여러 쪽 표는 행 높이로)"""
    page = 1; start = {}; prev_v = -1; prev_multi = False
    for j, it in enumerate(doc.info):
        v0 = it["ls"][0][0] if it["ls"] else 0
        if j > 0 and (it["pb"] or (v0 < prev_v and not (prev_multi and v0 > 0))):
            page += 1
        start[j] = page
        inner = sum(1 for a, b in zip(it["ls"], it["ls"][1:]) if b[0] < a[0])
        page += inner
        if it["tbl"] is not None:
            hs, hdr = row_heights(it["tbl"])
            y = v0; extra = 0
            for k, h in enumerate(hs):
                if y + h > BODY and y > 0: extra += 1; y = hdr if k > 0 else 0
                y += h
            nxt = doc.info[j + 1] if j + 1 < len(doc.info) else None
            if extra and nxt is not None and nxt["ls"] and not nxt["pb"] and nxt["ls"][0][0] > 0:
                # 표 소모 길이 = (BODY - v0) + (k-1)·BODY + 끝 위치 = 행 합 + k·머리행 + 낭비(0 ≤ 낭비 ≤ k·최대 행)
                y_end = nxt["ls"][0][0] - TBL_GAP; H = sum(hs); mx = max(hs)
                cands = [k for k in range(1, 8) if -1500 <= BODY * k - v0 + y_end - H - k * hdr <= k * mx + 1500]
                if cands: extra = min(cands, key=lambda k: abs(k - extra)); y = y_end
            page += extra; prev_v = y if extra else v0; prev_multi = extra > 0
        else:
            prev_v = it["ls"][-1][0] if it["ls"] else prev_v; prev_multi = False
    return start, page
