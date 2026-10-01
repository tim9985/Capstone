"""pages2.py — 한글 저장본의 줄 배치(linesegarray)와 표 행 높이로 최상위 문단의 시작·끝 쪽을 계산한다
   규칙: 쪽 나눔(pageBreak=1)은 항상 새 쪽 · 줄 위치가 앞 문단보다 작아지면 새 쪽
        표(쪽 경계 CELL)는 행 단위로 넘어가고 머리행 반복 · 첫 행이 안 들어가면 표 전체가 다음 쪽"""
import zipfile, re
from lxml import etree
HP = "http://www.hancom.co.kr/hwpml/2011/paragraph"; q = lambda t: "{%s}%s" % (HP, t)
BODY = 84188 - 4000 - 4000 - 2000 - 2000


def txt(e):
    return "".join(t.text or "" for t in e.iter(q("t")))


def table_pages(tb, y0, body=BODY):
    """y0 에서 시작하는 표가 차지하는 (추가 쪽 수, 마지막 쪽에서 끝나는 위치)"""
    trs = tb.findall(q("tr"))
    n = int(tb.get("rowCnt")); hs = [0] * n
    spans = []
    for tr in trs:
        for tc in tr.findall(q("tc")):
            r = int(tc.find(q("cellAddr")).get("rowAddr")); rs = int(tc.find(q("cellSpan")).get("rowSpan"))
            h = int(tc.find(q("cellSz")).get("height"))
            m = tc.find(q("cellMargin")) if tc.get("hasMargin") == "1" else tb.find(q("inMargin"))
            cb = 0
            for cp in tc.find(q("subList")).findall(q("p")):
                for sg in cp.findall(q("linesegarray") + "/" + q("lineseg")):
                    cb = max(cb, int(sg.get("vertpos")) + int(sg.get("vertsize")))
            if cb: h = max(h, cb + int(m.get("top")) + int(m.get("bottom")))
            if rs == 1: hs[r] = max(hs[r], h)
            else: spans.append((r, rs, h))
    for r, rs, h in spans:                         # 합친 칸이 더 크면 마지막 행에 모자란 만큼 더한다
        have = sum(hs[r:r + rs])
        if h > have: hs[r + rs - 1] += h - have
    hdr = hs[0] if (tb.get("repeatHeader") == "1" and trs and trs[0].find(q("tc")).get("header") == "1") else 0
    y = y0; extra = 0
    for k, h in enumerate(hs):
        if y + h > body and y > 0:
            extra += 1; y = (hdr if k > 0 else 0)
        y += h
    return extra, y


def compute(path):
    root = etree.fromstring(zipfile.ZipFile(path).read("Contents/section0.xml"))
    top = list(root.iterfind(q("p")))
    res = []
    page = 1; prev_end_page = 1; prev_v = 0; prev_multi = False
    for i, p in enumerate(top):
        ls = p.findall(q("linesegarray") + "/" + q("lineseg"))
        v0 = int(ls[0].get("vertpos")) if ls else 0
        if i == 0:
            start = 1
        elif p.get("pageBreak") == "1":
            start = prev_end_page + 1
        elif prev_multi and v0 > 0:
            start = prev_end_page                  # 여러 쪽 표 바로 뒤: 한글이 적은 위치(0 아님)는 표의 마지막 쪽
        elif v0 < prev_v:
            start = prev_end_page + 1
        else:
            start = prev_end_page
        inner = sum(1 for a, b in zip(ls, ls[1:]) if int(b.get("vertpos")) < int(a.get("vertpos")))
        end = start + inner
        tb = p.find(q("run") + "/" + q("tbl"))
        if tb is not None:
            extra, yend = table_pages(tb, v0)
            end = start + extra
            prev_v = yend if extra else v0
            prev_multi = extra > 0
        else:
            prev_v = int(ls[-1].get("vertpos")) if ls else prev_v
            prev_multi = False
        res.append(dict(i=i, start=start, end=end, v0=v0, text=txt(p), table=tb is not None))
        prev_end_page = end
    return res, top


def toc_entries(top):
    out = []
    for i, p in enumerate(top):
        t = p.find(".//" + q("t"))
        if t is None or t.find(q("tab")) is None: continue
        tail = (t.find(q("tab")).tail or "").strip()
        if re.fullmatch(r"\d+", tail): out.append((i, (t.text or "").strip(), int(tail)))
    return out


def heading_pages(res, toc):
    first_body = max(i for i, _, _ in toc) + 1
    got = {}
    for r in res:
        if r["i"] < first_body: continue
        t = r["text"].strip()
        if t and t not in got: got[t] = r["start"]
    return got
