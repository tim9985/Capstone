"""final2.py — 한글에서 저장한 SDD 의 배치 조정과 목차 쪽수 맞추기
   python final2.py <입력.hwpx> <출력.hwpx>
   1) 넉넉하게 잡혀 빈 줄이 남는 표 행을 실제 내용 높이로 (한 줄 행 최소 높이는 유지)
   2) 캡션·클래스명만 쪽 바닥에 남고 표가 다음 쪽으로 넘어가는 곳에 쪽 나눔
   3) 다시 흘려 계산한 쪽 번호로 목차 쪽수 (계산기는 현재 한글 배치와 목차 제목 66/66 일치를 확인한 규칙)"""
import json, re, sys, zipfile, collections
from lxml import etree
sys.path.insert(0, __file__.rsplit("/", 1)[0] if "/" in __file__ else ".")
from reflow import Doc, hangul_pages, q, txt
from tsplit import row_model

SRC, DST = sys.argv[1], sys.argv[2]
d = Doc(SRC)
root, top, info = d.root, d.top, d.info
zin = zipfile.ZipFile(SRC)

# ── 목차 항목과 본문 제목 ──
toc = []
for i, p in enumerate(top):
    t = p.find(".//" + q("t"))
    if t is not None and t.find(q("tab")) is not None and re.fullmatch(r"\d+", (t.find(q("tab")).tail or "").strip()):
        toc.append((i, (t.text or "").strip(), t.find(q("tab"))))
last_toc = max(i for i, _, _ in toc)
head = {}
for j, it in enumerate(info):
    if j > last_toc and it["text"] and it["text"] not in head: head[it["text"]] = j
assert all(title in head for _, title, _ in toc), [title for _, title, _ in toc if title not in head]
ch3 = next(j for j, it in enumerate(info) if j > last_toc and it["text"].startswith("3. 사용자 인터페이스"))
ch4 = next(j for j, it in enumerate(info) if j > last_toc and it["text"].startswith("4. 객체지향설계"))

before, n_before = hangul_pages(d)

# ── 1) 행 높이: 내용보다 1000 이상 큰 행을 내용 높이로 (한 줄 행 최소 높이 유지) ──
over = {}; n_rows = 0; saved = 0
for j, it in enumerate(info):
    tb = it["tbl"]
    if tb is None or j <= last_toc + 2: continue                 # 표지 표 제외
    if ch3 <= j < ch4: continue                                   # 3장 화면 설계 표는 손대지 않는다
    if tb.find(".//" + q("pic")) is not None: continue            # 그림을 담은 표 제외
    rows, _ = row_model(tb)
    singles = [r["cz"] for r in rows[1:] if max((len(c) for c in r["cells"]), default=1) == 1]
    floor = min(singles) if singles else rows[0]["cz"]
    ov = {}
    for k, r in enumerate(rows):
        if k == 0: continue                                       # 머리행은 그대로
        content = max((c[-1][1] for c in r["cells"] if c), default=0) + r["mt"] + r["mb"]
        new = max(content, floor)
        if r["cz"] - new >= 1000 or content > r["cz"]:
            ov[k] = new; n_rows += 1; saved += r["cz"] - new
    if ov: over[j] = ov


# ── 2) 고아 캡션: 캡션·클래스명('다음 문단과 함께' 사슬)은 이 쪽, 표 첫 행은 다음 쪽 → 사슬 앞에 쪽 나눔 ──
def chain_start(j):
    s = j
    while s - 1 > last_toc and info[s - 1]["kwn"] and info[s - 1]["tbl"] is None and not info[s - 1]["pic"] and info[s - 1]["text"]:
        s -= 1
    return s


breaks = []
for _ in range(400):
    start, total = d.flow(over, breaks)
    found = None
    for j, it in enumerate(info):
        if it["tbl"] is None or j not in d.tstart: continue
        s = chain_start(j)
        if s == j or info[s]["pb"] or s in breaks: continue
        if d.tstart[j] > start[s] and d.ystart.get(s, 0) > 0:
            found = s; break
    if found is None: break
    breaks.append(found)
start, total = d.flow(over, breaks)

# ── 3) 목차 쪽수 ──
toc_new = {title: start[head[title]] for _, title, _ in toc}
changed_toc = 0
for i, title, tab in toc:
    old = (tab.tail or "").strip()
    if old != str(toc_new[title]):
        tab.tail = str(toc_new[title]); changed_toc += 1

# ── 쓰기: 행 높이 · 쪽 나눔 · 줄 배치 정보 정리 ──
for j, ov in over.items():
    tb = info[j]["tbl"]; trs = tb.findall(q("tr")); hs = []
    rows, _ = row_model(tb)
    for k, tr in enumerate(trs):
        h = ov.get(k, rows[k]["cz"])
        for tc in tr.findall(q("tc")):
            if tc.find(q("cellSpan")).get("rowSpan") == "1": tc.find(q("cellSz")).set("height", str(h))
        hs.append(h)
    tb.find(q("sz")).set("height", str(sum(hs)))
for s in breaks:
    top[s].set("pageBreak", "1")
ORIG = {}
for p0 in etree.fromstring(zin.read("Contents/section0.xml")).iter(q("p")):
    ORIG.setdefault(p0.get("id"), []).append(etree.tostring(p0))
stripped = 0
for p in sorted(root.iter(q("p")), key=lambda e: -len(list(e.iterancestors()))):
    if p.find(q("linesegarray")) is not None and etree.tostring(p) not in ORIG.get(p.get("id"), []):
        for ls in p.findall(q("linesegarray")): p.remove(ls)
        stripped += 1
sec = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)
zout = zipfile.ZipFile(DST, "w")
for inf in zin.infolist():
    fn = inf.filename
    data = sec if fn == "Contents/section0.xml" else zin.read(fn)
    zi = zipfile.ZipInfo(fn, date_time=inf.date_time)
    zi.create_system, zi.create_version = inf.create_system, inf.create_version
    zi.compress_type = zipfile.ZIP_STORED if (fn in ("mimetype", "version.xml") or fn.endswith(".png")) else zipfile.ZIP_DEFLATED
    zout.writestr(zi, data)
zout.close()
rep = dict(rows=n_rows, saved=saved, breaks=[(s, info[s]["text"][:30], start[s]) for s in breaks], pages_before=n_before, pages_after=total,
           toc_changed=changed_toc, toc=[(title, toc_new[title]) for _, title, _ in toc], stripped=stripped)
json.dump(rep, open("final2.json", "w"), ensure_ascii=False, indent=1)
print(f"행 높이 줄임 {n_rows}행 (합 {saved}) · 쪽 나눔 추가 {len(breaks)} · 총 쪽수 {n_before} → {total} · 목차 바뀐 항목 {changed_toc}/66 · 줄 배치 지운 문단 {stripped}")
