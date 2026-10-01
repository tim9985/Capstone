"""final.py — 사용자 최신본(v2.1_쪽번호수정)을 검토·정리해 제출본을 만든다
   python final.py <입력.hwpx> <출력.hwpx> <v1.4 원본.hwpx>
   1) 4.2.6·4.2.7(패치로 만든 부분) 서식 번호를 문서 머리말 기준으로 바로잡기
   2) 문구 수정 · 3) 용어 해설 보강 · 4) 표 열 폭 재배분과 행 높이 재계산
   5) 클래스 설명 줄을 표와 같은 쪽에 묶기 · 6) 한글 저장 규칙(바꾼 문단 줄 배치 정보 제거 · PNG 무압축)"""
import copy, json, re, sys, zipfile, collections
from lxml import etree
sys.path.insert(0, __file__.rsplit("/", 1)[0] if "/" in __file__ else ".")
from stylemap import mapping
from wrap import lines

SRC, DST = sys.argv[1], sys.argv[2]
V14 = sys.argv[3] if len(sys.argv) > 3 else "v1.4.hwpx"          # build20 틀 문단의 원본 (서식 번호 대응용 · git 밖)
HP = "http://www.hancom.co.kr/hwpml/2011/paragraph"; HH = "http://www.hancom.co.kr/hwpml/2011/head"
q = lambda t: "{%s}%s" % (HP, t); h = lambda t: "{%s}%s" % (HH, t)
txt = lambda e: "".join(t.text or "" for t in e.iter(q("t")))
zin = zipfile.ZipFile(SRC)
root = etree.fromstring(zin.read("Contents/section0.xml"))
hdr = etree.fromstring(zin.read("Contents/header.xml"))
top = list(root.iterfind(q("p")))
LOG = collections.Counter()
NOTE = []                                            # 바꾼 곳 (위치 · 전 · 후)
TOUCH = set()                                        # 줄 배치 정보를 지울 문단
_pid = [max(int(p.get("id")) for p in root.iter(q("p"))) + 1]


def new_pid():
    _pid[0] += 1
    return str(_pid[0])


def touch(el):
    """el 을 담은 문단과 그 바깥 문단(표 문단)까지 줄 배치 정보를 지울 대상으로 둔다"""
    p = el
    while p is not None:
        if p.tag == q("p"): TOUCH.add(p)
        p = p.getparent()


# ── 1) 패치로 만든 부분의 서식 번호 (v1.4 머리말 번호 → 현재 머리말 번호) ──
ID0 = 2147493387
SKIP = ("RC, 텔레메트리", "Controller 와 다른 패키지")      # 원래 있던 문단에 글만 바꾼 것 (번호는 이미 현재 기준)
M = mapping(V14, SRC)
regen = [p for p in top if int(p.get("id")) >= ID0 and not txt(p).strip().startswith(SKIP)]
for p in regen:
    for e in p.iter():
        for attr, tag in (("paraPrIDRef", "paraPr"), ("charPrIDRef", "charPr"), ("borderFillIDRef", "borderFill")):
            v = e.get(attr)
            if v is not None and M[tag].get(v) not in (None, v):
                e.set(attr, M[tag][v]); LOG["서식 번호 바로잡음"] += 1
NOTE.append(("4.2.6 · 4.2.7 (패치로 만든 부분)", "다른 단위와 다른 서식 — 표 칸 줄간격 3000·왼쪽 정렬, 캡션 들여쓰기 2400·다음 문단과 함께 꺼짐",
             f"다른 단위와 같은 서식으로 통일 (최상위 문단 {len(regen)}개)"))

# ── 2) 문구 수정 ──
REPL = [
    (" · v1.4 클래스 유지", "", "클래스 설명 줄의 내부 표기 'v1.4 클래스 유지' 삭제"),
    ("관제 운용자", "관제 운영자", "용어 통일 (운용자 → 운영자)"),
    ("복원 크롭은 운용자 확인용", "복원 크롭은 운영자 확인용", "용어 통일 (운용자 → 운영자)"),
    ("명령· 영역", "명령·영역", "가운뎃점 뒤 공백"),
    ("단발· 오탐", "단발·오탐", "가운뎃점 뒤 공백"),
    ("위치·색상· 관측", "위치·색상·관측", "가운뎃점 뒤 공백"),
    ("이동· 관측", "이동·관측", "가운뎃점 뒤 공백"),
    ("캡스톤디자인에 대한 요구 명세서를", "캡스톤디자인에 대한 설계 명세서를", "표지 문구 — 요구 명세서 → 설계 명세서"),
    ("첨부 : 캡스톤디자인 요구 명세서", "첨부 : 캡스톤디자인 설계 명세서", "표지 문구 — 요구 명세서 → 설계 명세서"),
]
for t in root.iter(q("t")):
    if not t.text: continue
    s = t.text
    for a, b, why in REPL:
        if a in s:
            s = s.replace(a, b); LOG[why] += 1
    if s != t.text:
        t.text = s; touch(t)
for sub in root.iter(q("subList")):
    ps = sub.findall(q("p"))
    for a, b in zip(ps, ps[1:]):
        ta = [t for t in a.iter(q("t")) if t.text]; tb_ = [t for t in b.iter(q("t")) if t.text]
        if ta and tb_ and ta[-1].text.endswith("·") and tb_[0].text.startswith(" "):
            tb_[0].text = tb_[0].text.lstrip(" "); touch(tb_[0]); LOG["가운뎃점 뒤 공백"] += 1
# 1.3 소제목 — 아래 표 머리가 '구분 | 운용 조건' 이다
for p in top:
    if txt(p).strip() == "- 성능 목표":
        for t in p.iter(q("t")):
            if t.text and "성능 목표" in t.text: t.text = t.text.replace("성능 목표", "운용 조건"); touch(t)
        NOTE.append(("1.3 설계상 제약사항", "- 성능 목표 (아래 표 머리는 '구분 | 운용 조건')", "- 운용 조건")); break
# 6.2 식별자 문장
for p in top:
    s = txt(p).strip()
    if s.startswith("유스케이스는 UC-ggnn"):
        ts = [t for t in p.iter(q("t")) if t.text]
        new = ("유스케이스는 UC-ggnn, 대응 시퀀스는 SD-ggnn, 여러 유스케이스를 잇는 통합 시퀀스는 SD-X01~X09 를 사용한다. "
               "사용자 화면은 UI-01~07, 전체 ERD 는 ERD-00, 운용도는 DD-01, 패키지 다이어그램은 PD-01, 클래스 다이어그램은 관리단위별 CD-01~12, "
               "클래스는 C-ggnn(gg = 관리단위)으로 구분한다. 참조표는 5장에 정리한다.")
        ts[0].text = (ts[0].text[:len(ts[0].text) - len(ts[0].text.lstrip())]) + new
        for t in ts[1:]: t.text = ""
        touch(p); NOTE.append(("6.2 식별자 및 용어", s[:60] + "…", "SD-X · ERD-00 · DD-01 · PD-01 · CD-01~12 식별자 추가")); break
for why, n in LOG.items():
    if why != "서식 번호 바로잡음": NOTE.append(("문구", why, f"{n}곳"))


# ── 공통: 표 열 폭 · 행 높이 ──
def cell_margin(tb, tc):
    m = tc.find(q("cellMargin")) if tc.get("hasMargin") == "1" else tb.find(q("inMargin"))
    return int(m.get("left")) + int(m.get("right")), int(m.get("top")) + int(m.get("bottom"))


def cell_paras(tc):
    return [txt(cp) for cp in tc.find(q("subList")).findall(q("p"))]


def row_metrics(tb):
    """이 표에서 한 줄 행 높이(base)와 줄 간격(pitch)을 한글 줄 배치 정보로 잰다"""
    base, pitch = [], []
    for tr in tb.findall(q("tr")):
        single = True
        for tc in tr.findall(q("tc")):
            for cp in tc.find(q("subList")).findall(q("p")):
                ls = cp.findall(q("linesegarray") + "/" + q("lineseg"))
                if len(ls) != 1: single = False
                for a, b in zip(ls, ls[1:]): pitch.append(int(b.get("vertpos")) - int(a.get("vertpos")))
            if len(tc.find(q("subList")).findall(q("p"))) != 1: single = False
        if single: base.append(max(int(tc.find(q("cellSz")).get("height")) for tc in tr.findall(q("tc"))))
    b = collections.Counter(base).most_common(1)[0][0] if base else None
    pt = collections.Counter(pitch).most_common(1)[0][0] if pitch else None
    return b, pt


def set_widths(tb, widths, base, pitch):
    """열 폭을 바꾸고, 줄 수를 다시 어림해 행 높이(최소 높이)를 정한다 — 모자라면 한글이 늘린다"""
    total = 0
    for tr in tb.findall(q("tr")):
        need = 0
        for tc in tr.findall(q("tc")):
            c = int(tc.find(q("cellAddr")).get("colAddr")); cs = int(tc.find(q("cellSpan")).get("colSpan"))
            wd = sum(widths[c:c + cs]); tc.find(q("cellSz")).set("width", str(wd))
            mlr, mtb = cell_margin(tb, tc)
            n = sum(lines(t, int(wd - mlr - 100)) for t in cell_paras(tc))
            if tc.find(q("cellSpan")).get("rowSpan") == "1":
                need = max(need, base + (n - 1) * pitch)
        need = max(need, base)
        for tc in tr.findall(q("tc")):
            if tc.find(q("cellSpan")).get("rowSpan") == "1": tc.find(q("cellSz")).set("height", str(need))
        total += need
    tb.find(q("sz")).set("width", str(sum(widths)))
    tb.find(q("sz")).set("height", str(total))
    for cp in tb.iter(q("p")): TOUCH.add(cp)
    touch(tb)


def header_cells(tb):
    return [txt(tc).replace(" ", "") for tc in tb.find(q("tr")).findall(q("tc"))]


KIND = {
    ("속성/메소드명", "설명"): ("mem", [10500, 13028, 9500, 11500]),
    ("순서", "호출→반환"): ("step", [2800, 28000, 13728]),
    ("클래스식별자", "Entity·테이블"): ("ent", [5200, 11000, 28328]),
    ("ID", "항목"): ("design", [4008, 8800, 31720]),
}
widths_5 = [8000, 5600, 9500, 5600, 5600, 10228]          # 5장: 식별자 열을 줄이고 CD 열을 넓힌다
cnt = collections.Counter()
regen_ids = {id(p) for p in regen}
targets = []
for p in top:
    tb = p.find(q("run") + "/" + q("tbl"))
    if tb is None: continue
    hc = header_cells(tb)
    kind = KIND.get(tuple(hc[:2]))
    if kind is None and hc and hc[0].startswith("유스케이스다이어그램"):
        kind = ("ch5", widths_5)
    if kind is None or len(kind[1]) != int(tb.get("colCnt")): continue
    targets.append((p, tb, kind))
metric = collections.defaultdict(lambda: ([], []))
for p, tb, (name, wd) in targets:                     # 종류별 한 줄 높이·줄 간격 — 정상 서식 표에서만
    if id(p) in regen_ids: continue
    b, pt = row_metrics(tb)
    if b: metric[name][0].append(b)
    if pt: metric[name][1].append(pt)
MET = {k: (collections.Counter(v[0]).most_common(1)[0][0], collections.Counter(v[1]).most_common(1)[0][0]) for k, v in metric.items() if v[0] and v[1]}
print("종류별 (한 줄 행 높이, 줄 간격):", MET)
for p, tb, (name, wd) in targets:
    base, pitch = MET.get(name, MET.get("mem", (2400, 1400)))
    set_widths(tb, wd, base, pitch); cnt[name] += 1
LABEL = {"mem": "클래스 명세표 (속성/메소드 · 설명 · 매개변수 · 타입)", "step": "시퀀스 처리표 (순서 · 호출 → 반환 · 처리 내용)",
         "dto": "DTO 표", "ent": "Entity 표", "design": "1.3 설계 기준 표 (ID · 항목 · 적용 내용)", "ch5": "5장 요구분석 참조표"}
OLDW = {"mem": "11000 · 17528 · 8000 · 8000", "step": "2800 · 22400 · 19328", "dto": "5788 · 13358 · 25381", "ent": "5788 · 13358 · 25381",
        "design": "4008 · 13803 · 26717", "ch5": "8700 · 5600 · 11000 · 6500 · 6500 · 6228"}
NEWW = {v[0]: " · ".join(map(str, v[1])) for v in KIND.values()}; NEWW["ch5"] = " · ".join(map(str, widths_5))
for k, n in cnt.items():
    NOTE.append((LABEL[k], f"열 폭 {OLDW[k]}", f"열 폭 {NEWW[k]} ({n}개 표) · 행 높이 다시 계산"))

# ── 3) 용어 해설 보강 — 마지막 용어 표에 3행 ──
gloss = [p for p in top if (p.find(q("run") + "/" + q("tbl")) is not None and header_cells(p.find(q("run") + "/" + q("tbl")))[:2] == ["용어", "설명"])]
if gloss:
    tb = gloss[-1].find(q("run") + "/" + q("tbl"))
    last = tb.findall(q("tr"))[-1]
    base, pitch = row_metrics(tb)
    ADD = [("GRACE 크롭 제공 / CROP_GRACE · CROP_BASELINE",
            "서버가 원본 영상으로 만든 대상 크롭을 GRACE 로 인코딩·패킷화해 관제 단말에 보내는 방식이 CROP_GRACE, 조건을 만족하지 못할 때 쓰는 기존 크롭 제공 방식이 CROP_BASELINE 이다. 복원 크롭은 운영자 확인용이다."),
           ("FULL / PARTIAL / NONE",
            "크롭 프레임 패킷의 수신 상태이다. FULL 은 모두 받음, PARTIAL 은 일부만 받아 복원함, NONE 은 받은 패킷이 없어 복원하지 않음을 뜻한다."),
           ("Controller / Service / DAO / DTO / Entity",
            "요청을 받는 입구, 업무를 처리하는 인터페이스와 구현, 테이블 저장·조회, 계층 사이 전달 객체, 테이블 한 행을 나타내는 객체이다.")]
    for a, b in ADD:
        tr = copy.deepcopy(last)
        tcs = tr.findall(q("tc")); n = 1
        for tc, text in zip(tcs, (a, b)):
            sub = tc.find(q("subList")); ps = sub.findall(q("p"))
            for x in ps[1:]: sub.remove(x)
            p0 = ps[0]; p0.set("id", new_pid())
            runs = p0.findall(q("run"))
            for r in runs[1:]: p0.remove(r)
            for ch in list(runs[0]):
                if ch.tag == q("t"): runs[0].remove(ch)
            etree.SubElement(runs[0], q("t")).text = text
            for ls in p0.findall(q("linesegarray")): p0.remove(ls)
            mlr, _ = cell_margin(tb, tc)
            n = max(n, lines(text, int(int(tc.find(q("cellSz")).get("width")) - mlr - 100)))
        for tc in tcs: tc.find(q("cellSz")).set("height", str(base + (n - 1) * pitch))
        tb.append(tr)
    trs = tb.findall(q("tr"))
    for i, tr in enumerate(trs):
        for tc in tr.findall(q("tc")): tc.find(q("cellAddr")).set("rowAddr", str(i))
    tb.set("rowCnt", str(len(trs)))
    tb.find(q("sz")).set("height", str(sum(int(tr.find(q("tc")).find(q("cellSz")).get("height")) for tr in trs)))
    touch(tb)
    NOTE.append(("6.2 용어 표", "GRACE 관련 용어·계층 용어 없음", "3행 추가 (GRACE 크롭 제공 · FULL/PARTIAL/NONE · Controller/Service/DAO/DTO/Entity)"))

# ── 5) 클래스 설명 줄은 다음 표와 같은 쪽에 ──
pp = {e.get("id"): e for e in hdr.iter(h("paraPr"))}
props = hdr.find(".//" + h("paraProperties"))
clone = {}
desc_re = re.compile(r"^(«[^»]+» · )?[a-z_.]+ · CD-\d\d")
moved = 0
for i, p in enumerate(top[:-1]):
    if desc_re.match(txt(p).strip()) and top[i + 1].find(q("run") + "/" + q("tbl")) is not None:
        old = p.get("paraPrIDRef")
        if pp[old].find(h("breakSetting")).get("keepWithNext") == "1": continue
        if old not in clone:
            e = copy.deepcopy(pp[old]); nid = str(len(props.findall(h("paraPr"))))
            e.set("id", nid); e.find(h("breakSetting")).set("keepWithNext", "1")
            props.append(e); props.set("itemCnt", str(len(props.findall(h("paraPr"))))); clone[old] = nid
        p.set("paraPrIDRef", clone[old]); moved += 1
NOTE.append(("클래스 명세 배치", "클래스 설명 줄과 표가 다른 쪽으로 갈 수 있음", f"설명 줄 {moved}개에 '다음 문단과 함께' 적용 (클래스명 → 설명 → 표가 같은 쪽에서 시작)"))

# ── 6) 줄 배치 정보 정리 · 저장 ──
for p in TOUCH:
    for ls in p.findall(q("linesegarray")): p.remove(ls)
# 안전장치: 원본과 조금이라도 다른 문단(서식 번호만 바뀐 것 포함)은 줄 배치 정보를 지운다 — 안쪽부터
ORIG = {}
for p0 in etree.fromstring(zin.read("Contents/section0.xml")).iter(q("p")):
    ORIG.setdefault(p0.get("id"), []).append(etree.tostring(p0))
extra = 0
for p in sorted(root.iter(q("p")), key=lambda e: -len(list(e.iterancestors()))):
    if p.find(q("linesegarray")) is not None and etree.tostring(p) not in ORIG.get(p.get("id"), []):
        for ls in p.findall(q("linesegarray")): p.remove(ls)
        extra += 1
        a = p.getparent()
        while a is not None:
            if a.tag == q("p"):
                for ls in a.findall(q("linesegarray")): a.remove(ls)
            a = a.getparent()
print("추가로 줄 배치 정보를 지운 문단", extra)
sec = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)
hdr_b = etree.tostring(hdr, xml_declaration=True, encoding="UTF-8", standalone=True)
zout = zipfile.ZipFile(DST, "w")
for info in zin.infolist():
    fn = info.filename
    data = sec if fn == "Contents/section0.xml" else hdr_b if fn == "Contents/header.xml" else zin.read(fn)
    zi = zipfile.ZipInfo(fn, date_time=info.date_time)
    zi.create_system, zi.create_version = info.create_system, info.create_version
    zi.compress_type = zipfile.ZIP_STORED if (fn in ("mimetype", "version.xml") or fn.endswith(".png")) else zipfile.ZIP_DEFLATED
    zout.writestr(zi, data)
zout.close()
json.dump({"note": NOTE, "log": dict(LOG), "touched": len(TOUCH), "tables": dict(cnt)}, open("final.json", "w"), ensure_ascii=False, indent=1)
print(dict(cnt), "· 줄 배치 지운 문단", len(TOUCH), "· 저장", DST)
