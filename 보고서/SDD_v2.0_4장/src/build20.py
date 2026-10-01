"""build20.py — v1.4 (1).hwpx 의 4장을 객체지향 계층판으로 교체 → v2.0.hwpx
   1·2·3장 그대로 · 4.1 운용도 유지(그림만 무채색) · 4.2 클래스(패키지) · 4.3 시퀀스 · 4.4 클래스 설계 · 5장 CD 열 · 목차 · 6.2 문장"""
import copy, json, math, re, zipfile, importlib, io
from lxml import etree
from PIL import Image, ImageOps
from mdl import C, SEQS, CDS, UNITS, OLD
from seq import flatten

SRC = "/home/se/JupyterLAB/Capstone/3. 설계명세서_서성훈_이시원_정서인_v1.4 (1).hwpx"
DST = "/home/se/JupyterLAB/Capstone/3. 설계명세서_서성훈_이시원_정서인_v2.0.hwpx"
for i in range(1, 13):
    importlib.import_module(f"u{i:02d}")
SIZES = {}
HP = "http://www.hancom.co.kr/hwpml/2011/paragraph"; HC = "http://www.hancom.co.kr/hwpml/2011/core"
NS = {"hp": HP, "hc": HC}
q = lambda t: "{%s}%s" % (HP, t)

zin = zipfile.ZipFile(SRC)
root = etree.fromstring(zin.read("Contents/section0.xml"))
paras = list(root.iterfind("hp:p", NS))
TPL = {k: copy.deepcopy(paras[i]) for k, i in dict(h1=132, h2=141, h3=150, h4=151, cap=152, img=153, body=142, lst=154,
                                                       kv=157, mem=158, sdh=177, step=179, lab=180, ctx=181, toc1=19, toc2=20, toc3=22).items()}
_ids = [int(e.get("id")) for e in root.iter() if e.get("id", "").isdigit()]
NEXT = [max(_ids) + 10]


def nid():
    NEXT[0] += 1
    return str(NEXT[0])


def strip_ls(e):
    for ls in e.findall(".//" + q("linesegarray")):
        ls.getparent().remove(ls)


def set_text(p, text):
    runs = p.findall(q("run"))
    for r in runs[1:]:
        p.remove(r)
    r = runs[0]
    for ch in list(r):
        if ch.tag in (q("t"),):
            r.remove(ch)
    t = etree.SubElement(r, q("t")); t.text = text
    strip_ls(p)
    p.set("id", nid())
    return p


def para(kind, text, pb=None):
    p = copy.deepcopy(TPL[kind]); set_text(p, text)
    if pb is not None:
        p.set("pageBreak", "1" if pb else "0")
    return p


# ── 그림 ──
BIN = {}                 # 새 BinData 이름 → bytes
MAXH = 64000             # 쪽 본문 높이 72188 에서 캡션·여백을 뺀 그림 최대 높이


def img_para(png_path, name):
    im = Image.open(png_path)
    w, h = im.size
    nw = 44528; nh = round(nw * h / w)
    if nh > MAXH:
        nh = MAXH; nw = round(MAXH * w / h)
    buf = io.BytesIO(); im.convert("RGB").save(buf, "PNG", optimize=True)
    BIN[name] = buf.getvalue()
    p = copy.deepcopy(TPL["img"])
    pic = p.find(".//" + q("pic"))
    pic.set("id", nid()); pic.set("instid", nid())
    def st(tag, **kv):
        e = pic.find(q(tag))
        for k, v in kv.items(): e.set(k, str(v))
    st("orgSz", width=nw, height=nh)
    st("rotationInfo", centerX=nw // 2, centerY=nh // 2)
    rect = pic.find(q("imgRect")); pts = list(rect)
    for pt, (x, y) in zip(pts, [(0, 0), (nw, 0), (nw, nh), (0, nh)]):
        pt.set("x", str(x)); pt.set("y", str(y))
    st("imgClip", left=0, right=w * 75, top=0, bottom=h * 75)
    st("imgDim", dimwidth=w * 75, dimheight=h * 75)
    st("sz", width=nw, height=nh)
    pic.find(".//{%s}img" % HC).set("binaryItemIDRef", name)
    sc = pic.find(q("shapeComment"))
    if sc is not None: sc.text = name
    strip_ls(p); p.set("id", nid())
    return p, nh


# ── 표 ──
def text_w(s):
    return sum(950 if ord(ch) > 0x2E80 else 520 for ch in s)


def fill_cell(tc, text):
    sub = tc.find(q("subList"))
    ps = sub.findall(q("p"))
    for x in ps[1:]:
        sub.remove(x)
    lines = text.split("\n") if text else [""]
    first = ps[0]; set_text(first, lines[0])
    for ln in lines[1:]:
        p2 = copy.deepcopy(first); set_text(p2, ln); sub.append(p2)
    w = int(tc.find(q("cellSz")).get("width")) - 800
    nl = sum(max(1, math.ceil(text_w(ln) / max(w, 1000))) for ln in lines)
    return 800 + 1400 * nl


def table(kind, rows, header=None, widths=None):
    """kind: lst · mem · step (머리행 있음) · kv · sdh · ctx (머리행 없음) — rows: [[셀 문자열…]]"""
    p = copy.deepcopy(TPL[kind])
    tbl = p.find(".//" + q("tbl")); tbl.set("id", nid())
    trs = tbl.findall(q("tr"))
    has_h = kind in ("lst", "mem", "step")
    hdr, body = (trs[0], trs[-1]) if has_h else (None, trs[-1])
    for tr in trs:
        tbl.remove(tr)
    out = []
    if has_h:
        h = copy.deepcopy(hdr)
        if header:
            for tc, txt in zip(h.findall(q("tc")), header):
                fill_cell(tc, txt)
        out.append(h)
    if widths:
        for row_ in ([hdr] if has_h else []) + [body]:
            for tc, w_ in zip(row_.findall(q("tc")), widths):
                tc.find(q("cellSz")).set("width", str(w_))
        if has_h:
            out[0] = copy.deepcopy(hdr)
            if header:
                for tc, txt in zip(out[0].findall(q("tc")), header):
                    fill_cell(tc, txt)
    for r in rows:
        tr = copy.deepcopy(body)
        tcs = tr.findall(q("tc"))
        hh = 0
        for tc, txt in zip(tcs, r):
            hh = max(hh, fill_cell(tc, txt))
        for tc in tcs:
            tc.find(q("cellSz")).set("height", str(max(hh, 2200)))
        out.append(tr)
    total = 0
    for i, tr in enumerate(out):
        for tc in tr.findall(q("tc")):
            tc.find(q("cellAddr")).set("rowAddr", str(i))
        total += int(tr.find(q("tc")).find(q("cellSz")).get("height"))
        tbl.append(tr)
    tbl.set("rowCnt", str(len(out)))
    tbl.find(q("sz")).set("height", str(total))
    strip_ls(p); p.set("id", nid())
    return p, total


# ── 4장 조립 ──
OUT = []                 # (element, 높이 추정, 쪽나눔, 목차항목)
STEREO = {"controller": "«controller»", "interface": "«interface»", "service": "«service»", "component": "", "dao": "«DAO»",
          "entity": "«entity»", "dto": "«DTO»", "tool": "«tool»", "value": "«value»", "boundary": "«boundary»", "viewmodel": "«view model»"}
ORDER = ["controller", "boundary", "interface", "service", "component", "value", "viewmodel", "tool", "dao", "entity", "dto"]


def add(el, h, pb=False, toc=None):
    OUT.append((el, h, pb, toc))


def H(level, text, pb=False):
    kind = {2: "h2", 3: "h3", 4: "h4"}[level]
    add(para(kind, text, pb), 3000 if level <= 3 else 2300, pb, (level, text) if level <= 3 else None)


def body(text):
    n = max(1, math.ceil(text_w(text) / 44000))
    add(para("body", text), 600 + 1700 * n)


def cap(text):
    add(para("cap", text), 2300)


def img(path, name):
    p, h = img_para(path, name); add(p, h + 1200)


def tbl(kind, rows, header=None, widths=None):
    p, h = table(kind, rows, header, widths); add(p, h + 900)


OUT2 = "out2"
ucls = lambda u: sorted([c for c in C.values() if c["unit"] == u], key=lambda c: (ORDER.index(c["kind"]), c["id"]))
cd_of = lambda cid: f"CD-{C[cid]['unit']}"

# 4.2 Class Diagram
H(2, "4.2 Class Diagram(클래스 다이어그램)", pb=True)
for t in ["클래스 다이어그램은 소스 파일 구성을 나타낸다. 패키지는 폴더, 클래스는 파일이며 2.1 파일 구조의 논리 경로를 패키지 단위로 세분한다.",
          "계층: api(Controller) → services(«interface» I… ← 구현 → 구성요소) → storage.dao(DAO) → storage.entity(Entity). 계층 사이 자료는 contracts 의 DTO 로 주고받고, Entity 는 서비스 밖으로 내보내지 않는다.",
          "Controller 와 다른 패키지는 서비스의 인터페이스만 import 한다. HTTP·WSS·SRT 통신은 관계로 그리지 않고 Controller·EventPublisher·IGatewayLink(GatewayClient)·ServerReporter 가 경계를 맡는다.",
          "DAO 는 ERD 18개 테이블과 1:1 이며 SQL 은 DAO 에만 둔다. 게이트웨이(Pi)의 로컬 기록은 gateway.storage 의 DAO 가 맡는다 (중앙 DB 아님).",
          "표기: 속성 -, 공개 연산 +, 비공개 연산 -. 실선 화살표는 연관(보유), 점선 화살표는 의존(import), 점선 빈 삼각형은 구현, 채운 마름모는 구성, 점선 상자는 다른 도식의 클래스다. 모든 실패는 사유 코드·현재 상태·관측 시각을 가진 ResultDTO 또는 판정 객체로 반환한다."]:
    body(t)
H(3, "4.2.1 패키지 다이어그램(Package Diagram)")
cap("PD-01 — 전체 패키지와 계층 의존")
img(f"{OUT2}/pd01.png", "image900")
PKG_ROWS = [
    ("api", "services/api", "«controller» REST 입구 · EventPublisher(WebSocket) · GatewayController(Pi 보고)"),
    ("contracts", "shared/contracts", "«DTO» 계층·패키지 사이 전달 객체 (웹과 스키마 공유)"),
    ("services.auth", "services/auth", "사용자·제어권 — IAuthService · IControlAuthority · IPermissionPolicy"),
    ("services.vehicle", "services/vehicle", "기체·게이트웨이 상태 — IVehicleStateService · ILinkMonitor · IPreflightService"),
    ("services.spatial", "services/spatial", "지도·좌표 기준 — ISpatialDataService · ICoordinateTransform"),
    ("services.mission", "services/mission", "임무 실행 — IMissionService · IMissionRunner"),
    ("services.mission.global · .local", "services/mission/global · local", "전역 방문 계획 · 구역 IPP — IViewpointPlanner (계획 패키지 단일 진입점)"),
    ("services.media", "services/media", "영상 수신·저장 — IVideoIngestService · IMediaStore · IFrameSource · IFrameAnalysisLedger"),
    ("services.vision", "services/vision", "사람 탐지·좌표·후보·추적 — IPersonDetectionService · ITargetService · ITargetGeoLocator · ITargetTrackingService · IModelRegistry"),
    ("services.mission_map", "services/mission_map", "상황지도·관측·완료도 — IMissionMapService · ICameraModel"),
    ("services.command", "services/command", "비행 명령 — ICommandService · IGatewayLink(GatewayClient)"),
    ("services.safety · alert · history", "services/safety · alert · history", "안전·복구 · 알림 · 이력·결과 — ISafetySupervisor · IRecoveryCoordinator · IAlertService · IHistoryService · IStateSyncService"),
    ("policies", "shared/policies", "서버·Pi 공유 규칙 — CommandValidator · SafetyPolicy · DetectionLikelihoodModel"),
    ("storage.dao · storage.entity", "storage/", "«DAO» 18 · «entity» 18 — DB-01~18 과 1:1"),
    ("gateway.*", "gateway/ (Pi)", "api · command · flight · media · storage — 현장 명령 검증·MAVLink·현장 기록"),
    ("web", "apps/web", "«view model» 화면 상태 — api 만 호출"),
    ("vision_train · research.grace", "vision-train/ · research/grace/", "오프라인 학습(«tool») · GRACE 연구 경로 — 운용 경로 밖"),
]
tbl("lst", [list(r) for r in PKG_ROWS], header=["패키지", "경로", "책임 · 공개 인터페이스"])

for n, u in enumerate([f"{i:02d}" for i in range(1, 13)], start=2):
    H(3, f"4.2.{n} {UNITS[u]['title']}")
    cd = CDS[f"CD-{u}"]
    parts = cd["parts"]() if callable(cd["parts"]) else cd["parts"]
    for key, sub, *_ in parts:
        cap(f"CD-{u} — {UNITS[u]['cd_title']}" + (f" {sub}" if sub else ""))
        img(f"{OUT2}/{key}.png", f"image9{u}{key[-1] if key[-1].isalpha() else '0'}")
    body(f"클래스 {len(ucls(u))}개 — 식별자·책임·속성·연산은 4.4.{n - 1} 에 정리한다.")

# 4.3 Sequence Diagram
H(2, "4.3 Sequence Diagram(시퀀스 다이어그램)", pb=True)
for t in ["73개 SD 는 같은 번호의 UC 에 대응하고, SD-X01~X09 는 여러 유스케이스를 잇는 통합 시퀀스다. 첫 생명선은 실제 시작 주체(관제 운영자·현장 조종자·호출 서비스)이며 Controller 가 요청을 받는다.",
          "실선 화살표는 동기 호출, 점선 화살표는 반환(DTO·Entity)이다. 번호는 아래 처리표와 같다. 서비스는 호출자가 의존하는 인터페이스 이름(I…)으로 표시하고, 저장·조회는 DAO 를 거친다. alt·opt·loop 는 예외·선택·반복 흐름이다."]:
    body(t)
for n, u in enumerate([f"{i:02d}" for i in range(1, 13)], start=1):
    H(3, f"4.3.{n} {UNITS[u]['title']}")
    sds = sorted(UNITS[u]["sds"], key=lambda s: (s.startswith("SD-X"), s))
    for sd in sds:
        s = SEQS[sd]; inf = s["info"]
        cap(f"{sd} — {s['title']}")
        x = sd.startswith("SD-X")
        tbl("ctx", [["유스케이스 연결" if x else "유스케이스", inf.get("uc", "") + " — " + inf.get("개요", "")],
                    ["시작 주체/사건", inf.get("시작", "")], ["선행 조건", inf.get("선행", "")], ["사후 조건", inf.get("사후", "")],
                    ["예외/대안", inf.get("예외", "")], ["경계", inf.get("경계", "").replace(" | ", " · ")]], widths=[7800, 36728])
        img(f"{OUT2}/{sd}.png", "image" + sd.replace("SD-", "").replace("X", "8") + "1")
        _, steps = flatten(s)
        tbl("step", [list(r) for r in steps], header=["순서", "호출 → 반환", "처리 내용"], widths=[2800, 22400, 19328])

# 4.4 Design Classes
H(2, "4.4 Design Classes(클래스 설계)", pb=True)
body("관리단위별로 Controller · 인터페이스 · 구현 · 구성요소 · DAO 를 클래스마다 명세하고, DTO 와 Entity 는 단위마다 한 표로 정리한다. "
     "인터페이스는 공개 연산을, 구현 클래스는 의존 속성과 비공개 연산을 적는다 (공개 연산은 인터페이스 표와 같다). Entity 의 전체 컬럼은 2장 ERD 와 같다.")
for n, u in enumerate([f"{i:02d}" for i in range(1, 13)], start=1):
    H(3, f"4.4.{n} {UNITS[u]['title']}")
    cl = ucls(u)
    for c in [c for c in cl if c["kind"] not in ("dto", "entity")]:
        cap(f"{c['id']} {c['name']}")
        st = STEREO[c["kind"]]
        rel = ""
        if c["kind"] == "interface" and c.get("impl"):
            rel = f" · 구현 클래스: {C[c['impl']]['name']} ({c['impl']})"
        elif c.get("impl"):
            rel = f" · 구현 인터페이스: {C[c['impl']]['name']} ({c['impl']})"
        if c.get("old"):
            rel += " · v1.4 클래스 유지"
        body(f"{st + ' · ' if st else ''}{c['pkg']} · {cd_of(c['id'])}{rel} — {c['resp']}")
        mem = [[f"- {a[0]}", a[2] or "—", "—", a[1]] for a in c["attrs"]]
        mem += [[("- " if o[0].startswith("-") else "+ ") + o[0].lstrip("-") + "()", o[3], o[1] or "없음", o[2] or "void"] for o in c["ops"]]
        if c.get("impl") and c["kind"] not in ("interface",) and not any(not o[0].startswith("-") for o in c["ops"]):
            mem.append([f"+ ({C[c['impl']]['name']} 연산)", f"{c['impl']} {C[c['impl']]['name']} 의 공개 연산을 구현한다.", "—", "—"])
        if mem:
            tbl("mem", mem, header=["속성/메소드 명", "설명", "매개변수", "타입"], widths=[11000, 17528, 8000, 8000])
    dtos = [c for c in cl if c["kind"] == "dto"]
    if dtos:
        cap(f"{UNITS[u]['title']} — DTO (contracts)")
        tbl("lst", [[c["id"], c["name"] + "\n" + c["resp"], "\n".join(f"{a[0]} : {a[1]}" + (f" — {a[2]}" if a[2] else "") for a in c["attrs"])] for c in dtos],
            header=["클래스 식별자", "DTO", "필드"])
    ents = [c for c in cl if c["kind"] == "entity"]
    if ents:
        cap(f"{UNITS[u]['title']} — Entity (storage.entity)")
        rows = []
        for c in ents:
            f = ", ".join(a[0] for a in c["attrs"])
            ops = "\n".join(f"+ {o[0]}({o[1]}) : {o[2]} — {o[3]}" for o in c["ops"])
            rows.append([c["id"], f"{c['name']}\n{c['db']}", f"필드 {len(c['attrs'])}개: {f}" + (f"\n{ops}" if ops else "")])
        tbl("lst", rows, header=["클래스 식별자", "Entity · 테이블", "필드 · 연산"])

# ── 쪽 추정 (목차용) ──
BODY_H = 84188 - 4000 - 4000 - 2000 - 2000
page = 24; used = 0; TOC = []
first = True
for el, h, pb, toc in OUT:
    if pb and not first:
        page += 1; used = 0
    first = False
    if used + h > BODY_H and el.find(".//" + q("tbl")) is None:
        page += 1; used = 0
    if toc:
        TOC.append((toc[0], toc[1], page))
    if el.find(".//" + q("tbl")) is not None and used + h > BODY_H:
        h2 = h - (BODY_H - used); page += 1 + int(h2 // BODY_H); used = h2 % BODY_H
    else:
        used += h
ch4_end = page
print("4장 추정 쪽:", 24, "~", ch4_end, f"({ch4_end - 24 + 1}쪽)")

# ── section0 교체 ──
first_old, last_old = 141, 1192         # 4.2 설계 모델 ~ SD-X05 끝 (4.1 은 유지)
anchor = paras[first_old]
parent = anchor.getparent()
idx = parent.index(anchor)
for p in paras[first_old:last_old + 1]:
    parent.remove(p)
for k, (el, *_rest) in enumerate(OUT):
    parent.insert(idx + k, el)

# 5장 CD 열
names = {c["name"]: c["id"] for c in C.values()}
def sd_cds(sd):
    s = SEQS[sd]; out = {f"CD-{s['unit']}"}
    for _, nm, kd in s["parts"]:
        for part in nm.replace("\n", "").split(" · "):
            part = part.split(" (")[0].strip()
            if part in names: out.add(cd_of(names[part]))
    return " ".join(sorted(out))
paras2 = list(root.iterfind("hp:p", NS))
changed5 = 0
for p in paras2:
    tb = p.find(".//" + q("tbl"))
    if tb is None: continue
    for tr in tb.findall(q("tr")):
        tcs = tr.findall(q("tc")); txt = ["".join(x.text or "" for x in tc.iter(q("t"))) for tc in tcs]
        sdv = [t for t in txt if re.fullmatch(r"SD-\d{4}", t.strip())]
        if sdv and len(tcs) >= 3 and txt[-1].strip().startswith("CD-") and sdv[0].strip() in SEQS:
            fill_cell(tcs[-1], sd_cds(sdv[0].strip())); changed5 += 1
print("5장 CD 열 갱신:", changed5)

# 6.2 문장
for p in paras2:
    t = "".join(x.text or "" for x in p.iter(q("t")))
    if t.startswith("유스케이스는 UC-ggnn"):
        set_text(p, "유스케이스는 UC-ggnn, 대응 시퀀스는 SD-ggnn 을 사용한다. 사용자 화면은 UI-01~07, 테이블은 DB-01~18, 패키지 다이어그램은 PD-01, "
                    "클래스 다이어그램은 관리단위별 CD-01~12, 클래스는 C-ggnn(gg = 관리단위)으로 구분한다. 계층은 Controller · 인터페이스(I…) · 구현 · DAO · Entity · DTO 이다. 참조표는 5장에 정리한다.")
    if t.startswith("73개 유스케이스는 같은 숫자부의 SD에"):
        set_text(p, "73개 유스케이스는 같은 숫자부의 SD 에 연결한다. SD-X01~X09 는 여러 유스케이스를 연결하는 통합 시퀀스이다. 클래스 다이어그램 열은 시퀀스에 등장하는 클래스가 속한 관리단위의 CD 이다.")

# 목차 (문단 19~71 → 새 항목)
toc_old = paras2[19:72]
tparent = toc_old[0].getparent(); tidx = tparent.index(toc_old[0])
def toc_para(kind, text, pg):
    p = copy.deepcopy(TPL[kind]); t = p.find(".//" + q("t"))
    tab = t.find(q("tab"))
    t.text = text; tab.tail = str(pg)
    strip_ls(p); p.set("id", nid())
    return p
items = [toc_para("toc1", "4. 객체지향설계", 24), toc_para("toc2", "4.1 Deployment Diagram(시스템 운용도)", 24)]
for lv, text, pg in TOC:
    items.append(toc_para("toc2" if lv == 2 else "toc3", text, pg))
for p in toc_old:
    tparent.remove(p)
for k, p in enumerate(items):
    tparent.insert(tidx + k, p)
# 5·6 장 쪽 번호
for p in root.iterfind("hp:p", NS):
    t = p.find(".//" + q("t"))
    if t is None or t.find(q("tab")) is None: continue
    head_ = (t.text or "")
    if head_.startswith("5. 요구분석 참조표"): t.find(q("tab")).tail = str(ch4_end + 1)
    if head_.startswith("6. 참고 자료"): t.find(q("tab")).tail = str(ch4_end + 8)

# ── 패키지 쓰기 ──
sec = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)
used_refs = set(re.findall(rb'binaryItemIDRef="([^"]+)"', sec))
hpf = zin.read("Contents/content.hpf").decode("utf-8")
items_old = re.findall(r'<opf:item id="(image\d+)" href="BinData/[^"]+"[^>]*/>', hpf)
for iid in items_old:
    if iid.encode() not in used_refs:
        hpf = re.sub(r'<opf:item id="%s" href="[^"]+"[^>]*/>' % iid, "", hpf)
new_items = "".join(f'<opf:item id="{k}" href="BinData/{k}.png" media-type="image/png" isEmbeded="1"/>' for k in BIN)
hpf = hpf.replace("</opf:manifest>", new_items + "</opf:manifest>")
hpf = hpf.replace("설계 명세서 v1.1", "설계 명세서 v2.0")
zout = zipfile.ZipFile(DST, "w")
for info in zin.infolist():
    fn = info.filename
    if fn == "Contents/section0.xml":
        data = sec
    elif fn == "Contents/content.hpf":
        data = hpf.encode("utf-8")
    elif fn.startswith("BinData/"):
        iid = fn.split("/")[1].rsplit(".", 1)[0]
        if iid.encode() not in used_refs:
            continue
        data = zin.read(fn)
        if iid == "image14":                                   # 4.1 운용도 무채색
            im = Image.open(io.BytesIO(data)); g = ImageOps.grayscale(im).convert("RGB")
            b = io.BytesIO(); g.save(b, im.format or "BMP"); data = b.getvalue()
    else:
        data = zin.read(fn)
    zi = zipfile.ZipInfo(fn, date_time=info.date_time)
    zi.compress_type = zipfile.ZIP_STORED if fn == "mimetype" else zipfile.ZIP_DEFLATED
    zout.writestr(zi, data)
for k, v in BIN.items():
    zi = zipfile.ZipInfo(f"BinData/{k}.png", date_time=(2026, 10, 1, 12, 0, 0)); zi.compress_type = zipfile.ZIP_DEFLATED
    zout.writestr(zi, v)
zout.close()
print("그림", len(BIN), "· 4장 요소", len(OUT), "· 저장", DST)
json.dump({"toc": TOC, "ch4_end": ch4_end}, open("build20.json", "w"), ensure_ascii=False)
