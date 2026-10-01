"""patch22.py — v2.1 에 GRACE 적용 구간 변경(서버 → 관제 운용자 단말 크롭 제공)을 반영한다
   python patch22.py <입력 v2.1.hwpx> <출력.hwpx>
   2.1 파일 구조 · 4.1 DD-01 · 공통 기준 · PD-01 · 패키지 표 · 4.2.6(CD-06 · 클래스 명세 · SD-X06) · 4.2.7(CD-07 (1/4) · 클래스 명세)"""
import copy, io, math, re, sys, zipfile
from lxml import etree
from PIL import Image

SRC, DST = sys.argv[1], sys.argv[2]
HERE = __file__.rsplit("/", 1)[0] if "/" in __file__ else "."

# ── build20 의 도우미(문단·표·그림 틀)를 그대로 쓴다 — 틀 문단은 v1.4 (1) 원본에서 온다 ──
B = {"__name__": "b20"}
code = open(f"{HERE}/build20.py").read()
exec(code.split("# ── 4장 조립 ──")[0], B)
exec("\n".join(l for l in code.split("# ── 4장 조립 ──")[1].split("# 4.2 Class Diagram")[0].split("\n")), B)   # OUT · STEREO · ORDER · H/body/cap/img/tbl
C, SEQS, CDS, UNITS = B["C"], B["SEQS"], B["CDS"], B["UNITS"]
from seq import flatten

# patch21 의 표기 변환(액터 · DB-* 제거)을 새 요소에도 똑같이 적용
p21 = open(f"{HERE}/patch21.py").read()
P21 = {}
exec(p21.split("cnt = {")[0].split("SRC =")[0] + "\n" + p21[p21.index("ACT = {"):p21.index("cnt = {")], P21)
ACT, SUBS = P21["ACT"], P21["SUBS"]

HP = B["HP"]; q = B["q"]; NS = B["NS"]
zin = zipfile.ZipFile(SRC)
root = etree.fromstring(zin.read("Contents/section0.xml"))
_ids = [int(e.get("id")) for e in root.iter() if e.get("id", "").isdigit()]
B["NEXT"][0] = max(_ids) + 10
txt = lambda e: "".join(t.text or "" for t in e.iter(q("t")))
ctext = lambda tc: " ".join(txt(p).strip() for p in tc.find(q("subList")).findall(q("p")) if txt(p).strip())   # 셀 안 줄나눔 문단은 띄어 잇는다
LOG = []                                     # 변경 기록 (위치 · 전 · 후)


def conv(el):
    for t in el.iter(q("t")):
        if not t.text: continue
        s = t.text
        if s.strip() in ACT: s = s.replace(s.strip(), ACT[s.strip()])
        for a, b in SUBS:
            s = s.replace(a, b)
        s = re.sub(r"DB-\d{2} (?=[a-z_]+)", "", s)
        t.text = s
    return el


def top():
    return list(root.iterfind(q("p")))


def find_p(text, start=0, exact=True):
    for i, p in enumerate(top()):
        if i < start: continue
        t = txt(p)
        if (t.strip() == text) if exact else t.strip().startswith(text):
            return i, p
    raise KeyError(text)


def replace_range(i0, i1, new_els):
    """top() 의 [i0, i1) 문단을 new_els 로 바꾼다"""
    P = top(); parent = P[i0].getparent(); idx = parent.index(P[i0])
    for p in P[i0:i1]:
        parent.remove(p)
    for k, el in enumerate(new_els):
        parent.insert(idx + k, conv(el))


def gen(fn):
    B["OUT"].clear(); fn(); return [el for el, *_ in B["OUT"]]


ucls = lambda u: sorted([c for c in C.values() if c["unit"] == u], key=lambda c: (B["ORDER"].index(c["kind"]), c["id"]))
cd_of = lambda cid: f"CD-{C[cid]['unit']}"
IMGS = {}                                    # 그림 키 → BinData 이름


def cd_images(u, only=None):
    cd = CDS[f"CD-{u}"]; parts = cd["parts"]() if callable(cd["parts"]) else cd["parts"]
    for key, sub, *_ in parts:
        if only and key not in only: continue
        B["cap"](f"CD-{u} — {UNITS[u]['cd_title']}" + (f" {sub}" if sub else ""))
        name = f"image22{key[2:]}"; IMGS[key] = name
        B["img"](f"out2/{key}.png", name)


def class_specs(u):
    STEREO = B["STEREO"]; body, cap, tbl = B["body"], B["cap"], B["tbl"]
    cl = ucls(u)
    for c in [c for c in cl if c["kind"] not in ("dto", "entity")]:
        cap(f"{c['id']} {c['name']}")
        st = STEREO[c["kind"]]; rel = ""
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


def sd_block(sd):
    s = SEQS[sd]; inf = s["info"]
    B["cap"](f"{sd} — {s['title']}")
    B["tbl"]("ctx", [["유스케이스 연결", inf.get("uc", "") + " — " + inf.get("개요", "")], ["액터", inf.get("시작", "")],
                     ["선행 조건", inf.get("선행", "")], ["사후 조건", inf.get("사후", "")], ["예외/대안", inf.get("예외", "")],
                     ["경계", inf.get("경계", "").replace(" | ", " · ")]], widths=[7800, 36728])
    for i, (sub, keys, flow) in enumerate(s["split"], start=1):
        B["cap"](f"{sd} {sub}")
        name = f"image22x6{i}"; IMGS[f"{sd}_{i}"] = name
        B["img"](f"out2/{sd}_{i}.png", name)
    _, steps = flatten(s)
    B["tbl"]("step", [list(r) for r in steps], header=["순서", "호출 → 반환", "처리 내용"], widths=[2800, 22400, 19328])


# ── 1) 2.1 파일 구조 표 ──
def cell_rows(tbl_el):
    return [(tr, tr.findall(q("tc"))) for tr in tbl_el.findall(q("tr"))]


def renumber(tbl_el):
    p = tbl_el
    while p.tag != q("p") or p.getparent().tag != root.tag:      # 표를 담은 본문 문단의 줄 배치 정보도 지운다
        p = p.getparent()
    B["strip_ls"](p)
    total = 0
    for i, tr in enumerate(tbl_el.findall(q("tr"))):
        for tc in tr.findall(q("tc")):
            tc.find(q("cellAddr")).set("rowAddr", str(i))
        total += int(tr.find(q("tc")).find(q("cellSz")).get("height"))
    tbl_el.set("rowCnt", str(len(tbl_el.findall(q("tr")))))
    tbl_el.find(q("sz")).set("height", str(total))


def set_row(tr, texts):
    tcs = tr.findall(q("tc")); hh = 0
    for tc, t in zip(tcs, texts):
        if t is not None:
            hh = max(hh, B["fill_cell"](tc, t))
        else:
            hh = max(hh, int(tc.find(q("cellSz")).get("height")))
    for tc in tcs:
        tc.find(q("cellSz")).set("height", str(max(hh, 2200)))


def table_with(first_cell):
    for tb in root.iter(q("tbl")):
        for tr, tcs in cell_rows(tb):
            if tcs and txt(tcs[0]).strip() == first_cell:
                return tb
    raise KeyError(first_cell)


tb = table_with("research/grace/")
rows = {txt(tcs[0]).strip(): (tr, tcs) for tr, tcs in cell_rows(tb)}
before = {k: ctext(rows[k][1][1]) for k in ["apps/web/", "research/grace/", "tests/unit/ · integration/ · field/"]}
tr_g = rows["research/grace/"][0]
set_row(tr_g, ["services/media/crop-grace/",
               "서버 크롭 제공 모듈 — 원본 분석 결과로 대상 크롭 생성·원본 크롭 보존, GRACE 인코딩·패킷화, 제공 프로파일 협상(CROP_GRACE · CROP_BASELINE), "
               "단말 사용 패킷 피드백·참조 동기화, CropStreamEndpoint 송신 경계. 탐지·좌표 계산·관측 완료 판정은 원본 분석 경로(services/vision)가 맡는다."])
tb.remove(tr_g); tb.insert(tb.index(rows["services/media/"][0]) + 1, tr_g)
set_row(rows["apps/web/"][0], [None, before["apps/web/"] + " 관제 단말 전용 GRACE 디코더(apps/web/grace-decoder — 전용 실행 환경, PWA·WebCodecs 기본 지원은 가정하지 않음)와 "
                                     "후보 화면(UI-05) 크롭 표시 연계(복원 영상 표지 · 관측 시각 · 오래된 영상 표시)."])
set_row(rows["tests/unit/ · integration/ · field/"][0], [None, before["tests/unit/ · integration/ · field/"] +
        " GRACE 크롭 경로는 패킷 손실(FULL/PARTIAL/NONE)·참조 복구·단말 호환·성능 시험을 integration/crop-grace/ 에 둔다."])
renumber(tb)
LOG.append(("2.1 파일 구조", "research/grace/ — " + before["research/grace/"], "services/media/crop-grace/ (서버 크롭 제공 모듈) · 행을 services/media/ 아래로 이동"))
LOG.append(("2.1 파일 구조", "apps/web/ — " + before["apps/web/"], "+ 관제 단말 전용 GRACE 디코더 · UI-05 크롭 표시 연계"))
LOG.append(("2.1 파일 구조", "tests/ — " + before["tests/unit/ · integration/ · field/"], "+ 패킷 손실·참조 복구·단말 호환·성능 시험"))

# ── 2) 4.1 DD-01 그림 · 설명 ──
dd = Image.open("out2/dd01.png"); pw, ph = dd.size
pic = next(p for p in root.iter(q("pic")) if p.find(".//{%s}img" % B["HC"]).get("binaryItemIDRef") == "image14")
oldh = int(pic.find(q("sz")).get("height")); nw = int(pic.find(q("sz")).get("width")); nh = round(nw * ph / pw)
for tag in ("orgSz", "curSz"):
    pic.find(q(tag)).set("width", str(nw)); pic.find(q(tag)).set("height", str(nh))
pic.find(q("sz")).set("height", str(nh))
pic.find(q("offset")).set("y", "0")
ri = pic.find(q("rotationInfo")); ri.set("centerX", str(nw // 2)); ri.set("centerY", str(nh // 2))
HC = B["HC"]
rinfo = pic.find(q("renderingInfo"))
rinfo.find("{%s}transMatrix" % HC).set("e6", "0")
sm = rinfo.find("{%s}scaMatrix" % HC); sm.set("e1", "1"); sm.set("e5", "1"); sm.set("e6", "0")
for pt, (x, y) in zip(list(pic.find(q("imgRect"))), [(0, 0), (nw, 0), (nw, nh), (0, nh)]):
    pt.set("x", str(x)); pt.set("y", str(y))
pic.find(q("imgClip")).set("right", str(pw * 75)); pic.find(q("imgClip")).set("bottom", str(ph * 75))
pic.find(q("imgDim")).set("dimwidth", str(pw * 75)); pic.find(q("imgDim")).set("dimheight", str(ph * 75))
pic.find(".//{%s}img" % HC).set("binaryItemIDRef", "image22dd1")
pic.find(q("shapeComment")).text = "DD-01 실기체 3티어 운용도 (무채색 · GRACE 크롭 구간 반영)"
buf = io.BytesIO(); dd.convert("RGB").save(buf, "PNG", optimize=True); B["BIN"]["image22dd1"] = buf.getvalue()
tc = pic
while tc.tag != q("tc"): tc = tc.getparent()
csz = tc.find(q("cellSz")); csz.set("height", str(int(csz.get("height")) - (oldh - nh)))
ptbl = tc.getparent().getparent(); renumber(ptbl)
i_dd, p_dd = find_p("RC, 텔레메트리, 영상 링크를 분리한다.", exact=False)
t_old = txt(p_dd)
B["set_text"](p_dd, t_old + " 영상은 기체 → 게이트웨이 → 서버로 원본(SRT)을 보내고, 서버가 원본으로 탐지·좌표 계산을 한 뒤 대상 크롭을 GRACE 로 인코딩·패킷화해 "
              "관제 운용자 단말에 보낸다. 단말의 전용 GRACE 디코더가 복원·표시하고 실제 사용한 패킷을 서버에 피드백한다. 게이트웨이는 GRACE 인코딩을 하지 않는다. "
              "전용 디코더 실행 환경과 화면 연계는 배포 조건이며 PWA·WebCodecs 의 GRACE 기본 지원은 가정하지 않는다. 크롭 패킷 전송 기술은 구현·호환 검증 후 확정한다.")
LOG.append(("4.1 DD-01", "컬러 운용도 (GRACE 표기 없음) · 설명 3문장", "무채색 재작성 — 서버 GRACE 크롭 인코더 · 관제 운용자 단말 GRACE 디코더 · 크롭 패킷(서버→단말) · 사용 패킷 피드백(단말→서버) · 게이트웨이 'GRACE 인코딩 없음' · 설명 문장 추가"))

# ── 3) 4.2 공통 기준 문장 ──
_, p_cc = find_p("Controller 와 다른 패키지는 서비스의 인터페이스만 import 한다.", exact=False)
B["set_text"](p_cc, "Controller 와 다른 패키지는 서비스의 인터페이스만 import 한다. HTTP·WSS·SRT·크롭 스트림 통신은 관계로 그리지 않고 "
              "Controller·EventPublisher·CropStreamEndpoint·IGatewayLink(GatewayClient)·ServerReporter 가 경계를 맡는다. "
              "서버→관제 단말 크롭 패킷과 단말→서버 디코딩 결과 피드백은 CropStreamEndpoint 가 맡고, 구체 전송 기술은 구현·호환 검증 후 확정한다.")
_, p_sq = find_p("실선 화살표는 동기 호출, 점선 화살표는 반환", exact=False)
B["set_text"](p_sq, txt(p_sq) + " 생명선이 많은 SD-X06 은 세 장으로 나눠 그리고 번호와 처리표는 하나로 잇는다.")
LOG.append(("4.2 클래스 다이어그램 공통 기준", "HTTP·WSS·SRT … EventPublisher·IGatewayLink·ServerReporter 가 경계", "+ 크롭 스트림 · CropStreamEndpoint (서버→단말 패킷 · 단말→서버 피드백 · 전송 기술 미정)"))

# ── 4) 패키지 표 (PD-01 아래) ──
tb = table_with("api")
rows = {txt(tcs[0]).strip(): (tr, tcs) for tr, tcs in cell_rows(tb)}
api_old = ctext(rows["api"][1][2]); web_old = ctext(rows["web"][1][2])
set_row(rows["api"][0], [None, None, api_old + " · CropStreamEndpoint(관제 단말 크롭 패킷·피드백)"])
set_row(rows["web"][0], [None, None, web_old + " · GraceCropDecoder(관제 단말 전용 GRACE 디코더)"])
set_row(rows["vision_train · research.grace"][0], ["vision_train", "vision-train/", "오프라인 학습(«tool») — 운용 경로 밖"])
tr_new = copy.deepcopy(rows["services.media"][0])
set_row(tr_new, ["services.media.crop_grace", "services/media/crop-grace",
                 "관제 단말 대상 크롭 제공 — ICropDeliveryService (GRACE 인코딩·패킷화 · 제공 프로파일 협상 · 참조 동기화 · 기존 방식 대체)"])
tb.insert(tb.index(rows["services.media"][0]) + 1, tr_new)
renumber(tb)
LOG.append(("4.2 패키지 표", "vision_train · research.grace — GRACE 연구 경로 (운용 경로 밖)", "vision_train 만 · services.media.crop_grace 행 추가 · api 에 CropStreamEndpoint · web 에 GraceCropDecoder"))

# ── 5) 4.2.6 영상 관리 ──
i0, _ = find_p("4.2.6.1 클래스 다이어그램", start=100); i1, _ = find_p("4.2.6.2 클래스 명세", start=100)
replace_range(i0 + 1, i1, gen(lambda: (cd_images("06"), B["body"](f"클래스 {len(ucls('06'))}개 — 식별자·책임·속성·연산은 4.2.6.2 에 정리한다."))))
i0, _ = find_p("4.2.6.2 클래스 명세", start=100); i1, _ = find_p("4.2.6.3 시퀀스 다이어그램", start=100)
replace_range(i0 + 1, i1, gen(lambda: class_specs("06")))
i_sd, _ = find_p("SD-X06 — GRACE 부분 수신·참조 재동기화", start=100)
P = top(); kinds = [[e.tag for e in P[i_sd + k].iter() if e.tag in (q("tbl"), q("pic"))][:1] for k in (1, 2, 3)]
assert kinds == [[q("tbl")], [q("pic")], [q("tbl")]], kinds
replace_range(i_sd, i_sd + 4, gen(lambda: sd_block("SD-X06")))
LOG.append(("4.2.6.1 CD-06", "(1/2) 계층 구조 · (2/2) 현장 중계·원본 회수·GRACE 연구 경로", "(1/3) 계층 구조 · (2/3) 현장 중계·원본 회수 · (3/3) GRACE 크롭 제공 — 서버 인코딩 → 관제 단말 디코딩"))
LOG.append(("4.2.6.2 클래스 명세", "C-0606·0609·0610 research.grace (연구 경로) · C-0608 업무 조건만", "C-0606·0609·0610 services.media.crop_grace 로 재정의 · C-0608 크롭 표시 조건 추가 · 신규 C-0630~0638"))
LOG.append(("4.2.6.3 SD-X06", "GRACE 부분 수신·참조 재동기화 (게이트웨이 인코딩 → 서버 디코딩 · BASELINE_SRT 복귀)", "GRACE 대상 크롭 제공 — 서버 → 관제 운용자 단말 (3장 · 26단계)"))

# ── 6) 4.2.7 — TargetService 가 ICropDeliveryService 에 의존 ──
i_c, _ = find_p("CD-07 — 탐지·대상 관리 (1/4) 후보·추적 — 계층 구조", start=100)
assert top()[i_c + 1].find(".//" + q("pic")) is not None
replace_range(i_c, i_c + 2, gen(lambda: cd_images("07", only=["cd07a"])))
i0, _ = find_p("4.2.7.2 클래스 명세", start=100); i1, _ = find_p("4.2.7.3 시퀀스 다이어그램", start=100)
old07 = [txt(p) for p in top()[i0 + 1:i1]]
new07 = gen(lambda: class_specs("07"))
replace_range(i0 + 1, i1, new07)
LOG.append(("4.2.7 CD-07 (1/4) · C-0704", "TargetService 의존 8개", "+ cropDelivery : ICropDeliveryService (원본 분석 결과로 크롭 요청)"))

# ── 7) 6장·기타 문구 — 연구 경로 표현이 남았는지 ──
sec = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)
for bad in ["GRACE_LAB", "research/grace", "research.grace", "연구 경로", "BASELINE_SRT", "GraceCodecAdapter(Pi)", "GRACE 부분 수신"]:
    n = sec.decode().count(bad)
    print(f"남은 '{bad}':", n)

# ── 쓰기 ──
used = set(re.findall(rb'binaryItemIDRef="([^"]+)"', sec))
hpf = zin.read("Contents/content.hpf").decode("utf-8")
for iid in re.findall(r'<opf:item id="(image[^"]+)" href="BinData/[^"]+"[^>]*/>', hpf):
    if iid.encode() not in used:
        hpf = re.sub(r'<opf:item id="%s" href="[^"]+"[^>]*/>' % re.escape(iid), "", hpf)
hpf = hpf.replace("</opf:manifest>", "".join(f'<opf:item id="{k}" href="BinData/{k}.png" media-type="image/png" isEmbeded="1"/>' for k in B["BIN"]) + "</opf:manifest>")
zout = zipfile.ZipFile(DST, "w")
for info in zin.infolist():
    fn = info.filename
    if fn == "Contents/section0.xml":
        data = sec
    elif fn == "Contents/content.hpf":
        data = hpf.encode("utf-8")
    elif fn.startswith("BinData/"):
        iid = fn.split("/")[1].rsplit(".", 1)[0]
        if iid.encode() not in used:
            continue
        data = open("out2/pd01.png", "rb").read() if fn == "BinData/image15.png" else zin.read(fn)
    else:
        data = zin.read(fn)
    zi = zipfile.ZipInfo(fn, date_time=info.date_time)
    zi.compress_type = zipfile.ZIP_STORED if fn == "mimetype" else zipfile.ZIP_DEFLATED
    zout.writestr(zi, data)
for k, v in B["BIN"].items():
    zi = zipfile.ZipInfo(f"BinData/{k}.png", date_time=(2026, 10, 1, 21, 0, 0)); zi.compress_type = zipfile.ZIP_DEFLATED
    zout.writestr(zi, v)
zout.close()
LOG.append(("4.2 PD-01", "research.grace (점선 · 연구 경로) · 학습 서버 · 연구 경로", "services.media.crop_grace · web 에 GraceCropDecoder · api 에 CropStreamEndpoint · 크롭 경로 주석"))
import json
json.dump({"log": LOG, "old07": old07, "new07": [txt(e) for e in new07], "bin": sorted(B["BIN"])}, open("patch22.json", "w"), ensure_ascii=False, indent=1)
print("새 그림", sorted(B["BIN"]), "· 저장", DST)
