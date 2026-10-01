"""patch21b.py — 관리단위재구성(한글 저장본)에 v2.1 표기 수정('액터' · DB-* 제거 · PD-01 교체)
   python patch21b.py <원본.hwpx> <출력.hwpx> <PD-01.png>
   patch21 과 같은 내용 + 글자를 바꾼 문단의 줄 배치 정보(linesegarray)를 지운다 — 남겨 두면 위치값이 글자 수와 어긋나
   최신 한글이 '손상·변조'로 판단한다 · 압축은 한글 저장본 규칙(PNG·version.xml 무압축)을 따른다"""
import io, re, sys, zipfile
from lxml import etree
from PIL import Image

SRC, DST, PD = sys.argv[1], sys.argv[2], sys.argv[3]
HERE = __file__.rsplit("/", 1)[0] if "/" in __file__ else "."
P21 = {}
p21 = open(f"{HERE}/patch21.py").read()
exec(p21.split("SRC =")[0] + "\n" + p21[p21.index("ACT = {"):p21.index("cnt = {")], P21)
ACT, SUBS, HP = P21["ACT"], P21["SUBS"], "http://www.hancom.co.kr/hwpml/2011/paragraph"
q = lambda t: "{%s}%s" % (HP, t)
cnt = {"act": 0, "sub": 0, "db": 0, "ls": 0}
z = zipfile.ZipFile(SRC)
out = zipfile.ZipFile(DST, "w")
for info in z.infolist():
    data = z.read(info.filename)
    if re.match(r"Contents/section\d+\.xml", info.filename):
        root = etree.fromstring(data)
        touched = set()
        for t in root.iter(q("t")):
            if not t.text: continue
            s = t.text
            if s.strip() in ACT: s = s.replace(s.strip(), ACT[s.strip()]); cnt["act"] += 1
            for a, b in SUBS:
                if a in s: s = s.replace(a, b); cnt["sub"] += 1
            s2 = re.sub(r"DB-\d{2} (?=[a-z_]+)", "", s)
            if s2 != s: cnt["db"] += 1; s = s2
            if s != t.text:
                t.text = s
                p = t
                while p is not None:                       # 바뀐 문단과 그 문단을 담은 바깥 문단(표 문단)까지
                    if p.tag == q("p"): touched.add(p)
                    p = p.getparent()
        for p in touched:
            for ls in p.findall(q("linesegarray")):
                p.remove(ls); cnt["ls"] += 1
        data = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)
    elif info.filename == "BinData/image15.png":
        b = io.BytesIO(); Image.open(PD).convert("RGB").save(b, "PNG", optimize=True); data = b.getvalue()
    zi = zipfile.ZipInfo(info.filename, date_time=info.date_time)
    zi.create_system, zi.create_version = info.create_system, info.create_version
    zi.compress_type = zipfile.ZIP_STORED if (info.filename in ("mimetype", "version.xml") or info.filename.endswith(".png")) else zipfile.ZIP_DEFLATED
    out.writestr(zi, data)
out.close(); print(cnt)
