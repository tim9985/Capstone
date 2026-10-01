"""stylemap.py — v1.4 (1) 머리말(header.xml)의 서식 번호 → 현재 문서 서식 번호 대응"""
import zipfile
from lxml import etree
HH = "http://www.hancom.co.kr/hwpml/2011/head"
DROP = {"id", "textDir"}


def _sig(e):
    parts = []
    for c in e.iter():
        ln = etree.QName(c).localname
        if ln in ("switch", "case", "default"): continue
        parts.append(ln + str(sorted((k, v) for k, v in c.attrib.items() if k not in DROP)))
    return "|".join(parts)


def table(path, tag):
    hdr = etree.fromstring(zipfile.ZipFile(path).read("Contents/header.xml"))
    return {e.get("id"): _sig(e) for e in hdr.iter("{%s}%s" % (HH, tag))}


def mapping(old_path, new_path):
    res = {}
    for tag in ("paraPr", "charPr", "borderFill"):
        o, n = table(old_path, tag), table(new_path, tag)
        rev = {}
        for k, v in n.items(): rev.setdefault(v, []).append(k)
        m = {}
        for k, v in o.items():
            cands = rev.get(v, [])
            m[k] = (k if k in cands else (min(cands, key=lambda c: abs(int(c) - int(k))) if cands else None))
        res[tag] = m
    return res
