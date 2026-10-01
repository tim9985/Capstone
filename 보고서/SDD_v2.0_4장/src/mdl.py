"""mdl.py — 4장 객체지향 모델 레지스트리 (클래스 · 인터페이스 · DTO · DAO · Entity · 도식 · 시퀀스)"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ERD = json.load(open(HERE / "erd.json"))
OLD = json.load(open(HERE / "ch4.json"))

C = {}            # cid → class
UNITS = {}        # "01" → dict(title, cds=[…], sds=[…])
CDS = {}          # "CD-01" → dict(title, unit, parts=[(img_key, sub_title, packages, boxes, rels)])
SEQS = {}         # "SD-0101" → dict
PKGS = {}         # 패키지 key → dict(name, path, tier, resp)

TABLE_ID = {t: v["comment"].split()[0] for t, v in ERD.items()}          # user_account → DB-01


def unit(u, title, cd_title):
    UNITS[u] = dict(title=title, cd_title=cd_title, sds=[])


def pkg(key, path, tier, resp):
    PKGS[key] = dict(key=key, path=path, tier=tier, resp=resp)


def K(cid, name, kind, pk, resp, attrs=(), ops=(), impl=None, db=None, old=None, note=None):
    """attrs: (이름, 타입, 설명) · ops: (이름, 매개변수, 반환, 설명) — 이름이 '-' 로 시작하면 비공개"""
    assert cid not in C, cid
    C[cid] = dict(id=cid, name=name, kind=kind, pkg=pk, unit=cid[2:4], resp=resp,
                  attrs=[list(a) for a in attrs], ops=[list(o) for o in ops], impl=impl, db=db, old=old, note=note)
    return cid


def I(cid, name, pk, resp, ops, impl):
    """서비스 인터페이스 — impl = 구현 클래스 cid"""
    return K(cid, name, "interface", pk, resp, (), ops, impl=impl)


def DTO(cid, name, resp, fields):
    return K(cid, name, "dto", "contracts", resp, fields, ())


def ENT(cid, name, table, resp, ops=(), keys=None):
    cols = ERD[table]["cols"]
    attrs = [(camel(c[0]), c[1], f"{c[3]}" + (f" [{c[2]}]" if c[2] and c[2] not in ("NN", "NULL") else "")) for c in cols]
    cid_ = K(cid, name, "entity", "storage.entity", resp, attrs, ops, db=f"{TABLE_ID[table]} {table}")
    C[cid]["keys"] = keys or [camel(c[0]) for c in cols if "PK" in c[2] or "FK" in c[2]][:5]
    return cid_


def DAO(cid, name, table, ops, resp=None):
    return K(cid, name, "dao", "storage.dao", resp or f"{TABLE_ID[table]} {table} 테이블의 저장·조회 (SQL 은 이 클래스에만 둔다)",
             (("connection", "DbSession", "PostgreSQL/PostGIS 연결 (트랜잭션 경계는 서비스가 정한다)"),), ops,
             db=f"{TABLE_ID[table]} {table}")


def camel(s):
    p = s.split("_")
    return p[0] + "".join(x[:1].upper() + x[1:] for x in p[1:])


def by_name(name):
    for k, v in C.items():
        if v["name"] == name:
            return k
    raise KeyError(name)


def old_members(cid):
    """v1.4 명세 (속성/메소드) — [(vis, name, desc, params, type)]"""
    out = []
    for m in OLD["classes"].get(cid, {}).get("members", []):
        vis = m[0][0]; nm = m[0][1:].strip().rstrip("()").replace(" ", "")
        out.append((vis, nm, m[1], m[2], m[3]))
    return out


# ── 표시용 문자열 ────────────────────────────────────────────────
def sig(o, short=False):
    nm, params, ret = o[0].lstrip("-"), o[1], o[2]
    vis = "-" if o[0].startswith("-") else "+"
    if short:
        return f"{vis} {nm}({params})" + (f" : {ret}" if ret and ret != "void" else "")
    return f"{vis} {nm}({params}) : {ret or 'void'}"


def attr_line(a):
    return f"- {a[0]} : {a[1]}"
