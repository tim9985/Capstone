"""cdkit.py — 도식 등록 · 박스 생성 도우미"""
from mdl import C, CDS, sig, attr_line


def box(cid, attrs=None, ops=None, ref=False, compact=False, name=None, maxops=None):
    c = C[cid]
    if ref:
        return dict(kind=c["kind"], cid=cid, name=name or c["name"], ref=True)
    if compact:
        return dict(kind=c["kind"], cid=cid, name=c["name"], compact=True)
    if attrs is None:
        attrs = [f"{k}" for k in c.get("keys", [])] if c["kind"] == "entity" else [attr_line(x) for x in c["attrs"]]
    if ops is None:
        ops = [sig(x, True) for x in c["ops"] if not x[0].startswith("-")]
        if maxops: ops = ops[:maxops]
    return dict(kind=c["kind"], cid=cid, name=c["name"], attrs=attrs, ops=ops)


def CD(cd, title, unit, parts):
    """parts: [(그림 키, 소제목, packages, boxes, rels)]"""
    CDS[cd] = dict(cd=cd, title=title, unit=unit, parts=parts)


def _types(cid):
    out = set()
    for a in C[cid]["attrs"]:
        t = a[1].replace("?", "").replace("List<", "").replace(">", "").strip()
        out.add(t)
    return out


def _grid(keys, ncol):
    rows = []
    for i in range(0, len(keys), ncol):
        r = [(k, 1 / ncol) for k in keys[i:i + ncol]]
        rows.append(r)
    return rows


def layered(key, sub, *, svc_pkg, ctl=(), dto=(), pairs=(), comps=(), ext=(), daos=(), ents=(), extra=(), dto_cols=3,
            api_w=0.5, ext_w=0.24, comp_row=True, impl_ops=False, drop=()):
    """표준 계층 배치 — api | contracts / services(인터페이스·구현·구성요소) | 외부 인터페이스 / storage.dao / storage.entity
       pairs: [(인터페이스, 구현)] · comps: 같은 패키지 구성요소 · ext: 다른 패키지 참조 [(cid, 패키지)] · daos: [(cid, ref?)]"""
    B, P, R = {}, [], []
    kid = lambda cid: cid.replace("C-", "k")
    for c in ctl: B[kid(c)] = box(c)
    for d in dto: B[kid(d)] = box(d, compact=True)
    for i, m in pairs:
        B[kid(i)] = box(i)
        B[kid(m)] = box(m, ops=[sig(o, True) for o in C[m]["ops"] if not o[0].startswith("-")] if impl_ops else [])
    for c in comps: B[kid(c)] = box(c)
    for c, _ in ext: B[kid(c)] = box(c, ref=True)
    for d, ref in daos: B[kid(d)] = box(d, ref=True) if ref else box(d, attrs=[])
    for e in ents: B[kid(e)] = box(e)
    has_ext = bool(ext)
    sw = 0.99 - (ext_w + 0.01 if has_ext else 0)
    if ctl or dto:
        if ctl:
            P.append(dict(key="api", name="api", row=0, x=0.005, w=api_w if dto else 0.99,
                          rows=[[(kid(c), 1 / len(ctl)) for c in ctl]]))
        if dto:
            x0 = 0.005 + (api_w + 0.015 if ctl else 0)
            P.append(dict(key="ct", name="contracts", row=0, x=x0, w=0.995 - x0, rows=_grid([kid(d) for d in dto], dto_cols)))
    ncol = max(len(pairs) + (0 if comp_row else len(comps)), 1)
    rows = []
    if pairs:
        rows.append([(kid(i), 1 / ncol) for i, _ in pairs] + ([(None, 1 / ncol)] * (ncol - len(pairs))))
        rows.append([(kid(m), 1 / ncol) for _, m in pairs] + ([] if comp_row else [(kid(c), 1 / ncol) for c in comps]) +
                    ([(None, 1 / ncol)] * (ncol - len(pairs) - (0 if comp_row else len(comps)))))
    if comps and comp_row:
        rows += _grid([kid(c) for c in comps], max(ncol, min(len(comps), 4)))
    P.append(dict(key="svc", name=svc_pkg, row=1, x=0.005, w=sw, rows=rows))
    if has_ext:
        ec = 2 if len(ext) > 6 else 1
        rows_, tabs_ = [], []
        for i in range(0, len(ext), ec):
            grp = ext[i:i + ec]
            rows_.append([(kid(c), 1 / ec) for c, _ in grp] + [(None, 1 / ec)] * (ec - len(grp)))
            tabs_.append(" · ".join(dict.fromkeys(p for _, p in grp)))
        P.append(dict(key="ext", name="다른 패키지 (인터페이스)", row=1, x=sw + 0.015, w=ext_w, rows=rows_, tabs=tabs_))
    if daos:
        P.append(dict(name="storage.dao", row=2, x=0.005, w=0.99, rows=[[(kid(d), 1 / max(len(daos), 3)) for d, _ in daos]]))
    if ents:
        P.append(dict(name="storage.entity", row=3, x=0.005, w=0.99, rows=[[(kid(e), 1 / max(len(daos), 3)) for e in ents]]))
    # 관계 자동
    names = {C[k]["name"]: k for k in [*ctl, *[i for i, _ in pairs], *[m for _, m in pairs], *comps, *[c for c, _ in ext], *[d for d, _ in daos], *ents]}
    if ctl and dto: R.append(("api", "ct", "dep", "«import»", {}))
    for c in ctl:
        for t in _types(c):
            if t in names and C[names[t]]["kind"] == "interface": R.append((kid(c), kid(names[t]), "dep", "", {"elbow": 1}))
    for i, m in pairs:
        R.append((kid(m), kid(i), "real", "", {}))
    users = [m for _, m in pairs] + list(comps)
    for u in users:
        for t in _types(u):
            if t not in names or names[t] == u: continue
            tgt = names[t]; kd = C[tgt]["kind"]
            if kd in ("component", "service") and tgt in comps: R.append((kid(u), kid(tgt), "assoc", "", {"elbow": 1}))
            elif kd == "dao": R.append((kid(u), kid(tgt), "dep", "", {"elbow": 1}))
            elif kd == "interface" and tgt in [c for c, _ in ext]: pass   # 패키지 단위 «import» 하나로 표시
    for d, ref in daos:
        if ref: continue
        en = C[d]["name"][:-3]
        if en in names: R.append((kid(d), kid(names[en]), "dep", "CRUD", {}))
    if has_ext: R.append(("svc", "ext", "dep", "«import»", {}))
    R = [r for r in R if (r[0], r[1]) not in drop]
    R += list(extra)
    return (key, sub, P, B, R)
