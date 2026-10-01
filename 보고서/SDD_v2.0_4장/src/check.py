import re, importlib
from mdl import C, SEQS, OLD, CDS
for i in range(1, 13): importlib.import_module(f"u{i:02d}")
byname = {}
for k, v in C.items(): byname.setdefault(v["name"], []).append(k)
err = []
miss_old = [k for k in OLD["classes"] if k not in C]
if miss_old: err.append(f"v1.4 클래스 누락: {miss_old}")
old_sd = [s["sd"] for s in OLD["sds"]]
miss_sd = [s for s in old_sd if s not in SEQS]
if miss_sd: err.append(f"SD 누락: {miss_sd}")
dups = {n: v for n, v in byname.items() if len(v) > 1}
if dups: err.append(f"이름 중복: {dups}")
def ops_of(cid):
    c = C[cid]; names = {o[0].lstrip("-") for o in c["ops"]}
    if c.get("impl") and c["kind"] != "interface": names |= {o[0] for o in C[c["impl"]]["ops"]}
    if c["kind"] == "interface" and c.get("impl"): names |= {o[0].lstrip("-") for o in C[c["impl"]]["ops"]}
    return names
def walk(nodes, parts, sd):
    for nd in nodes:
        if nd[0] == "call":
            _, a, b, msg, ret, desc, sub = nd
            nm = parts[b].replace("\n", "").split(" (")[0].strip()
            meth = msg.split("(")[0].strip()
            if nm in byname and not ("." in meth):
                if meth not in ops_of(byname[nm][0]):
                    err.append(f"{sd}: {nm}.{meth} 없음")
            walk(sub, parts, sd)
        elif nd[0] in ("alt", "opt", "loop", "par"):
            for _, sub in nd[1]: walk(sub, parts, sd)
for sd, s in SEQS.items():
    parts = {k: n for k, n, _ in s["parts"]}
    walk(s["flow"], parts, sd)
# 타입 참조
tok = set()
for c in C.values():
    for a in c["attrs"]: tok |= set(re.findall(r"[A-Z]\w+", a[1]))
    for o in c["ops"]: tok |= set(re.findall(r"[A-Z]\w+", o[1] + " " + (o[2] or "")))
for t in sorted(tok):
    if (t.endswith("DTO") or t.endswith("DAO") or (t.startswith("I") and t[1:2].isupper() and t not in ("IoU",))) and t not in byname:
        err.append(f"정의 안 된 타입: {t}")
print(len(C), "classes ·", len(SEQS), "SDs ·", len(CDS), "CDs")
print("\n".join(err) if err else "OK")
