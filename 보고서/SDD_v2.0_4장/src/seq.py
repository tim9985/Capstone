"""seq.py — 시퀀스 DSL: 호출 트리 → (그림 행, 처리표 행)
   call(a, b, "method(args)", "Ret", "처리 내용", [하위 호출…])  · a == b 이면 자기 호출
   alt([("조건", [노드…]), ("조건2", [노드…])]) · opt("조건", [노드…]) · loop("조건", [노드…]) · note(a, b, "글")"""
from mdl import SEQS, UNITS


def call(a, b, msg, ret=None, desc="", sub=()):
    return ("call", a, b, msg, ret, desc, list(sub))


def alt(branches):
    return ("alt", branches)


def opt(cond, nodes):
    return ("opt", [(cond, nodes)])


def loop(cond, nodes):
    return ("loop", [(cond, nodes)])


def par(branches):
    return ("par", branches)


def note(a, b, text):
    return ("note", a, b, text)


def old_info(sd):
    from mdl import OLD
    for s in OLD["sds"]:
        if s["sd"] == sd:
            c = s.get("ctx", {})
            return {"uc": s["uc"], "개요": s["summary"], "시작": c.get("시작 주체/사건", ""), "선행": c.get("선행 조건", ""),
                    "사후": c.get("사후 조건", ""), "예외": c.get("예외/대안", ""), "경계": c.get("경계", "")}
    return {}


def S(sd, unit_, parts, flow, entry=None, **over):
    """parts: [(key, 이름, 종류)] · entry: 경계 첫 칸 (예: "AuthController.login") — 나머지 정보는 v1.4 문구를 기본으로 쓴다
       over: 개요 · 시작 · 선행 · 사후 · 예외 · 경계 · uc · title 덮어쓰기"""
    info = old_info(sd); info.update({k: v for k, v in over.items() if k not in ("title",)})
    b = info.get("경계", "")
    if entry and b:
        parts_b = [x.strip() for x in b.split("|")]
        http = parts_b[0]
        parts_b[0] = f"{entry} ({http})" if http and not http.startswith("내부") else entry
        info["경계"] = " | ".join(parts_b)
    title = over.get("title") or (info.get("uc", "").split(" ", 1)[1] if " " in info.get("uc", "") else sd)
    SEQS[sd] = dict(sd=sd, uc=info.get("uc", ""), title=title, unit=unit_, parts=parts, flow=flow, info=info)
    UNITS[unit_]["sds"].append(sd)


def flatten(s):
    names = {k: nm.replace("\n", "") for k, nm, _ in s["parts"]}
    items, steps = [], []
    cnt = [0]

    def walk(nodes):
        for nd in nodes:
            if nd[0] == "call":
                _, a, b, msg, ret, desc, sub = nd
                cnt[0] += 1; no = cnt[0]
                if a == b:
                    items.append(("self", a, f"{no}: {msg}"))
                    steps.append((str(no), f"{names[b]}.{msg}" + (f" → {ret}" if ret else ""), desc))
                    walk(sub)
                    continue
                items.append(("call", a, b, f"{no}: {msg}"))
                steps.append((str(no), f"{names[b]}.{msg}" + (f" → {ret}" if ret else ""), desc))
                walk(sub)
                if ret:
                    items.append(("ret", b, a, ret))
                else:
                    items.append(("act_end", b))
            elif nd[0] in ("alt", "opt", "loop", "par"):
                for i, (cond, sub) in enumerate(nd[1]):
                    items.append(("frame", nd[0], cond) if i == 0 else ("else", cond))
                    walk(sub)
                items.append(("end",))
            elif nd[0] == "note":
                items.append(nd)
    walk(s["flow"])
    return items, steps
