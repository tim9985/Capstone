"""mono.py — 무채색 UML 렌더러 (패키지 · 클래스 · 시퀀스) · 나눔고딕
   인쇄 폭 157 mm 기준 글자 ≥ 6 pt 가 되도록 W=2000 px · fs≈28 px 를 기본으로 쓴다."""
import math
import re
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Rectangle, Polygon, FancyBboxPatch, Circle

SP = Path("/tmp/claude-1001/-home-se-JupyterLAB/027ea9bd-abdf-476f-ad0d-25f3e2ea7ffc/scratchpad")
fm.fontManager.addfont(str(SP / "NanumGothic.ttf")); fm.fontManager.addfont(str(SP / "NanumGothicBold.ttf"))
FONT = fm.FontProperties(fname=str(SP / "NanumGothic.ttf")).get_name()
plt.rcParams["font.family"] = FONT
PX = 72 / 100

INK = "#1f1f1f"; LINE = "#3c3c3c"; MID = "#6e6e6e"; SOFT = "#9a9a9a"
PKG_BG = "#fbfbfb"; PKG_TAB = "#ececec"; BODY = "#ffffff"
HEAD = {"controller": "#e2e2e2", "interface": "#ffffff", "service": "#ececec", "component": "#f4f4f4",
        "dao": "#dadada", "entity": "#d2d2d2", "dto": "#f0f0f0", "tool": "#efefef", "value": "#f6f6f6",
        "boundary": "#e2e2e2", "viewmodel": "#f6f6f6"}
STEREO = {"controller": "«controller»", "interface": "«interface»", "service": "«service»", "component": "",
          "dao": "«DAO»", "entity": "«entity»", "dto": "«DTO»", "tool": "«tool»", "value": "«value»",
          "boundary": "«boundary»", "viewmodel": "«view model»"}


class Canvas:
    def __init__(self, w, h):
        self.w, self.h = w, h
        self.fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
        self.ax = self.fig.add_axes([0, 0, 1, 1]); self.ax.set_xlim(0, w); self.ax.set_ylim(h, 0); self.ax.axis("off")
        self.fig.patch.set_facecolor("white")
        self.r = self.fig.canvas.get_renderer()

    def text(self, x, y, s, size, ha="left", va="center", bold=False, color=INK, **kw):
        return self.ax.text(x, y, s, fontsize=size * PX, ha=ha, va=va, color=color,
                            fontweight="bold" if bold else "normal", **kw)

    def tw(self, s, size, bold=False):
        t = self.text(0, 0, s, size, bold=bold); bb = t.get_window_extent(self.r); t.remove()
        return bb.width

    def save(self, path):
        self.fig.savefig(path, dpi=100); plt.close(self.fig)


def _measure():
    c = Canvas(10, 10)
    return c


_MC = None
def tw(s, size, bold=False):
    global _MC
    if _MC is None:
        _MC = Canvas(10, 10)
    return _MC.tw(s, size, bold)


# ───────────────────────── 화살촉 ─────────────────────────
def head(ax, p_prev, p_end, kind, fs, col=INK):
    (x0, y0), (x1, y1) = p_prev, p_end
    ang = math.atan2(y1 - y0, x1 - x0)
    if kind == "real":          # 빈 삼각형
        L, Wd = fs * 0.95, fs * 0.55
        bx, by = x1 - L * math.cos(ang), y1 - L * math.sin(ang)
        ax.add_patch(Polygon([(x1, y1), (bx + Wd * math.sin(ang), by - Wd * math.cos(ang)),
                              (bx - Wd * math.sin(ang), by + Wd * math.cos(ang))], fc="white", ec=col, lw=2.0, zorder=6))
    elif kind in ("dep", "assoc", "open"):
        L, Wd = fs * 0.8, fs * 0.42
        bx, by = x1 - L * math.cos(ang), y1 - L * math.sin(ang)
        ax.plot([bx + Wd * math.sin(ang), x1, bx - Wd * math.sin(ang)],
                [by - Wd * math.cos(ang), y1, by + Wd * math.cos(ang)], color=col, lw=2.2, zorder=6, solid_capstyle="round")
    elif kind == "solid":       # 채운 삼각형 (동기 호출)
        L, Wd = fs * 0.75, fs * 0.33
        bx, by = x1 - L * math.cos(ang), y1 - L * math.sin(ang)
        ax.add_patch(Polygon([(x1, y1), (bx + Wd * math.sin(ang), by - Wd * math.cos(ang)),
                              (bx - Wd * math.sin(ang), by + Wd * math.cos(ang))], fc=col, ec=col, lw=1.2, zorder=6))


def diamond(ax, p0, p1, fs, filled=True):
    (x0, y0), (x1, y1) = p0, p1
    ang = math.atan2(y1 - y0, x1 - x0); L = fs * 0.7; Wd = fs * 0.36
    ax.add_patch(Polygon([(x0, y0), (x0 + L * math.cos(ang) + Wd * math.sin(ang), y0 + L * math.sin(ang) - Wd * math.cos(ang)),
                          (x0 + 2 * L * math.cos(ang), y0 + 2 * L * math.sin(ang)),
                          (x0 + L * math.cos(ang) - Wd * math.sin(ang), y0 + L * math.sin(ang) + Wd * math.cos(ang))],
                         fc=INK if filled else "white", ec=INK, lw=1.6, zorder=6))


# ───────────────────────── 클래스 · 패키지 ─────────────────────────
def class_diagram(path, W, packages, boxes, rels, fs=28, legend=True, title=None):
    """packages: [dict(key?, name, row, x, w, rows=[[(key|None, wfrac), …], …], h_full=False)]
       boxes: {key: dict(kind, cid, name, attrs=[], ops=[], ref=False)}
       rels: [(a, b, kind, label, opts)]  a·b = 박스 key 또는 패키지 key · kind: real · dep · assoc · comp
       opts: ax/bx · x · y · dy · elbow · pts(geo→점 목록) · lab_at · seg · lab_dx · lab_dy"""
    LS = fs * 1.3; pad = fs * 0.7; tabh = fs * 1.6; ig = fs * 1.4; rg = fs * 2.8; top = fs * 0.6

    def bh(b):
        if b.get("ref"):
            return fs * 3.1
        if b.get("compact"):
            return fs * 2.9
        n = lambda L: sum(t.count("\n") + 1 for t in L)
        h = fs * 2.9
        if b.get("attrs") or b["kind"] not in ("interface", "dao"):
            h += LS * n(b.get("attrs", [])) + fs * (0.6 if b.get("attrs") else 0.4)
        if b.get("ops"):
            h += LS * n(b["ops"]) + fs * 0.6
        return h

    # 폭을 먼저 정하고 넘치는 줄은 ' : 반환' 앞이나 ', ' 에서 줄바꿈
    wid = {}
    for p in packages:
        pw = p["w"] * W; inner = pw - 2 * pad; gx = fs * 0.9
        for r in p["rows"]:
            tot = sum(wf for _, wf in r)
            for k, wf in r:
                if k: wid[k] = wf * inner - (gx * (len(r) - 1) / len(r) if tot >= 0.999 else 0)

    def wrap(t, w):
        if tw(t, fs * 0.8) <= w - fs * 0.6 or "\n" in t:
            return t
        if " : " in t:
            a, b2 = t.rsplit(" : ", 1)
            t2 = a + "\n      : " + b2
            if all(tw(x, fs * 0.8) <= w - fs * 0.6 for x in t2.split("\n")):
                return t2
            if "(" in a and ", " in a:
                head_, rest = a.split("(", 1)
                ps = rest.rstrip(")").split(", ")
                lines = [head_ + "(" + ps[0]]
                for q in ps[1:]:
                    if tw(lines[-1] + ", " + q, fs * 0.8) <= w - fs * 0.6:
                        lines[-1] += ", " + q
                    else:
                        lines[-1] += ","; lines.append("      " + q)
                lines[-1] += ")"
                return "\n".join(lines) + "\n      : " + b2
            return t2
        return t
    for k, b in boxes.items():
        if k in wid and not b.get("ref") and not b.get("compact"):
            b["attrs"] = [wrap(t, wid[k]) for t in b.get("attrs", [])]
            b["ops"] = [wrap(t, wid[k]) for t in b.get("ops", [])]

    prow = sorted({p["row"] for p in packages})
    ph = {}
    for p in packages:
        h = tabh + pad
        for r in p["rows"]:
            h += max(bh(boxes[k]) for k, _ in r if k) + ig + (fs * 1.1 if p.get("tabs") else 0)
        ph[id(p)] = h - ig + pad
    rowh = {r: max(ph[id(p)] for p in packages if p["row"] == r) for r in prow}
    ry, y = {}, top + (fs * 1.9 if title else 0)
    for r in prow:
        ry[r] = y; y += rowh[r] + rg
    H = int(y - rg + fs * (2.3 if legend else 0.8))
    c = Canvas(W, H)
    if title:
        c.text(fs * 0.3, top + fs * 0.55, title, fs * 1.05, bold=True)
    geo = {}
    for p in packages:
        px, pw, py = p["x"] * W, p["w"] * W, ry[p["row"]]
        h = rowh[p["row"]] if p.get("h_full") else ph[id(p)]
        twd = c.tw(p["name"], fs * 0.9, bold=True) + fs * 1.3
        c.ax.add_patch(Rectangle((px, py + tabh), pw, h - tabh, fc=PKG_BG, ec=MID, lw=1.8, zorder=1))
        c.ax.add_patch(Rectangle((px, py), twd, tabh, fc=PKG_TAB, ec=MID, lw=1.8, zorder=1))
        c.text(px + fs * 0.65, py + tabh / 2, p["name"], fs * 0.9, bold=True, zorder=2)
        if p.get("key"):
            geo[p["key"]] = (px, py + tabh, pw, h - tabh)
        yy = py + tabh + pad + (fs * 1.1 if p.get("tabs") else 0)
        if p.get("tabs"):
            c.text(px + pw - fs * 0.4, py + tabh / 2, "", fs * 0.7)
        for ri, r in enumerate(p["rows"]):
            rh_ = max(bh(boxes[k]) for k, _ in r if k)
            if p.get("tabs"):
                c.text(px + pad, yy - fs * 0.2, p["tabs"][ri], fs * 0.68, color=MID, va="bottom", zorder=2)
            gx = fs * 0.9
            tot = sum(wf for _, wf in r)
            inner = pw - 2 * pad
            if tot >= 0.999:
                xx = px + pad
                for k, wf in r:
                    w = wf * inner - gx * (len(r) - 1) / len(r)
                    if k: geo[k] = (xx, yy, w, bh(boxes[k]))
                    xx += w + gx
            else:
                used = tot * inner + gx * (len(r) - 1)
                xx = px + pad + (inner - used) / 2
                for k, wf in r:
                    w = wf * inner
                    if k: geo[k] = (xx, yy, w, bh(boxes[k]))
                    xx += w + gx
            yy += rh_ + ig + (fs * 1.1 if p.get("tabs") else 0)
    for k, b in boxes.items():
        if k not in geo:
            continue
        x, yy, w, h = geo[k]
        st = STEREO.get(b["kind"], "")
        if b.get("ref"):
            c.ax.add_patch(Rectangle((x, yy), w, h, fc="white", ec=LINE, lw=1.7, ls=(0, (6, 4)), zorder=3))
            c.text(x + w / 2, yy + fs * 0.9, (st + "  " if st else "") + b["cid"], fs * 0.74, ha="center", color=MID, zorder=4)
            sz = fs * 0.9
            while c.tw(b["name"], sz) > w - fs * 0.4 and sz > fs * 0.62: sz -= 1
            c.text(x + w / 2, yy + fs * 2.1, b["name"], sz, ha="center", zorder=4)
            continue
        hh = fs * 2.9
        if b.get("compact"):
            c.ax.add_patch(Rectangle((x, yy), w, h, fc=HEAD.get(b["kind"], "#eeeeee"), ec=LINE, lw=1.8, zorder=3))
            c.text(x + w / 2, yy + fs * 0.85, (st + "  " if st else "") + b["cid"], fs * 0.72, ha="center", color=MID, zorder=4)
            sz = fs * 0.9
            while c.tw(b["name"], sz, bold=True) > w - fs * 0.3 and sz > fs * 0.6: sz -= 1
            c.text(x + w / 2, yy + fs * 2.0, b["name"], sz, ha="center", bold=True, zorder=4)
            continue
        c.ax.add_patch(Rectangle((x, yy), w, h, fc=BODY, ec=LINE, lw=2.0, zorder=3))
        c.ax.add_patch(Rectangle((x, yy), w, hh, fc=HEAD.get(b["kind"], "#eeeeee"), ec=LINE, lw=2.0, zorder=3))
        c.text(x + w / 2, yy + fs * 0.85, (st + "  " if st else "") + b["cid"], fs * 0.74, ha="center", color=MID, zorder=4)
        c.text(x + w / 2, yy + fs * 2.05, b["name"], fs * 0.98, ha="center", bold=True, zorder=4)
        if c.tw(b["name"], fs * 0.98, bold=True) > w - fs * 0.4:
            print(f"  ! {k}: 이름 넘침 {b['name']}")
        ty = yy + hh + fs * 0.3
        secs = ["attrs", "ops"] if (b.get("attrs") or b["kind"] not in ("interface", "dao")) else ["ops"]
        for sec in secs:
            L = b.get(sec) or []
            for t in L:
                for ln in t.split("\n"):
                    sz = fs * 0.8
                    while c.tw(ln, sz) > w - fs * 0.5 and sz > fs * 0.6: sz -= 1
                    if c.tw(ln, sz) > w - fs * 0.5: print(f"  ! {k}: 넘침 {ln[:60]}")
                    c.text(x + fs * 0.4, ty + LS / 2, ln, sz, zorder=4)
                    ty += LS
            ty += fs * (0.6 if L else 0.4)
            if sec == "attrs" and "ops" in secs and b.get("ops"):
                c.ax.plot([x, x + w], [ty - fs * 0.3] * 2, color=LINE, lw=1.3, zorder=4)
    used_my = []
    for rel in rels:
        a, b_, kind, lab = rel[:4]; o = rel[4] if len(rel) > 4 else {}
        xa, ya, wa, ha = geo[a]; xb, yb, wb, hb = geo[b_]
        if "pts" in o:
            p = o["pts"](geo)
        else:
            vo = (max(ya, yb), min(ya + ha, yb + hb)); ho = (max(xa, xb), min(xa + wa, xb + wb))
            if ho[1] - ho[0] > fs * 0.8 and not o.get("elbow"):
                xx = o.get("x", (ho[0] + ho[1]) / 2)
                if xx <= 1: xx = xx * W
                p = [(xx, ya + ha if yb > ya else ya), (xx, yb if yb > ya else yb + hb)]
            elif vo[1] - vo[0] > fs * 0.8 and not o.get("elbow"):
                yy = o.get("y", (vo[0] + vo[1]) / 2)
                p = [(xa + wa if xb > xa else xa, yy), (xb if xb > xa else xb + wb, yy)]
            else:
                sx = xa + wa * o.get("ax", 0.5); ex = xb + wb * o.get("bx", 0.5)
                y1 = ya + ha if yb > ya else ya; y2 = yb if yb > ya else yb + hb
                my = (y1 + y2) / 2 + o.get("dy", 0)
                if "dy" not in o:                       # 같은 틈을 지나는 꺾은선끼리 겹치지 않게 조금씩 비킨다
                    lo_, hi_ = min(y1, y2) + fs * 0.5, max(y1, y2) - fs * 0.5
                    for _ in range(12):
                        if not any(abs(my - u) < fs * 0.45 and min(sx, ex) < x2_ and max(sx, ex) > x1_ for u, x1_, x2_ in used_my):
                            break
                        my += fs * 0.5
                        if my > hi_: my = lo_
                    used_my.append((my, min(sx, ex), max(sx, ex)))
                p = [(sx, y1), (sx, my), (ex, my), (ex, y2)]
        col = INK if kind in ("assoc", "comp") else LINE
        ls = (0, (7, 5)) if kind in ("real", "dep") else "-"
        xs_, ys_ = zip(*p)
        c.ax.plot(xs_, ys_, color=col, lw=1.9, ls=ls, zorder=2, solid_capstyle="butt")
        if kind == "comp":
            diamond(c.ax, p[0], p[1], fs)
        else:
            head(c.ax, p[-2], p[-1], kind, fs, col)
        if o.get("mult"):
            (x1, y1) = p[-1]; (x0, y0) = p[-2]
            c.text(x1 + fs * 0.35, y1 - fs * 0.7 * (1 if y1 > y0 else -1), o["mult"], fs * 0.72, color=MID, zorder=6)
        if lab:
            segs = list(zip(p[:-1], p[1:]))
            s0, s1 = segs[o["seg"]] if "seg" in o else max(segs, key=lambda s: abs(s[1][0] - s[0][0]) + abs(s[1][1] - s[0][1]))
            t = o.get("lab_at", 0.5)
            c.text(s0[0] + (s1[0] - s0[0]) * t + o.get("lab_dx", 0), s0[1] + (s1[1] - s0[1]) * t + o.get("lab_dy", 0), lab,
                   fs * 0.72, ha="center", color=MID, zorder=7, linespacing=1.1,
                   bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))
    if legend:
        yL = H - fs * 0.95; xL = fs * 0.4; fl = fs * 0.7
        for kind, lab in (("real", "구현"), ("dep", "의존 · import"), ("assoc", "연관 · 보유"), ("comp", "구성"), ("ref", "다른 도식의 클래스")):
            if kind == "ref":
                c.ax.add_patch(Rectangle((xL, yL - fs * 0.36), fs * 1.5, fs * 0.72, fc="white", ec=LINE, lw=1.4, ls=(0, (4, 3))))
                c.text(xL + fs * 1.75, yL, lab, fl, color=MID); continue
            x0, x1 = xL, xL + fs * 2.1
            c.ax.plot([x0, x1], [yL, yL], color=LINE if kind in ("real", "dep") else INK, lw=1.8,
                      ls=(0, (6, 4)) if kind in ("real", "dep") else "-")
            if kind == "comp":
                diamond(c.ax, (x0, yL), (x1, yL), fs * 0.8)
            else:
                head(c.ax, (x0, yL), (x1, yL), kind, fs * 0.8, LINE if kind in ("real", "dep") else INK)
            c.text(x1 + fs * 0.3, yL, lab, fl, color=MID); xL = x1 + fs * 0.3 + c.tw(lab, fl) + fs * 0.9
    c.save(path)
    return H


# ───────────────────────── 시퀀스 ─────────────────────────
def sequence(path, W, parts, items, fs=28, numbered=True, fit_self=False):
    """parts: [(key, 이름, 종류)]  종류: actor · controller · interface · service · component · dao · entity · boundary · external
       items (평탄화된 행): ("call", a, b, label, depth_key) · ("ret", b, a, label) · ("self", a, label) · ("note", a, b, text)
                          · ("frame", kind, cond) · ("else", cond) · ("end",) · ("act_end", key)
       반환: H"""
    n = len(parts)
    keys = [k for k, _, _ in parts]
    LIM = fs * 15.5

    def wrapl(s, lim=LIM):
        if tw(s, fs * 0.86) <= lim or "(" not in s:
            return s
        head_, rest = s.split("(", 1)
        ps = rest[:-1].split(", ") if rest.endswith(")") else [rest]
        lines = [head_ + "("]
        for i, q in enumerate(ps):
            q2 = q + (", " if i < len(ps) - 1 else ")")
            if tw(lines[-1] + q2, fs * 0.86) <= lim or (lines[-1].endswith("(") and tw(lines[-1], fs * 0.86) < lim * 0.45):
                lines[-1] += q2
            else:
                lines.append("    " + q2)
        return "\n".join(lines)
    items = [(it[0], it[1], it[2], wrapl(it[3])) + tuple(it[4:]) if it[0] == "call" else
             ((it[0], it[1], wrapl(it[2], LIM * 0.9)) if it[0] == "self" else it) for it in items]
    # 열 간격 — 이름 폭과 인접 메시지 라벨 폭으로 최소 간격을 잡고 남는 폭은 고르게
    def split_name(nm):
        if "\n" in nm or tw(nm, fs, True) <= fs * 8.6:
            return nm
        caps = [m_.start() for m_ in re.finditer(r"(?<=[a-z])[A-Z]", nm)]
        if not caps:
            return nm
        cut = min(caps, key=lambda i: abs(i - len(nm) / 2))
        return nm[:cut] + "\n" + nm[cut:]
    parts = [(k, split_name(nm), kd) for k, nm, kd in parts]
    hw = {k: max(tw(s, fs, True) for s in nm.split("\n")) + fs * 1.2 for k, nm, _ in parts}
    need = [0.0] * (n - 1)
    for it in items:
        if it[0] in ("call", "ret"):
            a, b, lab = it[1], it[2], it[3]
            ia, ib = keys.index(a), keys.index(b)
            lo, hi = min(ia, ib), max(ia, ib)
            w = max(tw(x, fs * 0.86) for x in lab.split("\n")) + fs * 1.4
            span = hi - lo
            for j in range(lo, hi):
                need[j] = max(need[j], w / span)
        elif it[0] == "self":
            ia = keys.index(it[1]); w = max(tw(x, fs * 0.86) for x in it[2].split("\n")) + fs * 3.2
            if ia < n - 1: need[ia] = max(need[ia], w)
    gaps = [max(need[j], (hw[keys[j]] + hw[keys[j + 1]]) / 2 + fs * 0.4) for j in range(n - 1)]
    m0 = hw[keys[0]] / 2 + fs * 0.3; m1 = hw[keys[-1]] / 2 + fs * 0.3
    total = m0 + sum(gaps) + m1
    if total < W:
        extra = (W - total) / max(n - 1, 1); gaps = [g + extra for g in gaps]
    else:
        W = int(total + 2)
    xs = {}; x = m0
    for j, k in enumerate(keys):
        xs[k] = x
        if j < n - 1: x += gaps[j]
    kinds = {k: kd for k, _, kd in parts}
    names = {k: nm for k, nm, _ in parts}
    top = fs * 0.5; hh = fs * (4.2 if any("\n" in nm for _, nm, _ in parts) else 3.4)
    step = fs * 1.95
    U = {"call": 1.0, "ret": 0.85, "self": 1.35, "note": 1.15, "frame": 1.25, "else": 1.0, "end": 0.45, "act_end": 0.0}
    def rowu(it):
        u = U.get(it[0], 1.0)
        if it[0] in ("call", "self"):
            u += 0.62 * (it[-1].count("\n") if it[0] == "self" else it[3].count("\n"))
        return u
    nrow = sum(rowu(it) for it in items)
    y0 = top + hh + fs * 1.6
    H = int(y0 + nrow * step + fs * 1.6)
    c = Canvas(W, H)
    # 머리
    for k in keys:
        x = xs[k]; kd = kinds[k]; nm = names[k]
        if kd == "actor":
            cy = top + fs * 0.4
            c.ax.add_patch(Circle((x, cy), fs * 0.32, fc="white", ec=INK, lw=2, zorder=4))
            c.ax.plot([x, x], [cy + fs * 0.32, cy + fs * 1.1], color=INK, lw=2, zorder=4)
            c.ax.plot([x - fs * 0.5, x + fs * 0.5], [cy + fs * 0.62, cy + fs * 0.62], color=INK, lw=2, zorder=4)
            c.ax.plot([x - fs * 0.4, x, x + fs * 0.4], [cy + fs * 1.65, cy + fs * 1.1, cy + fs * 1.65], color=INK, lw=2, zorder=4)
            c.text(x, top + hh - fs * 0.45, nm, fs * 0.92, ha="center", bold=True)
            continue
        w = hw[k] - fs * 0.2
        fill = "#ffffff" if kd in ("interface", "boundary", "external") else HEAD.get(kd, "#f2f2f2")
        c.ax.add_patch(Rectangle((x - w / 2, top), w, hh, fc=fill, ec=LINE, lw=2.0,
                                 ls=(0, (6, 4)) if kd == "external" else "-", zorder=3))
        st = STEREO.get(kd, "") if kd not in ("external", "service", "component") else ("«external»" if kd == "external" else "")
        lines = nm.split("\n")
        if st:
            c.text(x, top + fs * 0.8, st, fs * 0.72, ha="center", color=MID, zorder=4)
            c.text(x, top + fs * 0.8 + (hh - fs * 0.8) / 2 + fs * 0.25, nm, fs, ha="center", bold=True, zorder=4, linespacing=1.05)
        else:
            c.text(x, top + hh / 2, nm, fs, ha="center", bold=True, zorder=4, linespacing=1.1)
    for k in keys:
        c.ax.plot([xs[k], xs[k]], [top + hh, H - fs * 0.5], color=SOFT, lw=1.5, ls=(0, (6, 5)), zorder=0)
    AW = fs * 0.62
    BB = dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.95)
    y = y0
    stack = {k: [] for k in keys}     # 활성 시작 y 목록 (중첩)
    spans = []                        # (key, depth, y0, y1)
    frames = []

    def depth(k): return len(stack[k])

    def edge(k, toward_right):
        d = max(depth(k), 1) if kinds[k] != "actor" else 0
        if kinds[k] == "actor":
            return xs[k]
        off = AW / 2 + (d - 1) * AW * 0.45
        return xs[k] + off if toward_right else xs[k] - AW / 2

    for it in items:
        kd = it[0]
        if kd in ("call", "self"):
            extra_ = 0.62 * step * ((it[3] if kd == "call" else it[2]).count("\n"))
            y += extra_
        if kd == "call":
            _, a, b, lab = it[:4]
            right = xs[b] > xs[a]
            xa = edge(a, right) if kinds[a] != "actor" else xs[a]
            stack[b].append(y)
            dB = depth(b)
            xb = xs[b] - AW / 2 if right else xs[b] + AW / 2 + (dB - 1) * AW * 0.45
            c.ax.plot([xa, xb], [y, y], color=INK, lw=2.0, zorder=5)
            head(c.ax, (xa, y), (xb, y), "solid", fs, INK)
            c.text((xa + xb) / 2, y - fs * 0.32, lab, fs * 0.86, ha="center", va="bottom", bbox=BB, zorder=7, linespacing=1.15)
        elif kd == "ret":
            _, b, a, lab = it[:4]
            right = xs[a] > xs[b]
            dB = depth(b)
            xb = xs[b] + AW / 2 + (dB - 1) * AW * 0.45 if right else xs[b] - AW / 2
            if stack[b]:
                s = stack[b].pop(); spans.append((b, dB, s, y))
            xa = edge(a, not right) if kinds[a] != "actor" else xs[a]
            if kinds[a] != "actor":
                xa = (xs[a] + AW / 2 + (max(depth(a), 1) - 1) * AW * 0.45) if not right else xs[a] - AW / 2
            c.ax.plot([xb, xa], [y, y], color=MID, lw=1.7, ls=(0, (6, 4)), zorder=5)
            head(c.ax, (xb, y), (xa, y), "open", fs * 0.9, MID)
            if lab:
                c.text((xa + xb) / 2, y - fs * 0.28, lab, fs * 0.8, ha="center", va="bottom", color=MID, bbox=BB, zorder=7)
        elif kd == "self":
            _, a, lab = it[:3]
            d = depth(a)
            x = xs[a] + AW / 2 + (max(d, 1) - 1) * AW * 0.45; w = fs * 1.5
            y1 = y - step * 0.22; y2 = y + step * 0.32
            c.ax.plot([x, x + w, x + w, x + AW * 0.45 + 2], [y1, y1, y2, y2], color=INK, lw=1.9, zorder=5)
            head(c.ax, (x + w, y2), (x + AW * 0.45 + 2, y2), "solid", fs * 0.9, INK)
            spans.append((a, d + 1, y + step * 0.1, y2 + step * 0.25))
            c.text(x + w + fs * 0.35, y1 + step * 0.05, lab, fs * 0.86, va="center", bbox=BB, zorder=7)
            if fit_self:                                # 틀 안 자기 호출 라벨이 틀 오른쪽 선에 걸리지 않게
                xr = x + w + fs * 0.35 + max(c.tw(t_, fs * 0.86) for t_ in lab.split("\n")) + fs * 0.6
                for fr in frames: fr[5] = max(fr[5], xr)
        elif kd == "note":
            _, a, b, txt = it[:4]
            tw_ = c.tw(txt, fs * 0.8) + fs * 1.2; cx = (xs[a] + xs[b]) / 2
            half = max(tw_ / 2, abs(xs[a] - xs[b]) / 2 + fs * 2)
            x1, x2 = max(6, cx - half), min(W - 6, cx + half)
            c.ax.add_patch(Polygon([(x1, y - step * 0.38), (x2 - fs * 0.6, y - step * 0.38), (x2, y - step * 0.38 + fs * 0.6),
                                    (x2, y + step * 0.38), (x1, y + step * 0.38)], fc="#f3f3f3", ec=MID, lw=1.4, zorder=6))
            c.text((x1 + x2) / 2, y, txt, fs * 0.8, ha="center", zorder=7)
        elif kd == "frame":
            frames.append([it[1], it[2], y - step * 0.3, [], set(), 0.0])
            y += step * U["frame"]; continue
        elif kd == "else":
            frames[-1][3].append((y - step * 0.42, it[1])); y += step * U["else"]; continue
        elif kd == "end":
            fk, cond, ys, elses, used, xr = frames.pop()
            ks = [k for k in keys if k in used] or keys
            if frames:
                frames[-1][4].update(used); frames[-1][5] = max(frames[-1][5], xr + fs * 0.6)
            ix = [keys.index(k) for k in ks]
            x1 = xs[keys[min(ix)]] - fs * 2.2; x2 = max(xs[keys[max(ix)]] + fs * 2.2, xr)
            x1 = max(x1, 4); x2 = min(x2, W - 4); ye = y - step * 0.1
            c.ax.add_patch(Rectangle((x1, ys), x2 - x1, ye - ys, fc="none", ec=MID, lw=1.6, zorder=1))
            twk = c.tw(fk, fs * 0.82, bold=True) + fs * 0.9
            c.ax.add_patch(Polygon([(x1, ys), (x1 + twk, ys), (x1 + twk, ys + fs * 0.95), (x1 + twk - fs * 0.45, ys + fs * 1.35),
                                    (x1, ys + fs * 1.35)], fc="#eeeeee", ec=MID, lw=1.4, zorder=7))
            c.text(x1 + fs * 0.3, ys + fs * 0.67, fk, fs * 0.82, bold=True, zorder=8)
            c.text(x1 + twk + fs * 0.35, ys + fs * 0.67, f"[{cond}]", fs * 0.8, color=MID, zorder=8, bbox=BB)
            for ey, ec in elses:
                c.ax.plot([x1, x2], [ey, ey], color=MID, lw=1.4, ls=(0, (7, 5)), zorder=1)
                c.text(x1 + fs * 0.35, ey + fs * 0.62, f"[{ec}]", fs * 0.8, color=MID, bbox=BB, zorder=8)
            y += step * U["end"]; continue
        elif kd == "act_end":
            k = it[1]
            if stack[k]:
                d = depth(k); s = stack[k].pop(); spans.append((k, d, s, y - step * 0.35))
            continue
        for fr in frames:
            if kd in ("call", "ret"):
                fr[4].update([it[1], it[2]])
            elif kd == "self":
                fr[4].add(it[1])
            elif kd == "note":
                fr[4].update([it[1], it[2]])
        y += step * U.get(kd, 1.0)
    for k in keys:
        while stack[k]:
            d = depth(k); s = stack[k].pop(); spans.append((k, d, s, y - step * 0.3))
    for k, d, s, e in spans:
        if kinds[k] == "actor":
            continue
        x = xs[k] - AW / 2 + (d - 1) * AW * 0.45
        c.ax.add_patch(Rectangle((x, s - fs * 0.15), AW, max(e - s + fs * 0.3, fs * 0.7), fc="#e6e6e6", ec=LINE, lw=1.3, zorder=4))
    c.save(path)
    return H, W
