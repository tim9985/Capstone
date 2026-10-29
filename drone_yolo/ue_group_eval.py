"""
ue_group_eval.py — 후보 묶기 규칙 비교 (10-15 G · 덤프 = ue_group_dump.py · 판정 기준 = _학습 큐 「10-15 G」)

  worker 의 후보 흐름 (추적 키 → 같은 후보 attach · 아니면 registry.update) 을 그대로 다시 돌리되 규칙만 바꾼다
  규칙 (cfg)
    min_gate · gate_k · gate_mode ("sum" = σa+σb · 지금 worker · "rss" = √(σa²+σb²))
    joint      같은 프레임의 좌표 있는 탐지 ↔ 후보 짝을 한꺼번에 (헝가리안 · 가까운 쪽 우선)
    veto_k     추적기가 붙이려는 후보가 좌표상 문 × veto_k 보다 멀면 붙이지 않음 (None = 끔)
    motion     후보 속도 (최근 두 위치) 로 지금 위치 예측 · 놓친 시간만큼 문을 걷는 속도 1.5 m/s × dt 만큼 넓힘
    merge_late 좌표 없던 후보가 처음 좌표를 얻을 때 문 안의 다른 후보가 있으면 그쪽으로 합침
    app        외형 특징 (10-15 R · ue_reid_feats.py) — None · "hist" (색) · "dino" (DINOv2 · feat 가 있으면 SigLIP 2) · "both"
    feat       (10-30 R2) 외형 임베딩 파일 — None (DINOv2) · "siglip_upper" · "siglip_person" (ue_reid_siglip.py)
    app_veto   외형 거리가 이보다 크면 그 후보에 붙이지 않음 (추적 키 · 좌표 짝 모두)
    app_w      좌표 짝 비용에 외형 거리 가중 (거리/문 + app_w × 외형)
    regain     외형 거리가 이보다 작으면 문을 regain_k 배로 넓힘 (오래 놓친 뒤 다시 잇기)
    pend_k     좌표 「보류」 탐지 (기울기 · 자세 변화 · 발끝 끝) 도 계산된 위치가 있으면 **같은 사람 판단에만** 씀 · 오차 σ × pend_k (보고 · 후보 위치 평균엔 안 씀) · None = 끔
  지표 (UE 정답 배우 기준)
    사람당 후보 · 섞인 후보 (두 배우가 각각 5장 이상) · 후보 IDF1 (배우 ↔ 후보 1:1 · 겹친 장 수)
실행  python ue_group_eval.py --runs S1_a16,S2_a16,S3_a16,S4_a16 [--grid]
"""
import argparse, itertools, json, math, pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

BASE = Path(__file__).resolve().parent
K = 111_320.0


def metres(a, b, c, d):
    return math.hypot((c - a) * K, (d - b) * K * math.cos(math.radians(a)))


class Cand:
    __slots__ = ("cid", "lat", "lon", "sig", "last_t", "last_box", "hist", "root", "ah", "ad")

    def __init__(self, cid, t):
        self.cid, self.lat, self.lon, self.sig, self.last_t, self.last_box, self.hist, self.root = cid, None, None, None, t, None, [], None
        self.ah, self.ad = None, None                                     # 외형 이동 평균 (색 히스토그램 · DINOv2)


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    it = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - it
    return it / u if u > 0 else 0.0


class Registry:
    def __init__(self, cfg):
        self.c, self.items, self.n = cfg, [], 0

    def root(self, c):
        while c.root is not None:
            c = c.root
        return c

    def new(self, t):
        self.n += 1; c = Cand(self.n, t); self.items.append(c); return c

    def pred(self, c, t):
        if self.c["motion"] and len(c.hist) >= 2:
            (t0, a0, b0), (t1, a1, b1) = c.hist[-2], c.hist[-1]
            if t1 - t0 > 0.05:
                v = 1.0 / (t1 - t0)
                dn, de = (a1 - a0) * K * v, (b1 - b0) * K * math.cos(math.radians(a1)) * v
                sp = math.hypot(dn, de)
                if sp > 3.0:                                          # 사람 걷기 이상은 잡음
                    dn, de = dn * 3.0 / sp, de * 3.0 / sp
                dt = t - c.last_t
                return c.lat + dn * dt / K, c.lon + de * dt / (K * math.cos(math.radians(c.lat)))
        return c.lat, c.lon

    def adist(self, c, d):
        """외형 거리 0~1 (없으면 None)"""
        mode, a = self.c.get("app"), d.get("app")
        if not mode or a is None or c.ah is None:
            return None
        h = float(np.sqrt(max(0.0, 1.0 - float(np.sum(np.sqrt(c.ah * a[0]))))))       # 바타차리야
        e = float(1.0 - np.dot(c.ad, a[1]) / max(np.linalg.norm(c.ad), 1e-6)) / 2   # (1 − cos)/2
        return {"hist": h, "dino": e, "both": (h + e) / 2}[mode]

    def app_ok(self, c, d):
        x = self.adist(c, d)
        return x is None or not self.c.get("app_veto") or x <= self.c["app_veto"]

    def gate(self, c, sig, t, d=None):
        if d is not None and self.c.get("regain"):
            x = self.adist(c, d)
            if x is not None and x <= self.c["regain"]:
                return self._gate(c, sig, t) * self.c.get("regain_k", 2.0)
        return self._gate(c, sig, t)

    def _gate(self, c, sig, t):
        s1, s2 = (c.sig or sig), sig
        g = self.c["gate_k"] * ((s1 + s2) if self.c["gate_mode"] == "sum" else math.hypot(s1, s2))
        g = max(self.c["min_gate"], g)
        if self.c["motion"]:
            g += 1.5 * max(0.0, t - c.last_t - 1.0)
        return g

    def dist(self, c, geo, t):
        la, lo = self.pred(c, t)
        return metres(la, lo, geo[1], geo[2])

    def attach(self, c, t, box, geo, app=None):
        c = self.root(c)
        if app is not None and self.c.get("app"):
            if c.ah is None:
                c.ah, c.ad = app[0].copy(), app[1].copy()
            else:
                c.ah, c.ad = 0.8 * c.ah + 0.2 * app[0], 0.8 * c.ad + 0.2 * app[1]
        first_geo = c.lat is None and geo is not None
        c.last_t, c.last_box = t, box
        if geo is not None:
            sig = max(geo[3][0], geo[3][1])
            if c.lat is None:
                c.lat, c.lon, c.sig = geo[1], geo[2], sig
            elif self.c["motion"]:                                   # 움직이는 사람은 최근 위치 쪽으로 (평균하면 뒤처진다)
                c.lat, c.lon, c.sig = 0.5 * c.lat + 0.5 * geo[1], 0.5 * c.lon + 0.5 * geo[2], min(c.sig, sig)
            else:
                wa, wb = 1 / c.sig ** 2, 1 / sig ** 2
                c.lat, c.lon = (c.lat * wa + geo[1] * wb) / (wa + wb), (c.lon * wa + geo[2] * wb) / (wa + wb)
                c.sig = (wa + wb) ** -0.5
            c.hist.append((t, geo[1], geo[2])); c.hist = c.hist[-6:]
            if first_geo and self.c["merge_late"]:
                for o in self.items:
                    o = self.root(o)
                    if o is c or o.lat is None or o.last_t == t:
                        continue
                    if metres(o.lat, o.lon, c.lat, c.lon) <= self.gate(o, c.sig, t):
                        c.root = o; o.last_t, o.last_box = t, box; return o
        return c

    def mg(self, d):
        """같은 사람 판단에 쓸 위치 — OK 는 그대로 · 보류는 pend_k 가 있으면 σ 를 키워서"""
        g = d["geo"]
        if g is None:
            return None
        if g[0] == "OK":
            return g
        if self.c.get("pend_k") and g[1] is not None and g[3]:
            return ("PEND", g[1], g[2], (g[3][0] * self.c["pend_k"], g[3][1] * self.c["pend_k"], g[3][2]))
        return None

    def frame(self, t, dets, tcand):
        """한 프레임 탐지 목록 → 후보 (root) 목록"""
        out = [None] * len(dets)
        used = set()
        free = []
        for i, d in enumerate(dets):                                  # ① 추적 키
            g = d["geo"] if d["geo"] and d["geo"][0] == "OK" else None
            k = tcand.get(d["tkey"]) if d["tkey"] else None
            k = self.root(k) if k is not None else None
            gm = self.mg(d)
            if k is not None and k.last_t != t and id(k) not in used and not self.app_ok(k, d):
                free.append(i); continue
            if k is not None and k.last_t != t and id(k) not in used:
                if self.c["veto_k"] and gm is not None and k.lat is not None and self.dist(k, gm, t) > self.c["veto_k"] * self.gate(k, max(gm[3]), t):
                    free.append(i); continue
                out[i] = self.attach(k, t, d["box"], g, d.get("app")); used.add(id(out[i]))
            else:
                free.append(i)
        geo_i = [i for i in free if self.mg(dets[i]) is not None]
        cands = [c for c in self.items if c.root is None and c.lat is not None and c.last_t != t and id(c) not in used]
        if self.c["joint"] and geo_i and cands:                       # ② 좌표 짝을 한꺼번에
            M = np.full((len(geo_i), len(cands)), 1e6)
            for a, i in enumerate(geo_i):
                g = self.mg(dets[i])
                for b, c in enumerate(cands):
                    dd = self.dist(c, g, t); gt = self.gate(c, max(g[3]), t, dets[i])
                    if dd <= gt and self.app_ok(c, dets[i]):
                        x = self.adist(c, dets[i])
                        M[a, b] = dd / gt + (self.c.get("app_w", 0.0) * x if x is not None else 0.0)
            for a, b in zip(*linear_sum_assignment(M)):
                if M[a, b] < 1e6:
                    i = geo_i[a]; gi = dets[i]["geo"] if dets[i]["geo"] and dets[i]["geo"][0] == "OK" else None
                    out[i] = self.attach(cands[b], t, dets[i]["box"], gi, dets[i].get("app")); used.add(id(out[i]))
        for i in free:                                                # ③ 남은 것 — 지금 worker 와 같은 순서 (좌표 → 화면 IoU → 새 후보)
            if out[i] is not None:
                continue
            d = dets[i]; g = d["geo"] if d["geo"] and d["geo"][0] == "OK" else None
            gm = self.mg(d)
            m = None
            if gm is not None:
                best = None
                for c in self.items:
                    if c.root is not None or c.lat is None or c.last_t == t or id(c) in used:
                        continue
                    dd = self.dist(c, gm, t)
                    if dd <= self.gate(c, max(gm[3]), t, d) and self.app_ok(c, d) and (best is None or dd < best[0]):
                        best = (dd, c)
                m = best[1] if best else None
            if m is None:
                best = None
                for c in self.items:
                    if c.root is None and c.last_box is not None and 0 < t - c.last_t <= 1.0 and id(c) not in used and self.app_ok(c, d):
                        o = iou(d["box"], c.last_box)
                        if o >= 0.3 and (best is None or o > best[0]):
                            best = (o, c)
                m = best[1] if best else None
            out[i] = self.attach(m if m is not None else self.new(t), t, d["box"], g, d.get("app")); used.add(id(out[i]))
        for i, d in enumerate(dets):
            if d["tkey"]:
                tcand[d["tkey"]] = out[i]
        return out


def evaluate(dump, cfg, feats=None):
    if feats is not None:
        for f, ff in zip(dump["frames"], feats):
            for d, a in zip(f["dets"], ff):
                d["app"] = a
    reg, tcand = Registry(cfg), {}
    obs = []                                                          # (배우, 후보 객체)
    n_vis = 0
    for f in dump["frames"]:
        n_vis += len(f["visible"])
        cs = reg.frame(f["t"], f["dets"], tcand)
        obs += [(d["actor"], c) for d, c in zip(f["dets"], cs)]
    obs = [(a, reg.root(c).cid) for a, c in obs]
    per_act = defaultdict(Counter); per_c = defaultdict(Counter)
    for a, c in obs:
        if a:
            per_act[a][c] += 1
        per_c[c][a] += 1
    acts = [a for a, cn in per_act.items() if sum(cn.values()) >= 10]
    cpa = float(np.mean([len(per_act[a]) for a in acts])) if acts else None
    mixed = sum(1 for c, cn in per_c.items() if sum(1 for a, k in cn.items() if a and k >= 5) >= 2)
    pairs = sorted(((k, a, c) for a, cn in per_act.items() for c, k in cn.items()), reverse=True)
    ua, uc, idtp = set(), set(), 0
    for k, a, c in pairs:
        if a not in ua and c not in uc:
            ua.add(a); uc.add(c); idtp += k
    idf1 = 2 * idtp / max(n_vis + len(obs), 1)
    return {"cand_per_actor": round(cpa, 2) if cpa else None, "mixed": mixed, "idf1": round(idf1, 3),
            "cands": len(per_c), "fp_cands": sum(1 for c, cn in per_c.items() if set(cn) == {None})}


BASECFG = dict(min_gate=3.0, gate_k=3.0, gate_mode="sum", joint=False, veto_k=None, motion=False, merge_late=False, pend_k=None)


_CACHE = {}


def load(r, feat=None):
    """feat (10-30 R2): None = R 의 DINOv2 (reid_<run>.pkl) · "siglip_upper" · "siglip_person" (reid_siglip_<part>_<run>.pkl · 외형 칸 「dino」 자리에 SigLIP 2)"""
    key = (r, feat)
    if key not in _CACHE:
        d = pickle.loads((BASE / "runs_state" / f"group_{r}.pkl").read_bytes())
        fp = BASE / "runs_state" / (f"reid_{r}.pkl" if not feat else f"reid_{feat}_{r}.pkl")
        _CACHE[key] = (d, pickle.loads(fp.read_bytes())["feats"] if fp.exists() else None)
    return _CACHE[key]


def run_all(runs, cfg):
    res = {r: evaluate(load(r, cfg.get("feat"))[0], cfg, load(r, cfg.get("feat"))[1]) for r in runs}
    agg = {"cand_per_actor": round(float(np.mean([v["cand_per_actor"] for v in res.values()])), 2),
           "mixed": sum(v["mixed"] for v in res.values()), "idf1": round(float(np.mean([v["idf1"] for v in res.values()])), 3)}
    return agg, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="S1_a16,S2_a16,S3_a16,S4_a16")
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--cfg", default=None, help="json — 하나만")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    runs = a.runs.split(",")
    if a.cfg:
        cfg = dict(BASECFG, **json.loads(a.cfg)); agg, res = run_all(runs, cfg)
        print(json.dumps({"cfg": cfg, "agg": agg, "runs": res}, ensure_ascii=False))
        if a.out:
            Path(a.out).write_text(json.dumps({"cfg": cfg, "agg": agg, "runs": res}, ensure_ascii=False, indent=1))
        return
    base, _ = run_all(runs, BASECFG); print("기준 (지금 worker)", base)
    rows = []
    grid = itertools.product((2.0, 3.0, 4.0), (2.0, 3.0), ("sum", "rss"), (False, True), (None, 2.0), (False, True), (False, True), (None, 1.5, 3.0)) if a.grid else []
    for mg, gk, gm, jo, ve, mo, ml, pk in grid:
        cfg = dict(min_gate=mg, gate_k=gk, gate_mode=gm, joint=jo, veto_k=ve, motion=mo, merge_late=ml, pend_k=pk)
        agg, _ = run_all(runs, cfg); rows.append((cfg, agg))
    ok = [r for r in rows if r[1]["mixed"] <= base["mixed"]]
    ok.sort(key=lambda r: (-r[1]["idf1"], r[1]["cand_per_actor"]))
    for cfg, agg in ok[:12]:
        print(agg, cfg)
    if a.out:
        Path(a.out).write_text(json.dumps({"base": base, "top": [{"cfg": c, "agg": g} for c, g in ok[:30]], "all": len(rows)}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
