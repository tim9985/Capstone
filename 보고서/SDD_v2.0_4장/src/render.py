"""render.py — 등록된 CD · SD 그림을 out2/ 로 렌더"""
import sys, json, importlib
from pathlib import Path
import mono
from mdl import CDS, SEQS
from seq import flatten
OUT = Path(__file__).resolve().parent / "out2"; OUT.mkdir(exist_ok=True)
UNITS_ = [f"u{i:02d}" for i in range(1, 13)]


def load(units):
    for u in units:
        importlib.import_module(u)


def render_cd(cd):
    sizes = {}
    parts = CDS[cd]["parts"]
    parts = parts() if callable(parts) else parts
    for key, sub, P, B, R in parts:
        H = mono.class_diagram(str(OUT / f"{key}.png"), 2000, P, B, R, fs=28)
        sizes[key] = (2000, H)
    return sizes


def render_sd(sd):
    if SEQS[sd].get("split"):                      # 큰 통합 시퀀스는 나눠 그리고 번호를 잇는다
        out, n0, P = {}, 0, SEQS[sd]["split_parts"]
        for i, (sub, keys, flow) in enumerate(SEQS[sd]["split"], start=1):
            part = dict(parts=[P[k] for k in keys], flow=flow)
            items, steps = flatten(part, start=n0); n0 += len(steps)
            H, W = mono.sequence(str(OUT / f"{sd}_{i}.png"), 2000, part["parts"], items, fs=28, fit_self=True)
            out[f"{sd}_{i}"] = (W, H)
        return out
    items, steps = flatten(SEQS[sd])
    H, W = mono.sequence(str(OUT / f"{sd}.png"), 2000, SEQS[sd]["parts"], items, fs=28)
    return {sd: (W, H)}


if __name__ == "__main__":
    units = [a for a in sys.argv[1:] if Path(a + ".py").exists()] or UNITS_
    only = [a for a in sys.argv[1:] if not Path(a + ".py").exists()]
    load(units)
    sizes = {}
    for cd in CDS:
        if not only or cd in only: sizes.update(render_cd(cd))
    for sd in SEQS:
        if not only or sd in only: sizes.update(render_sd(sd))
    print(json.dumps(sizes))
