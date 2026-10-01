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
