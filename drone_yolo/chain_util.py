"""
chain_util.py — 체인용 도우미 (2026-09-26 · chain_v12)

  pick  후보 기준1,기준2…  → "cand" 또는 "base" 출력 (다음 학습을 어느 데이터로 이을지 고르기)
        규칙: 세 평가셋 어디서도 **모든 기준 실행보다 유의하게 나쁘지 않고** (짝 구간 hi<0 가 아님)
              test_obl·test_v2 평균 AP 차 (후보 − 기준 실행 평균) > 0
        ※ '개선' 주장이 아니라 **이을 설정 고르기** — 개선 판정은 _학습 큐 규칙(반복 흔들림보다 큰가)으로 따로 한다
  mklist 이름 시드 목록1 [목록2 …] → configs/lists/train_<이름>.txt (합쳐 섞음) + configs/data_<이름>.yaml (val_v6b)
"""
import csv, random, sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
TAGS = ("test_obl", "test_v2", "test_kr")


def ap(model, tag):
    f = BASE / "metrics" / f"{tag}_{model}.csv"
    return float(next(csv.DictReader(open(f)))["AP50"]) if f.exists() else None


def ci_row(tag, base, model):
    f = BASE / "metrics" / f"compare_ci_{tag}.csv"
    rows = [r for r in csv.DictReader(open(f)) if r["base"] == base and r["model"] == model] if f.exists() else []
    return rows[-1] if rows else None


def pick(cand, bases):
    d, why = {}, []
    for t in TAGS:
        a = ap(cand, t); b = [x for x in (ap(m, t) for m in bases) if x is not None]
        if a is None or not b:
            why.append(f"{t} 없음"); continue
        d[t] = a - sum(b) / len(b)
        rows = [ci_row(t, m, cand) for m in bases]
        if rows and all(r is not None and float(r["hi"]) < 0 for r in rows):
            why.append(f"{t} 유의하게 나쁨 ({d[t]*100:+.1f} %p)")
            return "base", why
        why.append(f"{t} {d[t]*100:+.1f} %p")
    main = [d[t] for t in ("test_obl", "test_v2") if t in d]
    return ("cand" if main and sum(main) / len(main) > 0 else "base"), why


def mklist(name, seed, lists):
    items = []
    for l in lists:
        items += [x.strip() for x in open(BASE / "configs" / "lists" / l) if x.strip()]
    random.Random(int(seed)).shuffle(items)
    out = BASE / "configs" / "lists" / f"train_{name}.txt"
    out.write_text("\n".join(items) + "\n")
    (BASE / "configs" / f"data_{name}.yaml").write_text(
        f"# {name} (chain_v12 · 2026-09-26) — {' + '.join(lists)} · 순서 시드 {seed} · val = val_v6b\n"
        f"train: {out}\nval: {BASE / 'configs' / 'lists' / 'val_v6b.txt'}\nnc: 1\nnames: ['person']\n")
    print(f"{name}: {len(items)}장 → {out.name}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "pick":
        w, why = pick(sys.argv[2], sys.argv[3].split(","))
        print(w, "|", " · ".join(why))
    elif cmd == "mklist":
        mklist(sys.argv[2], sys.argv[3], sys.argv[4:])
