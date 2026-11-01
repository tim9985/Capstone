"""
c0_judge.py — 11-01 C0 (추론 전 흐림) 고르기 · 판정 (_학습 큐 「11-01 C0」 · 결과 보기 전에 적은 기준 그대로)
  pick   : pose_bench v12_f91 의 g06 · g08 · g10 · g12 중 Okutama 누움 · 앉음 · 서기 뜸 평균 최고 → σ 출력
  judge σ: bench_all.csv 흐림 없음 vs +g<σ> — ① 독립 8셋 AP50 평균 ≥ 0 · 어느 셋도 −2 %p 넘게 안 떨어짐 ② 도로 오탐 ≤ 1.0 · +0.2 넘게 안 늚 ③ f92 · f93 같은 방향
"""
import csv, json, sys
from pathlib import Path
B = Path(__file__).resolve().parent
IND = ["test_sar", "test_v2", "test_kr", "test_ladd_h", "test_visdrone", "test_nomad", "test_nii", "test_cy"]
if sys.argv[1] == "pick":
    d = json.load(open(B / "metrics/pose_bench_v12_f91_swa.json")); sc = {}
    for c in ("g06", "g08", "g10", "g12"):
        if c in d:
            sc[c] = sum(d[c][k]["뜸"] for k in ("OK-Lying", "OK-Sitting", "OK-Standing")) / 3
    best = max(sc, key=sc.get); print(json.dumps(sc), file=sys.stderr); print(int(best[1:]) / 10)
else:
    g = f"+g{round(float(sys.argv[2]) * 10):02d}"; R = {(r["model"], r["set"]): r for r in csv.DictReader(open(B / "metrics/bench_all.csv")) if r["kind"] == "ours"}
    ok = True; means = {}
    for m in ("v12_f91_swa", "v12_f92_swa", "v12_f93_swa"):
        d = {s: float(R[(m + g, s)]["AP50"]) - float(R[(m, s)]["AP50"]) for s in IND if (m + g, s) in R and (m, s) in R}
        means[m] = sum(d.values()) / len(d); fp0, fp1 = float(R[(m, "test_auair")]["FP/neg@0.2"]), float(R[(m + g, "test_auair")]["FP/neg@0.2"])
        obl = float(R[(m + g, "test_obl")]["AP50"]) - float(R[(m, "test_obl")]["AP50"])
        print(m, "셋별 %p", {s[5:]: round(v * 100, 1) for s, v in d.items()}, "평균", round(means[m] * 100, 2), "· 도로 오탐", fp0, "→", fp1, "· (참고) Okutama", round(obl * 100, 1))
        if m == "v12_f91_swa":
            c1 = means[m] >= 0 and min(d.values()) >= -0.02; c2 = fp1 <= 1.0 and fp1 - fp0 <= 0.2
            print("  ① 평균 ≥ 0 · 최저 ≥ −2 %p:", c1, "· ② 오탐:", c2); ok = c1 and c2
    c3 = all((v >= 0) == (means["v12_f91_swa"] >= 0) for v in means.values()); print("  ③ 반복 같은 방향:", c3)
    print("C0 판정:", "PASS ✅" if ok and c3 else "FAIL ❌")
