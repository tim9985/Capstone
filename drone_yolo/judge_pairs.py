"""
judge_pairs.py — compare_ci 출력 줄로 미리 정한 판정 규칙을 적용 (R3 · R5 · 2026-10-04)

  규칙 (_학습 큐 「10-04 R」): 짝마다 ① 한 평가셋 이상 구간 > 0  ② 평균 차 (세 평가셋) 가 반복 폭보다 큼 —
  반복 폭은 평가셋별 (obl 3.1 · v2 1.8 · kr 0.9 %p) 이라 "구간 > 0 인 평가셋에서 차 > 그 평가셋 반복 폭" 으로 본다
  ③ 어느 평가셋도 유의하게 나쁘지 않음 (구간 < 0 없음)  → 모든 짝이 ①②③ 이면 PASS
실행: python judge_pairs.py <compare_ci 출력 파일> 기준:비교 [기준:비교 …]  → 마지막 줄 PASS / FAIL (사유)
"""
import re
import sys

BAND = {"test_obl": 3.1, "test_v2": 1.8, "test_kr": 0.9}
LINE = re.compile(r"^(test_\w+)\s+(\S+) − (\S+) = ([+-][\d.]+) %p\s+\(95 % ([+-][\d.]+) ~ ([+-][\d.]+) %p")


def main():
    lines = open(sys.argv[1], encoding="utf-8").read().splitlines()
    res = {}
    for l in lines:
        m = LINE.match(l.strip())
        if m:
            tag, new, base, d, lo, hi = m.groups()
            res[(base, new, tag)] = (float(d), float(lo), float(hi))
    ok_all, why = True, []
    for pair in sys.argv[2:]:
        base, new = pair.split(":")
        rows = {t: res.get((base, new, t)) for t in BAND}
        if any(v is None for v in rows.values()):
            ok_all = False; why.append(f"{pair}: 비교 줄 없음"); continue
        gain = [t for t, (d, lo, hi) in rows.items() if lo > 0 and d > BAND[t]]
        loss = [t for t, (d, lo, hi) in rows.items() if hi < 0]
        ok = bool(gain) and not loss
        ok_all &= ok
        why.append(f"{pair}: {'통과' if ok else '탈락'} (유의 이득 {gain or '없음'} · 유의 손해 {loss or '없음'})")
    for w in why:
        print("   " + w)
    print("PASS" if ok_all else "FAIL")


if __name__ == "__main__":
    main()
