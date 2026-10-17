
# chain_c (10-14) — 선생 모델 의사 라벨

| 모델 | obl | v2 edge+무시 | kr edge |
|---|---|---|---|
| v15pl_f91 | 0.626 | 0.6445 | 0.9999 |
| v15pl_f91_swa | 0.6263 | 0.6493 | 1.0 |
| v15pl_f92 | 0.6214 | 0.6192 | 0.9989 |
| v15pl_f92_swa | 0.6313 | 0.6218 | 0.9896 |
- ★ 평균[v15pl swa 2] vs 평균[v12 swa 3]:
test_obl  평균[v15pl_f91_swa+v15pl_f92_swa] − 평균[v12_f91_swa+v12_f92_swa+v12_f93_swa] = +0.5 %p  (95 % -0.2 ~ +1.3 %p · 블록 48) → 구분 안 됨
test_v2_edge_ign  평균[v15pl_f91_swa+v15pl_f92_swa] − 평균[v12_f91_swa+v12_f92_swa+v12_f93_swa] = -0.2 %p  (95 % -1.6 ~ +1.2 %p · 블록 72) → 구분 안 됨
test_kr_edge  평균[v15pl_f91_swa+v15pl_f92_swa] − 평균[v12_f91_swa+v12_f92_swa+v12_f93_swa] = -0.5 %p  (95 % -0.5 ~ -0.0 %p · 블록 61) → 차이 있음
