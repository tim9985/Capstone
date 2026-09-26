# 60° 대기열 결과 (chain_v6) — AP50

판정: test_obl 이 기준선 v3_place 대비 +0.01 초과 = 개선

| 모델 | 평가셋 | AP50 |
|---|---|---:|
| p2m_place | test_v2 | 0.2697 |
| p2m_place | test_obl | 0.1309 |
| v3_place | test_obl | 0.1029 |
| fov_11m_1280_all | test_obl | 0.1661 |
| v6_obl | test_obl | 0.3311 |
| v6_obl | test_v2 | 0.5287 |

**v6_obl 판정 (test_obl)**: 기준선 v3_place 0.1029 → 0.3311 (+22.8 %p · ✅ 개선)
| v6_p2m | test_obl | 0.3333 |
| v6_p2m | test_v2 | 0.5381 |

**v6_p2m 판정 (test_obl)**: v6_obl 0.3311 → 0.3333 (+0.2 %p · ➖ 동등)
| v6_obl_r2 | test_obl | 0.6128 |
| v6_obl_r2 | test_v2 | 0.5156 |
| v6_nwd | test_obl | 0.5893 |
| v6_nwd | test_v2 | 0.5151 |
