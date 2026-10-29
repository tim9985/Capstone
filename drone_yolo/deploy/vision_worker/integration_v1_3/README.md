# 비전 worker 1.3 — 상의 색 판정 교체 (2026-10-29 · 10-30 잔디 문장 추가)

지금 배포 판 (SIM `drone-dev` · REAL `drone-real`) **위에** 상의 색 판정기만 바꿉니다. 탐지 · 추적 · 후보 · 좌표 · 이착륙 · 계약은 그대로입니다.
근거 · 수치 원본: `obsidian/11 서버 학습 계획/_학습 큐.md` 「10-29 K」 · 결정 `obsidian/05 결정/결정 - 상의 색 SigLIP 2 제로샷 (worker 1.3).md`

## 1. 무엇이 바뀌나

| | 지금 (규칙 V10) | 1.3 |
|---|---|---|
| 판정 | HSV 문턱 규칙 (GrabCut · 화이트밸런스 · 노출) · CPU ≈ 10 ms/명 | **SigLIP 2 base 제로샷** — 상의 크롭 (박스 위 10~62 %) ↔ 12색 문장 + **잔디 문장** (크롭이 「green grass」 에 「초록 셔츠」 보다 가까운 만큼 초록을 깎음 · 10-30) + **남색 문턱** (남색 − 검정 ≤ 0.95 면 검정) · GPU 3 ms/명 (한 명씩) |
| 누적 · 찾는 색 비교 | ColorAccumulator · match | **그대로** (반환 형식 같음) |
| 새 패키지 | — | **없음** — 이미지 인코더를 `color.py` 안에 plain torch 로 구현 · transformers 안 씀 |
| 모델 파일 | — | `/models/color_siglip2/` (186 MB · 읽기 전용 · 해시 선검사) |
| GPU 메모리 | — | 약 +0.4 GB |
| 상태 파일 | — | `color_method`: `siglip2-zs` (정상) · `rule-v10` (꾸러미 없음 · 실패 → 규칙으로 계속) |

색 균형 1순위 (색마다 정확도의 평균 · 규칙 → 1.3)

| 평가셋 | 규칙 V10 | 1.3 |
|---|---|---|
| UAV-Human 드론 시험 30명 (키 72 px) | 0.286 | 0.512 |
| NOMAD 드론 42명 (한 장) | 0.336 | 0.42 |
| PETA 지상 CCTV 1,606장 (확인용 · 처음 봄) | 0.270 | 0.560 |
| 디관 사진 106명 (확인용) | 0.314 | 0.744 |
| 디관 영상 45명 (확인용) | 0.422 | 0.885 |
| UE S1~S4 20 m (8색 · 정답 박스) | 0.393 | 0.536 |

- 풀밭 위 「초록」 질의 (누운 사람 크롭에 풀) — 잔디 문장으로 **다른 사람 일치 UE 관측 0.13 → 0.06 · 배포 경로 S1_a16 누적 0.32 → 0.085** (규칙 0.002 · 대신 규칙의 「검정」 질의 오일치 0.29~0.39 는 0.01) · 실사진 5셋 손해 없음 · 진짜 초록 옷 찾기 그대로
- 검정 ↔ 남색 · 흰 ↔ 회색 은 여전히 헷갈림 (남색 문턱으로 디관 사진 검정 → 남색 26 → 16)

## 2. 할 일 (통합 담당 · SIM · REAL 각각)

1. **모델 꾸러미 복사** — 서버 `drone_yolo/runs_person/_pack_v12_sim/color_siglip2/` (SIM) · `_pack_v12_real/color_siglip2/` (REAL) → 각 vision 컨테이너의 `/models/color_siglip2/` (`$INGEST_MODEL_DIR/color_siglip2`)
   - `vision.pt` sha256 `3b809f9b23e678a7…` · `text.npy` `5becc31667d016f5…` · `grass.npy` `37f928db3c8be558…` · `color_model.json` `b5334e9721ca9643…` (worker 가 json 안 해시로 다시 확인)
2. **`color.py` 덮어쓰기** — `integration_v1_3/app/vision_core/color.py` → `app/vision_core/color.py` (sha256 `069356e71c91b651…`)
   - 지금 배포 판 color.py 는 비전 10-05 판과 같음 (규칙 부분은 그대로 들어 있음 · `exp_clip` 인자만 추가)
3. **`ingest.py` 패치** (파일을 바꾸지 않고 3줄만 — 통합 담당이 고친 부분 보존)
   - SIM: `patch -p1 -d <SIM 배포 판 루트> -i ingest_dev.patch` (배포 판 sha256 앞 8자리 `a0fec6cc` 기준)
   - REAL: `patch -p1 -d <REAL 배포 판 루트> -i ingest_real.patch` (`ca6fbe81` 기준)
   - 바뀌는 것: `ColorProfiler()` → `make_colorer(...)` · `stats["color_method"]` 추가
4. 재시작 → 상태 파일 `color_method` = `siglip2-zs` 확인
- 되돌리기: 환경 변수 `VISION_COLOR=rule` (재시작) — 파일 그대로 규칙 V10 · 꾸러미를 빼도 같음
- `VISION_COLOR=siglip` 이면 꾸러미 문제 때 시작 실패 (조용히 규칙으로 가지 않게 하고 싶을 때)

## 3. 시험 (서버 · 10-29)

`tests_color_siglip.py` — 실제 배포 판 (`drone-dev` · `drone-real` vision_core) 복사 + 이 파일들 적용 · **transformers 없는 drone 환경** (= 배포 이미지 조건)

| | SIM (dev) | REAL (real) |
|---|---|---|
| 판정기 고르기 (꾸러미 · rule · 없음 · 해시 틀림) | ✅ | ✅ |
| 평가판 (color_lab) 과 1순위 일치 · 디관 영상 45명 | 1.000 | 1.000 |
| 한 명 판정 (3090 · 한 명씩) | 3.2 ms | 3.2 ms |
| UE S1_a16 배포 경로 · 백엔드 계약 (REAL · SIM) | 746 / 746 | 746 / 746 |
| 사람당 후보 | 3.33 (그대로) | 3.33 |
| 찾는 색 green — 대상 MATCH · 다른 사람 | 0.959 · **0.085** (잔디 문장 전 0.322 · 규칙 0.45 · 0.002) | 같음 |

- 이착륙 막기 (`tests_low_agl.py`) · 가장자리 규칙 (`tests_edge_confirm.py`) 숫자 그대로

## 4. 파일

| 파일 | 내용 |
|---|---|
| `app/vision_core/color.py` | 규칙 V10 (그대로) + `SiglipColorProfiler` (plain torch SigLIP 이미지 인코더 · 잔디 문장 · 남색 문턱) + `make_colorer` |
| `ingest_dev.patch` · `ingest_real.patch` | 배포 판 ingest.py 3줄 |
| `tests_color_siglip.py` | 위 시험 |
| 꾸러미 만들기 | `drone_yolo/export_color_pack.py` (git 밖 `runs_person/_pack_color_siglip2/` · 평가판과 같은 판정인지 검사 포함) |
| 평가 | `drone_yolo/color_lab.py` · `metrics/color_lab*.json` |

- 모델 라이선스: SigLIP 2 = Apache 2.0 (상업 가능) · 남색 문턱 δ 는 UAV-Human 학습셋 (연구용) 으로 정한 숫자 하나 — 데이터 자체는 꾸러미에 없음
