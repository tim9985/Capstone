# 비전 worker 1.2 교체 파일 — 통합 v1.1 판 위에 (2026-10-14)

통합 담당이 배포한 v1.1 (`app.vision_ingest` · SIM leased · REAL JSONL) 을 **바탕으로** 비전 1.2 기능만 얹은 파일입니다.
v1.1 규칙 (해시 선검사 · CUDA 필수 · 빠진 자세를 채우지 않음 · 좌표 기본 PENDING · 64 KiB · 재전송) 은 그대로입니다.

## 0. 준비물 (이 폴더 말고)

| 무엇 | 어디 | 비고 |
|---|---|---|
| 모델 꾸러미 (가중치 · 엔진 · posture.json) | 서버 `/home/se/JupyterLAB/Capstone/drone_yolo/runs_person/_pack_v12_real/` · `_pack_v12_sim/` | git 밖 (가중치 비공개) · `/home/se` 는 se 계정만 읽기 → sudo 로 복사 |
| 파이썬 패키지 `lap` (0.5.12 이상) | vision 이미지 | BoT-SORT 추적에 필요 · 이미지 안에 없으면 상태 (추적 · 누움 등급) 만 꺼지고 나머지는 동작 — 상태 파일 `state_disabled` 로 보임 · `network_mode: none` 이라 실행 중 설치 안 됨 → 이미지에 넣어야 함 |
| 나머지 모듈 (`color.py` · `coord_transform.py` · `common.py` …) | 지금 배포된 것 그대로 | 이 폴더 파일은 배포 판 위에 덮어쓰는 것만 |

## 1. 무엇이 바뀌나

| 기능 | v1.1 (지금) | 1.2 (이 파일) | 끄는 법 |
|---|---|---|---|
| 모델 | soup_v7r2 · conf 0.15 · PyTorch | **v12_f91_swa · conf 0.20 · TensorRT** (model.json 의 엔진 해시가 맞을 때만 · 아니면 PyTorch) | `VISION_TRT=0` |
| 후보 (REAL) | 화면 겹침만 → **사람당 후보 10~19** | 추적 키 + pose 로 계산한 위치 (내부용) → **4~5** | `VISION_GEO_GROUPING=0` |
| 후보 (SIM) | 프레임마다 새 id | 임무 안에서 같은 사람 = 같은 id (REAL 과 같은 코드) | — |
| 추적 · 상태 (요구조자) | `state: null` | BoT-SORT · 누움 점수 · 무동작 · 등급 (`/models/posture.json`) | posture.json 빼기 |
| 인상착의 비교 | 없음 | `appearance.match` (프레임 `appearance_query` 또는 `/scratch/appearance_query.json`) | 조건 없으면 안 함 |
| 좌표 보고 | 항상 PENDING | 기본 **그대로 PENDING** · `VISION_GEO_CLAIMS=1` 이면 pose 로 계산한 **worker 주장 VALID** (오차 타원 · method `FOOT_RAY_POSE…`) | 기본 꺼짐 |
| 이착륙 프레임 (10-17) | 후보 생김 (기체 그림자 오탐) | 지면 위 **`VISION_MIN_AGL_M` (기본 10 m)** 아래 프레임은 탐지만 · 후보 · 관측 없음 (REAL `SKIPPED` 사유 `LOW_ALTITUDE …` · SIM stats `low_altitude_frames` · 상태 파일 `dem_loaded` · `min_agl_m`) · 지면 위 높이 = MSL 고도 − DEM 또는 AGL 고도 · **근거 없으면 막지 않음** · SIM · REAL 둘 다 | `VISION_MIN_AGL_M=0` |

- 좌표 계산 = `frame-pose/1.0` 의 `camera_attitude` (CAMERA_FRD_TO_NED) + `camera_position_ned` 로 발끝 광선 · 짐벌 · 마운트 가정 없음
- 카메라 자세가 없으면 `body_attitude × mount` · 둘 다 없으면 보류 (`CAMERA_ATTITUDE_MISSING`)
- 지면: 지형 함수가 없으면 보류 (`GROUND_UNKNOWN`)
  - `VISION_GEO_AGL_FLAT=1` 이고 `altitude_datum = AGL` 이면 「기체 아래와 같은 높이」 가정으로 계산
  - 이때 가정은 method 에 `+LOCAL_FLAT_AGL` 로 표시
- **지면 = DEM (10-17 추가)**: `VISION_DEM_DIR` = Terrarium 타일 폴더 (z16 · 팀 캠퍼스 `drone-dev/data/spatial/campus-…/terrain` · 읽기 전용 마운트) · pose 고도 기준이 **MSL** 이면 광선 × DEM (method `FOOT_RAY_POSE+DEM`)
  - UE AirSim 고도 = PlayerStart MSL 71.429 − NED z (로컬 실측) = MSL 이 맞음 → **연결기 · settings 고칠 것 없음** · 팀 DEM = UE 지형 (사람 13명 발밑 차 −0.08 ~ −0.10 m)
  - UE 8회 정답 박스: 수평 오차 중앙 **0.34~0.61 m · P95 0.6~2.3 m** (장면 상수 평지 0.65~1.43 · 2.1~4.2 m)
  - SIM 시연에서 좌표를 보이려면 `VISION_DEM_DIR` + `VISION_GEO_CLAIMS=1` (SIM 만 · worker 주장 · 공식 승격은 백엔드 규칙대로)
- **이착륙 후보 막기 (10-17)**: 노트북 Kumoh 경로 녹화 298장 재생 (`tests_low_agl.py`) — 낮은 구간 관측 28 → **0** · 높은 구간 34 → 34 (그대로) · 후보 23 → 20 · DEM 없으면 298장 모두 「근거 없음」 → 막지 않음 (지금과 같음) · REAL `SKIPPED` 본문 계약 통과 · 남은 오탐 = 화면 위 가장자리 잘린 나무 줄기
- **나무 줄기 오탐 (10-17 T · 기본 끔)**: `VISION_EDGE_MARGIN_PX=3` → 영상 경계에 닿은 박스는 탐지 · 추적만 · 후보 · 관측 없음 (stats `edge_dets`) · 재생: Kumoh 경로 (사람 없음) 관측 34 → 1 · UE 8장면 배우 48/48 그대로 · 첫 관측 +0.5 s · 박스 재현율 −4.8 %p · `VISION_EMIT_CONFIRMED_ONLY=1` (확정 후보만) 은 0.5 s 간격에서 사람 관측까지 막아 **기각** (`tests_edge_confirm.py`)
- REAL 은 `clock_domain` UTC · DEVICE_MONOTONIC 만 · `sync_error_ms` > 50 ms 보류 · `missing_reasons` 는 사유로 (카메라 자세를 직접 받았으면 마운트 · 짐벌 항목은 막지 않음)

## 2. 파일

| 파일 | 내용 |
|---|---|
| `app/vision_ingest.py` | v1.1 + 엔진 (해시 확인) · `posture.json` · SIM 에 같은 worker 넘김 |
| `app/vision_core/ingest.py` | v1.1 + `observe()` (추적 · 상태 · 후보 · 색 · 질의 · pose 좌표) |
| `app/vision_sim.py` | v1.1 + `rich_observations` (임무 기억) · `SIM_RICH_FIELDS=1` 이면 appearance · state · track_id (**백엔드 계약 바뀐 뒤에만**) |
| `app/vision_core/detector.py` | v1.1 그대로 (비 1080p 되돌림 수정 포함) |
| `app/vision_core/` `pose.py` · `state.py` · `appearance.py` · `registry.py` · `geo.py` | 새 파일 · 비전 판 (registry = 추적 attach · geo = 방위) |
| `base/` | 비교용 v1.1 원본 (diff 용 · 배포하지 않음) |
| `tests_v1_2.py` | UE 촬영 → 이 파일들 → **백엔드 pydantic 계약** (`real_contracts.VisionResult` · `contracts.Observation`) 검사 |

## 3. 검증 (서버 · 10-14 · v12_f91_swa · TensorRT)

| | S4 16 m (746장) | S3 20 m (882장) |
|---|---|---|
| 백엔드 계약 통과 (REAL · SIM) | 746 / 746 | 882 / 882 |
| 사람당 후보 — v1.1 방식 → 1.2 | 10.1 → **4.1** | 19.3 → **4.9** |
| 상태가 붙은 관측 (REAL) | 1,202 | 3,306 |
| worker 주장 좌표 오차 중앙 · P95 (`VISION_GEO_CLAIMS=1`) | 1.0 · 2.6 m | 1.1 · 3.3 m |
| 찾는 색 green — 대상 MATCH · 다른 사람 MATCH | 0.50 · 0.00 | 1.00 · 0.01 |

- 모델 로더: REAL · SIM 꾸러미 둘 다 해시 검사 → **tensorrt** · 한 클래스 확인 통과
- 프레임 간격 영향 (UE 8회): 2 s 간격이면 추적 IDF1 0.43 → 0.07 · 누운 사람 URGENT/HIGH 3/10 → 0/10 — **상태 판단은 0.1~0.2 s 간격이 필요**

## 4. 모델 꾸러미 (서버 · git 밖)

- `drone_yolo/runs_person/_pack_v12_real/` · `_pack_v12_sim/`
  - `active.json` (name v12_f91_swa · conf 0.20 · model_config_id = 등록 후 채움)
  - `posture.json`
  - `v12_f91_swa/best.pt` · `model.json` (environment · sha256 · engine {file, sha256}) · `best_trt11.3.0.99_RTX3090.engine`
- best.pt SHA-256 `d2a4edc92931401e68e029e7ac193041eb9c45f852b596ab8c698611fd323772`
- 인수 검증: `verify_detector` test_v2 1,407장 AP50 0.6426 (.pt) · 0.6417 (엔진) ± 0.003
