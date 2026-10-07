# 비전 worker 인계 꾸러미 (2026-10-05 · 비전 정서인 → 통합 담당)

샌드박스의 **Vision worker 컨테이너**에 비전 파트(사람 탐지 · 좌표 · 후보 기억)를 넣기 위한 꾸러미다.
기반 이미지는 다시 빌드하지 않는다 — **패키지 한 층**을 얹은 이미지로 vision 서비스만 바꾼다. 비행 명령은 내지 않는다.

## 들어 있는 것

| 파일 | 내용 |
|---|---|
| `Dockerfile.vision` · `requirements-vision.txt` | 기존 이미지 + ultralytics 8.4.102 · TensorRT 11.3 (CUDA 12판) 등 7개 (기존 lock 과 겹치는 버전 없음) |
| `app/vision_worker.py` | 기존 `app/vision_worker.py` 교체본 — 상태 파일 규칙 (20초 헬스체크 · 기존 키) 유지 |
| `app/vision_core/` | `detector` 타일 4장 탐지 · `geo` 좌표 (발끝 광선 × 지면 · 오차 타원 · 보류 사유) · `registry` 후보 기억 v0 · `sources` RTSP · 파일 · `telemetry` 촬영 시각 · 자세 sidecar |
| `app/tools/verify_detector.py` | 인수 검증 — 평가셋에서 AP50 이 우리 값과 같은지 |
| `app/tools/build_engine.py` | TensorRT 엔진 미리 만들기 (worker 도 처음 뜰 때 자동으로 만든다) |

## 통합 담당이 할 일 (sudo 가 필요한 단계)

```bash
cd <샌드박스 루트>
# 1) 꾸러미 반영 — app/ 아래 (vision_worker.py · vision/ · tools/) 를 샌드박스 app/ 로 (sync-workspace.sh 가 app/ 을 옮긴다), Dockerfile.vision · requirements-vision.txt 는 루트로
# 2) 모델 — 가중치를 data/models/soup_v7r2/best.pt 로 (비전 담당이 서버 안에서 복사해 줌 · git 에 올리지 않는다)
# 3) 이미지 한 층 빌드
scripts/docker.sh build -f Dockerfile.vision -t drone-sandbox-vision:20261005 .
# 4) compose.yml 의 vision 서비스에 image: drone-sandbox-vision:20261005 한 줄 (나머지 설정 · GPU · 헬스체크는 그대로)
scripts/sync-workspace.sh && scripts/stack.sh up -d vision
# 5) 인수 검증 (평가셋을 data/eval/test_v2 에 두고 — 비전 담당이 서버 안에서 복사 · git 금지 데이터)
scripts/stack.sh exec -T vision python -m app.tools.verify_detector --data /data/eval/test_v2 \
    --weights /data/models/soup_v7r2/best.pt --engine /data/models/soup_v7r2/best_trt11.3.0.99_RTX3090.engine --expect 0.5927
```

- 처음 뜰 때 엔진을 만든다 (수 분 · 상태 `building_engine`) → `data/models/soup_v7r2/best_trt<버전>_<GPU>.engine`
- 엔진을 못 만들면 PyTorch FP16 으로 돈다 (상태 `backend`) — 한 장 약 40 ms 라 30 ms 기준 (NFR-V01) 은 넘지만 초당 5장 (NFR-V02) 은 된다

## 쓰는 법 — 임무 지정

```bash
echo '{"mission_id": "m001"}' > data/vision-control.json          # RTSP 기본: rtsp://127.0.0.1:18554/m001
echo '{"mission_id": "m001", "source": "/data/missions/m001/video/xxx.mp4"}' > data/vision-control.json   # 녹화 다시 보기
rm data/vision-control.json                                        # 멈춤
```

## 상태 (`data/worker-status.json` · 1초마다 · 쉬는 중 2초마다)

기존 키 `worker · updated_unix · cuda_available · model_loaded · flight_enabled(false) · state` 그대로 + 아래

| 키 | 뜻 |
|---|---|
| `state` | `starting` · `waiting_for_model` · `building_engine` · `idle_no_mission` · `running` · `waiting_for_video` · `finished_file` · `model_error` · `mission_error` |
| `model` · `backend` | `soup_v7r2` · `tensorrt` 또는 `pytorch_fp16` |
| `fps` · `latency_ms_p50` · `last_frame_age_s` | 처리 속도 · 한 장 탐지 시간 · 마지막 장이 얼마나 오래됐나 |
| `time_source` | `sidecar` (촬영 시각) · `receive` (받은 시각 — 좌표 못 냄) |
| `frames_processed` · `frames_dropped` · `detections` · `candidates` · `confirmed` · `geo_ok_ratio` | 누적 |

## 출력 (`data/missions/<id>/`)

| 파일 | 한 줄 |
|---|---|
| `results/detections.jsonl` | 처리한 장마다 `{"t_us", "time_source", "pts_ms", "infer_ms", "detections": [{"bbox", "conf", "candidate_id", "geo": {"status", "lat", "lon", "ellipse", "reasons"}}]}` |
| `results/candidates.jsonl` | 후보 사건 `new · best · confirmed` 마다 `{"event", "candidate_id", "first_seen_s", "last_seen_s", "n_hits", "best_conf", "confirmed", "geo_status", "lat", "lon", "sigma_m", "crop"}` |
| `crops/<후보 id>.jpg` | 그 후보의 가장 확신 높은 장면 크롭 |
| `logs/vision.log` | 시작 · 끝 |

- `ellipse` = (거리 방향 1σ m · 옆 방향 1σ m · 거리 방향 방위 °) · `status` `PENDING` 이면 `reasons` 에 사유 (지평선 위 · 화면 아래 끝 · 기울기 등)
- 후보는 지우지 않는다 — 운영자 판단 전까지 남는다

## 합의가 필요한 것

1. **촬영 시각 · 자세 sidecar** (`metadata/telemetry.jsonl` · `frames.jsonl` · 형식은 `app/vision_core/telemetry.py` 머리말 — **제안**)
   - 없으면 좌표를 못 낸다 (`time_source=receive` · geo 없음) · Pi · 서버 시계 동기 (NTP/PTP) 필요
   - ⚠ MediaMTX 를 거친 RTSP 의 PTS 가 Pi 가 적은 PTS 와 같은지 확인 필요 — 다르면 프레임에 시각을 싣는 방식으로 바꾼다
2. **DB** — 후보를 DB 에 넣는 쪽 (vision 이 직접 · API 가 `candidates.jsonl` 을 읽어서) · 테이블은 관제 담당 ERD 기준
3. **GPU 메모리** — 같은 GPU 를 학습과 나눠 쓴다 (vision 2~3 GB) · 시연 · 측정 땐 학습을 멈춘다

## 검증 (비전 담당 환경 · RTX 3090 · 2026-10-05)

| 항목 | 결과 |
|---|---|
| `verify_detector` test_v2 1,407장 · .pt FP16 | AP50 **0.5927** (평가 스크립트와 같음) · 재현율@0.15 0.682 |
| 같은 셋 · TensorRT FP16 | AP50 **0.5933** (+0.0006) · 한 장 21.5 ms (확신도 0.01 기준) |
| worker 전체 (녹화 200장 · 가짜 sidecar) | 200장 처리 · TensorRT · 탐지 중앙 24.6 ms · 좌표 OK 100 % · 상태 파일 · 재실행 없음 확인 |
| 샌드박스 녹화 파일 (sample) 읽기 | 1920×1080 · 초당 337장 디코딩 (CPU) |

- 아직 안 한 것: **컨테이너 안 (Python 3.12 · torch 2.8) 검증** · 실시간 RTSP · 실제 sidecar · 후보 묶기 문턱 (시뮬레이션 · UE 비행으로 정할 예정)

---

## v1.0 — vision-ingest/1.0 (10-07) · 이 절이 최신

결정된 연동 규격 (`schema/vision_backend_handoff_v1.0.md` · `schema/openapi-vision.json`) 에 맞춘 worker. **v0 (`app/vision_worker.py` · RTSP · 임무 폴더 JSONL) 은 쓰지 않는다.**

| 파일 | 내용 |
|---|---|
| `app/vision_ingest.py` | 실행 파일 — `python -m app.vision_ingest` (compose 의 vision `command` 를 이것으로) |
| `app/vision_core/ingest.py` | 핵심 — 프레임 API → 탐지 → 좌표 → 후보 → 색 → 크롭 업로드 → `/scratch/outbox.jsonl` → 전송 |
| `tools/mock_backend.py` | 서버 밖 왕복 시험용 가짜 백엔드 (OpenAPI 스키마 · 백엔드 규칙 흉내) |
| `tools/make_model_pack.py` | 후보 모델 폴더 (가중치 + `model.json` + `active.json`) — 데이터셋 없이 모델만 |

### 모델 폴더 (`/models` · 읽기 전용) — 갈아 끼우기

```
/models/active.json          {"name": "soup_v7r2", "model_config_id": "<등록 uuid>", "conf": 0.15, "upload_crops": true}
/models/soup_v7r2/best.pt     (같은 폴더 best*.engine 이 있으면 TensorRT)
/models/soup_v7r2/model.json  이름 · SHA-256 · 세 평가셋 AP50 · 파이프라인 판
/models/soup_v9x2/…           후보
```
- 교체 = `active.json` 의 `name` · `model_config_id` 를 바꾸고 vision 재시작 · 후보마다 `POST /api/v1/configs/models` 로 한 번 등록 (`environment: REAL` · `pipeline_version`)

### 서버 밖 시험 (10-07 통과)

```bash
python tools/mock_backend.py <1920×1080 JPEG 폴더> --n=30 [--pose] &
INGEST_BASE_URL=http://127.0.0.1:18080 INGEST_TOKEN=test INGEST_MODEL_DIR=<모델 폴더> INGEST_SCRATCH=<빈 폴더> python -m app.vision_ingest
```
- UE level01 30장 (pose 없음): 결과 30 · 관측 737 · 계약 위반 0 · 좌표 전부 PENDING/NO_POSE · 크롭 업로드 681
- 20장 + 가짜 pose: 관측 477 전부 VALID (오차 타원 포함) · 후보 묶기 동작 (60 후보) · worker_revision 증가 위반 0 · outbox 전부 전송 (ack)

### 샌드박스에 넣기 (통합 담당 · sudo)

1. `app/vision_ingest.py` · `app/vision_core/` → 샌드박스 `app/` (기존 `app/common.py` · `app/vision_outbox.py` 는 그대로 — worker 가 `send_pending` 을 가져다 씀)
2. 모델 폴더 → `data/models/` (UID 2202 읽기)
3. 이미지 한 층 (`Dockerfile.vision` · ultralytics 8.4.102 · TensorRT) + compose vision: `command: ["python", "-m", "app.vision_ingest"]` · **GPU 연결**
4. 모델 설정 등록 → `active.json` 의 `model_config_id`

### 남은 것 (백엔드 쪽)

- **`GET /internal/v1/frames` 응답에 `pose` 가 없다** → 좌표 전부 `PENDING/NO_POSE` (worker 는 `pose` 가 오면 바로 씀 · 키: `lat · lon · alt_agl_m · roll_deg · pitch_deg · yaw_deg · gimbal_pitch_deg · gimbal_yaw_deg · gimbal_stabilized`)
- 결과가 422/409 로 거절되면 그 프레임은 다시 처리하지 않는다 (outbox 전송기가 그 줄에서 멈춤 → 확인 · 격리는 사람이)
- 상태 인지 (`state`) · 근거 클립 (`clip_asset_id`) · 재관측 제안은 다음 판

## SIM 어댑터 — 학교 SIM 업무 서비스 (10-07) · 시연은 이쪽

SIM 은 **분석 작업 방식** (`schema/openapi-sim-vision.json` · SIM 백엔드 `analysis.py` 규칙). 탐지 · 좌표 · 후보 · 색 코드는 REAL 과 같고 주고받기만 다르다.

| 파일 | 내용 |
|---|---|
| `app/vision_sim.py` | 실행 — `python -m app.vision_sim` (SIM compose 의 vision `command`) |
| `app/vision_core/sim.py` | claim → input (SHA-256 = `input_hash`) → `IngestWorker.observe` → SIM 관측 → completion · 실패는 attempt/final-failure |
| `tools/mock_sim_backend.py` | 가짜 SIM 백엔드 — 임대 · 시도 한도 3 · 멱등 · 후보 임무 · geo DB 검사 흉내 · `--drop` 응답 끊기 · `--expire` 임대 만료 · `--missions` |

```bash
python tools/mock_sim_backend.py <JPEG 폴더> --model=<active.json 의 id> --n=30 [--missions=2 --drop=2 --expire=1] &
INGEST_BASE_URL=http://127.0.0.1:18081 INGEST_MODEL_DIR=<모델 폴더> INGEST_SCRATCH=<빈 폴더> INGEST_STOP_AFTER_IDLE=6 python -m app.vision_sim
```
- 10-07 통과 (UE level01): 기본 30장 → 완료 30 · 관측 737 · 계약 위반 0 · 시도 1회씩
- 거친 조건 20장 (임무 2 · 응답 끊김 2 · 임대 만료 1) → 완료 20 · 재전송 2건 모두 duplicate 로 받아들여짐 · 만료 작업은 2번째 시도로 완료 · 임무 바뀜 403 1건 → 후보 기억 새로 시작 후 통과 · 임무 사이 후보 섞임 0
- 모델 id 가 다르면 작업마다 attempt-failure (`MODEL_CONFIG_NOT_LOADED`) — `active.json` 의 id 를 비우면 작업이 고른 것을 받음

SIM 에서 못 하는 것 (SIM 백엔드 요청 사항)

| 무엇 | 지금 SIM | 결과 |
|---|---|---|
| 좌표 | `geo_result` VALID 면 503 · 프레임에 자세 없음 (`SIM_HAS_NO_REAL_POSE`) | 늘 PENDING → **지도에 위치 안 뜸** · 계산되면 `sim_estimate` 에만 |
| 색 · 상태 · 크롭 | 관측 칸 없음 (`additionalProperties: false`) | 후보 카드에 색 · 상태 · 사진 없음 |
| 임무 id · 촬영 시각 | 작업 (Claim) 에 없음 | 후보 기억 범위 하나 (`sim`) · 임무 바뀌면 403 보고 다시 시작 · 시각 = 받은 시각 |
| 작업 만들기 | 운영자가 프레임마다 `POST /api/v1/analysis-runs` | 프레임이 들어오면 자동으로 작업이 생겨야 시연이 돈다 |
| 모델 설정 | `environment: SIM` · APPROVED 인 MODEL 설정 | 등록 필요 |
