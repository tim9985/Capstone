# 비전 worker 인계 꾸러미 (2026-10-05 · 비전 정서인 → 통합 담당)

샌드박스의 **Vision worker 컨테이너**에 비전 파트(사람 탐지 · 좌표 · 후보 기억)를 넣기 위한 꾸러미다.
기반 이미지는 다시 빌드하지 않는다 — **패키지 한 층**을 얹은 이미지로 vision 서비스만 바꾼다. 비행 명령은 내지 않는다.

## 들어 있는 것

| 파일 | 내용 |
|---|---|
| `Dockerfile.vision` · `requirements-vision.txt` | 기존 이미지 + ultralytics 8.4.102 · TensorRT 11.3 (CUDA 12판) 등 7개 (기존 lock 과 겹치는 버전 없음) |
| `app/vision_worker.py` | 기존 `app/vision_worker.py` 교체본 — 상태 파일 규칙 (20초 헬스체크 · 기존 키) 유지 |
| `app/vision/` | `detector` 타일 4장 탐지 · `geo` 좌표 (발끝 광선 × 지면 · 오차 타원 · 보류 사유) · `registry` 후보 기억 v0 · `sources` RTSP · 파일 · `telemetry` 촬영 시각 · 자세 sidecar |
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

1. **촬영 시각 · 자세 sidecar** (`metadata/telemetry.jsonl` · `frames.jsonl` · 형식은 `app/vision/telemetry.py` 머리말 — **제안**)
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
