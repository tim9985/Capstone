# CLAUDE.md — drone_yolo (비전 파트)

「자율 정찰 드론 관제 시스템」의 비전 코드 — **사람 1클래스 탐지 · 좌표 산정 · 상의 색상 비교**.
경로는 Capstone 저장소 루트 기준. **수치·경위의 원본은 볼트(`obsidian/`)** 이고, 여기에는 작업 규칙과 함정만 둔다.
갱신 2026-10-07

## 1. 먼저 볼 곳

| 무엇 | 어디 |
|---|---|
| 현재 상태 · 일정 | `obsidian/00 홈.md` |
| 학습 계획 · 실제로 간 길 | `obsidian/11 서버 학습 계획/00 학습 계획 한눈에.md` |
| **최종 모델 (09-28)** | `obsidian/05 결정/결정 - 최종 탐지 모델 soup_v7r2.md` — `runs_person/soup_v7r2/weights/best.pt` · 엔진 `best.engine` |
| GPU 대기열 · 판정 기준 | `obsidian/11 서버 학습 계획/_학습 큐.md` · 실행 기록 `logs/QUEUE.md` |
| 할 일 | `obsidian/09 기록/다음 할 일.md` |
| 실험 · 결정 색인 | `obsidian/04 실험/_실험 색인.md` · `obsidian/05 결정/_결정 색인.md` |
| 서버 설정 · 가중치 · 데이터 받는 법 | `SERVER.md` · `WEIGHTS.md` · `DATASETS.md` |

## 2. 고정된 전제 — 다시 논의하지 않는다

- **사람 1클래스** (누운 사람 · 부분 가림 포함) — 탐지기는 그대로. 자세 자동 판별은 09-13 에 뺐고, **09-30 발표 피드백으로 탐지 위에 얹는 상태 인지 층**(자세 + 움직임 + 주변 상황 · 후보를 거르지 않고 순위만)을 계획 중 → `obsidian/11 서버 학습 계획/06 요구조자 상태 인지 계획.md`
- 우선순위: **미탐 → 오탐 → 상의 색상(12색) → 좌표**
- **목적에 데이터를 맞춘다** — 데이터가 그렇다는 이유로 운용 조건을 정하지 않는다
- 카메라 IMX415 · 1920×1080 · 주 렌즈 LN012 **대각 88° = 수평 80.2° · f 1,141 px** · 보조 62°·78° 는 호환 확인 중
- 짐벌 GM3 V2 · **기본 마운트각 45°** (09-29 · 60° 비교 · 재학습 불필요) · 대응 45~75° · 고도 16~20 m 권장 (30 m 까지) · 합성 데이터 보류
- 기준 문서: 캡스톤 디자인 계획서 §3.2 (최종 예산안) · 요구명세서 v1.7 — 볼트와 어긋나면 이쪽이 우선

## 3. 합격선 (요구명세서 v1.7) — 현재값은 `obsidian/10 성능 요구사항/`

| ID | 기준 | 재는 법 |
|---|---|---|
| NFR-V01 | 1920×1080 한 장 ≤30 ms | 타일 4장 · TensorRT FP16 · 전후처리 포함 |
| NFR-V02 | ≥5 FPS | |
| **NFR-V03** | **사람 AP50 ≥0.80 · 학습/평가 장소 분리** | `eval_test_v2.py` (test_v2) · test_obl · test_kr |
| NFR-V04 | 안정 프레임 수평 오차 10~15 m | `coord_error.py` |
| NFR-V05 | F2 로 임계값 · 기본 0.15 | |
| NFR-V06 | 사람 없는 영상 오탐 ≤1건/프레임 | |

## 4. 학습 · 평가 규칙

- **판정은 장소 분리 평가셋으로만** — val mAP50 으로 모델을 고르면 실전과 반대로 간다 (11l 사례)
- 재현율은 **AP 와 짝으로** 본다 · 소표본(수십 장)으로 결론 내지 않는다
- 한 번에 하나만 바꾼다 · 판정 기준은 **돌리기 전에** `_학습 큐.md` 에 적는다
- **판정 = 짝 부트스트랩 95 % 구간이 0 을 넘는가** — `eval_test_v2.py`(블록 부트스트랩 · `runs_person/<모델>/eval_<tag>.npz` 저장) → `compare_ci.py --tag test_obl 기준:비교`. 단일 AP 차 ±0.01 로 판정하지 않는다
- ⚠ 짝 구간은 **평가셋** 흔들림만 잰다 — 같은 설정을 두 번 돌리면 **최대 3.4 %p** 달라진다 (test_kr · 09-26). 개선 = 구간 > 0 **그리고** 반복 차이보다 큼 **그리고** 세 평가셋(test_obl · test_v2 · test_kr) 어디서도 유의하게 나빠지지 않음
- test_obl 만 크게 오르면 **박스 너비비**를 먼저 본다 — Okutama 는 박스를 넓게 그린다 (AP30 0.845 ↔ AP50 0.626)
- 최종 모델 = **같은 설정 반복들의 가중치 평균** (`soup.py`) — 같은 구조·손실·hyp·데이터만 섞는다. ⚠ 같은 설정끼리도 **BN 통계가 어긋나 무너질 수 있다** (v7_r3 · 09-27) → 수프마다 세 평가셋 + 박스 너비비 확인 · `--bn` 재계산은 박스를 넓혀 채택 안 함
- 다음 학습의 val 은 `configs/data_v6b.yaml` (장소가 겹치지 않는 val) — `data_v6.yaml` 의 val 은 학습과 같은 영상이 섞였다
- `ultralytics==8.4.102` 고정 — fitness = **mAP50-95 만** · 초반 2~4에폭 하락은 warmup
- 기본 설정: COCO YOLO11m · imgsz 1280 (학습은 1280 까지) · SGD lr0 0.01 · **close_mosaic 0** · scale 0.3 · translate 0.15
- 설정은 **`configs/hyp/*.yaml` 로 묶는다** — 기준선 `v3_nadir.yaml` · 60° `v6_oblique.yaml` (둘은 degrees · flipud 만 다르다). 우선순위: 기본값 < `--hyp` < 명시 인자 < `--set 키=값`
- ⚠ `--hyp` 없이 `--stage 1` 만 주면 **옛 수직 전제**(degrees 180 · flipud 0.5 · imgsz 960) 로 돈다 — 체인 호환용
- 커스텀 구조는 `--model-yaml` + `.load()` · yaml **파일명에 크기 글자**가 있어야 한다 (`yolo11m-p2.yaml`, 없으면 nano 로 학습된다)
- 추론: 1920 → **타일 4장 1280×720 (겹침 50 %) + NMS 0.6** · 항상 **FP16** · TensorRT 는 `imgsz=[736,1280]` (정수 하나면 정사각 엔진) · 엔진은 GPU 마다 재빌드
- AI-Hub 182 는 **학습용** — 가림이 없어 검증에 쓰면 부풀려진다 (AP50 0.995)
- Okutama 는 **평가 전용** (test_obl) · WiSARD val 앞 291장은 음성 — `--limit` 으로 앞에서 자르지 않는다
- `det_fullframe` 등 옛 시험셋 칸 이름(s38 · l128)은 **사람 px** 다 — 붙어 있는 "54° · 25 m" 표기는 틀린 가정
- 노트북 가중치는 `--resume` 금지 (Windows 경로) — `--weights` 로
- ⚠ **`seed` 를 바꿔도 · 학습 목록 순서를 섞어도 같은 학습이 된다** — ultralytics 8.4 가 목록을 **파일 이름순 정렬** (`data/base.py:177`) · 고정 상수로 섞음 (09-25 · 10-06 확인: ft10_s51 = s52 소수점 다섯째 자리까지). 지금까지의 "반복" 차이는 GPU 비결정성뿐 → 진짜 반복은 **목록 95 % 무작위 뽑기** 또는 `deterministic=False`

### 상태 인지 (10-03~)

- 자세 판정기는 **정답 박스 크롭**으로 고르고, **탐지 박스 파이프라인**(`state_pipeline.py --posture=`)으로 다시 확인한다 — 둘이 다르게 나온다
- ⚠ 추적 이력을 쓰는 신호(자세 누적 · 무동작 · 속도)는 **추적 ID 바뀜**에 오염된다 — 누운 사람은 확신도가 낮아 띄엄띄엄 잡히고 주변 추적에 붙는다 (5초 중앙값 누움 0.98 → 0.80 · `state_diag.py`)
- **추적기 문턱 = 탐지 운용 임계 0.15** (BoT-SORT 기본 0.25 면 누운 사람이 새 추적을 못 만든다) + `--reset-jump` (ID 바뀜 끊기) + 자세 항은 지금 한 장 → 점수 누움 AUROC 0.78 → 0.95 (10-03 Q1 · `state_trackfix.py`)
- ⚠ **`model.track(img, persist=k > 0)` 는 추적이 안 된다** (ultralytics 8.4.102) — 첫 호출의 `persist=False` 가 콜백에 고정돼 **매 프레임 추적기를 새로 만든다** → ID = 확신도 순번 (10-07 발견 · 10-01~10-07 상태 연구의 이력 · 무동작 · 추적 단위 지표가 오염) · 추적은 **처음부터 `persist=True`** 로 부르거나 `BOTSORT` 를 직접 이어 쓴다 (`deploy/vision_worker/app/vision_core/state.py`)
- 출처 하나 빼기로 고를 땐 **클래스마다 진짜 출처가 2개 이상인지** 먼저 본다 — 비스듬 앉음은 SARD 뿐이라 합성 C2A 가 구조적으로 뽑혔다
- Okutama 자세 크롭 `crops.csv` 의 w1080 · h1080 은 **720p px 그대로** → 1080p 로 쓰려면 ×1.5 (`posture_v2.load_set` 이 보정) · `eval_posture.auc` 는 동점 미처리 → `posture_v2.auc`

## 5. 서버 운영 (RTX 3090 · 대여 서버)

- **GPU 를 쉬게 두지 않는다** — 긴 작업은 `chain_*.sh` 로 잇고 `logs/QUEUE.md` 에 남긴다
- 체인을 걸기 전 **연기 실행**: 같은 명령에 `--smoke` (학습 64장 · 평가 20장 · 1에폭 · 실패면 exit 2 · 1에폭 시간 추정) · 설정 확인만은 `--dry-run`
- **실행 중인 체인 스크립트는 고치지 않는다** — bash 가 실행 중에 파일을 다시 읽는다. 새 스크립트로 이어 건다
- 학습이 끝나면 `train_person.py` 가 **자동 백업** — `best.pt` → `weights/` · `results.csv`·`args.yaml`·`command.txt` → `metrics/train_runs/<이름>/` (commit 하면 서버 밖에 남는다)
- 프로세스는 **PID 로** 끈다 — `pgrep -f`/`pkill -f` 는 자기 셸(heredoc 본문의 스크립트 이름까지)을 잡아 exit 144 로 죽는다. 파일 편집도 heredoc 대신 별도 스크립트로
- 서버 시계는 **UTC** — 시각은 `TZ=Asia/Seoul date '+%F %T'` · 기록은 KST
- GPU 가 사라지는 장애(Xid 79) 이력 → `autoheal/` · `obsidian/05 결정/결정 - GPU 완화 설정과 자동 복구.md`

## 6. 기록 (볼트)

- 작업이 끝나면 볼트 갱신 → commit · push (저장소 하나: `tim9985/Capstone` · 서버 push 는 deploy key `github-capstone`)
- **장(폴더)별 + 날짜별**(`09 기록/일자별/`) · `타임라인` 한 줄 추가
- **개조식 · 짧게 · 설정값과 수량**을 적는다 · %·%p·지표 이름 → `obsidian/10 성능 요구사항/수치 표기 규칙.md`
- 지난 판은 지우지 않는다 — `지난 …` 하위 폴더로 옮기고 맨 위에 대체 링크
- 그림은 `obsidian/_첨부/` (svg) · 같은 노트를 노트북과 동시에 고치지 않는다

## 7. 막다른 길 — 다시 하지 말 것 (근거 → `_실험 색인`)

- 모델 확대 11m→11l · VisDrone 항공 사전학습(640) · 음성 15.8 % · translate 0.30 · close_mosaic
- NWD 손실·할당 (확신도 부풀림 · −2.4 %p) · P2 머리 (v6 데이터 위에선 ±0) — 작은 사람용 손실·구조는 반복 없이 판정 불가 (09-26)
- YOLO26m (10-04 · v9 목록 짝 ×2 + 수프) — 숲 +2.8 · 비스듬 −1.8 · 한국 −1.9 %p 맞바꿈 · 1회차 숲 +3.9 는 재현 안 됨 · 구조 바꾸기는 또 반복 폭 안
- AU-AIR 로 재현율 판정 — 정답 박스가 헐렁하고 어긋난다 (10-04) · 대신 사람 없는 장 오탐 (도로 시설물 3.24 건/장) 확인용으로 쓴다
- 1920 초과 업스케일 추론 · 2×(1280×1080) 타일 (겹침 없으면 이득 없음)
- 한 장 외형만으로 하는 자세 탐지 (3클래스 탐지기) — 마스크 기하 · 200 px 확대 · lr 만 낮추기 · 섞인 분할로 판정
- ForestPersons (지상 1.5~2 m 시점) · CloudTrack 수치 비교 (제로샷 VLM)
- 조건이 안 맞는 데이터 추가 — NII-CU 45° 야구장 (±0) · AI-Hub 190 (09-28 승인 · 고도 70~80 m · 사람 1920 기준 18 px)
- 수프에 BN 재계산 (`soup.py --bn`) — 무너진 조합은 살리지만 박스가 넓어져 test_obl 만 오른다 (라벨 습관)
- 학습 길이 60 에폭 — val mAP50-95 는 0.424 → 0.441 로 오르는데 **test_obl −4.0 %p** (09-29) · 30 에폭 유지
- 자세 분류기 (10-03 · `chain_p.sh`): 합성 C2A 로 비스듬 앉음 메우기 (**앉음 F1 −0.47**) · 같은 장면 상대 키 (전부 손해) · DINOv2 부분 미세조정 (한국 수직만 오르고 Okutama 앉음 재현율 0.07~0.13) — 앉음은 진짜 비스듬 앉음 데이터가 생길 때까지 손대지 않는다

## 8. 지켜야 할 규칙

- **NOMAD · WiSARD · Okutama · AI-Hub 원본과 파생물(크롭·인스턴스 은행·추론 결과 이미지)은 git 에 올리지 않는다.** 연구용·약관 제한이다 (Unicamp-UAV · VisDrone 도 같게 다룬다)
- rclone OAuth 토큰 · Roboflow/Kaggle 키는 문서·채팅·커밋에 남기지 않는다 — AI-Hub API 키(`~/.config/aihub/api_key`)도 같다. 키가 보이는 명령줄·`ps` 출력을 채팅에 찍지 않는다
- 파일 삭제 · 전역 설정 변경 · 패키지 설치 전에는 무엇을 왜 하는지 사용자에게 먼저 설명한다
- 노트북 기준선과 비교하려면 `ultralytics==8.4.102` 고정
- 학습이 끝날 때마다 `results.csv` 와 `best.pt` 를 바로 백업한다 (대여 서버)
- 이 저장소는 **공개**다 — 팀 내부 배포 주소·개인 연락처·학번을 넣지 않는다. 학번이 든 보고서(`*.hwpx` · `*.docx`)는 gitignore
- `팀드라이브/` (전자서명 · 회의록) 는 gitignore — 커밋하지 않는다. 팀 Google Drive 에 쓰는 건 사용자가 요청할 때만

## 9. 코드 지도 (자주 쓰는 것)

| 스크립트 | 용도 |
|---|---|
| `train_person.py` | 학습 — `--hyp` · `--set` · `--smoke` · `--dry-run` · `--model-yaml`(크기 글자 검사) · 끝나면 자동 백업 |
| `configs/hyp/` | 학습 설정 묶음 — `v3_nadir.yaml` (기준선) · `v6_oblique.yaml` (60°) |
| `soup.py` | 같은 구조 모델 가중치 평균 → `runs_person/<이름>/weights/best.pt` |
| `analyze_models.py` · `diag_misses.py` | 모델 차이 · 놓친 원인 · 크기/자세/가림 층 · 박스 너비비 (`--out=파일명` 으로 따로 저장) |
| `eval_test_v2.py` | **NFR-V03 판정** — 1920 · 타일 4장 · NMS 0.6 · FP16 · AP50 → `metrics/test_v2_<이름>.csv` |
| `make_testset_v2.py` · `make_place_split.py` | 장소 분리 평가셋 · 학습 목록 `configs/lists/` |
| `chain_v12.sh` · `chain_util.py` | 데이터 고르기 → 반복 → 수프 자동 체인 (`pick` · `mklist`) · 최신 체인 `chain_v16.sh` (v9 ×2 → 수프 · 층별) |
| `bench_trt.py` · `f2_tune.py` | NFR-V01 (TensorRT 엔진 빌드 · 한 장 ms) · NFR-V05 (val 로 F2 임계값) |
| `make_v7_data.py` · `make_v8_data.py` · `make_v9_data.py` · `make_testset_kr.py` | v7 (SARD·HERIDAL) · NII-CU 크롭 · NOMAD 가림·누움 부분집합 · test_kr |
| `geo_resolver.py` · `ue_geo_validate.py` | 좌표 산정 (발끝 광선 × 지면 · 좌표보류 · 오차 타원) · UE 검증 (노트북 · `configs/settings_geo.json`) |
| `obs_time.py` | 관측 시간 — test_obl 추적 ID 로 k회 관측 시 찾을 확률 |
| `eval_tile.py` · `eval_tile_edge.py` · `tune_threshold_fullframe.py` | 타일 추론 · 가장자리 · F2 임계값 |
| `coord_error.py` · `make_px_table.py` | 좌표 오차 예산 · 사람 px 표 |
| `survey_tilt2.py` · `make_tilt_tags.py` · `eval_tilt_groups.py` | 마운트각 추정 · 태그 · 층화 평가 |
| `nomad_prep.py` · `wisard_prep.py` · `aihub_prep.py` · `okutama3_prep.py` | 원본 → 크롭 |
| `video_infer.py` | 영상 일괄 추론 (`live_view.py` 는 화면이 필요해 서버에서 못 쓴다) |
| `state_pipeline.py` · `posture_runtime.py` | 상태 인지 — 탐지 + BoT-SORT + 자세 + 무동작 → 점수 · `--posture=box·p3·ft_s0` · `--dump` (1초 표본 → `runs_state/`, git 밖) |
| `posture_v2_data.py` · `posture_v2.py` · `posture_ft.py` · `chain_p.sh` | 자세 판별 P0~P5 — C2A 크롭 · 상대 키 · 출처 빼기 고르기 · 짝 판정 · DINOv2 부분 미세조정 · 보고 `metrics/AUTO_RESULT_posture.md` (state 환경 `/home/se/venvs/state/bin/python`) |
| `state_diag.py` | 파이프라인 1초 표본 진단 — 중앙값이 누움을 놓친 원인 (ID 바뀜 · 전환) · 창 규칙 관찰 |
| `state_trackfix.py` · `configs/trackers/` · `chain_q.sh` | 추적 고치기 Q1 (추적 문턱 0.15 · ID 바뀜 끊기 · ReID · Okutama 영상 반반 A 고르기 / B 판정) · YOLO26m 반복 (10-03) |
| `chain_r.sh` · `chain_r3.sh` · `judge_pairs.py` | R2 공통 출발 수프 → R3 v10 (+LADD) → R5 · 판정 규칙을 compare_ci 출력에 자동 적용 (Q2 기각 재현) |
| `bn_dist.py` · `make_v10_data.py` | 수프 재료끼리 BN · 가중치 거리 (붕괴 진단) · LADD 크롭 + `test_ladd_h` (번호 뒤 30 % 참고 평가) |
| `deploy/vision_worker/` | **동료 서버 샌드박스 Vision worker 인계 꾸러미** — 탐지 · 좌표 · 후보 기억 · RTSP · sidecar · 패키지 한 층 Dockerfile · 인수 검증 |
| `ue_posture_capture.py` | UE (Cosys-AirSim) 자세 데이터 캡처 — 배우 이름 `Person_<자세>_<번호>` · 마운트 × 고도 × 방위 · 서버 자세 크롭 형식 (노트북) |

결과: `runs_person/<이름>/` · `metrics/*.csv` · `metrics/AUTO_RESULT*.md` · 서버 진행 기록 `SERVER_PROGRESS.md`
옛 문서(쓰지 않음): `EXPERIMENTS.md` (E1~E8) · `MODELS.md` · `REBOOT_PLAN.md` (자세 트랙)
