# CLAUDE.md — drone_yolo (비전 파트)

「자율 정찰 드론 관제 시스템」의 비전 코드 — **사람 1클래스 탐지 · 좌표 산정 · 상의 색상 비교**.
경로는 Capstone 저장소 루트 기준. **수치·경위의 원본은 볼트(`obsidian/`)** 이고, 여기에는 작업 규칙과 함정만 둔다.
갱신 2026-09-24

## 1. 먼저 볼 곳

| 무엇 | 어디 |
|---|---|
| 현재 상태 · 일정 | `obsidian/00 홈.md` |
| 학습 계획 (현행 v6 · 60°) | `obsidian/11 서버 학습 계획/00 학습 계획 한눈에.md` → `02 학습 계획 v6 (60° 기준).md` |
| GPU 대기열 · 판정 기준 | `obsidian/11 서버 학습 계획/_학습 큐.md` · 실행 기록 `logs/QUEUE.md` |
| 할 일 | `obsidian/09 기록/다음 할 일.md` |
| 실험 · 결정 색인 | `obsidian/04 실험/_실험 색인.md` · `obsidian/05 결정/_결정 색인.md` |
| 서버 설정 · 가중치 · 데이터 받는 법 | `SERVER.md` · `WEIGHTS.md` · `DATASETS.md` |

## 2. 고정된 전제 — 다시 논의하지 않는다

- **사람 1클래스** (누운 사람 · 부분 가림 포함). 자세 자동 판별은 하지 않는다 (09-13 · 재제안 금지)
- 우선순위: **미탐 → 오탐 → 상의 색상(12색) → 좌표**
- **목적에 데이터를 맞춘다** — 데이터가 그렇다는 이유로 운용 조건을 정하지 않는다
- 카메라 IMX415 · 1920×1080 · 주 렌즈 LN012 **대각 88° = 수평 80.2° · f 1,141 px** · 보조 62°·78° 는 호환 확인 중
- 짐벌 GM3 V2 · **기본 마운트각 60° (잠정)** · 대응 45~75° · 고도 16~30 m · 합성 데이터 보류
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
- ⚠ **`seed` 를 바꿔도 같은 학습이 된다** — ultralytics 8.4 데이터 로더가 고정 상수로 섞는다 (09-25 확인). 반복 학습은 **학습 목록 순서를 섞어** 만든다 (`configs/lists/train_v6_r2.txt`)

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
- 1920 초과 업스케일 추론 · 2×(1280×1080) 타일 (겹침 없으면 이득 없음)
- 자세 판별 전반 — 마스크 기하 · 200 px 확대 · lr 만 낮추기
- ForestPersons (지상 1.5~2 m 시점) · CloudTrack 수치 비교 (제로샷 VLM)
- 참고: AI-Hub `dataSetSn=190` (45°) 은 90° 시절 기각 — **60° 운용에선 재검토 후보**

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
| `eval_test_v2.py` | **NFR-V03 판정** — 1920 · 타일 4장 · NMS 0.6 · FP16 · AP50 → `metrics/test_v2_<이름>.csv` |
| `make_testset_v2.py` · `make_place_split.py` | 장소 분리 평가셋 · 학습 목록 `configs/lists/` |
| `chain_v2.sh` | 현재 GPU 대기열 |
| `eval_tile.py` · `eval_tile_edge.py` · `tune_threshold_fullframe.py` | 타일 추론 · 가장자리 · F2 임계값 |
| `coord_error.py` · `make_px_table.py` | 좌표 오차 예산 · 사람 px 표 |
| `survey_tilt2.py` · `make_tilt_tags.py` · `eval_tilt_groups.py` | 마운트각 추정 · 태그 · 층화 평가 |
| `nomad_prep.py` · `wisard_prep.py` · `aihub_prep.py` · `okutama3_prep.py` | 원본 → 크롭 |
| `video_infer.py` | 영상 일괄 추론 (`live_view.py` 는 화면이 필요해 서버에서 못 쓴다) |

결과: `runs_person/<이름>/` · `metrics/*.csv` · `metrics/AUTO_RESULT*.md` · 서버 진행 기록 `SERVER_PROGRESS.md`
옛 문서(쓰지 않음): `EXPERIMENTS.md` (E1~E8) · `MODELS.md` · `REBOOT_PLAN.md` (자세 트랙)
