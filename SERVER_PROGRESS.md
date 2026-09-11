# 서버(3090) 이관 진행 현황

`SERVER.md` · `EXPERIMENTS.md` · `DATASETS.md` 대로 진행하면서 실제로 겪은 것들과
현재 상태를 기록한다. 로컬(노트북) 작업 이어가기 전에 이 파일부터 볼 것.

작성 2026-09-11 · Claude Code (se 계정 세션)

---

## 0. 요약

**환경 세팅 완료, 데이터 준비 완료, 학습 1차 스윕 진행 중 GPU 하드웨어 장애로 중단.**
재부팅 전까지는 GPU를 쓰는 어떤 작업도 불가능하다 (아래 6절).

---

## 1. 환경

- conda env `drone` (python 3.10), `/home/se/miniconda3/envs/drone/bin/python`
- torch 2.13.0+cu130, ultralytics 8.4.102, opencv 5.0.0.93 — `requirements.txt`대로 설치됨
- **주의**: 이 머신은 `/home/se/JupyterLAB/bin`이 `PATH`에서 conda보다 앞에 있다.
  `pip install`을 conda env 활성화 없이 돌리면 JupyterLab 쪽 venv로 잘못 들어간다.
  반드시 `/home/se/miniconda3/envs/drone/bin/pip` 절대경로로 설치할 것.
- 드라이버 580.173.02 / CUDA 13.0 / RTX 3090 24GB. 서버 접속 초기엔 드라이버 커널모듈
  버전 불일치(580.159 로드됨 vs 580.173 유저스페이스)로 `nvidia-smi`가 죽어있었는데
  `sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia && sudo modprobe nvidia_uvm`로
  재부팅 없이 해결했었다 (이번 6절의 장애와는 다른 문제였음).

## 2. 데이터 — 확보 완료분

`data/`는 `/home/se/drone_dev/data`를 가리키는 심볼릭 링크다 (레포 밖에 둠, `.gitignore`).

| 데이터셋 | 상태 | 방법 |
|---|---|---|
| **NOMAD** | 배우 1~30, 거리 a10/a30만 | Google Drive `gdown --folder`가 API 레이트리밋에 걸려 실패 → 이 서버에 이미 설정된 `rclone gdrive:` 원격으로 전환, 성공. `--filter-from`으로 불필요한 배우71~100·거리a50/70/90 제외해서 다운로드 시간 단축 |
| **WiSARD** | VIS 전체(39개 비행), IR 삭제 | `gdown`으로 43.5GB zip, 압축해제 후 IR 폴더 삭제(용량 절반 이상 감소) |
| SARD | **미확보** | Roboflow API 키 필요 — 아직 안 받음. pose3(3클래스 자세) 실험엔 필수 |
| Okutama | **미확보** | 공식 Dropbox 링크 막힘. 노트북에서 scp로 옮겨야 함 (평가 전용이라 탐지 학습 자체는 안 막힘) |

### NOMAD 전처리시 겪은 함정
- `nomad_prep.py`는 `--out` 하나만 받고 배우 범위 필터가 없다. 배우10명씩 3배치로 나눠
  각각 다른 서브셋 디렉토리를 가리키게 해야 `nomad_actor01_10/11_20/21_30` 구조가 나온다.
- 배치 디렉토리를 **심볼릭 링크**로 구성했더니 Python 3.10의 `Path.rglob('**')`가
  심볼릭 링크된 디렉토리를 따라가지 않아 이미지 0장으로 나왔다(3.13부터 해결된 이슈).
  **하드링크(`cp -al`)**로 바꿔서 해결 — 디스크 추가 사용 없음.

### WiSARD 원본 라벨 품질 이슈 (발견, 미수정)
`data/raw/WiSARD/**/*.txt` 전수조사 결과 **20,541개 중 5,599개(27.3%)**가 완전히 동일한
줄이 중복 저장돼 있다(좌표까지 일치). Ultralytics 데이터로더가 캐싱 시점에 자동으로
제거해줘서 학습엔 무해하지만, 원본 배포 자체의 결함으로 보인다. `DATASETS.md`에
"겪은 함정"으로 추가할지는 미정.

## 3. 코드 수정 — 서버(3090) 사양 반영

`train_person.py`, `train_pose_cls.py` 둘 다 노트북(RTX 3050 4GB) 기준 하드코딩이었던
`workers=4`, `cache=False`를 인자로 뺐고(`--workers` 기본 8), `--batch`가 **소수(0~1)를
받으면 그 비율만큼 VRAM을 목표로 자동산정**하게 고쳤다(ultralytics
`check_train_batch_size`의 `fraction` 인자 활용).

**중요**: `--batch -1`(완전자동)은 목표 60%로 너무 보수적이라 batch=6~9(24GB 중
13GB, 57%)로 낮게 잡혔다. **`--batch 0.85`를 쓸 것** — batch 13(19GB, 82%)으로
훨씬 GPU를 제대로 쓴다. 다음 실행부터는 이 값을 기본으로 삼는 게 나을 수 있다.

## 4. 학습 — 1차 스윕(해상도·백본) 결과

`EXPERIMENTS.md` 2절의 E1~E3 (`data_all.yaml` = NOMAD 배우1~30 + WiSARD, train 15,292장
/ val 3,900장, nc=1 person). `weights/yolov8s_stage1_all.pt`(안전자산)는 노트북에만
있어 못 가져왔으므로, **공개 COCO 사전학습 가중치에서 새로 시작하는 백본 비교**로
방향을 틀었다 (사용자 지시).

| 실험 | 조건 | 상태 | mAP50 | mAP50-95 |
|---|---|---|---:|---:|
| E1 | yolo11s / imgsz960 | ✅ 완료 (54ep 조기종료) | 0.647 | 0.340 |
| E2 | yolov8s / imgsz1280 | ✅ 완료 (31ep 조기종료) | 0.639 | **0.350** |
| E3 | yolo11s / imgsz1280 | ⚠️ **14에폭에서 GPU 크래시로 중단** | 0.641(13ep 기준) | 0.341 |
| E7 | yolo11m / imgsz1280 | ❌ 미실행 (GPU 죽은 채로 시작해 8초만에 실패) | - | - |
| E8 | yolo11l / imgsz1280 | ❌ 미실행 (동일 사유, 2초만에 실패) | - | - |

E2(mAP50-95 0.350)가 E1보다 나아서 `EXPERIMENTS.md`의 "imgsz 1280이 핵심" 가설과
방향이 맞다. E3·E7·E8은 GPU 복구 후 재실행 필요.

가중치·결과: `runs_person/e1_11s_960/`, `runs_person/e2_v8s_1280/`,
`runs_person/e3_11s_1280/`(13에폭까지). 체인 실행 스크립트는
`run_remaining_experiments.sh` (이번 커밋에 포함).

## 5. 다음 실험 순서(이대로 이어가면 됨)

1. GPU 복구 후 `run_remaining_experiments.sh`를 E3부터 다시 (배치 0.85 유지)
2. E7·E8까지 끝나면 `EXPERIMENTS.md` 3절 기준으로 m/l 용량 효과 판단
3. 승자 조합으로 `tune_threshold.py` / `eval_scale.py` / `eval_altitude.py` 재평가
4. SARD(Roboflow 키 필요) 받으면 `make_pose3_dataset.py`로 pose3 데이터 구성,
   `EXPERIMENTS.md` 5절 실험 A(회전증강) 진행 가능

## 6. 현재 블로커 — GPU 하드웨어 장애 (중요, 최우선 확인)

E3 14에폭 시작 직후 `torch.AcceleratorError: CUDA error: unspecified launch failure`
발생, 직후 `nvidia-smi`가 `No devices were found`로 GPU를 완전히 잃음.

```
NVRM: Xid (PCI:0000:02:00): 154, GPU recovery action changed from 0x0 (None) to 0x2 (Node Reboot Required)
```

드라이버가 스스로 "재부팅 필요"로 판정한 상태(GSP RPC 타임아웃 계열). 커널 모듈
`rmmod`도 걸린 채 `SIGKILL`에도 안 죽어 재부팅 외엔 복구 방법이 없다고 결론.

**재부팅이 막힌 이유**: 이 서버를 `bz2149` 계정과 공유 중인데, 그쪽이 CPU로
17.8일짜리 장기 작업(`linux_score_first.py`)을 막 돌리기 시작한 상태였다. `write`
명령으로 접속 중인 세 터미널(pts/0,2,3)에 재부팅 양해를 구하는 메시지를 보내놨으나
(2026-09-11 기준) 아직 응답 없음(세션 idle 상태 유지 중).

**재부팅 가능해지면 할 일**:
```bash
# 재부팅 후 검증
nvidia-smi   # 정상 출력 확인
cd ~/drone_yolo && git checkout server-3090-setup  # 또는 병합된 main
bash run_remaining_experiments.sh   # E3부터 이어서 (E1·E2는 이미 완료라 재실행 낭비니
                                     # 스크립트에서 E1 줄은 지우고 돌릴 것)
```
