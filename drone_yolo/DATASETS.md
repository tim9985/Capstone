# 데이터셋 확보 안내

서버(Ubuntu 22.04)에서 원본 데이터를 받는 절차. 모두 `drone_dev/data/raw/` 아래에 둔다.

> **재배포 금지** — NOMAD·WiSARD·Okutama 는 연구용 라이선스다. 우리 저장소에 올리지 않는다.
> `.gitignore` 에 `data/` 가 들어 있다.

---

## 0. 준비

```bash
cd ~/drone_yolo          # 저장소 루트
mkdir -p data/raw && cd data/raw
sudo apt update && sudo apt install -y unzip p7zip-full aria2 tmux rclone
pip install gdown
df -h .                  # 여유 공간 확인 — 원본만 200GB 안팎, 산출물 포함 시 더
```

**받는 동안 세션이 끊겨도 되게 `tmux` 안에서 할 것.**

```bash
tmux new -s dl
```

---

## 1. NOMAD — 배우 100명 · 우리 인물 다양성의 핵심

Bernal 외, *NOMAD: A Natural, Occluded, Multi-scale Aerial Dataset for Emergency Response
Scenarios*, WACV 2024. 42,825 프레임 · 5.4K 영상에서 추출 · 가시도 10단계 라벨.

- 논문 <https://arxiv.org/abs/2309.09518>
- 배포 <https://github.com/ArtRuss/NOMAD>

**배포는 Google Drive 폴더 하나다.** 저장소에는 안내만 있고 데이터가 없다.

**rclone 으로 받는다.** 서버에서 실제로 받은 방법이다. `gdown --folder` 는
폴더당 파일 50개 제한이 있고, gdown 5.x 에는 `--remaining-ok` 옵션이 없어 실패한다.

```bash
# 1) 원격 등록 — 서버엔 브라우저가 없으므로 토큰은 노트북에서 받는다
rclone config                 # n → 이름 gdrive → drive → 스코프 1 → auto config 'n'
#   노트북에서: rclone authorize "drive"  → 나온 토큰을 서버 프롬프트에만 붙여넣는다
#   (토큰은 채팅·문서에 남기지 말 것)

# 2) 공유 폴더를 루트로 지정 — 'config update' 는 OAuth 를 다시 띄우므로 환경변수로
export RCLONE_DRIVE_ROOT_FOLDER_ID=1zRiOzedR-PzO1bps5I1vb6jtVoQHFWzg
rclone lsd gdrive:            # annotations / images / labels / videos 가 보여야 한다
                              # '내 드라이브' 폴더가 보이면 export 가 빠진 것

# 3) 받기 — 끊기면 같은 명령을 다시 돌리면 받은 파일은 건너뛴다
cd ~/drone_yolo/data/raw
rclone copy gdrive:annotations ./NOMAD -P
rclone copy gdrive:labels ./NOMAD/labels -P \
  --transfers 32 --checkers 32 --drive-pacer-min-sleep 10ms --drive-pacer-burst 200
rclone copy gdrive:images ./NOMAD/images -P \
  --transfers 16 --checkers 32 --drive-pacer-min-sleep 10ms --drive-pacer-burst 200
```

`labels/` 는 작은 파일 수천 개라 pacer 옵션 없이는 매우 느리다.

**우리가 실제로 쓰는 것은 `annotations/` 와 `images/` 뿐이다.**
`videos/`(5.4K 원본)는 용량만 크고 `nomad_prep.py` 가 쓰지 않는다. 용량을 아끼려면 건너뛴다.

**50 · 70 m 원본은 `download_nomad_far.sh`** (09-14 서버). 서버에 30 m 원본이 있는 배우(학습 구성 60명)만
필터(`data/raw/nomad_filter_a50_a70.txt`, 스크립트가 만든다)로 골라 labels → images 순서로 받는다.
끊기면 같은 명령을 다시 실행한다 — 받은 파일은 건너뛴다. 끝나면 `data/raw/nomad_far_dl.done`.

```bash
cd ~/JupyterLAB/Capstone/drone_yolo && setsid nohup ./download_nomad_far.sh > logs/nomad_far_dl.log 2>&1 < /dev/null &
```

결과: a50 · a70 각 이미지 5,165 · 라벨 5,165 · 71 GB · 약 6시간 (Drive 속도가 0.3 ~ 9 MiB/s 로 출렁인다).
`annotations.json` 은 원래 a10 ~ a90 전 거리를 담고 있어 따로 받을 것이 없다.

**우리 스크립트가 기대하는 구조** (`nomad_prep.py`)

```
data/raw/NOMAD/
├── activityLabels.json      # 배우·거리별 활동 구간 (자세 라벨의 근거)
├── annotations.json
├── metadata.json
├── images/Actor001..Actor100/
└── labels/Actor001..Actor100/
```

원본은 `annotations/` 하위에 json 이 모여 있다. 위 구조가 되도록 옮긴다.

```bash
mv NOMAD/annotations/*.json NOMAD/
```

> `activityLabels.json` 에 형식이 깨진 구간이 4개 있다(4,606개 중).
> `make_pose3_dataset.py` 의 `parse_span()` 이 복구하므로 그대로 두면 된다.

---

## 2. WiSARD — 겨울·가을 산지 · 56,000장

Broyles, Hayner, Leung, *WiSARD: A Labeled Visual and Thermal Image Dataset for
Wilderness Search and Rescue*, IROS 2022. 가시광 + 열화상.

- 논문 <https://arxiv.org/abs/2309.04453>
- 프로젝트 <https://sites.google.com/uw.edu/wisard/>

**Google Drive 단일 파일 40.54 GB.**

```bash
cd ~/drone_yolo/data/raw
gdown --continue 1PKjGCqUszHH1nMbXUBTwPSDqRabAt_ht -O WiSARDv1.zip   # 끊기면 같은 명령으로 이어받기
unzip -tq WiSARDv1.zip && unzip -q WiSARDv1.zip -d WiSARD && rm WiSARDv1.zip
```

`unzip -t` 에서 오류가 나면 풀지 말고 지운 뒤 다시 받는다. (gdown 5.x 는 파일 id 를
그대로 받으므로 `--fuzzy` 는 필요 없다.)

먼저 시험해 보려면 표본(971.6 MB)이 있다.

```bash
gdown 1uSgMXuZGVCrWM_151UcykyHejxxaVHOo -O WiSARD_sample.zip
```

**우리는 가시광(VIS)만 쓴다** — 열화상은 우리 카메라(IMX415)에 없다.
받은 뒤 `*_IR_*` 폴더를 지우면 용량이 절반 가까이 준다.

> 배포 페이지에 **MIT 계열 라이선스**로 적혀 있다. 다른 셋보다 조건이 느슨하지만,
> 재배포 전에 페이지에서 한 번 더 확인할 것. 인용은 IROS 2022 논문으로 한다.

**기대 구조** (`wisard_prep.py`)

```
data/raw/WiSARD/
├── 200704_Baker_FLIR_IR_1/          # 1월 설경 (겨울)
├── 200910_Carnation_FLIR_IR_1/
├── 210924_FHL_Enterprise_VIS_0403/  # 9월 해안 (가을)
└── ...
```

폴더명이 `날짜_장소_센서_번호` 규약이다. `VIS` 가 가시광, `IR` 이 열화상.

---

## 3. Okutama-Action — **평가 전용** · 우리 고도와 일치

Barekatain 외, CVPR 2017 Workshops. 4K 드론 영상 43편 + 프레임별 자세 라벨.

- 공식 <http://okutama-action.org/>

**공식 배포는 Dropbox 폴더 하나인데, 현재 접근이 막혀 있다.**

<https://www.dropbox.com/scl/fo/9qvpsb3fsamvqzsa12149/APTyV-f01XLnJ0WFpZSBLOE>

이미 노트북에 받아 둔 것이 있으므로 **서버로 직접 복사하는 편이 빠르다** (약 9.6 GB).

```bash
# 노트북(Windows)에서 — MobaXterm 터미널 또는 PowerShell
scp -r "C:/Users/timjj/Desktop/캡스톤/drone_dev/data/raw/okutama"     <계정>@<서버IP>:~/drone_yolo/data/raw/
```

MobaXterm 왼쪽 SFTP 패널에 끌어다 놓아도 된다. 링크가 다시 열리면 아래를 쓴다.

```bash
pip install dropbox-downloader   # 또는 브라우저로 받아 scp
```

**기대 구조** (`okutama_prep.py`, `okutama3_prep.py`)

```
data/raw/okutama/
├── Drone1/{Morning,Noon}/*.mp4
├── Drone2/{Morning,Noon}/*.mp4
├── TrainSetVideos (2)/Drone{1,2}/{Morning,Noon}/*.mp4   # ← 30편이 여기 있다
└── Labels/MultiActionLabels/3840x2160/*.txt
```

> **주의 두 가지**
> ① 영상이 40편이다. `Drone*/*/` 만 보면 10편만 잡힌다 — 우리가 겪은 함정이다.
> ② 영상과 라벨의 프레임 번호가 **영상마다 0~21 어긋나 있다.**
> `okutama_align.py` 로 오프셋을 먼저 추정하지 않으면 상자가 사람에서 빗나간다.

**이 데이터는 학습에 쓰지 않는다.** 배우가 10명 미만이라 다양성이 없고,
우리 운용 고도(환산 17 m)와 정확히 맞는 유일한 외부 데이터라 **정직한 잣대**로 남긴다.

---

## 4. SARD — 자세 학습의 핵심 · 유일하게 재배포 가능

Sambolek & Ivasic-Kos, *Search and Rescue Image Dataset for Person Detection — SARD*,
IEEE DataPort 2021. **CC BY 4.0.** 1,981장에 사람이 직접 붙인 6클래스 자세 라벨.

- 원본 <https://ieee-dataport.org/documents/search-and-rescue-image-dataset-person-detection-sard>
  (DOI `10.21227/ahxm-k331`)
- 우리가 쓴 재배포본 — Roboflow `rescuedby/sard-peykp-lxuf9` v1

Roboflow 계정의 API 키가 필요하다. **키는 파일로만 넘긴다** — 채팅 · 문서 · 명령줄 · 커밋에 쓰지 않는다.

```bash
# 1) 키 파일 (별도 터미널에서 — 입력이 화면 · 셸 기록에 안 남는다)
mkdir -p ~/.config/roboflow && (umask 077; read -rsp 'Roboflow key: ' K; echo; printf '%s' "$K" > ~/.config/roboflow/api_key)

# 2) 서버 (09-14): roboflow 패키지 없이 REST API 로 받는다. NOMAD 받기가 돌고 있으면 끝나길 기다린다
cd ~/JupyterLAB/Capstone/drone_yolo && setsid nohup ./download_sard.sh > logs/sard_dl.log 2>&1 < /dev/null &
tail -n 8 logs/sard_dl.log     # 장수 · 해상도 · 박스 수 · 끝나면 data/raw/sard_dl.done
```

`download_sard.sh` 는 키를 인증 헤더로만 보내 명령줄 · `ps` · 로그에 드러나지 않는다.
zip 검사 → 장수가 문서와 다르거나 받을 자리에 이미 뭔가 있으면 옮기지 않고 멈춘다.
패키지를 쓸 수 있는 곳에서는 `Roboflow(api_key=...).workspace("rescuedby").project("sard-peykp-lxuf9").version(1).download("yolov8", location="data/raw/sard2")` 도 된다.

**기대 구조**

```
data/raw/sard2/search-and-rescue-2/
├── train/{images,labels}/    # 1,386장 · 박스 4,424
├── valid/{images,labels}/    #   396장 · 박스 1,312
├── test/{images,labels}/     #   198장 · 박스   618   ← ⚠ 아래 누수 — 재분할 전엔 판정에 쓰지 않는다
└── data.yaml                 # nc: 6
```

클래스 순서 — `['Running','Walking','laying_down','not_defined','seated','stands']`
해상도 전부 **1920×1080** (축소 재배포 아님) · 사람 박스 긴 변 중앙값 57 px (입력 1280 에서 약 38 px).

> ⚠ **분할 누수 (09-14 서버 확인)** — 파일명이 `gssNNNN` 연속 프레임이고 Roboflow 가 프레임 단위 무작위로 나눴다.
> test 의 87 % · valid 의 91 % 가 train 프레임과 번호 차이 1 이내 → 점수가 부풀려진다.
> test 로 쓰려면 프레임 번호 **구간 단위**로 다시 나눈다.

1클래스 탐지 학습용은 `data/det/sard/` — train 만, 이미지는 raw 하드링크, 6클래스 전부 0 (`not_defined` 포함 · 계획 3절 5항).

> 자세 라벨은 **SARD 원본 고유**다. Roboflow 사용자가 붙인 것이 아니다.
> 인용은 Sambolek & Ivasic-Kos 2021 로 한다.

---

## 5. 받은 뒤 — 전처리

```bash
cd ~/drone_yolo

python nomad_prep.py
python wisard_prep.py

# Okutama 는 정렬이 먼저다
python okutama_align.py --videos 2.2.10 1.2.10 2.1.2 2.1.7 2.1.4 2.1.1 \
                                  1.2.6 1.1.7 1.1.4 1.2.8 1.2.2 2.2.2 1.1.5 1.1.8 \
                        --range 60 --samples 12
python okutama3_prep.py --videos 2.2.10 1.2.10 2.1.2 2.1.7 2.1.4 2.1.1 \
                                  1.2.6 1.1.7 1.1.4 1.2.8 1.2.2 2.2.2 1.1.5 1.1.8 \
                        --val-videos 2.2.10 2.1.2 --stride 15

python make_pose3_dataset.py --wisard exclude    # 학습셋 (SARD+NOMAD 3클래스)
python make_configs.py                            # ★ yaml 절대경로를 이 서버 기준으로
```

**`make_configs.py` 를 빼먹으면 안 된다.** 저장소의 yaml 에는 Windows 절대경로가 박혀 있다.

### 확인

```bash
python - <<'PY'
from pathlib import Path
for d in ["det/pose3_sn", "det/okutama3", "det/wisard",
          "det/nomad_actor01_10", "raw/sard2/search-and-rescue-2"]:
    p = Path("data") / d
    print(f"{d:<38}{'있음' if p.exists() else '없음'}")
PY
```

기대값 — `pose3_sn` train 9,008장 / val 1,592장 (person 7,562 · fallen 2,755 · ambiguous 1,731)

---

## 6. 추가 후보 (현재 미사용)

우리 조건(하향 90° + 자세 라벨)에 **둘 다 맞는 공개 데이터는 위 넷이 전부**다.
아래는 한쪽이 어긋나 보류한 것들이며, 목적이 바뀌면 다시 볼 것.

| 데이터셋 | 규모 | 왜 보류했나 | 링크 |
|---|---|---|---|
| ForestPersons (ETRI) | 96,482장 · Standing/Sitting/**Lying** | 카메라 높이 **1.5~2 m** — 지상 시점 | <https://huggingface.co/datasets/etri/ForestPersons> |
| AI-Hub 자율주행드론 | 320시간 4K · **한국** | 각도 45° 예시, 90° 존재 미확인 · 내국인 승인 필요 | <https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=190> |
| UAV-Human | 피험자 119명 | 라벨이 **클립 단위**, 넘어짐 클래스 없음 | <https://github.com/sutdcv/UAV-Human> |
| CPD-UAV | 1,061장 · 하향 · 픽셀 마스크 | 자세 라벨 없음 (위장 탐지용) | <https://doi.org/10.3390/drones10060447> |

> ForestPersons 는 **한국 산림·사계절(눈 포함)** 이라 아깝지만 시점이 다르다.
> 지상 1.5 m 에서 본 사람과 상공 16 m 에서 내려다본 사람은 화면에서 아예 다르게 생겼다.
