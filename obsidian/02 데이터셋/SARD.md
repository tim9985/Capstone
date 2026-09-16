---
tags: [데이터셋]
라이선스: CC BY 4.0
---

> [!summary] 사람이 **직접 자세를 라벨링**한 유일한 항공 데이터 — 쓰러짐 판별의 핵심
> 09-13 자세 판별 제외 이후에는 **누운 사람이 많은 사람 탐지 데이터**로 쓰고, 자세 라벨은 자세별 재현율 평가에만 쓴다 → [[결정 - 자세 판별 제외]]
> Roboflow `rescuedby/sard-peykp-lxuf9` v1 · **CC BY 4.0** (재배포 가능)

## 사양

채석장 · 풀밭 · 숲 · 도로 · 약 20 m 고도 · 배우는 소수.

| 분할 | 이미지 | 라벨 |
|---|---:|---:|
| train | 1,386 | 4,424 |
| valid | 396 | 1,312 |
| test | 198 | 618 |

> [!warning] 분할 누수 (09-14 서버 확인)
> 파일명이 `gssNNNN` 연속 프레임이고 Roboflow 가 **프레임 단위 무작위**로 나눴다 → test 의 87 % · valid 의 91 % 가 train 프레임과 번호 차이 1 이내.
> 아래 SARD test 수치(자세 7~9차 쓰러짐 재현율 0.960~0.987)는 부풀려졌을 수 있다. 번호 구간 단위로 다시 나눠야 test 로 쓸 수 있다.

서버 확보 (09-14): `data/raw/sard2/search-and-rescue-2` · 해상도 전부 **1920×1080** · 사람 박스 긴 변 중앙값 57 px (입력 1280 에서 약 38 px).
1클래스 탐지용 `data/det/sard` (train 만 · 6클래스 전부 `person`) 로 서버 M2 학습에 넣었다 → `drone_yolo/SERVER_PROGRESS.md` 4-2절

| 원래 클래스 (train) | 개수 | 우리 3클래스 |
|---|---:|---|
| stands | 1,265 | person |
| laying_down | 1,096 | **fallen** |
| Walking | 907 | person |
| not_defined | 698 | **ambiguous** |
| seated | 414 | person |
| Running | 44 | person |

`not_defined` 전체 976개 (15.4%) — 무엇인지 애매한 것. → [[ambiguous 물체 오탐]]

### 서버 확보 (2026-09-14)

| 항목 | 값 |
|---|---|
| 경로 | `data/raw/sard2/search-and-rescue-2` (`download_sard.sh` · Roboflow REST · 키는 파일로만) |
| 해상도 | **전부 1920×1080** (축소 재배포 아님) |
| 사람 크기 | 긴 변 중앙값 **57 px** (모델 입력 1280 환산 38 px) |
| 1클래스 변환 | `data/det/sard` — train 만 · 이미지는 하드링크 · 6클래스를 전부 0(person) |

> [!danger] test · valid 가 train 과 연속 프레임이다
> 파일명이 `gssNNNN` 연속 프레임인데 Roboflow 가 **프레임 단위로 무작위 분할**했다.
> test 의 87 % · valid 의 91 % 가 train 프레임과 번호 차이 1 이내 → **test 점수를 쓸 수 없다.**
> 위의 쓰러짐 재현율 0.96~0.99 도 이 영향을 받았을 수 있다. 쓰려면 **프레임 번호 구간 단위**로 다시 나눠야 한다.


## 이 데이터로 알게 된 것

> [!success] 쓰러짐 판별이 **여기서만** 된다
> SARD test 쓰러짐 재현율 [[자세 7차 - SARD 6클래스|7차]] 0.960 · [[자세 8차 - 3클래스 동결|8차]] 0.974 · [[자세 9차 - not_defined 제외|9차]] 0.987

> [!failure] 다른 도메인으로 안 옮겨간다
> 같은 모델이 Okutama 에서 0.091 → [[쓰러짐 판별 도메인 이전 실패]]

> [!note] CloudTrack 과 같은 데이터
> arXiv 2409.16111 (ICRA 2025) 이 이 SARD 로 평가했다. 제로샷 VLM 이라 수치 비교 대상은 아니다 → [[막다른 길]]

## 연결

[[결정 - 3클래스 체계]] · [[NOMAD]] · [[Okutama-Action]]
