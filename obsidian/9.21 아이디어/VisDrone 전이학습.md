---
tags: [아이디어, 학습]
날짜: 2026-09-21
상태: 미실행 — 큐 대기
---

> [!summary] 지금 우리 학습은 **COCO(지상 시점)** 에서 출발한다 — 항공 사전학습을 되찾는다
> NOMAD 원논문 저자가 *"training data bias toward ground views rather than aerial ones"* 를
> 성능 저하의 1순위 원인으로 지목했다 → [[문헌 대조 - 비슷한 조건의 공개 실험]]

## 확인된 사실

`runs_person/*/args.yaml` 을 열어 보니 서버 학습 전부가 COCO 가중치에서 출발한다.

| 학습 | 시작 가중치 |
|---|---|
| `fov_11m_1280_all` | `yolo11m.pt` (COCO) |
| `l1_11l_1280` | `yolo11l.pt` (COCO) |
| `v3a_11m_neg` (진행 중) | `yolo11m.pt` (COCO) |

> [!warning] 의식적인 결정이 아니었다
> 노트북 시절 계보는 **공개 YOLOv8s-VisDrone → NOMAD+WiSARD → stage1_all** 로
> 중간에 **항공 시점을 한 번 거쳤다** (`WEIGHTS.md`).
> 서버에서 YOLO11 로 아키텍처를 바꾸면서 **YOLO11용 VisDrone 가중치가 없어** COCO 에서 직접 출발하게 됐다.
> **아키텍처 교체의 부산물**이지 판단의 결과가 아니다.

그래서 **"항공 사전학습을 잃은 손해가 얼마인가" 를 분리해서 잰 적이 없다.**
stage1_all 의 AP50 0.850 과 현재 val mAP50 0.478 을 직접 비교할 수 없는 이유이기도 하다
(시험셋·아키텍처·전처리가 동시에 바뀌었다).

## 설계 — 클래스를 몇 개로 하나

| | **10클래스 전부** | 사람(pedestrian+people)만 |
|---|---|---|
| 장점 | 이미지당 객체가 수십~수백 → **백본이 항공 소형 객체 특징을 풍부하게 배움** | 우리 헤드와 일치 |
| 단점 | 헤드는 어차피 버린다 | 차·오토바이가 **음성**이 된다 → "작은 덩어리 = 배경" 을 배울 위험 |

**10클래스 전부를 쓴다.** 가져올 것은 백본·넥이고 헤드는 `nc=1` 로 갈 때 ultralytics 가 자동 재초기화한다.
우리 실패 양상이 이미 "작은 물체를 배경으로 판단" 하는 것이라 사람만 남기면 역효과가 난다.

## 실행

VisDrone 은 서버에 없지만 ultralytics 가 `VisDrone.yaml` 을 내장해 **자동으로 받는다** (약 2 GB).

```bash
# 1단계 — 항공 사전학습 (약 3시간 · train 6,471장은 우리 30,056장의 1/5)
yolo detect train data=VisDrone.yaml model=yolo11m.pt imgsz=1280 \
    epochs=30 batch=8 optimizer=SGD lr0=0.01 close_mosaic=0 \
    project=runs_person name=pre_visdrone_11m

# 2단계 — 우리 데이터 (약 13시간)
python train_person.py --stage 1 --data configs/data_fov.yaml \
    --weights runs_person/pre_visdrone_11m/weights/best.pt \
    --imgsz 1280 --batch 8 --close-mosaic 0 --optimizer SGD --lr0 0.01 \
    --scale 0.3 --translate 0.15 --name vd_11m --epochs 30
```

기준선 `fov_11m` 과 **사전학습 하나만** 다르다.

## 판정 기준 (돌리기 전에 적는다)

| 지표 | 기준선 `fov_11m` | 성공 조건 |
|---|---:|---|
| 가림 지형 40칸 평균 재현율@0.15 | 0.4780 | **오른다** |
| 서 있는 38 px 재현율@0.15 (타일) | 0.3729 | 오른다 |
| val mAP50 | 0.4779 | 참고만 — [[수치 표기 규칙]] 대로 판정 근거로 쓰지 않는다 |
| 음성 프레임 오탐 | 0.594 건/프레임 | **오르지 않는다** |

## 더 싼 선행 확인 (10분)

본 실험 전에 **COCO 원본 `yolo11m.pt` 를 우리 시험셋에 그대로** 돌려 본다.
COCO 가 항공 사람을 얼마나 아는지 재는 것이다.

- 가림 지형 재현율@0.15 이 **0.05 수준**이면 → 사전학습 기여가 거의 없다는 뜻, 항공 사전학습 여지가 크다
- **0.20 수준**이면 → COCO 도 꽤 알고 있다는 뜻, 이득이 작을 수 있다

## 연결
[[문헌 대조 - 비슷한 조건의 공개 실험]] · [[손실 가중치 재조정]] · [[2 환경과 기준선]] · [[가중치 계보]]
