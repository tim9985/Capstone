---
tags: [모델]
파일: weights/yolov8s_visdrone.pt
상태: 시작 가중치
---

> [!summary] 공개 사전학습 모델 — 우리 모든 학습의 출발점이자 "기성 기술로는 안 된다" 의 근거
> HuggingFace `dronefreak/visdrone-yolov8s` · YOLOv8s · 11클래스 · 21.5 MB

## 그대로 썼을 때 (NOMAD 검증 1,198장)

| 지표 | 값 |
|---|---:|
| mAP50 | 0.133 |
| 재현율 | 0.123 |
| 보행 재현율 | 0.504 |
| **쓰러진 사람 재현율** | **0.048** — 20명 중 1명 |
| 겨울 mAP50 | 0.544 |

> [!quote] 해석
> 도시 지상에 가까운 시점으로 학습돼, **위에서 본 누운 자세를 사람으로 보지 않는다.**

> [!warning] 저장소 이름 주의
> `dronefreak/yolov8s-visdrone` 은 없다(401). `dronefreak/visdrone-yolov8s` 가 맞다.

## 연결

[[stage1_all]] · [[가중치 계보]] · [[탐지 1 - NOMAD 배우 확장]]
