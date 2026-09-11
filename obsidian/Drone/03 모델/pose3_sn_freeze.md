---
tags: [모델, 핵심]
파일: weights/yolov8s_pose3_sn_freeze.pt
상태: 자세 운용안
학습일: 2026-09-05
---

> [!summary] 3클래스 자세 모델 (8차) — person / fallen / ambiguous
> [[stage1_all]] 에서 **백본 동결** 전이학습 · SARD + NOMAD 9,008장 · WiSARD 제외

## 성능

| 지표 | 값 |
|---|---:|
| **SARD test 쓰러짐 재현율** | **0.974** (정밀도 0.919) |
| NOMAD 쓰러짐 재현율 | 0.227 |
| Okutama 쓰러짐 재현율 | **0.091** |
| 발견율 NOMAD · WiSARD · Okutama | 0.589 · 0.804 · **0.682** |
| 추론 1280×720 | **20.6 ms · 48.6 fps** |

| 검증 클래스 | P | R | mAP50 |
|---|---:|---:|---:|
| person | 0.771 | 0.648 | 0.668 |
| fallen | 0.738 | 0.679 | 0.702 |
| ambiguous | 0.550 | 0.516 | 0.473 |
| 전체 | | | 0.614 |

## 강점과 약점

> [!success] 강점
> - 동결 덕분에 탐지 손실이 **−11%** 에 그쳤다 (동결 없는 7차는 −39%)
> - **미학습 Okutama 발견율은 원본보다 높다** (0.633 → 0.682)

> [!failure] 약점
> - 쓰러짐 판별이 **SARD 밖에서 무너진다** → [[쓰러짐 판별 도메인 이전 실패]]
> - `ambiguous` 가 벤치·가방에 붙는다 (실영상 탐지의 47%) → [[ambiguous 물체 오탐]]

## 학습

```
python train_person.py --stage 1 --data configs/data_pose3_sn.yaml \
  --weights weights/yolov8s_stage1_all.pt --name pose3_sn_freeze \
  --epochs 35 --imgsz 960 --batch 6 --lr0 0.001 --patience 10 --freeze 10
```

28 epoch (best 18) · 2시간 33분 · 노트북

## 연결

[[자세 8차 - 3클래스 동결]] · [[결정 - 3클래스 체계]] · [[결정 - 백본 동결]] · [[가중치 계보]]
