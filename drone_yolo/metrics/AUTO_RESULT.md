# 자동 실행 결과 — 2026-09-22 21:27 KST

## D — 항공 사전학습 (vd_11m)

```
26   mAP50 0.4426  mAP50-95 0.2361
27   mAP50 0.4389  mAP50-95 0.2352
28   mAP50 0.4423  mAP50-95 0.2365
29   mAP50 0.4397  mAP50-95 0.2353
30   mAP50 0.4390  mAP50-95 0.2346
```

### 가림 지형 40칸 평균 재현율@0.15

| 모델 | 값 | 기준선 대비 |
|---|---:|---:|
| 기준선 fov_11m | 0.4780 | — |
| vd_11m | 0.4548 | **-2.3 %p** |

판정: ❌ 실패 (성공 조건: 기준선 초과)

## 마운트각 태그 검증 (Okutama 비스듬 vs 수직)

```
비스듬 시퀀스 16: ['1.1.2', '1.2.1', '1.2.2', '1.2.4', '1.2.6', '1.2.7', '1.2.8', '1.2.9', '2.1.1', '2.1.10', '2.1.2', '2.1.5', '2.1.7', '2.1.8', '2.1.9', '2.2.11']
수직   시퀀스 7: ['1.1.11', '1.1.4', '1.2.11', '2.1.3', '2.2.1', '2.2.10', '2.2.3']

비스듬: 정답 31,504 · 재현율@0.15 0.1695 · 사람 px 중앙 38.3
수직: 정답 11,086 · 재현율@0.15 0.2117 · 사람 px 중앙 36.7

차이(비스듬−수직): -4.2 %p
→ |차이| ≥ 5 %p 면 태그가 의미 있다 · 미만이면 잡음
→ /home/se/JupyterLAB/Capstone/drone_yolo/metrics/tilt_groups_vd_11m.csv
```

## 고도별 층화

```
고도별 검증셋
    a10    320장 · 객체   320개
    a30    273장 · 객체   273개

============================================================
vd_11m  (imgsz 1280 · conf 0.15)
============================================================
Ultralytics 8.4.102 🚀 Python-3.10.21 torch-2.13.0+cu130 CUDA:0 (NVIDIA GeForce RTX 3090, 24124MiB)
YOLO11m summary (fused): 126 layers, 20,030,803 parameters, 0 gradients, 67.6 GFLOPs
[34m[1mval: [0mFast image access ✅ (ping: 0.0±0.0 ms, read: 51.0±20.3 MB/s, size: 374.7 KB)
[K[34m[1mval: [0mScanning /home/se/JupyterLAB/Capstone/drone_yolo/data/det/nomad_actor01_10/labels/val... 14 images, 0 backgrounds, 0 corrupt: 4% ╸─────────── 14/320 33.4it/s 0.1s<9.2s[K[34m[1mval: [0mScanning /home/se/JupyterLAB/Capstone/drone_yolo/data/det/nomad_actor01_10/labels/val... 51 images, 0 backgrounds, 0 corrupt: 15% ━╸────────── 51/320 109.1it/s 0.3s<2.5s[K[34m[1mval: [0mScanning /home/se/JupyterLAB/Capstone/drone_yolo/data/det/nomad_actor01_10/labels/val... 86 images, 0 backgrounds, 0 corrupt: 26% ━━━───────── 86/320 178.9it/s 0.4s<1.3s[K[34m[1mval: [0mScanning /home/se/JupyterLAB/Capstone/drone_yolo/data/det/nomad_actor01_10/labels/val... 124 images, 0 backgrounds, 0 corrupt: 38% ━━━━╸─────── 124/320 221.6it/s 0.5s<0.9s[K[34m[1mval: [0mScanning /home/se/JupyterLAB/Capstone/drone_yolo/data/det/nomad_actor01_10/labels/val... 153 images, 0 backgrounds, 0 corrupt: 47% ━━━━━╸────── 153/320 239.9it/s 0.6s<0.7s[K[34m[1mval: [0mScanning /home/se/JupyterLAB/Capstone/drone_yolo/data/det/nomad_actor01_10/labels/val... 179 images, 0 backgrounds, 0 corrupt: 55% ━━━━━━╸───── 179/320 225.1it/s 0.7s<0.6s[K[34m[1mval: [0mScanning /home/se/JupyterLAB/Capstone/drone_yolo/data/det/nomad_actor01_10/labels/val... 220 images, 0 backgrounds, 0 corrupt: 68% ━━━━━━━━──── 220/320 278.2it/s 0.8s<0.4s[K[34m[1mval: [0mScanning /home/se/JupyterLAB/Capstone/drone_yolo/data/det/nomad_actor01_10/labels/val... 247 images, 0 backgrounds, 0 corrupt: 77% ━━━━━━━━━─── 247/320 258.3it/s 0.9s<0.3s[K[34m[1mval: [0mScanning /home/se/JupyterLAB/Capstone/drone_yolo/data/det/nomad_actor01_10/labels/val... 254 images, 0 backgrounds, 0 corrupt: 79% ━━━━━━━━━╸── 254/320 194.7it/s 1.1s<0.3s[K[34m[1mval: [0mScanning /home/se/JupyterLAB/Capstone/drone_yolo/data/det/nomad_actor01_10/labels/val... 319 images, 0 backgrounds, 0 corrupt: 99% ━━━━━━━━━━━╸ 319/320 322.3it/s 1.2s<0.0s[K[34m[1mval: [0mScanning /home/se/JupyterLAB/Capstone/drone_yolo/data/det/nomad_actor01_10/labels/val... 320 images, 0 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 320/320 266.6it/s 1.2s
[34m[1mval: [0mNew cache created: /home/se/JupyterLAB/Capstone/drone_yolo/data/det/nomad_actor01_10/labels/val.cache
[K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 5% ╸─────────── 1/20 10.9s/it 3.3s<3:27[K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 10% ━─────────── 2/20 1.5it/s 3.5s<12.4s[K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 15% ━╸────────── 3/20 2.4it/s 3.7s<7.1s[K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 20% ━━────────── 4/20 3.1it/s 3.9s<5.2s[K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 25% ━━━───────── 5/20 3.5it/s 4.1s<4.3s[K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 30% ━━━╸──────── 6/20 3.9it/s 4.3s<3.6s[K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 35% ━━━━──────── 7/20 4.1it/s 4.6s<3.2s[K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 40% ━━━━╸─────── 8/20 4.3it/s 4.8s<2.8s[K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 45% ━━━━━─────── 9/20 3.3it/s 5.7s<3.3s[K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 50% ━━━━━━────── 10/20 3.7it/s 5.9s<2.7s[K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 55% ━━━━━━╸───── 11/20 4.0it/s 6.1s<2.3s[K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 60% ━━━━━━━───── 12/20 4.2it/s 6.3s<1.9s[K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 65% ━━━━━━━╸──── 13/20 4.4it/s 6.5s<1.6s[K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 70% ━━━━━━━━──── 14/20 4.5it/s 6.8s<1.3s[K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 75% ━━━━━━━━━─── 15/20 4.5it/s 7.0s<1.1s[K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 80% ━━━━━━━━━╸── 16/20 4.5it/s 7.2s<0.9s[K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 85% ━━━━━━━━━━── 17/20 4.5it/s 7.4s<0.7s[K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 90% ━━━━━━━━━━╸─ 18/20 4.6it/s 7.6s<0.4s[K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 95% ━━━━━━━━━━━─ 19/20 4.6it/s 7.8s<0.2s[K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 20/20 2.5it/s 8.1s
                   all        320        320      0.761      0.594      0.623      0.433
Speed: 0.8ms preprocess, 11.7ms inference, 0.0ms loss, 0.3ms postprocess per image
Ultralytics 8.4.102 🚀 Python-3.10.21 torch-2.13.0+cu130 CUDA:0 (NVIDIA GeForce RTX 3090, 24124MiB)
[34m[1mval: [0mFast image access ✅ (ping: 0.0±0.0 ms, read: 96.1±86.7 MB/s, size: 315.3 KB)
[K[34m[1mval: [0mScanning /home/se/JupyterLAB/Capstone/drone_yolo/data/det/nomad_actor01_10/labels/val... 16 images, 0 backgrounds, 0 corrupt: 5% ╸─────────── 16/273 45.5it/s 0.1s<5.6s[K[34m[1mval: [0mScanning /home/se/JupyterLAB/Capstone/drone_yolo/data/det/nomad_actor01_10/labels/val... 50 images, 0 backgrounds, 0 corrupt: 18% ━━────────── 50/273 124.4it/s 0.2s<1.8s[K[34m[1mval: [0mScanning /home/se/JupyterLAB/Capstone/drone_yolo/data/det/nomad_actor01_10/labels/val... 85 images, 0 backgrounds, 0 corrupt: 31% ━━━╸──────── 85/273 183.8it/s 0.3s<1.0s[K[34m[1mval: [0mScanning /home/se/JupyterLAB/Capstone/drone_yolo/data/det/nomad_actor01_10/labels/val... 112 images, 0 backgrounds, 0 corrupt: 41% ━━━━╸─────── 112/273 209.3it/s 0.4s<0.8s[K[34m[1mval: [0mScanning /home/se/JupyterLAB/Capstone/drone_yolo/data/det/nomad_actor01_10/labels/val... 135 images, 0 backgrounds, 0 corrupt: 49% ━━━━━╸────── 135/273 209.4it/s 0.5s<0.7s[K[34m[1mval: [0mScanning /home/se/JupyterLAB/Capstone/drone_yolo/data/det/nomad_actor01_10/labels/val... 161 images, 0 backgrounds, 0 corrupt: 58% ━━━━━━━───── 161/273 218.5it/s 0.6s<0.5s[K[34m[1mval: [0mScanning /home/se/JupyterLAB/Capstone/drone_yolo/data/det/nomad_actor01_10/labels/val... 196 images, 0 backgrounds, 0 corrupt: 71% ━━━━━━━━╸─── 196/273 241.0it/s 0.8s<0.3s[K[34m[1mval: [0mScanning /home/se/JupyterLAB/Capstone/drone_yolo/data/det/nomad_actor01_10/labels/val... 205 images, 0 backgrounds, 0 corrupt: 75% ━━━━━━━━━─── 205/273 193.6it/s 0.9s<0.4s[K[34m[1mval: [0mScanning /home/se/JupyterLAB/Capstone/drone_yolo/data/det/nomad_actor01_10/labels/val... 235 images, 0 backgrounds, 0 corrupt: 86% ━━━━━━━━━━── 235/273 215.1it/s 1.0s<0.2s[K[34m[1mval: [0mScanning /home/se/JupyterLAB/Capstone/drone_yolo/data/det/nomad_actor01_10/labels/val... 252 images, 0 backgrounds, 0 corrupt: 92% ━━━━━━━━━━━─ 252/273 190.6it/s 1.1s<0.1s[K[34m[1mval: [0mScanning /home/se/JupyterLAB/Capstone/drone_yolo/data/det/nomad_actor01_10/labels/val... 273 images, 0 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 273/273 234.6it/s 1.2s
[34m[1mval: [0mNew cache created: /home/se/JupyterLAB/Capstone/drone_yolo/data/det/nomad_actor01_10/labels/val.cache
[K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 5% ╸─────────── 1/18 6.6s/it 2.0s<1:51[K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 11% ━─────────── 2/18 1.7s/it 2.6s<27.1s[K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 16% ━━────────── 3/18 1.9it/s 2.8s<8.1s[K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 22% ━━╸───────── 4/18 1.8it/s 3.4s<7.8s[K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 27% ━━━───────── 5/18 2.7it/s 3.6s<4.9s[K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 33% ━━━━──────── 6/18 3.3it/s 3.8s<3.7s[K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 38% ━━━━╸─────── 7/18 3.7it/s 4.1s<3.0s[K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 44% ━━━━━─────── 8/18 4.0it/s 4.3s<2.5s[K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 50% ━━━━━━────── 9/18 4.2it/s 4.5s<2.1s[K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 55% ━━━━━━╸───── 10/18 3.2it/s 5.8s<2.5s[K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 61% ━━━━━━━───── 11/18 3.6it/s 6.0s<1.9s[K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 66% ━━━━━━━━──── 12/18 4.0it/s 6.2s<1.5s[K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 72% ━━━━━━━━╸─── 13/18 4.2it/s 6.4s<1.2s[K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 77% ━━━━━━━━━─── 14/18 4.4it/s 6.6s<0.9s[K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 83% ━━━━━━━━━━── 15/18 4.5it/s 6.9s<0.7s[K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 88% ━━━━━━━━━━╸─ 16/18 4.4it/s 7.1s<0.5s[K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 94% ━━━━━━━━━━━─ 17/18 4.5it/s 7.3s<0.2s[K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 18/18 2.4it/s 7.4s
                   all        273        273       0.81      0.609      0.623      0.369
Speed: 0.8ms preprocess, 11.9ms inference, 0.0ms loss, 0.3ms postprocess per image

모델                        고도    장수    mAP50  mAP50-95      정밀도      재현율
vd_11m                   a10   320    0.623     0.433    0.761    0.594
vd_11m                   a30   273    0.623     0.369    0.810    0.609

vd_11m: a10 → a30 mAP50 0.623 → 0.623 (-0.000)

저장: /home/se/JupyterLAB/Capstone/drone_yolo/metrics/eval_altitude.csv
주의: 사람 크기는 이미 같게 리샘플링돼 있다. 이 격차는 '크기 차이'가 아니라
      '같은 크기일 때 촬영 거리에 따른 광학 정보량 차이'다.
```

## Unicamp-UAV 크기 분포

```
표본 800장 · 사람 12,486
긴 변 px 5/25/50/75/95: [ 30.  40.  51.  77. 111.]
운용 구간(18~50 px) 비율: 49.4%
```
