#!/bin/bash
# E1~E8 순차 실행 — 세션이 끊겨도 서버에서 끝까지 진행되도록 묶은 스크립트
#
# 2026-09-11 재구성 사유
#   - ~/drone_dev 를 ~/JupyterLAB/ 로 이동 (경로 전체 변경)
#   - 증강 프로필 수정(degrees=180/flipud=0.5, train_person.py) — 이전 E1·E2·E3
#     결과와 비교 불가하므로 전부 재실행
#   - NOMAD 배우 30→42명 확대(metrics/nomad_actor_selection.csv 선정)
#   - cache='disk' 유지. 42명 확장 후 데이터셋(~68GB)이 RAM(62GB)을 넘어서
#     'ram' 은 ultralytics 자체 안전장치(check_cache_ram, 50% 안전마진)로
#     거의 확실히 disk 로 자동 폴백된다 — 애초에 명시적으로 disk 로 시작
cd /home/se/JupyterLAB/drone_dev/drone_yolo
PY=/home/se/miniconda3/envs/drone/bin/python

echo "=== E1 (yolo11s, imgsz960) 시작 $(date) ==="
$PY train_person.py --stage 1 --data configs/data_all.yaml --weights yolo11s.pt \
  --imgsz 960 --batch 0.85 --name e1_11s_960 --workers 8 --cache disk
echo "=== E1 종료 $(date) ==="

echo "=== E2 (yolov8s, imgsz1280) 시작 $(date) ==="
$PY train_person.py --stage 1 --data configs/data_all.yaml --weights yolov8s.pt \
  --imgsz 1280 --batch 0.85 --name e2_v8s_1280 --workers 8 --cache disk
echo "=== E2 종료 $(date) ==="

echo "=== E3 (yolo11s, imgsz1280) 시작 $(date) ==="
$PY train_person.py --stage 1 --data configs/data_all.yaml --weights yolo11s.pt \
  --imgsz 1280 --batch 0.85 --name e3_11s_1280 --workers 8 --cache disk
echo "=== E3 종료 $(date) ==="

echo "=== E7 (yolo11m, imgsz1280) 시작 $(date) ==="
$PY train_person.py --stage 1 --data configs/data_all.yaml --weights yolo11m.pt \
  --imgsz 1280 --batch 0.85 --name e7_11m_1280 --workers 8 --cache disk
echo "=== E7 종료 $(date) ==="

echo "=== E8 (yolo11l, imgsz1280) 시작 $(date) ==="
$PY train_person.py --stage 1 --data configs/data_all.yaml --weights yolo11l.pt \
  --imgsz 1280 --batch 0.85 --name e8_11l_1280 --workers 8 --cache disk
echo "=== E8 종료 $(date) ==="

echo "=== 전체 실험 완료 $(date) ==="
