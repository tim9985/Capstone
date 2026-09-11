#!/bin/bash
# E2·E3·E7·E8 순차 실행 — 세션이 끊겨도 서버에서 끝까지 진행되도록 묶은 스크립트
cd /home/se/drone_dev/drone_yolo
PY=/home/se/miniconda3/envs/drone/bin/python

echo "=== E1 재실행 (yolo11s, imgsz960, batch0.85) 시작 $(date) ==="
$PY train_person.py --stage 1 --data configs/data_all.yaml --weights yolo11s.pt \
  --imgsz 960 --batch 0.85 --name e1_11s_960 --workers 8 --cache disk
echo "=== E1 재실행 종료 $(date) ==="

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
