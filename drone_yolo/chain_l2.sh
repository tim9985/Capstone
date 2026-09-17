#!/bin/bash
# chain_l2.sh — L1 이 끝나면(logs/train_l.done) 곧바로 L2 를 잇는다 (2026-09-18 밤 무인 운전용)
# train_l.sh 를 실행 중에 고치면 bash 가 남은 부분을 엉뚱하게 읽으므로 이렇게 밖에서 잇는다.
cd /home/se/JupyterLAB/Capstone/drone_yolo
echo "[$(date -u '+%F %T') UTC] L1 완료 대기 시작"
while [ ! -f logs/train_l.done ]; do sleep 120; done
echo "[$(date -u '+%F %T') UTC] L1 완료 확인 → L2 시작"
exec ./train_l2_boot.sh
