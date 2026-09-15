#!/bin/bash
# train_fov_boot.sh — autoheal 이 재부팅 뒤 부르는 화각 실험 입구 (autoheal/job.conf JOB_SCRIPT)
# fov_guard.sh(체크포인트 백업 · JOB_NAME 단계 전환)가 꺼져 있으면 켜고, train_fov.sh 를 이어서 실행한다.
# guard 없이 train_fov.sh 만 다시 불리면 11s 가중치 복사 뒤 JOB_NAME 이 넘어가지 않아
# 11m 학습 중 두 번째 장애 때 autoheal 이 "끝남" 으로 보고 재개하지 않는다 (로컬 세션 검토 09-15)
cd /home/se/JupyterLAB/Capstone/drone_yolo
if ! pgrep -f 'fov_guard\.sh$' > /dev/null; then
  setsid nohup ./fov_guard.sh >> logs/fov_guard.log 2>&1 < /dev/null &
  echo "fov_guard 시작 (PID $!)"
fi
exec ./train_fov.sh
