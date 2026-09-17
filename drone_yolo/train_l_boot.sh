#!/bin/bash
# train_l_boot.sh — autoheal 이 재부팅 뒤 부르는 L 계열 입구 (autoheal/job.conf JOB_SCRIPT)
cd /home/se/JupyterLAB/Capstone/drone_yolo
if ! pgrep -f 'l_guard\.sh$' > /dev/null; then
  setsid nohup ./l_guard.sh >> logs/l_guard.log 2>&1 < /dev/null &
  echo "l_guard 시작 (PID $!)"
fi
exec ./train_l.sh
