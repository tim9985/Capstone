#!/bin/bash
# train_l2_boot.sh — autoheal 이 재부팅 뒤 부르는 L2 입구
cd /home/se/JupyterLAB/Capstone/drone_yolo
pgrep -f 'l2_guard\.sh$' > /dev/null || { setsid nohup ./l2_guard.sh >> logs/l2_guard.log 2>&1 < /dev/null & echo "l2_guard 시작"; }
exec ./train_l2.sh
