#!/bin/bash
# start_fiftyone_app.sh — FiftyOne 앱을 독립 프로세스로 띄운다 (09-30)
#   crontab @reboot 로 호출 — gpu-autoheal 이 트리거한 재부팅을 포함해 모든 재부팅에서 동작
#   (gpu-autoheal.service 는 root 소유라 손댈 수 없어, 사용자 crontab 으로 분리)
#   이미 떠 있으면 아무것도 안 한다 (중복 실행 방지)
cd /home/se/JupyterLAB/Capstone/drone_yolo
mkdir -p logs
FO=/home/se/miniconda3/envs/drone/bin/fiftyone
PIDF=logs/fiftyone_app.pid

if [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then
  echo "[$(TZ=Asia/Seoul date '+%F %T') KST] 이미 실행 중 (PID $(cat "$PIDF")) — 건너뜀" >> logs/fiftyone_app.log
  exit 0
fi

echo "[$(TZ=Asia/Seoul date '+%F %T') KST] FiftyOne 앱 시작" >> logs/fiftyone_app.log
setsid nohup "$FO" app launch --address 0.0.0.0 --port 5151 --remote --wait -1 \
  >> logs/fiftyone_app.log 2>&1 < /dev/null &
disown
sleep 3
# fiftyone CLI 는 실제 서버를 자식으로 띄운다 — 포트로 실제 PID 를 찾아 기록
for _ in $(seq 1 10); do
  REAL=$(ss -ltnp 2>/dev/null | awk '/:5151/{print $NF}' | grep -oP 'pid=\K[0-9]+')
  [ -n "$REAL" ] && break
  sleep 1
done
[ -n "$REAL" ] && echo "$REAL" > "$PIDF" && \
  echo "[$(TZ=Asia/Seoul date '+%F %T') KST] PID $REAL · 0.0.0.0:5151" >> logs/fiftyone_app.log
