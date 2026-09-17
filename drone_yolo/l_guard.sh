#!/bin/bash
# l_guard.sh — train_l.sh 동안 체크포인트 백업 + autoheal job.conf 단계 전환 (2026-09-18)
# fov_guard.sh 와 같은 이유 · 같은 방식. 단계가 늘면 STAGES 에 이름을 추가한다.
set -u
cd /home/se/JupyterLAB/Capstone/drone_yolo
STAGES="l1_11l_1280"
log() { echo "[$(date -u '+%F %T') UTC] $*"; }
declare -A seen
log "시작"
while [ ! -f logs/train_l.done ]; do
  stage=""
  for s in $STAGES; do [ -f weights/$s.pt ] || { stage=$s; break; }; done
  [ -n "$stage" ] || stage=${STAGES##* }
  cur=$(grep -E '^JOB_NAME=' autoheal/job.conf | cut -d= -f2)
  [ "$cur" = "$stage" ] || sed -i "s/^JOB_NAME=.*/JOB_NAME=$stage/" autoheal/job.conf && log "job.conf JOB_NAME → $stage"
  for run in $STAGES; do
    last=runs_person/$run/weights/last.pt
    [ -f "$last" ] || continue
    m=$(stat -c %Y "$last"); [ "$m" = "${seen[$run]:-}" ] && continue
    sleep 5; [ "$(stat -c %Y "$last")" = "$m" ] || continue
    cp -f "$last" "${last%last.pt}last_backup.pt.tmp" && mv -f "${last%last.pt}last_backup.pt.tmp" "${last%last.pt}last_backup.pt" \
      && { seen[$run]=$m; log "$run last.pt → last_backup.pt"; }
  done
  sleep 30
done
log "대기열 완료 — 종료"
