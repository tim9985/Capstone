#!/bin/bash
# l2_guard.sh — L2 동안 체크포인트 백업 + autoheal job.conf 전환 (l_guard.sh 와 같은 방식)
set -u
cd /home/se/JupyterLAB/Capstone/drone_yolo
NAME=l2_11l_1280_aihub
log() { echo "[$(date -u '+%F %T') UTC] $*"; }
seen=""
log "시작"
while [ ! -f logs/train_l2.done ]; do
  cur=$(grep -E '^JOB_NAME=' autoheal/job.conf | cut -d= -f2)
  if [ "$cur" != "$NAME" ]; then
    sed -i "s/^JOB_NAME=.*/JOB_NAME=$NAME/" autoheal/job.conf \
      && sed -i "s/^JOB_SCRIPT=.*/JOB_SCRIPT=train_l2_boot.sh/" autoheal/job.conf \
      && log "job.conf $cur → $NAME (train_l2_boot.sh)"
  fi
  last=runs_person/$NAME/weights/last.pt
  if [ -f "$last" ]; then
    m=$(stat -c %Y "$last")
    if [ "$m" != "$seen" ]; then
      sleep 5
      if [ "$(stat -c %Y "$last")" = "$m" ]; then
        cp -f "$last" "${last%last.pt}last_backup.pt.tmp" && mv -f "${last%last.pt}last_backup.pt.tmp" "${last%last.pt}last_backup.pt" \
          && { seen=$m; log "last.pt → last_backup.pt"; }
      fi
    fi
  fi
  sleep 30
done
log "완료 — 종료"
