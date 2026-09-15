#!/bin/bash
# fov_guard.sh — 화각 실험 대기열(train_fov.sh) 동안 체크포인트 백업 + autoheal job.conf 단계 전환 (09-15)
#
# 왜 필요한가 (로컬 세션 검토)
#   gpu_autoheal.sh 는 부팅 때 한 번 읽은 job.conf 의 JOB_NAME 으로만 last.pt 를 백업 · 복구한다.
#   대기열용 이름(fov_queue)에는 결과 폴더가 없어 실제 학습의 last_backup.pt 가 생기지 않는다
#   → GPU 가 체크포인트를 쓰는 중에 떨어지면 이어 학습이 막힌다.
# 하는 일 (30초마다)
#   1) 지금 단계 학습의 last.pt 가 바뀌고 5초 동안 그대로면 last_backup.pt 로 복사 (autoheal 과 같은 방식)
#   2) job.conf JOB_NAME 을 지금 단계 이름으로 — 재부팅 뒤 autoheal 이 그 last.pt 를 검사 · 복구하고 train_fov.sh 를 다시 부른다
#      11s 가중치 복사가 끝나면 곧바로 11m 이름으로 넘겨 "11s 끝남 = 대기열 끝남" 으로 오판하지 않게 한다
#   대기열이 끝나면(logs/train_fov.done) 스스로 멈춘다. 재부팅 뒤에는 다시 켜야 한다.
#
# 실행: cd ~/JupyterLAB/Capstone/drone_yolo && setsid nohup ./fov_guard.sh > logs/fov_guard.log 2>&1 < /dev/null &
set -u
cd /home/se/JupyterLAB/Capstone/drone_yolo
log() { echo "[$(date -u '+%F %T') UTC] $*"; }
declare -A seen

log "시작"
while [ ! -f logs/train_fov.done ]; do
  if [ -f weights/fov_11s_1280_all.pt ]; then stage=fov_11m_1280_all; else stage=fov_11s_1280_all; fi
  cur=$(grep -E '^JOB_NAME=' autoheal/job.conf | cut -d= -f2)
  if [ "$cur" != "$stage" ]; then
    sed -i "s/^JOB_NAME=.*/JOB_NAME=$stage/" autoheal/job.conf && log "job.conf JOB_NAME $cur → $stage"
  fi
  for run in fov_11s_1280_all fov_11m_1280_all; do
    last=runs_person/$run/weights/last.pt
    [ -f "$last" ] || continue
    m=$(stat -c %Y "$last")
    [ "$m" = "${seen[$run]:-}" ] && continue
    sleep 5
    [ "$(stat -c %Y "$last")" = "$m" ] || continue          # 아직 쓰는 중
    cp -f "$last" "${last%last.pt}last_backup.pt.tmp" && mv -f "${last%last.pt}last_backup.pt.tmp" "${last%last.pt}last_backup.pt" \
      && { seen[$run]=$m; log "$run last.pt → last_backup.pt"; }
  done
  sleep 30
done
log "대기열 완료 — 종료"
