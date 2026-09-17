#!/bin/bash
# chain_night.sh — 09-18 밤 무인 운전 사슬
#   1) aihub_prep.py 끝나기를 기다리고 결과를 검사한다
#   2) L1(train_l.sh) 끝나기를 기다린다
#   3) L2(train_l2_boot.sh)를 잇는다 — 자체 가드가 autoheal job.conf 를 L2 로 넘긴다
cd /home/se/JupyterLAB/Capstone/drone_yolo
log() { echo "[$(date -u '+%F %T') UTC] $*"; }
log "시작"
while pgrep -f 'aihub_prep\.py' > /dev/null; do sleep 60; done
n=$(find data/det_aihub/images/train -name '*.jpg' | wc -l)
v=$(find data/det_aihub/images/val -name '*.jpg' | wc -l)
log "AI-Hub 전처리 완료 — train $n · val $v"
if [ "$n" -lt 15000 ] || [ "$v" -lt 500 ]; then
  log "크롭 수가 예상(train≥15000 · val≥500)보다 적다 — L2 를 걸지 않는다. 아침에 확인할 것"
  exit 1
fi
touch data/det_aihub/.prep_done
log "L1 완료 대기"
while [ ! -f logs/train_l.done ]; do sleep 120; done
log "L1 완료 → L2 시작"
exec ./train_l2_boot.sh
