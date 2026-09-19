#!/bin/bash
# chain_night.sh — L1 이 끝나면 L2 를 잇는다 (2026-09-18 · v3)
#
# 09-18 교훈: 프로세스 생존을 pgrep 문자열로 판정하면 **그 문자열을 명령줄에 담은 셸**(이 스크립트를
# 띄운 Claude Bash 래퍼, 주석까지 포함)을 같이 잡는다. 대괄호 관용구로도 주석에 원문이 남아 또 잡혔다.
# → 프로세스를 세지 않고 **결과물(크롭 수)** 로만 판정한다. 끝난 일을 다시 기다리지 않는다.
cd /home/se/JupyterLAB/Capstone/drone_yolo
log() { echo "[$(date -u '+%F %T') UTC] $*"; }
log "시작 (v3)"
n=$(find data/det_aihub/images/train -name '*.jpg' | wc -l)
v=$(find data/det_aihub/images/val -name '*.jpg' | wc -l)
log "AI-Hub 크롭 확인 — train $n · val $v"
if [ "$n" -lt 15000 ] || [ "$v" -lt 500 ]; then
  log "크롭 수 미달(train>=15000 · val>=500) — L2 를 걸지 않는다"
  exit 1
fi
touch data/det_aihub/.prep_done
log "L1 완료 대기"
while [ ! -f logs/train_l.done ]; do sleep 120; done
log "L1 완료 → L2 시작"
exec ./train_l2_boot.sh
