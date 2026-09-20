#!/bin/bash
# chain_tile.sh — F2 임계값 탐색이 끝나면 타일 추론 평가를 잇는다 (2026-09-19)
# 프로세스 문자열로 판정하지 않는다 — 결과 파일로만 본다 (09-18 자기 매칭 버그 교훈)
cd /home/se/JupyterLAB/Capstone/drone_yolo
PY=/home/se/miniconda3/envs/drone/bin/python
W=runs_person/fov_11m_1280_all/weights/best.pt
log(){ echo "[$(date -u '+%F %T') UTC] $*"; }
log "F2 결과 파일 대기"
for i in $(seq 1 240); do [ -f metrics/threshold_fullframe_fov_11m_1280_all.csv ] && break; sleep 60; done
if [ ! -f metrics/threshold_fullframe_fov_11m_1280_all.csv ]; then log "4시간 내 F2 결과 없음 — 중단"; exit 1; fi
log "F2 완료 확인 → 타일 추론 시작"
$PY eval_tile.py --weights $W 2>&1 | grep -vE 'Scanning|━' | tee -a logs/tile_eval.log | tail -10
log "타일 추론 완료"
touch logs/tile_eval.done
