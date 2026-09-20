#!/bin/bash
# chain_edge.sh — ① 가장자리 측정이 끝나면 ② translate 0.30 학습을 잇는다 (2026-09-20)
# 판정은 결과 파일로만 한다 (프로세스 문자열 pgrep 은 자기 부모 셸을 잡는다 · 09-18 교훈)
cd /home/se/JupyterLAB/Capstone/drone_yolo
log(){ echo "[$(date -u '+%F %T') UTC] $*"; }
log "① 가장자리 측정 결과 대기"
for i in $(seq 1 120); do [ -f metrics/tile_edge_fov_11m_1280_all.csv ] && break; sleep 60; done
[ -f metrics/tile_edge_fov_11m_1280_all.csv ] || { log "2시간 내 결과 없음 — 학습만 진행"; }
sleep 20
log "② translate 0.30 학습 시작"
exec ./train_edge.sh
