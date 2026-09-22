#!/bin/bash
# v3a 판정 — 미리 정한 4개 지표 (2026-09-22)
cd /home/se/JupyterLAB/Capstone/drone_yolo
P=/home/se/miniconda3/envs/drone/bin/python
W=runs_person/v3a_11m_neg/weights/best.pt
log(){ echo "[$(TZ=Asia/Seoul date '+%F %T') KST] $*"; }
log "① 가림 지형 40칸 (재현율@0.15 · 기준선 0.4780)"
$P eval_fov.py --weights $W --tag fovbudget 2>&1 | tail -5
log "② 전체 프레임 + 음성 오탐 (기준선 0.594 건/프레임)"
$P eval_fullframe.py --weights $W --imgsz 1920 2>&1 | tail -6
log "③ 타일 추론 (재현율 0.3729 · 오탐 1.718)"
$P eval_tile.py --weights $W 2>&1 | tail -8
log "완료"
