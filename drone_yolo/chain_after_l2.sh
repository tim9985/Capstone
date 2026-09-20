#!/bin/bash
# chain_after_l2.sh — L2 가 끝나면 ① F2 임계값 탐색 ② 타일 추론 을 잇는다 (2026-09-20)
# 09-19 교훈: L2 와 GPU 를 나눠 쓰면 1600 이상 추론이 OOM 으로 죽는다 → 학습 종료 후로 미룬다
# 판정은 프로세스 이름이 아니라 결과 파일로만 한다
cd /home/se/JupyterLAB/Capstone/drone_yolo
PY=/home/se/miniconda3/envs/drone/bin/python
W=runs_person/fov_11m_1280_all/weights/best.pt
log(){ echo "[$(date -u '+%F %T') UTC] $*"; }

log "L2 완료(logs/train_l2.done) 대기"
for i in $(seq 1 720); do [ -f logs/train_l2.done ] && break; sleep 60; done
[ -f logs/train_l2.done ] || { log "12시간 내 L2 미완료 — 중단"; exit 1; }
log "L2 완료 확인"
for i in $(seq 1 30); do pgrep -f '[t]rain_person\.py' > /dev/null || break; sleep 60; done   # 잔여 정리 대기
sleep 30

log "① F2 임계값 탐색 (1280 · 1600 · 1920)"
$PY tune_threshold_fullframe.py --weights $W --imgsz 1280 1600 1920 2>&1 \
  | grep -vE 'Scanning|━' | tee logs/f2_sweep.log | tail -12
[ -f metrics/threshold_fullframe_fov_11m_1280_all.csv ] || { log "F2 실패 — 타일은 계속 진행"; }

log "② 타일 추론"
$PY eval_tile.py --weights $W 2>&1 | grep -vE 'Scanning|━' | tee logs/tile_eval.log | tail -10

log "전체 완료"
touch logs/after_l2.done
