#!/bin/bash
# chain_a5.sh — 10-07 A5: A4 (방위 변화) 를 실전 파이프라인에서 · 판정 기준은 _학습 큐 「10-07 A5」
cd /home/se/JupyterLAB/Capstone/drone_yolo
S=/home/se/venvs/state/bin/python; Q=logs/QUEUE.md
log(){ echo "[$(TZ=Asia/Seoul date '+%F %T') KST] $*" | tee -a $Q; }
log "chain_a5 시작 — 파이프라인 1초 표본 (박스 이력 포함) → A4' 판정"
if $S state_pipeline.py --weights=runs_person/soup_v9x2/weights/best.pt --posture=box --tracker=configs/trackers/botsort_t015.yaml \
     --reset-jump --tag=a5 --no-render --dump > logs/a5_pipeline.log 2>&1; then
  log "   파이프라인 끝 (runs_state/samples_a5.json)"
  $S posture_a5.py > logs/a5_eval.log 2>&1 && log "   A5 판정 끝" || log "   ❌ A5 판정 실패"
else log "   ❌ 파이프라인 실패 (logs/a5_pipeline.log)"; fi
log "chain_a5 완료 → metrics/AUTO_RESULT_archangel.md"
