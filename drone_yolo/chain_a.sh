#!/bin/bash
# chain_a.sh — 10-07 자세 판정기 개선 A3 → A4 → A6 (Archangel-Real) · 판정 기준은 _학습 큐 「10-07 A」 에 미리 적음
cd /home/se/JupyterLAB/Capstone/drone_yolo
S=/home/se/venvs/state/bin/python; Q=logs/QUEUE.md
log(){ echo "[$(TZ=Asia/Seoul date '+%F %T') KST] $*" | tee -a $Q; }
log "chain_a 시작 — 자세 A3 (φ · 크기) → A4 (방위 변화) → A6 (DINOv2 외형)"
echo -e "\n# chain_a 자동 결과 ($(TZ=Asia/Seoul date '+%F %H:%M') KST)" >> metrics/AUTO_RESULT_archangel.md
for k in a3 a4 a6; do
  if $S posture_archangel_eval.py $k > logs/archangel_$k.log 2>&1; then log "   $k 끝"; else log "   ❌ $k 실패 (logs/archangel_$k.log)"; fi
done
log "chain_a 완료 → metrics/AUTO_RESULT_archangel.md"
