#!/bin/bash
# chain_p.sh — 자세 판별 P4 (미세조정 3회) → 판정 → P5 (파이프라인) → 보고 (2026-10-03)
#   계획: obsidian 「11 서버 학습 계획/08 자세 판별 학습 계획」 · P3 는 수동 선행 (logs/qp_p3.log)
#   사용: nohup bash chain_p.sh <P3 PID> &
cd /home/se/JupyterLAB/Capstone/drone_yolo
P=/home/se/venvs/state/bin/python
log(){ echo "[$(TZ=Asia/Seoul date '+%F %T') KST] $*" >> logs/QUEUE.md; }
WAIT=${1:-0}
while [ "$WAIT" -gt 0 ] && kill -0 "$WAIT" 2>/dev/null; do sleep 30; done
[ -f metrics/posture_v2_p3.json ] || { log "❌ chain_p: P3 결과 없음 — 중단 (logs/qp_p3.log)"; exit 1; }
log "chain_p: P3 완료 — 고른 것 $(python3 -c "import json;print(json.load(open('metrics/posture_v2_p3.json'))['고른 것'])")"
for s in 0 1 2; do
  $P posture_ft.py --seed=$s > logs/qp_ft_s$s.log 2>&1 && log "   P4 ft_s$s 완료" || log "   ❌ P4 ft_s$s 실패 (logs/qp_ft_s$s.log)"
done
$P posture_v2.py judge4 > logs/qp_judge4.log 2>&1 || log "   ❌ P4 판정 실패"
log "   P4 판정: $(tail -1 logs/qp_judge4.log)"
for v in box p3 $(grep -o 'ft_s0' weights/posture_v2_best.txt); do
  $P state_pipeline.py --weights=runs_person/soup_v9x2/weights/best.pt --posture=$v --tag=v9x2_pose_$v --no-render \
    > logs/qp_state_$v.log 2>&1 && log "   P5 $v 완료" || log "   ❌ P5 $v 실패 (logs/qp_state_$v.log)"
done
$P posture_v2.py report > logs/qp_report.log 2>&1 || log "   ❌ 보고 실패"
log "chain_p 완료 → metrics/AUTO_RESULT_posture.md"
