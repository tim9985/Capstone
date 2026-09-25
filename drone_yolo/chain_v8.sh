#!/bin/bash
# chain_v8.sh — v6_obl 시드 반복 (2026-09-25): 같은 설정 · seed 43 → 판정 잡음 폭 측정
cd /home/se/JupyterLAB/Capstone/drone_yolo
P=/home/se/miniconda3/envs/drone/bin/python
Q=logs/QUEUE.md; R=metrics/AUTO_RESULT_v6.md
log(){ echo "[$(TZ=Asia/Seoul date '+%F %T') KST] $*" | tee -a $Q; }
ap(){ awk -F, 'NR==2{print $3}' "$1" 2>/dev/null; }
A="--stage 1 --hyp configs/hyp/v6_oblique.yaml --weights yolo11m.pt --data configs/data_v6.yaml --set seed=43 --name v6_obl_s43"
log "⑤ v6_obl_s43 (시드 반복 · 잡음 폭) 연기 실행"
if $P train_person.py $A --smoke >> logs/q_v6_obl_s43_smoke.log 2>&1; then
  log "   통과 → 30 에폭"; $P train_person.py $A >> logs/q_v6_obl_s43.log 2>&1
  for t in test_obl test_v2; do
    if [ $t = test_obl ]; then X="--data data/test_obl --mode single --tag test_obl"; else X=""; fi
    $P eval_test_v2.py --weights runs_person/v6_obl_s43/weights/best.pt $X >> logs/q_v6_obl_s43_$t.log 2>&1
    V=$(ap metrics/${t}_v6_obl_s43.csv); log "   v6_obl_s43 $t AP50 $V (seed 42: $(ap metrics/${t}_v6_obl.csv))"; echo "| v6_obl_s43 | $t | $V |" >> $R
  done
else log "   ❌ 연기 실행 실패"; fi
log "chain_v8 완료"
