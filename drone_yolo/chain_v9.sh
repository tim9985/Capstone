#!/bin/bash
# chain_v9.sh — v6_obl 진짜 반복 (2026-09-25): 학습 목록 순서만 섞고 val 은 val_v6b → 학습 흔들림 폭 · 새 val 로 고른 best.pt
cd /home/se/JupyterLAB/Capstone/drone_yolo
P=/home/se/miniconda3/envs/drone/bin/python
Q=logs/QUEUE.md; R=metrics/AUTO_RESULT_v6.md
log(){ echo "[$(TZ=Asia/Seoul date '+%F %T') KST] $*" | tee -a $Q; }
ap(){ awk -F, 'NR==2{print $3}' "$1" 2>/dev/null; }
A="--stage 1 --hyp configs/hyp/v6_oblique.yaml --weights yolo11m.pt --data configs/data_v6b_r2.yaml --name v6_obl_r2"
log "⑥ v6_obl_r2 (순서 섞은 반복 · 새 val) 연기 실행"
if $P train_person.py $A --smoke >> logs/q_v6_obl_r2_smoke.log 2>&1; then
  log "   통과 → 30 에폭"; $P train_person.py $A >> logs/q_v6_obl_r2.log 2>&1
  for t in test_obl test_v2; do
    if [ $t = test_obl ]; then X="--data data/test_obl --mode single --tag test_obl"; else X=""; fi
    $P eval_test_v2.py --weights runs_person/v6_obl_r2/weights/best.pt $X >> logs/q_v6_obl_r2_$t.log 2>&1
    V=$(ap metrics/${t}_v6_obl_r2.csv); log "   v6_obl_r2 $t AP50 $V (v6_obl $(ap metrics/${t}_v6_obl.csv))"; echo "| v6_obl_r2 | $t | $V |" >> $R
    $P compare_ci.py --tag $t v6_obl:v6_obl_r2 | tee -a $Q
  done
else log "   ❌ 연기 실행 실패"; fi
log "chain_v9 완료"
