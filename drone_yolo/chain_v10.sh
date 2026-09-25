#!/bin/bash
# chain_v10.sh — NWD 손실 (2026-09-26): v6_obl_r2 와 데이터·순서·val 이 같고 손실만 다르다 → 차이 = NWD 효과
# 판정 (돌리기 전): 짝 부트스트랩 95 % 구간 (v6_nwd − v6_obl_r2) 이 test_obl 또는 test_v2 에서 0 을 넘고, 다른 쪽이 0 아래로 안 가면 채택
cd /home/se/JupyterLAB/Capstone/drone_yolo
P=/home/se/miniconda3/envs/drone/bin/python
Q=logs/QUEUE.md; R=metrics/AUTO_RESULT_v6.md
WAIT_PID=${WAIT_PID:?chain_v9 PID}
log(){ echo "[$(TZ=Asia/Seoul date '+%F %T') KST] $*" | tee -a $Q; }
ap(){ awk -F, 'NR==2{print $3}' "$1" 2>/dev/null; }
while kill -0 $WAIT_PID 2>/dev/null; do sleep 120; done
A="--stage 1 --hyp configs/hyp/v6_oblique.yaml --weights yolo11m.pt --data configs/data_v6b_r2.yaml --nwd 0.5 --nwd-c 32 --name v6_nwd"
log "⑦ v6_nwd (NWD α 0.5 · C 32 px · 나머지는 v6_obl_r2 와 같음) 연기 실행"
if $P train_person.py $A --smoke >> logs/q_v6_nwd_smoke.log 2>&1; then
  log "   통과 → 30 에폭"; $P train_person.py $A >> logs/q_v6_nwd.log 2>&1
  for t in test_obl test_v2; do
    if [ $t = test_obl ]; then X="--data data/test_obl --mode single --tag test_obl"; else X=""; fi
    $P eval_test_v2.py --weights runs_person/v6_nwd/weights/best.pt $X >> logs/q_v6_nwd_$t.log 2>&1
    V=$(ap metrics/${t}_v6_nwd.csv); log "   v6_nwd $t AP50 $V (v6_obl_r2 $(ap metrics/${t}_v6_obl_r2.csv))"; echo "| v6_nwd | $t | $V |" >> $R
    $P compare_ci.py --tag $t v6_obl_r2:v6_nwd | tee -a $Q
  done
  $P diag_misses.py --weights runs_person/v6_nwd/weights/best.pt > logs/q_v6_nwd_diag.log 2>&1 && log "   진단 → metrics/diag_misses_v6_nwd.json"
else log "   ❌ 연기 실행 실패 — logs/q_v6_nwd_smoke.log"; fi
log "chain_v10 완료"
