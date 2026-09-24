#!/bin/bash
# chain_v7.sh — chain_v6 뒤에 잇는다 (2026-09-24): v6 데이터 + P2 헤드 (v6_obl 대비 바꾼 것은 P2 하나)
# 판정: test_obl AP50 이 v6_obl 보다 +0.01 초과면 P2 채택
cd /home/se/JupyterLAB/Capstone/drone_yolo
P=/home/se/miniconda3/envs/drone/bin/python
Q=logs/QUEUE.md; R=metrics/AUTO_RESULT_v6.md
WAIT_PID=${WAIT_PID:?chain_v6 PID}
log(){ echo "[$(TZ=Asia/Seoul date '+%F %T') KST] $*" | tee -a $Q; }
ap(){ awk -F, 'NR==2{print $3}' "$1" 2>/dev/null; }
while kill -0 $WAIT_PID 2>/dev/null; do sleep 120; done
A="--stage 1 --hyp configs/hyp/v6_oblique.yaml --weights yolo11m.pt --data configs/data_v6.yaml --model-yaml configs/models/yolo11m-p2.yaml --batch 6 --name v6_p2m"
log "④ v6_p2m 연기 실행 (v6 데이터 + P2 헤드)"
if $P train_person.py $A --smoke >> logs/q_v6_p2m_smoke.log 2>&1; then
  log "   연기 실행 통과 → v6_p2m 30 에폭 시작"
  $P train_person.py $A >> logs/q_v6_p2m.log 2>&1
  for t in test_obl test_v2; do
    if [ $t = test_obl ]; then X="--data data/test_obl --mode single --tag test_obl"; else X=""; fi
    $P eval_test_v2.py --weights runs_person/v6_p2m/weights/best.pt $X >> logs/q_v6_p2m_$t.log 2>&1
    V=$(ap metrics/${t}_v6_p2m.csv); log "   v6_p2m $t AP50 $V"; echo "| v6_p2m | $t | $V |" >> $R
  done
  B=$(ap metrics/test_obl_v6_obl.csv); V=$(ap metrics/test_obl_v6_p2m.csv)
  J=$($P -c "a=float('$V' or 0);b=float('$B' or 0);print(f'{(a-b)*100:+.1f} %p · ' + ('✅ P2 채택' if a>b+0.01 else ('➖ 동등' if a>=b-0.01 else '❌ 악화')))")
  log "   v6_p2m test_obl 판정: v6_obl $B 대비 $J"; { echo; echo "**v6_p2m 판정 (test_obl)**: v6_obl $B → $V ($J)"; } >> $R
else
  log "   ❌ v6_p2m 연기 실행 실패 — logs/q_v6_p2m_smoke.log"
fi
log "chain_v7 완료"
