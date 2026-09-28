#!/bin/bash
# chain_v15.sh — 09-29 새벽: v7_e60 판정 → 좋으면 60 에폭 반복(e60_r2) → 수프 · 아니면 멈추고 사람 판단 기다림
#   고르기 = chain_util.py pick v7_e60 (기준 v7 · v7_r2) · 개선 주장은 _학습 큐 규칙(반복 폭 · 너비비)으로 따로
cd /home/se/JupyterLAB/Capstone/drone_yolo
P=/home/se/miniconda3/envs/drone/bin/python; Q=logs/QUEUE.md; R=metrics/AUTO_RESULT_v12.md
WAIT_PID=${WAIT_PID:?chain_v14 PID}
log(){ echo "[$(TZ=Asia/Seoul date '+%F %T') KST] $*" | tee -a $Q; }
ap(){ awk -F, 'NR==2{print $3}' "$1" 2>/dev/null; }
eval3(){ local m=$1 t Y
  for t in test_obl test_v2 test_kr; do
    case $t in test_obl) Y="--data data/test_obl --mode single --tag test_obl";; test_v2) Y="";; test_kr) Y="--data data/test_kr --tag test_kr";; esac
    [ -f metrics/${t}_$m.csv ] || $P eval_test_v2.py --weights runs_person/$m/weights/best.pt $Y >> logs/q_${m}_$t.log 2>&1
  done
  log "   $m AP50 obl $(ap metrics/test_obl_$m.csv) · v2 $(ap metrics/test_v2_$m.csv) · kr $(ap metrics/test_kr_$m.csv)"
  echo "| $m | $(ap metrics/test_obl_$m.csv) | $(ap metrics/test_v2_$m.csv) | $(ap metrics/test_kr_$m.csv) |" >> $R; }
while kill -0 $WAIT_PID 2>/dev/null; do sleep 120; done
log "chain_v15 시작 — v7_e60 판정"
[ -f runs_person/v7_e60/weights/best.pt ] || { log "   ❌ v7_e60 없음 — 멈춤"; exit 1; }
for t in test_obl test_v2 test_kr; do $P compare_ci.py --tag $t v7_r2:v7_e60 | tail -1 | tee -a $Q; done
W=$($P chain_util.py pick v7_e60 v7,v7_r2); log "   v7_e60 고르기 (기준 v7 · v7_r2): $W"
if [ "${W%% *}" != cand ]; then log "   ➖ 60 에폭 이득 없음 — GPU 대기 (다음 실험은 사람이 고른다)"; exit 0; fi
$P chain_util.py mklist v7_e60_r2 24 train_v7.txt | tee -a $Q
A="--stage 1 --hyp configs/hyp/v6_oblique.yaml --weights yolo11m.pt --data configs/data_v7_e60_r2.yaml --name v7_e60_r2 --epochs 60"
if $P train_person.py $A --smoke >> logs/q_v7_e60_r2_smoke.log 2>&1; then
  log "   v7_e60_r2 연기 통과 → 60 에폭"; $P train_person.py $A >> logs/q_v7_e60_r2.log 2>&1
  if [ -f runs_person/v7_e60_r2/weights/best.pt ]; then
    eval3 v7_e60_r2
    $P soup.py soup_e60x2 v7_e60 v7_e60_r2 >> $Q 2>&1 && eval3 soup_e60x2
    for t in test_obl test_v2 test_kr; do $P compare_ci.py --tag $t soup_v7r2:soup_e60x2 | tail -1 | tee -a $Q; done
  fi
else log "   ❌ v7_e60_r2 연기 실행 실패"; fi
log "chain_v15 완료"
