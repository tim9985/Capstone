#!/bin/bash
# chain_v13.sh — 09-27 밤: v7 네 번째 반복 (GPU 빈 시간) · 끝나면 BN 재계산 없는 수프 조합을 세 평가셋에서 잰다
#   배경: v7+v7_r2 수프는 정상 · v7_r3 가 들어간 수프는 BN 통계가 어긋나 무너짐 · BN 재계산은 박스를 넓혀 test_kr 손해
cd /home/se/JupyterLAB/Capstone/drone_yolo
P=/home/se/miniconda3/envs/drone/bin/python
Q=logs/QUEUE.md; R=metrics/AUTO_RESULT_v12.md
log(){ echo "[$(TZ=Asia/Seoul date '+%F %T') KST] $*" | tee -a $Q; }
ap(){ awk -F, 'NR==2{print $3}' "$1" 2>/dev/null; }
eval3(){ local m=$1 t Y
  for t in test_obl test_v2 test_kr; do
    case $t in test_obl) Y="--data data/test_obl --mode single --tag test_obl";; test_v2) Y="";; test_kr) Y="--data data/test_kr --tag test_kr";; esac
    [ -f metrics/${t}_$m.csv ] || $P eval_test_v2.py --weights runs_person/$m/weights/best.pt $Y >> logs/q_${m}_$t.log 2>&1
  done
  log "   $m AP50 obl $(ap metrics/test_obl_$m.csv) · v2 $(ap metrics/test_v2_$m.csv) · kr $(ap metrics/test_kr_$m.csv)"
  echo "| $m | $(ap metrics/test_obl_$m.csv) | $(ap metrics/test_v2_$m.csv) | $(ap metrics/test_kr_$m.csv) |" >> $R; }
log "chain_v13 시작 — v7_r4"
$P chain_util.py mklist v7_r4 23 train_v7.txt | tee -a $Q
A="--stage 1 --hyp configs/hyp/v6_oblique.yaml --weights yolo11m.pt --data configs/data_v7_r4.yaml --name v7_r4"
if $P train_person.py $A --smoke >> logs/q_v7_r4_smoke.log 2>&1; then
  log "   v7_r4 연기 통과 → 30 에폭"; $P train_person.py $A >> logs/q_v7_r4.log 2>&1
  if [ -f runs_person/v7_r4/weights/best.pt ]; then
    eval3 v7_r4
    for s in "soup_v7p14 v7 v7_r4" "soup_v7p24 v7_r2 v7_r4" "soup_v7t124 v7 v7_r2 v7_r4"; do
      set -- $s; $P soup.py $@ >> $Q 2>&1 && eval3 $1
    done
  else log "   ❌ v7_r4 학습 실패"; fi
else log "   ❌ v7_r4 연기 실행 실패"; fi
log "chain_v13 완료"
