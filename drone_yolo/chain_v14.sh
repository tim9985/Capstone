#!/bin/bash
# chain_v14.sh — 09-28: v7 을 60 에폭으로 (30 에폭 학습에서 val 이 끝까지 오름 → 덜 학습됐나) · 바꾸는 것은 에폭 수 하나
#   판정: soup_v7r2 · v7 대비 세 평가셋 — 반복 폭(test_obl 2.8 · test_v2 1.3 %p) 보다 커야 반영 · 09-29 07:30 쯤 결과
cd /home/se/JupyterLAB/Capstone/drone_yolo
P=/home/se/miniconda3/envs/drone/bin/python; Q=logs/QUEUE.md; R=metrics/AUTO_RESULT_v12.md
log(){ echo "[$(TZ=Asia/Seoul date '+%F %T') KST] $*" | tee -a $Q; }
ap(){ awk -F, 'NR==2{print $3}' "$1" 2>/dev/null; }
log "chain_v14 시작 — v7_e60 (60 에폭)"
A="--stage 1 --hyp configs/hyp/v6_oblique.yaml --weights yolo11m.pt --data configs/data_v7.yaml --name v7_e60 --epochs 60"
if $P train_person.py $A --smoke >> logs/q_v7_e60_smoke.log 2>&1; then
  log "   v7_e60 연기 통과 → 60 에폭"; $P train_person.py $A >> logs/q_v7_e60.log 2>&1
  for t in test_obl test_v2 test_kr; do
    case $t in test_obl) Y="--data data/test_obl --mode single --tag test_obl";; test_v2) Y="";; test_kr) Y="--data data/test_kr --tag test_kr";; esac
    $P eval_test_v2.py --weights runs_person/v7_e60/weights/best.pt $Y >> logs/q_v7_e60_$t.log 2>&1
    for b in v7 soup_v7r2; do $P compare_ci.py --tag $t $b:v7_e60 | tail -1 | tee -a $Q; done
  done
  log "   v7_e60 AP50 obl $(ap metrics/test_obl_v7_e60.csv) · v2 $(ap metrics/test_v2_v7_e60.csv) · kr $(ap metrics/test_kr_v7_e60.csv)"
else log "   ❌ v7_e60 연기 실행 실패"; fi
log "chain_v14 완료"
