#!/bin/bash
# chain_r12c.sh — 10-14 v12_f91 에서 30 에폭 더 (사용자 제안) · lr0 0.002 · warmup 1 · 같은 목록 (train_v12_f91) · 판정 기준 _학습 큐 「10-13 E」
cd /home/se/JupyterLAB/Capstone/drone_yolo
P=/home/se/miniconda3/envs/drone/bin/python; Q=logs/QUEUE.md; IGN=data/test_v2/ignore_v1.json
log(){ echo "[$(TZ=Asia/Seoul date '+%F %T') KST] $*" | tee -a $Q; }
ap(){ awk -F, 'NR==2{print $3}' "$1" 2>/dev/null; }
while kill -0 591036 2>/dev/null; do sleep 300; done
log "chain_r12c 시작 — v12_f91 + 30 에폭 (v12_f91e60)"
A="--stage 1 --hyp configs/hyp/v6_oblique.yaml --weights runs_person/v12_f91/weights/best.pt --data configs/data_v12_f91.yaml --name v12_f91e60 --epochs 30 --lr0 0.002 --warmup-epochs 1"
if $P train_person.py $A --smoke >> logs/r12c_smoke.log 2>&1 && $P train_person.py $A --set save_period=1 >> logs/r12c.log 2>&1; then
  $P swa_ckpt.py v12_f91e60 >> $Q 2>&1
  for m in v12_f91e60 v12_f91e60_swa; do
    $P eval_test_v2.py --weights runs_person/$m/weights/best.pt --data data/test_obl --mode single --tag test_obl >> logs/r12c_$m.log 2>&1
    $P eval_test_v2.py --weights runs_person/$m/weights/best.pt --merge edge --ignore $IGN >> logs/r12c_$m.log 2>&1
    $P eval_test_v2.py --weights runs_person/$m/weights/best.pt --data data/test_kr --tag test_kr --merge edge >> logs/r12c_$m.log 2>&1
    log "   $m AP50 obl $(ap metrics/test_obl_$m.csv) · v2 edge+무시 $(ap metrics/test_v2_edge_ign_$m.csv) · kr edge $(ap metrics/test_kr_edge_$m.csv)"
  done
  : > logs/r12c_cmp.txt
  for t in test_obl test_v2_edge_ign test_kr_edge; do $P compare_ci.py --tag $t v12_f91_swa:v12_f91e60_swa | tail -1 | tee -a $Q logs/r12c_cmp.txt; done
  log "   ★ +30 에폭 판정 (v12_f91_swa 대비 · 반복 폭 전체): $($P judge_pairs.py --band=test_obl:3.1,test_v2_edge_ign:3.3,test_kr_edge:1.2 logs/r12c_cmp.txt v12_f91_swa:v12_f91e60_swa | tail -1)"
else log "   ❌ v12_f91e60 실패"; fi
log "chain_r12c 완료"
