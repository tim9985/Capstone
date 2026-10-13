#!/bin/bash
# chain_r12b.sh — 10-13 v12 세 번째 반복 (시드 93) → 학습 안 평균 → 평균[v12 3] vs 평균[v7 4] (설명용 · 교체 후보 고르기)
cd /home/se/JupyterLAB/Capstone/drone_yolo
P=/home/se/miniconda3/envs/drone/bin/python; Q=logs/QUEUE.md; R=metrics/AUTO_RESULT_r12.md; IGN=data/test_v2/ignore_v1.json
log(){ echo "[$(TZ=Asia/Seoul date '+%F %T') KST] $*" | tee -a $Q; }
ap(){ awk -F, 'NR==2{print $3}' "$1" 2>/dev/null; }
ev(){ local m=$1
  $P eval_test_v2.py --weights runs_person/$m/weights/best.pt --data data/test_obl --mode single --tag test_obl >> logs/r12b_$m.log 2>&1
  $P eval_test_v2.py --weights runs_person/$m/weights/best.pt --merge edge --ignore $IGN >> logs/r12b_$m.log 2>&1
  $P eval_test_v2.py --weights runs_person/$m/weights/best.pt --data data/test_kr --tag test_kr --merge edge >> logs/r12b_$m.log 2>&1
  log "   $m AP50 obl $(ap metrics/test_obl_$m.csv) · v2 edge+무시 $(ap metrics/test_v2_edge_ign_$m.csv) · kr edge $(ap metrics/test_kr_edge_$m.csv)"; }
log "chain_r12b 시작 — v12_f93 (시드 93)"
$P chain_util.py mklistf v12_f93 93 0.95 train_v12.txt | tee -a $Q
printf "# v12_f93 · val = val_v12\ntrain: %s\nval: %s\nnc: 1\nnames: ['person']\n" "$PWD/configs/lists/train_v12_f93.txt" "$PWD/configs/lists/val_v12.txt" > configs/data_v12_f93.yaml
A="--stage 1 --hyp configs/hyp/v6_oblique.yaml --weights yolo11m.pt --data configs/data_v12_f93.yaml --name v12_f93 --epochs 30"
if $P train_person.py $A --smoke >> logs/r12b_smoke.log 2>&1 && $P train_person.py $A --set save_period=1 >> logs/r12b_v12_f93.log 2>&1; then
  ev v12_f93; $P swa_ckpt.py v12_f93 >> $Q 2>&1 && ev v12_f93_swa
  echo "- 세 반복 평균 (설명용):" >> $R
  for t in test_obl test_v2_edge_ign test_kr_edge; do $P compare_means.py --tag $t v7,v7_r2,v7_r3,v7_r4 v12_f91_swa,v12_f92_swa,v12_f93_swa | tail -1 | tee -a $Q $R; done
else log "   ❌ v12_f93 실패"; fi
log "chain_r12b 완료"
