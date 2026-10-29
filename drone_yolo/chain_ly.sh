#!/bin/bash
# chain_ly.sh — 10-30 L 누운 사람 보강 (v12ly) → SWA → 세 평가셋 · 짝 비교 · 누움 탐지 · 상태 판단 (판정 기준 = _학습 큐 「10-30 L」)
cd /home/se/JupyterLAB/Capstone/drone_yolo
P=/home/se/miniconda3/envs/drone/bin/python; Q=logs/QUEUE.md; R=metrics/AUTO_RESULT_ly.md; IGN=data/test_v2/ignore_v1.json
log(){ echo "[$(TZ=Asia/Seoul date '+%F %T') KST] $*" | tee -a $Q $R; }
ap(){ awk -F, 'NR==2{print $3}' "$1" 2>/dev/null; }
ev(){ local m=$1
  $P eval_test_v2.py --weights runs_person/$m/weights/best.pt --data data/test_obl --mode single --tag test_obl >> logs/ly_$m.log 2>&1
  $P eval_test_v2.py --weights runs_person/$m/weights/best.pt --merge edge --ignore $IGN >> logs/ly_$m.log 2>&1
  $P eval_test_v2.py --weights runs_person/$m/weights/best.pt --data data/test_kr --tag test_kr --merge edge >> logs/ly_$m.log 2>&1
  log "   $m AP50 obl $(ap metrics/test_obl_$m.csv) · v2 edge+무시 $(ap metrics/test_v2_edge_ign_$m.csv) · kr edge $(ap metrics/test_kr_edge_$m.csv)"; }
log "chain_ly 시작 — v12ly (누운 사람 2,040장 ×3)"
A="--stage 1 --hyp configs/hyp/v6_oblique.yaml --weights yolo11m.pt --data configs/data_v12ly.yaml --name v12ly --epochs 30"
if $P train_person.py $A --smoke >> logs/ly_smoke.log 2>&1 && $P train_person.py $A --set save_period=1 >> logs/ly_train.log 2>&1; then
  $P swa_ckpt.py v12ly >> $Q 2>&1 && ev v12ly_swa
  for t in test_obl test_v2_edge_ign test_kr_edge; do $P compare_ci.py --tag $t v12_f91_swa:v12ly_swa 2>&1 | tail -2 | tee -a $Q $R; done
  $P diag_misses.py --weights runs_person/v12ly_swa/weights/best.pt --sets test_obl >> logs/ly_diag.log 2>&1
  $P lying_det_check.py v12_f91_swa v12ly_swa 2>&1 | tail -3 | tee -a $Q $R
  $P state_worker_check.py --weights=runs_person/v12ly_swa/weights/best.pt --tag=v12ly_c20 --posture=runs_person/_pack_v12/posture.json --conf=0.20 >> logs/ly_state.log 2>&1
  log "   상태: $(python3 -c "import json;d=json.load(open('metrics/state_worker_check_v12ly_c20.json'));print(d['점수 누움 AUROC'], d['누움 잡힌 비율'])")"
else log "   ❌ v12ly 학습 실패"; fi
log "chain_ly 완료"
