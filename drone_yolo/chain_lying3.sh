#!/bin/bash
# chain_lying3.sh — 10-30 (2판 · CP 재료 품질 기준 뒤) 누운 사람 강건성: CP (복사-붙여넣기) → EL (타원 회전 박스) · 판정 기준 = _학습 큐 「10-30 CP」 · 「10-30 EL」
cd /home/se/JupyterLAB/Capstone/drone_yolo
P=/home/se/miniconda3/envs/drone/bin/python; S=/home/se/venvs/state/bin/python; Q=logs/QUEUE.md; R=metrics/AUTO_RESULT_lying3.md; IGN=data/test_v2/ignore_v1.json
log(){ echo "[$(TZ=Asia/Seoul date '+%F %T') KST] $*" | tee -a $Q $R; }
ap(){ awk -F, 'NR==2{print $3}' "$1" 2>/dev/null; }
ev(){ local m=$1
  $P eval_test_v2.py --weights runs_person/$m/weights/best.pt --data data/test_obl --mode single --tag test_obl >> logs/l2_$m.log 2>&1
  $P eval_test_v2.py --weights runs_person/$m/weights/best.pt --merge edge --ignore $IGN >> logs/l2_$m.log 2>&1
  $P eval_test_v2.py --weights runs_person/$m/weights/best.pt --data data/test_kr --tag test_kr --merge edge >> logs/l2_$m.log 2>&1
  $P eval_test_v2.py --weights runs_person/$m/weights/best.pt --data data/test_auair --tag test_auair --boot 0 >> logs/l2_$m.log 2>&1
  log "   $m AP50 obl $(ap metrics/test_obl_$m.csv) · v2 edge+무시 $(ap metrics/test_v2_edge_ign_$m.csv) · kr edge $(ap metrics/test_kr_edge_$m.csv) · 도로 오탐 $(awk -F, 'NR==2{print $6}' metrics/test_auair_$m.csv) 건/장"
  for t in test_obl test_v2_edge_ign test_kr_edge; do $P compare_ci.py --tag $t v12_f91_swa:$m 2>&1 | tail -1 | tee -a $Q $R; done
  $P lying_bench.py $m 2>&1 | grep -v -i warn | tail -1 | tee -a $Q $R
  $P diag_misses.py --weights runs_person/$m/weights/best.pt --sets test_obl >> logs/l2_$m.log 2>&1
  $P state_worker_check.py --weights=runs_person/$m/weights/best.pt --tag=${m}_c20 --posture=runs_person/_pack_v12/posture.json --conf=0.20 >> logs/l2_$m.log 2>&1
  log "   $m 상태 (AUROC · 잡힌 비율): $(python3 -c "import json;d=json.load(open('metrics/state_worker_check_${m}_c20.json'));print(d['점수 누움 AUROC'], d['누움 잡힌 비율'])")"; }
tr(){ local name=$1; shift
  A="--stage 1 --hyp configs/hyp/v6_oblique.yaml --weights yolo11m.pt --name $name --epochs 30 $*"
  if $P train_person.py $A --smoke >> logs/l2_${name}_smoke.log 2>&1 && $P train_person.py $A --set save_period=1 >> logs/l2_${name}_train.log 2>&1; then
    $P swa_ckpt.py $name >> $Q 2>&1 && ev ${name}_swa
  else log "   ❌ $name 학습 실패"; fi; }
log "chain_lying3 시작 — CP 2판 (재료 품질 기준 · 1.3배 넘게 안 키움)"
log "CP — v12cp (v12_f91 + 누운 사람 붙인 비스듬 창)"
tr v12cp --data configs/data_v12cp.yaml
log "EL — v12el (v12_f91 · 회전 박스 타원)"
tr v12el --data configs/data_v12_f91.yaml --ellipse-box
log "chain_lying3 완료"
