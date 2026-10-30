#!/bin/bash
# chain_c.sh — 10-31 C 화질 열화 증강 (--degrade) × 반복 2 (데이터 95 % 시드 91 · 92)
#   왜: v12 같은 설정 3회 Okutama 누움 뜸 0.692 · 0.582 · 0.734 (표준편차 7.9 %p) → 한 번 학습으로는 누움 개선을 못 가림
#   판정 = _학습 큐 「10-31 C」 (돌리기 전에 적음) · 비교 기준 = v12_f91 · f92 · f93 _swa 세 회 평균
#   ⚠ chain_lying3 (CP → EL) 이 끝난 뒤에 건다 (GPU 학습 하나씩)
cd /home/se/JupyterLAB/Capstone/drone_yolo
P=/home/se/miniconda3/envs/drone/bin/python; Q=logs/QUEUE.md; R=metrics/AUTO_RESULT_c.md; IGN=data/test_v2/ignore_v1.json
BASEM=v12_f91_swa,v12_f92_swa,v12_f93_swa
log(){ echo "[$(TZ=Asia/Seoul date '+%F %T') KST] $*" | tee -a $Q $R; }
ap(){ awk -F, 'NR==2{print $3}' "$1" 2>/dev/null; }
ev(){ local m=$1
  $P eval_test_v2.py --weights runs_person/$m/weights/best.pt --data data/test_obl --mode single --tag test_obl >> logs/c_$m.log 2>&1
  $P eval_test_v2.py --weights runs_person/$m/weights/best.pt --merge edge --ignore $IGN >> logs/c_$m.log 2>&1
  $P eval_test_v2.py --weights runs_person/$m/weights/best.pt --data data/test_kr --tag test_kr --merge edge >> logs/c_$m.log 2>&1
  $P eval_test_v2.py --weights runs_person/$m/weights/best.pt --data data/test_auair --tag test_auair --boot 0 >> logs/c_$m.log 2>&1
  log "   $m AP50 obl $(ap metrics/test_obl_$m.csv) · v2 edge+무시 $(ap metrics/test_v2_edge_ign_$m.csv) · kr edge $(ap metrics/test_kr_edge_$m.csv) · 도로 오탐 $(awk -F, 'NR==2{print $6}' metrics/test_auair_$m.csv) 건/장"
  $P lying_bench.py $m 2>&1 | grep -v -i warn | tail -1 | tee -a $Q $R
  $P pose_bench.py $m --cond=raw,j30,w30 2>&1 | grep -v -i warn | tail -1 | tee -a $Q $R; }
tr(){ local name=$1; shift
  A="--stage 1 --hyp configs/hyp/v6_oblique.yaml --weights yolo11m.pt --name $name --epochs 30 $*"
  if $P train_person.py $A --smoke >> logs/c_${name}_smoke.log 2>&1 && $P train_person.py $A --set save_period=1 >> logs/c_${name}_train.log 2>&1; then
    $P swa_ckpt.py $name >> $Q 2>&1 && ev ${name}_swa
  else log "   ❌ $name 학습 실패"; fi; }
log "chain_c 시작 — 화질 열화 증강 × 반복 2"
log "C1 — v12dg (train_v12_f91 · 화질 열화)"
tr v12dg --data configs/data_v12_f91.yaml --degrade
log "C2 — v12dg92 (train_v12_f92 · 화질 열화)"
tr v12dg92 --data configs/data_v12_f92.yaml --degrade
log "C 평균 비교"
for t in test_obl test_v2_edge_ign test_kr_edge; do $P compare_means.py --tag $t $BASEM v12dg_swa,v12dg92_swa 2>&1 | tail -1 | tee -a $Q $R; done
log "chain_c 완료"
