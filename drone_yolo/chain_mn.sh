#!/bin/bash
# chain_mn.sh — 11-02 MN 마네킹 창 + 같은 촬영장 빈 창 × 반복 2 · 판정 = _학습 큐 「11-02 MN」
cd /home/se/JupyterLAB/Capstone/drone_yolo
P=/home/se/miniconda3/envs/drone/bin/python; Q=logs/QUEUE.md; R=metrics/AUTO_RESULT_mn.md; IGN=data/test_v2/ignore_v1.json
HO=20-25m,20-25m_2,30-35m
log(){ echo "[$(TZ=Asia/Seoul date '+%F %T') KST] $*" | tee -a $Q $R; }
ap(){ awk -F, 'NR==2{print $3}' "$1" 2>/dev/null; }
mq(){ $P -c "
import json,sys;d=json.load(open('metrics/mannequin_bench_$1.json'))
t={p:[0,0] for p in ('누움','서기')}
for g in '$HO'.split(','):
    for p in t:
        if p in d.get(g,{}): t[p][0]+=d[g][p]['n']*d[g][p]['뜸@0.20']; t[p][1]+=d[g][p]['n']
print('$1 평가 회차 뜸', {p:round(a/b,3) for p,(a,b) in t.items() if b})" | tee -a $Q $R; }
ev(){ local m=$1
  $P eval_test_v2.py --weights runs_person/$m/weights/best.pt --data data/test_obl --mode single --tag test_obl >> logs/mn_$m.log 2>&1
  $P eval_test_v2.py --weights runs_person/$m/weights/best.pt --merge edge --ignore $IGN >> logs/mn_$m.log 2>&1
  $P eval_test_v2.py --weights runs_person/$m/weights/best.pt --data data/test_kr --tag test_kr --merge edge >> logs/mn_$m.log 2>&1
  $P eval_test_v2.py --weights runs_person/$m/weights/best.pt --data data/test_auair --tag test_auair --boot 0 >> logs/mn_$m.log 2>&1
  log "   $m AP50 obl $(ap metrics/test_obl_$m.csv) · v2 edge+무시 $(ap metrics/test_v2_edge_ign_$m.csv) · kr edge $(ap metrics/test_kr_edge_$m.csv) · 도로 오탐 $(awk -F, 'NR==2{print $6}' metrics/test_auair_$m.csv) 건/장"
  $P lying_bench.py $m 2>&1 | grep -v -i warn | tail -1 | tee -a $Q $R
  $P bench_all.py --models $m --sets test_sar >> logs/mn_$m.log 2>&1; grep "^$m,ours,1280,test_sar," metrics/bench_all.csv | cut -d, -f1,4,7 | tee -a $Q $R
  $P mannequin_bench.py $m $HO 60 >> logs/mn_$m.log 2>&1; mq $m; }
tr(){ local name=$1; shift
  A="--stage 1 --hyp configs/hyp/v6_oblique.yaml --weights yolo11m.pt --name $name --epochs 30 $*"
  if $P train_person.py $A --smoke >> logs/mn_${name}_smoke.log 2>&1 && $P train_person.py $A --set save_period=1 >> logs/mn_${name}_train.log 2>&1; then
    $P swa_ckpt.py $name >> $Q 2>&1 && ev ${name}_swa
  else log "   ❌ $name 학습 실패"; fi; }
log "chain_mn 시작 — 마네킹 창 811 + 빈 창 521 × 반복 2"
log "MN1 — v12mqn (train_v12mq + 빈 창)"
tr v12mqn --data configs/data_v12mqn.yaml
log "MN2 — v12mqn92 (train_v12mq92 + 빈 창)"
tr v12mqn92 --data configs/data_v12mqn92.yaml
log "MN 평균 비교"
for t in test_obl test_v2_edge_ign test_kr_edge; do $P compare_means.py --tag $t v12_f91_swa,v12_f92_swa,v12_f93_swa v12mqn_swa,v12mqn92_swa 2>&1 | tail -1 | tee -a $Q $R; done
log "chain_mn 완료"
