#!/bin/bash
# chain_d0b.sh — 11-01 D0b 현장 적응 연습 2 (v12 창 1:10 · lr 0.001) · 판정 = _학습 큐 「11-01 D0b」
cd /home/se/JupyterLAB/Capstone/drone_yolo
P=/home/se/miniconda3/envs/drone/bin/python; Q=logs/QUEUE.md
log(){ echo "[$(TZ=Asia/Seoul date '+%F %T') KST] $*" | tee -a $Q; }
A="--stage 1 --hyp configs/hyp/v6_oblique.yaml --weights runs_person/v12_f91_swa/weights/best.pt --name d0b_sar90 --epochs 10 --lr0 0.001 --warmup-epochs 1 --data configs/data_d0b_sar90.yaml"
log "chain_d0b 시작 — UAV-SAR 90° 150 + v12 창 1500 · lr 0.001 · 10 에폭"
if $P train_person.py $A --smoke >> logs/d0b_smoke.log 2>&1 && $P train_person.py $A >> logs/d0b_train.log 2>&1; then
  $P bench_all.py --models v12_f91_swa,d0b_sar90 --sets test_sar45,test_obl,test_v2,test_kr,test_auair 2>&1 | grep -v -i warn >> logs/d0b_eval.log
  grep -E "^d0b_sar90," metrics/bench_all.csv | cut -d, -f1,4,7,11 | tee -a $Q
else log "   ❌ d0b 학습 실패"; fi
log "chain_d0b 완료"
