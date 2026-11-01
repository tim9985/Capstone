#!/bin/bash
# chain_d0.sh — 11-01 D0 현장 적응 연습 (보고만 · D1 설정 고르기) · _학습 큐 「11-01 D0」
cd /home/se/JupyterLAB/Capstone/drone_yolo
P=/home/se/miniconda3/envs/drone/bin/python; Q=logs/QUEUE.md; R=metrics/AUTO_RESULT_d0.md
log(){ echo "[$(TZ=Asia/Seoul date '+%F %T') KST] $*" | tee -a $Q $R; }
A="--stage 1 --hyp configs/hyp/v6_oblique.yaml --weights runs_person/v12_f91_swa/weights/best.pt --name d0_sar90 --epochs 10 --lr0 0.002 --warmup-epochs 1 --data configs/data_d0_sar90.yaml"
log "chain_d0 시작 — v12_f91 에서 UAV-SAR 90° 150 + v12 창 600 · lr 0.002 · 10 에폭"
if $P train_person.py $A --smoke >> logs/d0_smoke.log 2>&1 && $P train_person.py $A >> logs/d0_train.log 2>&1; then
  until ! pgrep -f "chain_c0.sh" > /dev/null; do sleep 60; done          # 평가는 하나씩 (C0 끝난 뒤)
  $P bench_all.py --models v12_f91_swa,d0_sar90 --sets test_sar45,test_obl,test_v2,test_kr,test_auair 2>&1 | grep -v -i warn | tee -a $R
  log "   d0 결과는 metrics/bench_all.csv (d0_sar90 vs v12_f91_swa · UAV-SAR 45° = 새 장소의 다른 각도)"
else log "   ❌ d0 학습 실패"; fi
log "chain_d0 완료"
