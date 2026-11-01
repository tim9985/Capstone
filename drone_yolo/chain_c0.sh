#!/bin/bash
# chain_c0.sh — 11-01 C0 추론 전 흐림 (학습 없음) · 판정 = _학습 큐 「11-01 C0」
cd /home/se/JupyterLAB/Capstone/drone_yolo
P=/home/se/miniconda3/envs/drone/bin/python; Q=logs/QUEUE.md; R=metrics/AUTO_RESULT_c0.md
log(){ echo "[$(TZ=Asia/Seoul date '+%F %T') KST] $*" | tee -a $Q $R; }
log "chain_c0 시작 — 흐림 σ 고르기 (Okutama · pose_bench)"
$P pose_bench.py v12_f91_swa --cond=g08,g12 2>&1 | grep -v -i warn | tail -2 >> $R
S=$($P c0_judge.py pick 2>>$R); log "   고른 σ = $S"
$P bench_all.py --models v12_f91_swa,v12_f92_swa,v12_f93_swa --blur $S --sets test_sar,test_v2,test_kr,test_ladd_h,test_visdrone,test_nomad,test_nii,test_cy,test_auair,test_obl 2>&1 | grep -v -i warn >> logs/c0.log
$P c0_judge.py judge $S 2>&1 | tee -a $Q $R
log "chain_c0 완료"
