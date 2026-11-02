#!/bin/bash
# run_bench_m.sh — 11-02 M · MN (· DG) 모델을 12셋 같은 틀 비교에 더함 (평가만 · 학습과 같이 돌아도 됨 · 하나씩)
cd /home/se/JupyterLAB/Capstone/drone_yolo
P=/home/se/miniconda3/envs/drone/bin/python
$P bench_all.py --models v12mq_swa,v12mq92_swa,v12dg_swa,v12dg92_swa 2>&1 | grep -v -i warn >> logs/bench_m.log
while kill -0 178121 2>/dev/null; do sleep 300; done          # chain_mn (PID) 끝날 때까지
$P bench_all.py --models v12mqn_swa,v12mqn92_swa 2>&1 | grep -v -i warn >> logs/bench_m.log
$P bench_table.py > metrics/bench_all.md 2>/dev/null
echo "[$(TZ=Asia/Seoul date '+%F %T') KST] bench M · MN · DG 12셋 완료" >> logs/QUEUE.md
