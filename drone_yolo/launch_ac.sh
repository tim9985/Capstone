#!/bin/bash
# launch_ac.sh — 10-31 사용자 「1 · 2 · 3 다 진행」: chain_lying3 끝나면 A (chain_lying4) → C (chain_c)
#   A 거는 조건 (큰 손해 없음): 단일 CP · EL 도로 오탐 ≤ 1.0 건/장 — 어기면 A 건너뛰고 C 만
cd /home/se/JupyterLAB/Capstone/drone_yolo; Q=logs/QUEUE.md
until grep -q "chain_lying3 완료" $Q; do sleep 300; done
fp(){ awk -F, 'NR==2{print $6}' metrics/test_auair_$1.csv 2>/dev/null; }
ok=1; for m in v12cp_swa v12el_swa; do v=$(fp $m); [ -n "$v" ] && awk "BEGIN{exit !($v <= 1.0)}" || ok=0; done
if [ $ok = 1 ]; then ./chain_lying4.sh; else echo "[$(TZ=Asia/Seoul date '+%F %T') KST] A 건너뜀 — CP · EL 도로 오탐 > 1.0 또는 없음" | tee -a $Q; fi
./chain_c.sh
