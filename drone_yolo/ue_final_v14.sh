#!/usr/bin/env bash
# ue_final_v14.sh — ue_watch 가 끝나면 20 m 를 상의색 v14 로 한 번 (판정용 · _학습 큐 U) · 16 m 도 참고로
cd "$(dirname "$0")"
P=/home/se/miniconda3/envs/drone/bin/python; W=runs_person/v12_f91_swa/weights; Q=logs/QUEUE.md
while kill -0 1753348 2>/dev/null; do sleep 60; done
echo "[$(TZ=Asia/Seoul date '+%F %T') KST] ue_final_v14 시작 (상의색 v14 · 20 m 판정 · 16 m 참고)" | tee -a $Q
for r in S1_a20 S2_a20 S3_a20 S4_a20 S1_a16 S2_a16 S3_a16 S4_a16; do
  q=""; case $r in S4_*) q="--query green";; esac
  [ -d ue_flight/$r ] || { echo "  $r 없음" | tee -a $Q; continue; }
  $P ue_vision_eval.py ue_flight/$r --weights $W/best.pt --engine $W/best.engine --conf 0.20 --color v14 --tag _v14 $q > logs/ue_eval_${r}_v14.log 2>&1
  echo "  $r v14 exit $? · $(grep '상의색' metrics/ue_vision_${r}_v14.md)" | tee -a $Q
done
echo "[$(TZ=Asia/Seoul date '+%F %T') KST] ue_final_v14 끝" | tee -a $Q
