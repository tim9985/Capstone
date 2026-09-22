#!/bin/bash
# chain_d.sh — v3a 평가가 끝나면 D(항공 사전학습) 학습 (2026-09-22)
# 완료 판정은 **결과 파일**로 한다 (pgrep -f 는 자기 셸을 잡는다 · CLAUDE.md §4-17)
cd /home/se/JupyterLAB/Capstone/drone_yolo
P=/home/se/miniconda3/envs/drone/bin/python
log(){ echo "[$(TZ=Asia/Seoul date '+%F %T') KST] $*"; }

log "① v3a 평가 완료 대기 (logs/eval_v3a.log 에 '완료')"
for i in $(seq 1 120); do
  grep -q "^\[.*\] 완료$" logs/eval_v3a.log 2>/dev/null && break
  sleep 60
done
grep -q "^\[.*\] 완료$" logs/eval_v3a.log 2>/dev/null || log "2시간 내 평가 미완 — 학습은 그대로 진행"
sleep 20

W=weights/visdrone_11m/best.pt
if [ ! -s "$W" ]; then log "가중치 없음: $W — 중단"; exit 1; fi
log "② D 학습 시작 — 시작 가중치 $W (VisDrone 300ep · imgsz 640 · COCO→VisDrone)"
exec $P train_person.py --stage 1 --data configs/data_fov.yaml \
  --weights "$W" \
  --imgsz 1280 --batch 8 --epochs 30 --patience 100 --close-mosaic 0 \
  --optimizer SGD --lr0 0.01 --scale 0.3 --translate 0.15 \
  --name vd_11m
