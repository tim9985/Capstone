#!/bin/bash
# chain_v3a.sh — 마이닝이 끝나면 ① 중복률 측정 ② v3a 학습 (2026-09-21)
# 판정은 **결과 파일로만** 한다 (pgrep -f 는 자기 셸을 잡는다 · CLAUDE.md §4-17)
cd /home/se/JupyterLAB/Capstone/drone_yolo
P=/home/se/miniconda3/envs/drone/bin/python
log(){ echo "[$(TZ=Asia/Seoul date '+%F %T') KST] $*"; }   # 서버는 UTC 다 — 반드시 TZ 를 준다

log "① 마이닝 완료 대기 (data/det_neg/mine_stats.json)"
for i in $(seq 1 180); do
  [ -f data/det_neg/mine_stats.json ] && break
  sleep 60
done
if [ ! -f data/det_neg/mine_stats.json ]; then
  log "3시간 내 완료 신호 없음 — 중단"; exit 1
fi
sleep 20
log "② 중복률 측정"
$P check_dup.py data/det_neg/images/train 2>&1 | tee logs/v3a_dup.log

n=$(ls data/det_neg/images/train/*.jpg 2>/dev/null | wc -l)
log "음성 ${n}장 확보 → 학습 시작"
exec $P train_person.py --stage 1 --data configs/data_fov_neg.yaml \
  --weights yolo11m.pt --imgsz 1280 --batch 8 --patience 100 \
  --close-mosaic 0 --optimizer SGD --lr0 0.01 --scale 0.3 --translate 0.15 \
  --name v3a_11m_neg --epochs 30
