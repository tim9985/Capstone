#!/bin/bash
# train_edge.sh — 가장자리 노출을 늘린 학습 (2026-09-20)
#
# 왜
#   학습 크롭은 사람이 거의 가운데에 온다 (저장된 크롭에서 가장자리 100 px 안 1.8 %).
#   타일 추론에서는 사람이 타일 어디에나 오므로 학습/추론 불일치가 생긴다.
#   크롭을 다시 만들지 않고 **translate 0.15 → 0.30** 으로 학습 중 이미지를 더 흔든다.
#
# 함께 바꾸는 것
#   close_mosaic 5 → 0 — 모델 3종에서 일관되게 성능을 떨어뜨렸다 (09-20 확인)
#   비교 대상 fov_11m_1280_all 은 close_mosaic 5 였지만, 둘 다 best.pt(정점)로 비교하므로 영향이 작다
set -u
cd /home/se/JupyterLAB/Capstone/drone_yolo
PY=/home/se/miniconda3/envs/drone/bin/python
NAME=e1_11m_translate30
log(){ echo "=== $* [$(date -u '+%F %T') UTC] ==="; }
pgrep -af "train_person\.py" | awk '$2 ~ /python/' | grep . && { log "다른 학습 중 — 중단"; exit 1; }

if [ -f weights/$NAME.pt ]; then log "$NAME 이미 완료"
elif [ -f runs_person/$NAME/weights/last.pt ]; then
  log "이어 학습"; $PY train_person.py --stage 1 --resume --name $NAME
else
  log "새 학습 $NAME (translate 0.30 · close_mosaic 0)"
  $PY train_person.py --stage 1 --data configs/data_fov.yaml --weights yolo11m.pt --imgsz 1280 --batch 8 \
    --patience 100 --close-mosaic 0 --optimizer SGD --lr0 0.01 --scale 0.3 --translate 0.30 \
    --name $NAME --epochs 30
fi
[ -f weights/$NAME.pt ] || { log "가중치 복사 없음"; exit 1; }
log "가림 지형 평가"
$PY eval_fov.py --testsets data/det_fov_test_budget --tag budget --weights runs_person/$NAME/weights/best.pt 2>&1 \
  | grep -vE 'Scanning|━' | tee logs/eval_fovbudget_$NAME.log | tail -20
log "타일 가장자리 평가"
$PY eval_tile_edge.py --weights runs_person/$NAME/weights/best.pt 2>&1 | grep -vE 'Scanning|━' | tail -15
log "완료"; touch logs/train_edge.done
