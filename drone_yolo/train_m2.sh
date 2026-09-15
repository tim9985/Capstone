#!/bin/bash
# train_m2.sh — M2: yolo11m @1280 · batch 8 · data_all + SARD train (09-14 밤, 사용자 결정)
#
# M1(m1_11m_1280 · batch 3 · best mAP50 0.651) 대비 바뀐 것 두 가지 — 원인 분리는 안 된다
#   1) batch AutoBatch 3 → 8 고정 (batch 3 은 BN 통계가 흔들린다 · 22에폭부터 과적합 신호)
#   2) SARD train 1,386장 1클래스 추가 (data/det/sard · configs/data_m2_sard.yaml). val 은 data_all 그대로
# 시간 제한 7.5시간 (에폭 수 · 학습률 스케줄 자동) · close_mosaic 5 · patience 10
#
# autoheal/job.conf 에 등록 — GPU 장애로 재부팅되면 이 스크립트가 다시 불려 last.pt 에서 이어 학습한다
# (--time 0: 이어 학습 시점부터 시간을 다시 재지 않고 저장된 에폭 수까지 간다 · resume_m1.sh 와 같은 방식)
#
# 실행: cd ~/JupyterLAB/Capstone/drone_yolo && setsid nohup ./train_m2.sh > logs/train_m2.log 2>&1 < /dev/null &
# 진행 확인: python train_status.py · tail -f logs/train_m2.log
set -u
cd /home/se/JupyterLAB/Capstone/drone_yolo
export MPLBACKEND=Agg   # 화면 없는 서버에서 최종 검증 PR 곡선 그리기가 죽지 않게 (E1 에서 겪음)
PY=/home/se/miniconda3/envs/drone/bin/python
NAME=m2_11m_1280_sard

log()  { echo "=== $* [$(date '+%F %T %Z')] ==="; }
fail() { echo "!! $*"; echo "[$(date)] train_m2: $*" >> overnight_incident.log; log "중단 — 사람 확인 필요"; exit 1; }

log "시작"
$PY -c "import torch; assert torch.cuda.is_available(); x=torch.randn(2048,2048,device='cuda'); assert torch.isfinite(x@x).all(); print('CUDA OK')" \
  || fail "torch 에서 CUDA 사용 불가"
pgrep -af "train_person\.py" | awk '$2 ~ /python/' | grep . && fail "다른 train_person.py 가 실행 중 — 중복 학습 방지"

if [ -f runs_person/$NAME/weights/last.pt ]; then
  log "이어 학습 $NAME"
  $PY train_person.py --stage 1 --resume --name $NAME --time 0
else
  log "새 학습 $NAME"
  $PY train_person.py --stage 1 --data configs/data_m2_sard.yaml --weights yolo11m.pt \
    --imgsz 1280 --batch 8 --time 7.5 --close-mosaic 5 --patience 10 --name $NAME
fi
rc=$?
log "학습 종료 (rc=$rc)"
[ -f runs_person/$NAME/weights/best.pt ] || fail "best.pt 없음"
tail -4 runs_person/$NAME/results.csv

if [ "$rc" -eq 0 ] && nvidia-smi > /dev/null 2>&1; then
  log "도메인별 평가 $NAME @1280"
  $PY eval_domain.py --weights runs_person/$NAME/weights/best.pt --imgsz 1280 \
    --out metrics/eval_domain_${NAME}.csv > logs/eval_${NAME}.log 2>&1 || echo "경고: 평가 실패 (logs/eval_${NAME}.log)"
fi
log "끝"
