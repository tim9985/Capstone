#!/bin/bash
# finish_m1.sh — 재부팅 후 M1 마무리 학습 (2026-09-12 회의 전)
#
# 배경: m1_11m_1280 이 GPU 장애로 34에폭 스케줄의 13에폭에서 멈췄다(학습률 높음 · 미수렴).
#       last.pt 에서 짧은 스케줄로 학습률을 끝까지 내려 수렴시키고, GPU 로 도메인 평가한다.
#
# 설정 근거
#   · 옵티마이저 MuSGD 명시 — auto 면 lr0 를 무시한다. 원 실행도 auto → MuSGD(lr 0.01) 였다
#   · lr0 0.006 — 원 스케줄(34에폭 선형)의 13에폭 지점 학습률 ≈ 0.01 × 0.625
#   · warmup 0.5 에폭 — 처음부터가 아니므로 3에폭 워밍업은 낭비
#   · batch 6 고정 — AutoBatch 가 3 을 골랐다(메모리 2.5배 과대 예측). BatchNorm 안정용
#   · close_mosaic 3 — 남은 에폭이 약 9개라 마지막 3에폭만 모자이크 끔
#   · GPU 온도 · 전력 · 스로틀 원인을 30초마다 기록 — 24시간 안에 두 번 죽은 원인 추적용
#
# 실행 (재부팅 후): cd ~/JupyterLAB/drone_dev/drone_yolo && nohup ./finish_m1.sh > finish_m1.log 2>&1 &
set -u
cd /home/se/JupyterLAB/drone_dev/drone_yolo
export MPLBACKEND=Agg
PY=/home/se/miniconda3/envs/drone/bin/python
SRC=runs_person/m1_11m_1280/weights/last.pt
NAME=m1f_11m_1280
# 학습 마감 12:40 KST → 최종 검증 + GPU 평가(~15분) 후 13:00 전 결과
TRAIN_DEADLINE_UTC="2026-09-12 03:40 UTC"

log()  { echo "=== $* [$(date '+%F %T %Z')] ==="; }
fail() { echo "!! $*"; echo "[$(date)] finish_m1: $*" >> overnight_incident.log; log "중단 — 사람 확인 필요"; exit 1; }

log "시작"
nvidia-smi --query-gpu=name,driver_version,power.limit,temperature.gpu --format=csv || fail "nvidia-smi 실패 — GPU 없음"
$PY -c "import torch; assert torch.cuda.is_available(); x=torch.randn(2048,2048,device='cuda'); print('CUDA OK', float((x@x).sum()) != 0)" \
  || fail "torch 에서 CUDA 사용 불가"
pgrep -af "train_person\.py" | awk '$2 ~ /python/' | grep . && fail "다른 train_person.py 가 실행 중"
[ -f "$SRC" ] || fail "시작 가중치 없음: $SRC"

# 감시 스크립트 · GPU 기록 (재부팅으로 둘 다 사라졌다)
pgrep -f "bash gpu_watchdog.sh" >/dev/null || { nohup bash gpu_watchdog.sh > gpu_watchdog.log 2>&1 & echo "gpu_watchdog 재시작"; }
nvidia-smi --query-gpu=timestamp,temperature.gpu,power.draw,power.limit,utilization.gpu,memory.used,fan.speed,pcie.link.gen.current,clocks_event_reasons.active \
  --format=csv -l 30 > gpu_telemetry_20260912.csv 2>&1 &
TELE=$!

HOURS=$($PY -c "import time,subprocess;e=int(subprocess.check_output(['date','-d','$TRAIN_DEADLINE_UTC','+%s']));print(round((e-time.time())/3600-0.05,2))")
echo "학습 시간 예산: ${HOURS}시간 (마감 $TRAIN_DEADLINE_UTC)"
$PY -c "import sys; sys.exit(0 if float('$HOURS') >= 1.0 else 1)" || { kill $TELE; fail "남은 시간이 1시간 미만 (${HOURS}h) — 학습 생략"; }

log "마무리 학습 시작 $NAME (from $SRC)"
$PY train_person.py --stage 1 --data configs/data_all.yaml --weights "$SRC" \
  --imgsz 1280 --batch 6 --name $NAME --workers 8 --cache disk \
  --time "$HOURS" --close-mosaic 3 --optimizer MuSGD --lr0 0.006 --warmup-epochs 0.5
rc=$?
log "학습 종료 (rc=$rc)"
W=runs_person/$NAME/weights/best.pt
[ -f "$W" ] || { kill $TELE; fail "best.pt 없음"; }
cut -d, -f1,2,6-9 runs_person/$NAME/results.csv

log "평가 $NAME @1280"
$PY eval_domain.py --weights "$W" --imgsz 1280 --out metrics/eval_domain_${NAME}.csv > eval_${NAME}.log 2>&1 \
  || echo "경고: 평가 실패 (eval_${NAME}.log)"
grep -E "mAP50 |재현율" eval_${NAME}.log

kill $TELE 2>/dev/null
log "전체 완료"
