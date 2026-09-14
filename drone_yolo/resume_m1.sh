#!/bin/bash
# resume_m1.sh — M1(m1_11m_1280) 을 원래 스케줄 그대로 이어 학습 (2026-09-14)
#
# 배경: 09-12 07:16 KST GPU 장애로 34에폭 스케줄의 13에폭(last.pt)에서 멈췄다. 서버가 이틀 꺼져 있어
#       회의용 짧은 마무리(finish_m1.sh)는 실행되지 않았다. 마감이 없으므로 --resume 으로
#       옵티마이저 상태 · 학습률 스케줄을 그대로 이어, 중단이 없었던 것과 같은 결과를 만든다.
#
# 순서
#   1) GPU · CUDA 확인, 중복 학습 확인
#   2) 전력 제한 POWER_TARGET(250 W) 적용 대기 — 사용자가 직접 `sudo nvidia-smi -pl 250` 실행 (최대 60분)
#   3) GPU 감시 스크립트 · 온도/전력/스로틀 기록(30초) 시작 — 24시간 안에 두 번 죽은 원인 추적
#   4) train_person.py --resume → 끝나면 GPU 로 도메인 평가
#
# 실행: cd ~/JupyterLAB/drone_dev/drone_yolo && nohup ./resume_m1.sh > resume_m1.log 2>&1 &
set -u
cd /home/se/JupyterLAB/drone_dev/drone_yolo
export MPLBACKEND=Agg   # 화면 없는 서버에서 최종 검증 PR 곡선 그리기가 죽지 않게 (E1 에서 겪음)
PY=/home/se/miniconda3/envs/drone/bin/python
NAME=m1_11m_1280
# 09-14 안정성 테스트(200/250/300 W 각 30분 모두 통과, 장애 재현 안 됨) 후 사용자 결정으로 250 W.
# 300 W 대비 속도 −7~11 %, 최고 61 °C. 효과가 입증된 건 아니고 위험 노출을 줄이는 쪽의 선택.
POWER_TARGET=250

log()  { echo "=== $* [$(date '+%F %T %Z')] ==="; }
fail() { echo "!! $*"; echo "[$(date)] resume_m1: $*" >> overnight_incident.log; log "중단 — 사람 확인 필요"; exit 1; }

log "시작"
nvidia-smi --query-gpu=name,driver_version,power.limit,temperature.gpu --format=csv || fail "nvidia-smi 실패 — GPU 없음"
$PY -c "import torch; assert torch.cuda.is_available(); x=torch.randn(2048,2048,device='cuda'); assert torch.isfinite(x@x).all(); print('CUDA OK')" \
  || fail "torch 에서 CUDA 사용 불가"
pgrep -af "train_person\.py" | awk '$2 ~ /python/' | grep . && fail "다른 train_person.py 가 실행 중"
[ -f runs_person/$NAME/weights/last.pt ] || fail "last.pt 없음"

log "전력 제한 ${POWER_TARGET}W 적용 대기 (sudo nvidia-smi -pl ${POWER_TARGET})"
for i in $(seq 1 240); do
  pl=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits | head -1)
  $PY -c "import sys; sys.exit(0 if float('$pl') <= $POWER_TARGET + 0.5 else 1)" && break
  [ "$i" -eq 240 ] && fail "60분 안에 전력 제한이 적용되지 않았다 (현재 ${pl}W)"
  sleep 15
done
log "전력 제한 확인: ${pl}W"

# ^ 로 명령줄 시작에 고정 — 그냥 -f 는 이 스크립트를 띄운 셸의 명령줄(문자열 포함)까지 잡아
# "이미 켜져 있음"으로 오판하고 감시를 시작하지 않았다 (09-14 안정성 테스트에서 발견)
pgrep -f '^bash gpu_watchdog\.sh' >/dev/null || { nohup bash gpu_watchdog.sh > gpu_watchdog.log 2>&1 & echo "gpu_watchdog 시작"; }
TELE_CSV=gpu_telemetry_$(date +%Y%m%d_%H%M).csv
# clocks.current.graphics: 클럭 상한(nvidia-smi -lgc) 적용 확인용 · pcie.link.gen.current: Gen3 고정(setpci) 확인용
nvidia-smi --query-gpu=timestamp,temperature.gpu,power.draw,power.limit,utilization.gpu,memory.used,fan.speed,pcie.link.gen.current,clocks.current.graphics,clocks_event_reasons.active \
  --format=csv -l 30 > "$TELE_CSV" 2>&1 &
TELE=$!
echo "GPU 기록: $TELE_CSV"

log "이어 학습 시작 $NAME"
# --time 0: 체크포인트의 time 8.56h 를 끈다. 그대로 두면 resume 시점부터 시간을 다시 재고
# 이어 학습한 에폭 평균으로 총 에폭을 재계산해, 300W 로 느려지면 34 보다 일찍 끝난다.
$PY train_person.py --stage 1 --resume --name $NAME --time 0
rc=$?
log "학습 종료 (rc=$rc)"
W=runs_person/$NAME/weights/best.pt
[ -f "$W" ] || { kill $TELE; fail "best.pt 없음"; }
cut -d, -f1,2,6-9 runs_person/$NAME/results.csv | tail -6
[ "$rc" -eq 0 ] || echo "경고: 학습이 비정상 종료 (rc=$rc) — 마지막 저장 시점의 best.pt 로 평가한다"

if nvidia-smi >/dev/null 2>&1; then
  log "평가 $NAME @1280"
  $PY eval_domain.py --weights "$W" --imgsz 1280 --out metrics/eval_domain_${NAME}_full.csv > eval_${NAME}_full.log 2>&1 \
    || echo "경고: 평가 실패 (eval_${NAME}_full.log)"
  grep -E "mAP50 |재현율" eval_${NAME}_full.log
else
  echo "경고: GPU 응답 없음 — 평가 생략 (재부팅 후 eval_domain.py 로 따로 실행)"
fi

kill $TELE 2>/dev/null
log "전체 완료"
