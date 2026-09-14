#!/bin/bash
# gpu_stability_test.sh — 전력 제한 단계별 GPU 안정성 테스트 (2026-09-14)
#
# 배경
#   GPU 장애 3회: 09-11 E3(350 W · 약 75분) · 09-12 M1(350 W · 약 3.6시간) · 09-14 M1 이어 학습(300 W · 29분).
#   마지막 장애 직전 온도 65 °C · 스로틀은 전력 한도뿐 → 과열 아님. 전력을 낮추면 버티는지 확인한다.
#
# 부하
#   실제 학습과 같은 조건(yolo11m @1280 · batch 3 · NOMAD+WiSARD · 검증 포함).
#   합성 부하(행렬곱)는 데이터로더 · 검증이 섞인 실제 부하 패턴을 재현하지 못할 수 있다.
#   M1 체크포인트를 건드리지 않도록 별도 이름(stabtest_<W>W) · COCO 가중치에서 새로 시작한다
#   (장애가 last.pt 저장 순간과 겹치면 M1 파일이 깨질 수 있어서). 결과물은 테스트 후 지워도 된다.
#
# 진행
#   단계마다 사용자가 `sudo nvidia-smi -pl <W>` 를 실행 → 스크립트가 감지하면 MINUTES 동안 부하.
#   GPU 가 사라지면 그 단계에서 끝낸다 (sysrq 재시작 필요). 결과는 gpu_stability_<시각>.txt.
#
# 한계
#   장애 간격이 29분 ~ 3.6시간으로 들쭉날쭉하다. 30분 통과는 "그 전력에서 안전하다"는 증거가 약하다.
#   통과한 가장 낮은 단계로 더 길게 돌려봐야 확신할 수 있다.
#
# 실행: cd ~/JupyterLAB/Capstone/drone_yolo && nohup ./gpu_stability_test.sh > gpu_stability_test.log 2>&1 &
#       단계 · 시간 바꾸기: LEVELS="200 250" MINUTES=45 nohup ./gpu_stability_test.sh > ... &
set -u
cd /home/se/JupyterLAB/Capstone/drone_yolo
export MPLBACKEND=Agg
PY=/home/se/miniconda3/envs/drone/bin/python
LEVELS=${LEVELS:-"200 250 300"}
MINUTES=${MINUTES:-30}
SUMMARY=gpu_stability_$(date +%Y%m%d_%H%M).txt

log()    { echo "=== $* [$(date '+%F %T %Z')] ==="; }
note()   { echo "$*" | tee -a "$SUMMARY"; }
gpu_ok() { timeout 10 nvidia-smi >/dev/null 2>&1; }

gpu_ok || { note "시작 시 GPU 없음 — 재시작 후 다시 실행"; exit 1; }
pgrep -af "train_person\.py" | awk '$2 ~ /python/' | grep . && { note "다른 train_person.py 실행 중 — 종료"; exit 1; }
# ^ 고정 — 그냥 -f 는 이 스크립트를 띄운 셸의 명령줄까지 잡아 감시를 시작하지 않았다 (09-14 첫 실행)
pgrep -f '^bash gpu_watchdog\.sh' >/dev/null || { nohup bash gpu_watchdog.sh > gpu_watchdog.log 2>&1 & echo "gpu_watchdog 시작"; }

note "# GPU 안정성 테스트 $(date '+%F %H:%M %Z') · 단계 ${LEVELS} W · 단계당 ${MINUTES}분 · 부하 yolo11m@1280 batch 3"
HOURS=$($PY -c "print(round(${MINUTES}/60, 3))")

for W in $LEVELS; do
  log "단계 ${W}W — 'sudo nvidia-smi -pl ${W}' 실행 대기 (최대 60분)"
  for i in $(seq 1 240); do
    pl=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits 2>/dev/null | head -1)
    $PY -c "import sys; sys.exit(0 if abs(float('${pl:-0}') - ${W}) < 1 else 1)" 2>/dev/null && break
    gpu_ok || { note "${W}W: 대기 중 GPU 사라짐 — 종료"; exit 1; }
    [ "$i" -eq 240 ] && { note "${W}W: 60분 안에 전력 제한이 적용되지 않아 종료 (현재 ${pl}W)"; exit 1; }
    sleep 15
  done

  TELE=gpu_telemetry_stab_${W}W_$(date +%m%d_%H%M).csv
  # 장애 직전 해상도를 높이려고 10초 간격
  nvidia-smi --query-gpu=timestamp,temperature.gpu,power.draw,power.limit,utilization.gpu,memory.used,fan.speed,pcie.link.gen.current,clocks_event_reasons.active \
    --format=csv -l 10 > "$TELE" 2>&1 &
  TELE_PID=$!
  START=$(date +%s)
  log "단계 ${W}W 부하 시작 (${MINUTES}분)"
  # 캐시 · 최종 검증 여유 15분. GPU 가 사라진 채 프로세스가 매달려도 여기서 끊는다
  timeout -k 60 $(( MINUTES * 60 + 900 )) \
    $PY train_person.py --stage 1 --data configs/data_all.yaml --weights yolo11m.pt \
      --imgsz 1280 --batch 3 --name stabtest_${W}W --workers 8 --cache disk \
      --time "$HOURS" --patience 0 > stabtest_${W}W.log 2>&1
  rc=$?
  ELAPSED=$(( ($(date +%s) - START) / 60 ))
  kill "$TELE_PID" 2>/dev/null
  MAXT=$(awk -F', ' 'NR>1 && $2+0>m{m=$2+0} END{print m+0}' "$TELE")

  if gpu_ok; then
    note "${W}W: 통과 — ${ELAPSED}분 (rc=${rc}, 최고 ${MAXT}°C) · 기록 ${TELE}"
    [ "$rc" -ne 0 ] && note "   주의: rc=${rc} — stabtest_${W}W.log 확인 (시간 초과로 끊겼을 수 있음)"
  else
    note "${W}W: ✗ GPU 사라짐 — 부하 ${ELAPSED}분 만에 (rc=${rc}, 최고 ${MAXT}°C) · 기록 ${TELE}"
    note "→ sysrq 재시작 필요. 테스트 종료"
    log "종료 (장애)"
    exit 1
  fi
done

note "모든 단계 통과 — 가장 낮은 통과 단계로 더 길게 돌려 확인할 것"
log "종료"
