#!/bin/bash
# gpu_watchdog.sh — GPU 하드웨어 장애(Xid154류) 조기 감지 및 안전 정지
#
# 왜 필요한가
#   2026-09-11 E3(yolo11s@1280) 14에폭에서 GPU가 응답불능(Xid 154)이 됐는데,
#   아무도 감시하지 않아 한참 뒤에야 발견했다. 그 사이 죽은 GPU를 계속 찌르던
#   프로세스(rmmod)가 SIGKILL 도 안 먹는 상태로 수십 분 CPU 를 물고 있었다.
#
# 이 스크립트가 하는 일 (안전한 것만)
#   - nvidia-smi 를 주기적으로 타임아웃 걸고 호출해 GPU 응답성을 확인
#   - 연속 2회 실패하면: 학습 체인(run_remaining_experiments.sh 및 그 자식
#     train_person.py)을 즉시 종료 시도하고, 사고 시각·마지막 실행 중이던
#     실험 이름을 gpu_watchdog_incident.log 에 남긴다
#   - **재부팅이나 모듈 리로드는 절대 자동으로 하지 않는다** — 그건 항상
#     사람이 다른 사용자와 조율해서 결정할 문제다 (2026-09-11 에 겪은 일)
#
# 사고 발생 후 사람이 할 일
#   1) gpu_watchdog_incident.log 확인 — 어느 실험(--name)이 몇 에폭에서 죽었는지
#   2) bz2149 등 동시 사용자와 재부팅 조율
#      ★ sudo reboot / sudo shutdown 금지 — 이 서버는 꺼지면 사람이 직접 켜야 한다 (관리자 공지, 09-14).
#        커널 강제 재시작: sync → echo 1 | sudo tee /proc/sys/kernel/sysrq
#                          → echo s | sudo tee /proc/sysrq-trigger → echo b | sudo tee /proc/sysrq-trigger
#   3) 재부팅 후, 죽은 실험은 처음부터 말고 이어서:
#        python train_person.py --resume --name <죽은 실험 이름>
#      (last.pt 가 있으면 그 에폭부터 재개된다. 없으면 처음부터.)
#
# 실행: nohup bash gpu_watchdog.sh > gpu_watchdog.log 2>&1 &
# 중지: pkill -f gpu_watchdog.sh

cd "$(dirname "$0")"
INCIDENT_LOG="gpu_watchdog_incident.log"
CHECK_INTERVAL=30      # 초
FAIL_THRESHOLD=2        # 연속 실패 횟수

fail_count=0
echo "[$(date)] watchdog 시작 (PID $$)"

while true; do
  if timeout 10 nvidia-smi >/dev/null 2>&1; then
    fail_count=0
  else
    fail_count=$((fail_count + 1))
    echo "[$(date)] nvidia-smi 실패 ${fail_count}/${FAIL_THRESHOLD}"
  fi

  if [ "$fail_count" -ge "$FAIL_THRESHOLD" ]; then
    NOW=$(date)
    RUNNING_NAME=$(pgrep -af "train_person.py" | grep -oP '(?<=--name )\S+' | head -1)
    {
      echo "===================================================="
      echo "[$NOW] GPU 응답불능 감지 (nvidia-smi 연속 ${fail_count}회 실패)"
      echo "  마지막 실행 중이던 실험: ${RUNNING_NAME:-알수없음}"
      echo "  → 재부팅 전까지 이 실험은 진행 불가. 사람이 확인 필요."
      echo "  → 재부팅 후 이어하려면:"
      echo "      python train_person.py --resume --name ${RUNNING_NAME:-<실험이름>}"
      echo "===================================================="
    } | tee -a "$INCIDENT_LOG"

    # 학습 체인·프로세스 정지 시도 (재부팅/모듈 조작은 하지 않는다)
    # 주의: 죽은 GPU를 붙잡은 채 커널 안에서 멈춘 프로세스는 SIGKILL 도
    # 안 먹을 수 있다(2026-09-11 rmmod 가 그랬다). 그래도 시도는 한다.
    pkill -TERM -f "run_remaining_experiments.sh" 2>/dev/null
    pkill -TERM -f "train_person.py" 2>/dev/null
    sleep 5
    pkill -KILL -f "run_remaining_experiments.sh" 2>/dev/null
    pkill -KILL -f "train_person.py" 2>/dev/null
    if pgrep -f "train_person.py" >/dev/null 2>&1; then
      echo "[$NOW] 경고: train_person.py 가 SIGKILL 에도 안 죽음 — 커널에 멈춘 것으로 추정, 재부팅 필요"
    else
      echo "[$NOW] 학습 프로세스 종료됨."
    fi

    # 복구될 때까지 대기 후(재부팅 감지) 다시 감시 재개 — 자동 재시작은 안 함
    until timeout 10 nvidia-smi >/dev/null 2>&1; do
      sleep 30
    done
    echo "[$(date)] GPU 복구 확인됨 (nvidia-smi 정상). 학습 재개는 사람이 판단."
    fail_count=0
  fi

  sleep "$CHECK_INTERVAL"
done
