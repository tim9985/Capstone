#!/bin/bash
# gpu_autoheal.sh — GPU Xid 79 자동 복구 데몬 (root · systemd 가 부팅 때 실행)
#
# 배경 (SERVER_PROGRESS.md · GPU_XID79_REPORT.md)
#   이 서버의 RTX 3090 은 학습 중 무작위로 PCIe 버스에서 떨어진다 (Xid 79, 09-11 ~ 09-14 4회).
#   관리자 답변(09-14): 하드웨어로는 해결 불가, 재부팅을 자동화할 것. 서버 사용자는 한 명.
#   → 죽으면 사람 없이 재부팅 → 완화 설정 적용 → 학습 이어가기.
#
# 하는 일
#   [시작 단계] (부팅 직후, 또는 서비스가 다시 시작될 때 한 번)
#     1) 드라이버 준비 대기 (nvidia-smi 성공까지 최대 5분)
#     2) 완화 설정: persistence · PCIe Gen3(상위 포트 00:01.0) · 코어 클럭 상한 · 전력 제한
#        PCIe 는 이미 Gen3 거나 학습 중이면 건드리지 않는다 (학습 중 링크 재협상은 GPU 를 떨어뜨릴 수 있다)
#     3) job.conf 의 학습이 안 끝났고 돌고 있지 않으면 se 계정으로 이어 학습 시작
#        학습이 끝났는데 평가 결과가 없으면 평가만 실행
#   [감시 단계] 30초마다
#     · last.pt 가 새로 저장되면 읽어 보고, 정상이면 last_backup.pt 로 백업
#     · GPU 소실 판정 = nvidia-smi 연속 2회 실패 AND 커널 로그에 Xid 79 ("fallen off the bus")
#       → 커널 로그 기록 → 재부팅 횟수 확인 → sync (최대 30초) → sysrq b
#
# 안전장치
#   · autoheal/ENABLED 가 없으면 아무것도 하지 않는다          끄기: rm autoheal/ENABLED
#   · 최근 2시간 안에 자동 재부팅 3번이면 멈추고 autoheal/HALTED 를 남긴다 (사람 확인 후 삭제)
#   · Xid 79 없이 nvidia-smi 만 실패하면 재부팅하지 않는다 (오판 방지)
#   · 학습 · 평가 명령은 root 가 아니라 se 계정으로 실행한다
#   · 재개 전에 last.pt 를 읽어 보고, 깨졌으면 last_backup.pt 로 되돌린다
#
# 재부팅 절차: 관리자 매뉴얼(sysrq 1 → b)에 사용자 결정으로 sync 를 앞에 더했다 (09-14).
#   사람이 타이밍을 볼 수 없는 자동 재부팅이라, 체크포인트를 쓰던 중이면 파일이 깨질 수 있어서다.
#
# 설치: sudo bash autoheal/install.sh   → /usr/local/sbin/gpu_autoheal.sh 로 복사되어 실행된다
#       (레포의 이 파일을 고친 뒤에는 install.sh 를 다시 실행해야 반영된다)
# 점검: bash autoheal/gpu_autoheal.sh --selftest        (root 불필요, 실제 조치 없음)
set -u

REPO=/home/se/JupyterLAB/drone_dev/drone_yolo
AH=$REPO/autoheal
USER_NAME=se
PY=/home/se/miniconda3/envs/drone/bin/python
PCI_PORT=00:01.0          # GPU(02:00.0)가 꽂힌 CPU 직결 PCIe 루트 포트
CLOCK_MIN=300
CLOCK_MAX=1600            # 최대 2130 MHz 의 약 75 %
POWER_W=250
MAX_REBOOTS=3
REBOOT_WINDOW=7200        # 초

DRY_RUN=${DRY_RUN:-0}
STATE=${STATE:-/var/lib/gpu-autoheal}
LOG=${AH_LOG:-$AH/logs/autoheal.log}
HALTED=${HALTED:-$AH/HALTED}

log() { echo "[$(date '+%F %T %Z')] $*" >> "$LOG"; }
run() {   # 실제 조치. DRY_RUN 이면 기록만
  if [ "$DRY_RUN" = 1 ]; then log "(DRY_RUN) $*"; return 0; fi
  "$@" >> "$LOG" 2>&1
}
as_se() { # se 계정으로 실행 (root 가 아니면 그대로 실행 — 점검용)
  if [ "$(id -u)" -eq 0 ]; then runuser -u "$USER_NAME" -- "$@"; else "$@"; fi
}
gpu_ok() { timeout 20 nvidia-smi > /dev/null 2>&1; }
xid79_seen() { dmesg 2>/dev/null | grep -qE "Xid \(PCI:[^)]*\): 79,|fallen off the bus"; }
training_running() { pgrep -af "train_person\.py" | awk '$2 ~ /python/' | grep -q .; }
eval_running() { pgrep -af "eval_domain\.py" | awk '$2 ~ /python/' | grep -q .; }
conf() { grep -E "^$1=" "$AH/job.conf" 2>/dev/null | head -1 | cut -d= -f2-; }

load_job() {
  JOB_NAME=$(conf JOB_NAME); JOB_SCRIPT=$(conf JOB_SCRIPT)
  EVAL_CSV=$(conf EVAL_CSV); EVAL_IMGSZ=$(conf EVAL_IMGSZ)
  if ! [[ "$JOB_NAME" =~ ^[A-Za-z0-9_.-]+$ && "$JOB_SCRIPT" =~ ^[A-Za-z0-9_.-]+\.sh$ ]]; then
    log "job.conf 형식 오류 (JOB_NAME='$JOB_NAME' JOB_SCRIPT='$JOB_SCRIPT') — 학습 자동 재개 안 함"
    return 1
  fi
  [[ "$EVAL_CSV" =~ ^[A-Za-z0-9_./-]*$ && "$EVAL_IMGSZ" =~ ^[0-9]*$ ]] || { EVAL_CSV=""; log "EVAL_* 형식 오류 — 평가 자동 실행 안 함"; }
  RUN_DIR=$REPO/runs_person/$JOB_NAME
  LAST=$RUN_DIR/weights/last.pt
  BAK=$RUN_DIR/weights/last_backup.pt
  return 0
}

job_done() {  # 전체 epoch 소화, 또는 학습 끝의 weights/<name>.pt 복사(조기 종료 포함)
  [ -f "$RUN_DIR/results.csv" ] || return 1
  local rows total
  rows=$(( $(wc -l < "$RUN_DIR/results.csv") - 1 ))
  total=$(awk '/^epochs:/{print $2}' "$RUN_DIR/args.yaml" 2>/dev/null)
  [ -n "$total" ] && [ "$rows" -ge "$total" ] && return 0
  [ -f "$REPO/weights/$JOB_NAME.pt" ] && [ "$REPO/weights/$JOB_NAME.pt" -nt "$RUN_DIR/results.csv" ]
}

ckpt_ok() {
  as_se env CUDA_VISIBLE_DEVICES= "$PY" -c \
    "import sys, torch; ck = torch.load(sys.argv[1], map_location='cpu', weights_only=False); sys.exit(0 if ck.get('epoch') is not None else 1)" \
    "$1" > /dev/null 2>&1
}

apply_mitigations() {
  run nvidia-smi -pm 1
  local cur
  cur=$(setpci -s "$PCI_PORT" CAP_EXP+30.w 2>/dev/null)
  if [ -n "$cur" ] && [ $(( 0x$cur & 0xf )) -eq 3 ]; then
    log "PCIe 목표 속도 이미 Gen3 (LnkCtl2=0x$cur)"
  elif training_running; then
    log "경고: PCIe 목표 속도 확인 불가/Gen3 아님(LnkCtl2=0x${cur:-?}) — 학습 중이라 링크 재협상 건너뜀"
  else
    run setpci -s "$PCI_PORT" CAP_EXP+30.w=0003:000f && run setpci -s "$PCI_PORT" CAP_EXP+10.w=0020:0020
    sleep 3
    log "PCIe Gen3 적용 (LnkCtl2=0x$(setpci -s "$PCI_PORT" CAP_EXP+30.w 2>/dev/null))"
    gpu_ok || log "경고: 링크 재협상 뒤 nvidia-smi 실패 — 감시 단계에서 판정"
  fi
  run nvidia-smi -lgc "$CLOCK_MIN,$CLOCK_MAX" || log "경고: 클럭 상한 적용 실패"
  local pl
  pl=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits 2>/dev/null | head -1)
  if [ "${pl%.*}" != "$POWER_W" ]; then
    run nvidia-smi -pl "$POWER_W" || log "경고: 전력 제한 적용 실패"
  fi
  log "설정 상태: $(nvidia-smi --query-gpu=power.limit,persistence_mode,pcie.link.gen.current,clocks.current.graphics --format=csv,noheader 2>&1 | head -1)"
}

start_eval_if_needed() {
  [ -n "$EVAL_CSV" ] || return
  if [ -f "$REPO/$EVAL_CSV" ]; then log "학습 · 평가 모두 완료 — 할 일 없음"; return; fi
  if eval_running; then log "평가 실행 중 — 건드리지 않음"; return; fi
  local ts; ts=$(date +%Y%m%d_%H%M)
  run as_se bash -c "cd '$REPO' && MPLBACKEND=Agg nohup '$PY' eval_domain.py --weights runs_person/$JOB_NAME/weights/best.pt --imgsz ${EVAL_IMGSZ:-1280} --out $EVAL_CSV > autoheal/logs/eval_${JOB_NAME}_$ts.log 2>&1 &"
  log "학습은 끝났고 평가 결과가 없어 평가 시작 → autoheal/logs/eval_${JOB_NAME}_$ts.log"
}

bootstrap() {
  log "=== 시작 (부팅 $(uptime -s) · DRY_RUN=$DRY_RUN) ==="
  local i
  for i in $(seq 1 30); do gpu_ok && break; sleep 10; done
  if ! gpu_ok; then log "GPU 준비 안 됨 — 설정 · 재개 생략, 감시만 한다"; return; fi
  apply_mitigations
  load_job || return
  if [ -f "$HALTED" ]; then log "HALTED 있음 — 학습 자동 재개 안 함 ($(cat "$HALTED"))"; return; fi
  if job_done; then start_eval_if_needed; return; fi
  if training_running || pgrep -f "^/bin/bash \./$JOB_SCRIPT" > /dev/null; then
    log "학습 이미 실행 중 — 건드리지 않음"; return
  fi
  if [ -f "$LAST" ] && ! ckpt_ok "$LAST"; then
    if [ -f "$BAK" ] && ckpt_ok "$BAK"; then
      run as_se cp -f "$BAK" "$LAST"
      log "last.pt 가 깨져 last_backup.pt 로 복구 (1 epoch 이내 되돌림 · results.csv 에 중복 행이 생길 수 있음)"
    else
      log "!! last.pt 가 깨졌고 쓸 수 있는 백업도 없음 — 학습 자동 재개 안 함"; return
    fi
  fi
  local ts; ts=$(date +%Y%m%d_%H%M)
  run as_se bash -c "cd '$REPO' && nohup ./$JOB_SCRIPT > autoheal/logs/job_${JOB_NAME}_$ts.log 2>&1 &"
  log "학습 자동 재개: $JOB_SCRIPT → autoheal/logs/job_${JOB_NAME}_$ts.log"
}

maybe_backup() {
  [ -n "${LAST:-}" ] && [ -f "$LAST" ] || return
  local m; m=$(stat -c %Y "$LAST")
  [ "$m" = "${LAST_MTIME:-}" ] && return
  sleep 5
  [ "$(stat -c %Y "$LAST")" = "$m" ] || return      # 아직 쓰는 중 → 다음 주기
  if ckpt_ok "$LAST"; then
    run as_se cp -f "$LAST" "$BAK.tmp" && run as_se mv -f "$BAK.tmp" "$BAK"
    LAST_MTIME=$m
    log "last.pt 백업 → last_backup.pt"
  else
    log "경고: last.pt 읽기 실패 — 백업하지 않음"
  fi
}

keep_power_limit() {  # 관리자 nvidia-powerlimit.service 가 부팅 때 350 W 로 거는 것 등으로 바뀌면 되돌린다
  local pl
  pl=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits 2>/dev/null | head -1)
  [ -n "$pl" ] && [ "${pl%.*}" != "$POWER_W" ] || return 0
  log "전력 제한이 ${pl} W 로 바뀌어 있어 ${POWER_W} W 로 다시 적용"
  run nvidia-smi -pl "$POWER_W" || log "경고: 전력 제한 재적용 실패"
}

reboot_now() {
  mkdir -p "$STATE"; touch "$STATE/reboots"
  local now recent
  now=$(date +%s)
  recent=$(awk -v n="$now" -v w="$REBOOT_WINDOW" '$1 > n - w' "$STATE/reboots" | wc -l)
  if [ "$recent" -ge "$MAX_REBOOTS" ]; then
    log "!! 최근 $((REBOOT_WINDOW / 3600))시간 안에 자동 재부팅 ${recent}회 — 더 하지 않는다. 사람 확인 필요"
    echo "$(date '+%F %T %Z') 최근 $((REBOOT_WINDOW / 3600))시간 자동 재부팅 ${recent}회" > "$HALTED"
    [ "$(id -u)" -eq 0 ] && chown "$USER_NAME": "$HALTED"
    return 1
  fi
  echo "$now" >> "$STATE/reboots"
  log "자동 재부팅 $((recent + 1))/$MAX_REBOOTS: sync (최대 30초) → sysrq b"
  if [ "$DRY_RUN" = 1 ]; then log "(DRY_RUN) sync; echo 1 > /proc/sys/kernel/sysrq; echo b > /proc/sysrq-trigger"; return 0; fi
  timeout 30 sync
  echo 1 > /proc/sys/kernel/sysrq
  echo b > /proc/sysrq-trigger
  sleep 60
}

selftest() {
  local tmp; tmp=$(mktemp -d)
  export DRY_RUN=1 STATE=$tmp/state AH_LOG=$tmp/selftest.log HALTED=$tmp/HALTED
  DRY_RUN=1; STATE=$tmp/state; LOG=$tmp/selftest.log; HALTED=$tmp/HALTED
  echo "== 점검 (DRY_RUN · 임시 폴더 $tmp · 실제 조치 없음) =="
  bootstrap
  maybe_backup
  echo "-- 재부팅 횟수 제한: 4번 연속 요청 → 3번 실행, 4번째는 멈춰야 한다"
  for i in 1 2 3 4; do reboot_now && echo "  요청 $i: 재부팅 실행" || echo "  요청 $i: 거부됨 (HALTED)"; done
  echo "-- GPU 상태: $(gpu_ok && echo 정상 || echo 실패) · Xid 79 기록: $(xid79_seen && echo 있음 || echo '없음(또는 root 아님)')"
  echo "-- 기록:"; cat "$LOG"
  [ -f "$HALTED" ] && echo "-- HALTED: $(cat "$HALTED")"
  rm -rf "$tmp"
}

if [ "${1:-}" = "--selftest" ]; then selftest; exit 0; fi

mkdir -p "$STATE" "$(dirname "$LOG")"
BOOTSTRAPPED=0
fails=0
while true; do
  if [ ! -f "$AH/ENABLED" ]; then BOOTSTRAPPED=0; sleep 30; continue; fi
  if [ "$BOOTSTRAPPED" != 1 ]; then bootstrap; BOOTSTRAPPED=1; fi
  if gpu_ok; then
    fails=0
    keep_power_limit
    maybe_backup
  else
    fails=$((fails + 1))
    log "nvidia-smi 실패 $fails/2"
    if [ "$fails" -ge 2 ]; then
      if xid79_seen; then
        log "GPU 소실 확정 (Xid 79). 커널 로그:"
        dmesg -T 2>/dev/null | grep -E "NVRM|Xid" | tail -15 >> "$LOG"
        if [ -f "$HALTED" ]; then log "HALTED 있음 — 재부팅 안 함"; sleep 300; continue; fi
        reboot_now || { sleep 300; continue; }
      else
        log "nvidia-smi 는 실패하지만 Xid 79 기록 없음 — 재부팅하지 않는다 (사람 확인)"
        sleep 270
      fi
    fi
  fi
  sleep 30
done
