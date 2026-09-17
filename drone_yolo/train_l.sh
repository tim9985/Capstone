#!/bin/bash
# train_l.sh — L 계열(yolo11l) 학습 대기열 (2026-09-18)
#
# 왜 11l 인가
#   3090 실측(1920×1080 한 장 전 과정): 11l fp16 27.0 ms → NFR-V01(≤30 ms) 통과. fp32 는 1600 부터 초과다.
#   즉 **FP16 을 쓴다는 조건**에서만 11l 이 실기체 후보가 된다 → 학습해 볼 가치가 있다.
#
# 단계
#   L1  yolo11l · data_fov.yaml(NOMAD+WiSARD · 16~160 px 로그 균등) · 1280 · batch 6 · 30 에폭
#       fov_11s(30 에폭) · fov_11m(8시간=19 에폭) 과 같은 조리법 → **모델 크기 효과만** 본다
#   L2  L1 + AI-Hub 182 (aihub_prep.py 뒤에 추가한다 — 학습에만 쓰고 검증엔 안 쓴다)
#   L3  L2 + 모션 블러 증강 (지금은 블러 증강이 전혀 없다 · albumentations 미설치)
#
# 판정은 **가림 지형** 시험셋(data/det_fov_test_budget)으로 한다.
#   AI-Hub 시험셋은 같은 30 px 에서 0.86 vs 0.33 으로 너무 쉬워 검증에 쓰면 자기기만이다 (09-17 확인)
#
# 실행: cd ~/JupyterLAB/Capstone/drone_yolo && setsid nohup ./train_l_boot.sh > logs/train_l.log 2>&1 < /dev/null &
set -u
cd /home/se/JupyterLAB/Capstone/drone_yolo
PY=/home/se/miniconda3/envs/drone/bin/python
TEST=data/det_fov_test_budget
log() { echo "=== $* [$(date -u '+%F %T') UTC] ==="; }
fail() { log "실패: $*"; exit 1; }

pgrep -af "train_person\.py" | awk '$2 ~ /python/' | grep . && fail "다른 train_person.py 실행 중 — 중복 학습 방지"
[ -d $TEST ] || fail "시험셋 없음: $TEST"

train() {  # 이름 · 가중치 · batch · 추가 인자
  local name=$1 w=$2 bs=$3; shift 3
  if [ -f weights/$name.pt ]; then log "$name 학습 이미 완료"; return 0; fi
  if [ -f runs_person/$name/weights/last.pt ]; then
    log "이어 학습 $name"
    $PY train_person.py --stage 1 --resume --name $name --time 0
  else
    log "새 학습 $name"
    $PY train_person.py --stage 1 --data configs/data_fov.yaml --weights $w --imgsz 1280 --batch $bs \
      --patience 100 --close-mosaic 5 --optimizer SGD --lr0 0.01 --scale 0.3 --name $name "$@"
  fi
  log "학습 종료 $name (rc=$?)"
  [ -f weights/$name.pt ] || fail "$name 가중치 복사 없음 — GPU 장애면 재부팅 뒤 autoheal 이 이어간다"
}

evaluate() {  # 이름 · 가중치 경로
  local name=$1 w=$2
  if [ -f metrics/eval_fovbudget_$name.csv ]; then log "평가 이미 완료 $name"; return 0; fi
  nvidia-smi > /dev/null 2>&1 || fail "평가 전 GPU 없음"
  log "가림 지형 화각 평가 $name"
  $PY eval_fov.py --testsets $TEST --tag budget --weights $w 2>&1 | grep -vE 'Scanning|━' \
    | tee logs/eval_fovbudget_$name.log | tail -30 || fail "평가 실패 $name"
}

# ── L1 ──
train l1_11l_1280 yolo11l.pt 6 --epochs 30
evaluate l1_11l_1280 runs_person/l1_11l_1280/weights/best.pt

log "L1 완료 — L2(AI-Hub 추가)는 aihub_prep.py 뒤에 이 스크립트에 붙인다"
touch logs/train_l.done
