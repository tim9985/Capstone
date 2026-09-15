#!/bin/bash
# train_fov.sh — 화각 비교 실험 대기열 (09-15, 사용자 결정 · 20시간 안)
#
# 목적: 렌즈 화각(54·65·80·90°) × 고도(20·25·30 m) 별 사람 탐지율 표
#   1) 데이터: data/det_fov — 사람 크기 16~160 px 로그 균등 (54~90° · 20~30 m 에서 나올 크기 전부)
#   2) 시험셋: data/det_fov_test — 같은 원본에서 화각·고도별 크기로 맞춘 구간 (make_fov_testsets.py)
#   3) 학습 1: yolo11s · 1280 · 30에폭 → eval_fov.py (새 모델 + M1 기준)
#   4) 학습 2: yolo11m · 1280 · 시간 제한 08시간 → eval_fov.py
# 단계마다 끝난 흔적(weights/<이름>.pt · metrics CSV)이 있으면 건너뛴다 → GPU 장애로 재부팅돼도 같은 스크립트로 이어진다
# autoheal/job.conf: JOB_NAME=fov_queue (결과 폴더가 없는 이름이라 "끝남"으로 판정되지 않고 부팅마다 이 스크립트를 다시 부른다)
#
# 실행: cd ~/JupyterLAB/Capstone/drone_yolo && setsid nohup ./train_fov.sh > logs/train_fov.log 2>&1 < /dev/null &
set -u
set -o pipefail   # "명령 | tail" 에서 앞 명령 실패를 놓치지 않게
cd /home/se/JupyterLAB/Capstone/drone_yolo
export MPLBACKEND=Agg
PY=/home/se/miniconda3/envs/drone/bin/python
RAW=../data/raw
DET=data/det_fov
TEST=data/det_fov_test

log()  { echo "=== $* [$(date '+%F %T %Z')] ==="; }
fail() { echo "!! $*"; echo "[$(date)] train_fov: $*" >> overnight_incident.log; log "중단 — 사람 확인 필요"; exit 1; }

log "시작"
pgrep -af "train_person\.py" | awk '$2 ~ /python/' | grep . && fail "다른 train_person.py 가 실행 중 — 중복 학습 방지"

# ── 1) 학습 데이터 ──
PREP="--target-dist loguniform --target-min 16 --target-max 160 --workers 16"
if [ ! -f $DET/.done ]; then
  [ -e $DET ] && fail "$DET 가 불완전하게 남아 있다 — 확인 후 지우고 다시 (옛 .npy 캐시가 섞이지 않게)"
  for b in "NOMAD_b1_10 nomad_actor01_10" "NOMAD_b11_20 nomad_actor11_20" "NOMAD_b21_30 nomad_actor21_30" "NOMAD_sel31_100 nomad_actor_sel31_100"; do
    set -- $b
    log "nomad_prep $2"
    $PY nomad_prep.py --nomad $RAW/$1 --out $DET/$2 --distances 10,30,50,70 $PREP | tail -6 || fail "nomad_prep $2 실패"
  done
  log "wisard_prep"
  $PY wisard_prep.py --out $DET/wisard $PREP | tail -8 || fail "wisard_prep 실패"
  # 분할이 기존 data/det 와 같은지 (val 배우 · 비행)
  $PY - <<'PY' || fail "val 분할이 기존과 다르다"
import json, glob, sys
old = lambda p: json.load(open(p))
for b in ("nomad_actor01_10", "nomad_actor11_20", "nomad_actor21_30", "nomad_actor_sel31_100"):
    a, n = old(f"data/det/{b}/nomad_prep_stats.json")["val_actors"], old(f"data/det_fov/{b}/nomad_prep_stats.json")["val_actors"]
    print(b, n, "같음" if a == n else f"다름 (기존 {a})"); assert a == n
a, n = old("data/det/wisard/wisard_prep_stats.json")["val_flights"], old("data/det_fov/wisard/wisard_prep_stats.json")["val_flights"]
print("wisard val 비행", "같음" if a == n else "다름"); assert a == n
PY
  touch $DET/.done
fi

# ── 2) 화각·고도 시험셋 ──
if [ ! -f $TEST/manifest.json ]; then
  [ -e $TEST ] && fail "$TEST 가 불완전하게 남아 있다 — 확인 후 지우고 다시"
  log "make_fov_testsets"
  $PY make_fov_testsets.py --det $DET --out $TEST || fail "시험셋 생성 실패"
fi

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
  local rc=$?
  log "학습 종료 $name (rc=$rc)"
  [ -f weights/$name.pt ] || fail "$name 가중치 복사 없음 (rc=$rc) — GPU 장애면 재부팅 뒤 autoheal 이 이어간다"
}

evaluate() {  # 이름 · 가중치 경로
  local name=$1 w=$2
  if [ -f metrics/eval_fov_$name.csv ]; then log "평가 이미 완료 $name"; return 0; fi
  nvidia-smi > /dev/null 2>&1 || fail "평가 전 GPU 없음"
  log "화각 평가 $name"
  $PY eval_fov.py --weights $w 2>&1 | grep -vE 'Scanning|━' | tee logs/eval_fov_$name.log | tail -30 || fail "평가 실패 $name"
}

# ── 3) 학습 1: yolo11s ──
train fov_11s_1280_all yolo11s.pt 16 --epochs 30
evaluate fov_11s_1280_all runs_person/fov_11s_1280_all/weights/best.pt
evaluate m1_11m_1280 runs_person/m1_11m_1280/weights/best.pt

# ── 4) 학습 2: yolo11m (시간 제한 08시간) ──
train fov_11m_1280_all yolo11m.pt 8 --time 08
evaluate fov_11m_1280_all runs_person/fov_11m_1280_all/weights/best.pt

log "전체 완료"
touch logs/train_fov.done
