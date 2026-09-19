#!/bin/bash
# train_l2.sh — L2: L1 데이터 + AI-Hub 182 (2026-09-18)
#
# 왜 별도 스크립트인가
#   train_l.sh 가 돌고 있는 동안 그 파일을 고치면 bash 가 남은 부분을 엉뚱한 위치에서 읽는다.
#   그래서 L2 는 따로 두고 chain_l2.sh 가 logs/train_l.done 을 기다렸다가 부른다.
#
# 무엇이 달라지나
#   학습 데이터 +68 % (30,056 → 약 50,600장). val 은 L1 과 똑같이 둬서 best.pt 기준을 맞춘다.
#   AI-Hub 는 **장소 단위**로 나눴다 — train 저수지2 · 평지(흙)3 · 산악6 / val 저수지1 / 제외 산악5(화각 시험 원본)
#
# 판정
#   ① 가림 지형 시험셋(det_fov_test_budget) — L1 · fov_11m · fov_11s 와 직접 비교
#   ② AI-Hub 장소 분리 val — NFR-V03 의 "장소 분리" 를 만족하는 유일한 수치 (다만 쉬운 데이터다)
set -u
cd /home/se/JupyterLAB/Capstone/drone_yolo
PY=/home/se/miniconda3/envs/drone/bin/python
NAME=l2_11l_1280_aihub
log() { echo "=== $* [$(date -u '+%F %T') UTC] ==="; }
fail() { log "실패: $*"; exit 1; }

pgrep -af "train_person\.py" | awk '$2 ~ /python/' | grep . && fail "다른 train_person.py 실행 중"
[ -d data/det_aihub/images/train ] || fail "AI-Hub 크롭 없음 — aihub_prep.py 먼저"

if [ -f weights/$NAME.pt ]; then
  log "$NAME 학습 이미 완료"
elif [ -f runs_person/$NAME/weights/last.pt ]; then
  # 09-19: 20 h 상한 때문에 ultralytics 가 목표 에폭을 30→23 으로 줄여 놨다.
  # 체크포인트의 train_args(epochs 30 · time None)를 직접 고쳐 뒀으므로 --time 을 넘기지 않는다
  # (ultralytics resume 은 imgsz·batch·device·close_mosaic 외의 덮어쓰기를 버린다)
  log "이어 학습 $NAME"; $PY train_person.py --stage 1 --resume --name $NAME
else
  log "새 학습 $NAME"
  $PY train_person.py --stage 1 --data configs/data_l2.yaml --weights yolo11l.pt --imgsz 1280 --batch 6 \
    --patience 100 --close-mosaic 5 --optimizer SGD --lr0 0.01 --scale 0.3 --name $NAME --epochs 30 --time 20
fi
[ -f weights/$NAME.pt ] || fail "$NAME 가중치 복사 없음 — 재부팅 뒤 autoheal 이 이어간다"

W=runs_person/$NAME/weights/best.pt
if [ ! -f metrics/eval_fovbudget_$NAME.csv ]; then
  log "① 가림 지형 평가"
  $PY eval_fov.py --testsets data/det_fov_test_budget --tag budget --weights $W 2>&1 \
    | grep -vE 'Scanning|━' | tee logs/eval_fovbudget_$NAME.log | tail -30 || fail "가림 지형 평가 실패"
fi
if [ ! -f metrics/aihub_placeval_$NAME.txt ]; then
  log "② AI-Hub 장소 분리 val 평가"
  MPLBACKEND=Agg $PY -c "
from ultralytics import YOLO
r = YOLO('$W').val(data='configs/data_aihub_val.yaml', imgsz=1280, batch=8, plots=False, verbose=False)
open('metrics/aihub_placeval_$NAME.txt','w').write(
  f'model=$NAME\nAP50={r.box.map50:.4f}\nAP50-95={r.box.map:.4f}\nP={r.box.mp:.4f}\nR={r.box.mr:.4f}\n')
print(open('metrics/aihub_placeval_$NAME.txt').read())" 2>&1 | grep -vE 'Scanning|━' | tail -8 || fail "AI-Hub 평가 실패"
fi

log "L2 완료"
touch logs/train_l2.done
