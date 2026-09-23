#!/bin/bash
# chain_v2.sh — 재검토 후 큐 (2026-09-23)
# 기준선: v3_place (장소 분리 · test_v2 AP50 0.2233). 모든 실험을 같은 장소 분리 데이터로 돌리고
# test_v2 AP50 으로 판정한다. 판정 임계값을 스크립트에 넣는다.
cd /home/se/JupyterLAB/Capstone/drone_yolo
P=/home/se/miniconda3/envs/drone/bin/python
Q=logs/QUEUE.md; R=metrics/AUTO_RESULT_v2.md
BASE_AP=0.2233
log(){ echo "[$(TZ=Asia/Seoul date '+%F %T') KST] $*" | tee -a $Q; }
done30(){ [ -f "runs_person/$1/results.csv" ] && [ "$(awk -F, 'END{print $1}' runs_person/$1/results.csv)" = "30" ]; }
COMMON="--stage 1 --data configs/data_v3_place_neg.yaml --weights yolo11m.pt --imgsz 1280 --epochs 30 --patience 100 --close-mosaic 0 --optimizer SGD --lr0 0.01 --scale 0.3 --translate 0.15"

judge(){  # $1 = run 이름
  W=runs_person/$1/weights/best.pt
  [ -s "$W" ] || { log "   $1 가중치 없음"; return; }
  $P eval_test_v2.py --weights $W >> logs/q_$1_testv2.log 2>&1 || { log "   $1 test_v2 평가 실패"; return; }
  AP=$(awk -F, 'NR==2{print $3}' metrics/test_v2_$1.csv)
  V=$($P -c "a=$AP;b=$BASE_AP;print(f'{(a-b)*100:+.1f} %p · ' + ('✅ 개선' if a>b+0.01 else ('➖ 동등' if a>=b-0.01 else '❌ 악화')))")
  log "   $1 test_v2 AP50 $AP (기준선 $BASE_AP 대비 $V)"
  echo "| $1 | $AP | $V |" >> $R
}

: > $R
echo "# 재검토 큐 결과 — test_v2 AP50 (기준선 v3_place $BASE_AP)" >> $R
echo "" >> $R; echo "| 실험 | AP50 | 판정 |" >> $R; echo "|---|---:|---|" >> $R
log "재검토 큐 시작 (chain_v2)"

# ① P2 헤드 — 11m-p2 + COCO 이식 · 앵커 4배라 batch 6
if ! done30 p2m_place; then
  log "① p2m_place 시작 — yolo11m-p2 (20.55 M) + COCO 이식"
  $P train_person.py $COMMON --model-yaml configs/models/yolo11m-p2.yaml --batch 6 --name p2m_place >> logs/q_p2m_place.log 2>&1
fi
judge p2m_place

# ② rect — 학습 모양을 추론 엔진 [736,1280] 에 맞춘다
if ! done30 rect_place; then
  log "② rect_place 시작 — rect=True"
  $P train_person.py $COMMON --rect --batch 8 --name rect_place >> logs/q_rect_place.log 2>&1
fi
judge rect_place

log "재검토 큐 완료 → $R"
