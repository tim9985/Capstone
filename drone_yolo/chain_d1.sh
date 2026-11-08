#!/bin/bash
# chain_d1.sh — 11-08 D1 실기체 데이터로 처음부터 다시 학습 × 반복 2 · 판정 = _학습 큐 「11-08 D1」
#   준비  configs/d1_flights.json (비행 → 라벨 폴더 · 장소 A · B · C) — 사람이 고친 라벨 (label_draft.py 초안 → 수정)
#   ① make_d1_data.py → 학습 목록 (v12_f91 · f92 + 실기체 창 ≈ 20 %) · 평가셋 data/test_real_c (C 장소)
#   ② d1_f91 · d1_f92 — COCO YOLO11m 부터 v12 와 같은 설정 30 에폭 → swa_ckpt
#   ③ 평가: 세 공개 평가셋 · 도로 헛탐지 · 누움 묶음 · **실기체 C** — v12 반복 3개도 실기체 C 를 잼 → 평균 비교
#   연기 실행만: SMOKE=1 TAG=d1smoke TEST=test_real_smoke FLIGHTS=<json> bash chain_d1.sh  (학습 64장 1에폭 · 평가 v12 하나)
cd /home/se/JupyterLAB/Capstone/drone_yolo
P=/home/se/miniconda3/envs/drone/bin/python; Q=logs/QUEUE.md; R=metrics/AUTO_RESULT_d1.md; IGN=data/test_v2/ignore_v1.json
TAG=${TAG:-d1}; TEST=${TEST:-test_real_c}; FLIGHTS=${FLIGHTS:-configs/d1_flights.json}; SMOKE=${SMOKE:-0}
[ $SMOKE = 1 ] && { Q=logs/d1_smoke.md; R=/dev/null; }
log(){ echo "[$(TZ=Asia/Seoul date '+%F %T') KST] $*" | tee -a $Q $R; }
ap(){ awk -F, 'NR==2{print $3}' "$1" 2>/dev/null; }
real(){ local m=$1   # 실기체 C 평가 (1920 원본 · 4조각 · 가장자리 합치기)
  [ -f metrics/${TEST}_edge_$m.csv ] || $P eval_test_v2.py --weights runs_person/$m/weights/best.pt --data data/$TEST --tag $TEST --merge edge --op-conf 0.20 >> logs/d1_$m.log 2>&1
  log "   $m 실기체 C AP50 $(ap metrics/${TEST}_edge_$m.csv)"; }
ev(){ local m=$1
  $P eval_test_v2.py --weights runs_person/$m/weights/best.pt --data data/test_obl --mode single --tag test_obl >> logs/d1_$m.log 2>&1
  $P eval_test_v2.py --weights runs_person/$m/weights/best.pt --merge edge --ignore $IGN >> logs/d1_$m.log 2>&1
  $P eval_test_v2.py --weights runs_person/$m/weights/best.pt --data data/test_kr --tag test_kr --merge edge >> logs/d1_$m.log 2>&1
  $P eval_test_v2.py --weights runs_person/$m/weights/best.pt --data data/test_auair --tag test_auair --boot 0 >> logs/d1_$m.log 2>&1
  log "   $m AP50 obl $(ap metrics/test_obl_$m.csv) · v2 edge+무시 $(ap metrics/test_v2_edge_ign_$m.csv) · kr edge $(ap metrics/test_kr_edge_$m.csv) · 도로 오탐 $(awk -F, 'NR==2{print $6}' metrics/test_auair_$m.csv) 건/장"
  $P lying_bench.py $m 2>&1 | grep -v -i warn | tail -1 | tee -a $Q $R
  real $m; }
tr(){ local name=$1; shift
  A="--stage 1 --hyp configs/hyp/v6_oblique.yaml --weights yolo11m.pt --name $name --epochs 30 $*"
  if ! $P train_person.py $A --smoke >> logs/d1_${name}_smoke.log 2>&1; then log "   ❌ $name 연기 실행 실패 — logs/d1_${name}_smoke.log"; return 1; fi
  [ $SMOKE = 1 ] && { log "   $name 연기 통과 ($(grep -o '1에폭[^·]*' logs/d1_${name}_smoke.log | tail -1))"; return 0; }
  if $P train_person.py $A --set save_period=1 >> logs/d1_${name}_train.log 2>&1; then $P swa_ckpt.py $name >> $Q 2>&1 && ev ${name}_swa
  else log "   ❌ $name 학습 실패"; return 1; fi; }

log "chain_d1 시작 (TAG=$TAG · 평가 $TEST · 연기 $SMOKE)"
$P make_d1_data.py --flights $FLIGHTS --tag $TAG --test-dir $TEST --check logs/${TAG}_check.jpg 2>&1 | tail -1 | tee -a $Q $R || { log "❌ 데이터 실패"; exit 1; }
if [ $SMOKE = 1 ]; then
  tr ${TAG}_f91 --data configs/data_${TAG}_f91.yaml; real v12_f91_swa; log "chain_d1 연기 끝"; exit 0
fi
for b in v12_f91_swa v12_f92_swa v12_f93_swa; do real $b; done
log "D1-1 — ${TAG}_f91 (train_v12_f91 + 실기체)"; tr ${TAG}_f91 --data configs/data_${TAG}_f91.yaml
log "D1-2 — ${TAG}_f92 (train_v12_f92 + 같은 창)"; tr ${TAG}_f92 --data configs/data_${TAG}_f92.yaml
log "D1 평균 비교 (평균[v12 3회] → 평균[D1 2회])"
for t in ${TEST}_edge test_obl test_v2_edge_ign test_kr_edge; do
  $P compare_means.py --tag $t v12_f91_swa,v12_f92_swa,v12_f93_swa ${TAG}_f91_swa,${TAG}_f92_swa 2>&1 | tail -1 | tee -a $Q $R; done
log "chain_d1 완료 — 판정은 _학습 큐 「11-08 D1」 · 누움 · 빈 장면은 real_flight_eval.py 로 같은 녹화 다시"
