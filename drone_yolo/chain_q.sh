#!/bin/bash
# chain_q.sh — 10-03: Q1 추적 고치기 (분류기가 실제로 쓰이는 파이프라인) → Q2 YOLO26m 탐지기 (v9 데이터 · 반복 2회)
#   계획 · 판정 기준 (돌리기 전): obsidian 「11 서버 학습 계획/_학습 큐」 10-03 Q 항목
#   Q1  soup_v9x2 + 자세 box · V0 기본 / V1 추적 문턱 0.15 / V2 V1+ID 바뀜 끊기 / V3 V2+ReID → state_trackfix.py (A 에서 고르고 B 에서 판정)
#   Q2  y26_s31 · y26_s32 = v9_s31 · v9_s32 와 같은 학습 목록 · 같은 hyp · 모델만 yolo26m (한 번에 하나만 바꾼다)
#       → 3 평가셋 · 짝 비교 (v9_s3x:y26_s3x) · 수프 · 수프 비교 (soup_v9x2:soup_y26x2) · 층별 (누움 · 가림 · 너비비)
#       → 파이프라인 (Q1 에서 고른 추적 설정) · TensorRT 속도 (NFR-V01)
cd /home/se/JupyterLAB/Capstone/drone_yolo
P=/home/se/miniconda3/envs/drone/bin/python; S=/home/se/venvs/state/bin/python
Q=logs/QUEUE.md; R=metrics/AUTO_RESULT_q.md
log(){ echo "[$(TZ=Asia/Seoul date '+%F %T') KST] $*" | tee -a $Q; }
ap(){ awk -F, 'NR==2{print $3}' "$1" 2>/dev/null; }
eval3(){ local m=$1 t Y
  for t in test_obl test_v2 test_kr; do
    case $t in test_obl) Y="--data data/test_obl --mode single --tag test_obl";; test_v2) Y="";; test_kr) Y="--data data/test_kr --tag test_kr";; esac
    [ -f metrics/${t}_$m.csv ] || $P eval_test_v2.py --weights runs_person/$m/weights/best.pt $Y >> logs/q_${m}_$t.log 2>&1
  done
  log "   $m AP50 obl $(ap metrics/test_obl_$m.csv) · v2 $(ap metrics/test_v2_$m.csv) · kr $(ap metrics/test_kr_$m.csv)"
  echo "| $m | $(ap metrics/test_obl_$m.csv) | $(ap metrics/test_v2_$m.csv) | $(ap metrics/test_kr_$m.csv) |" >> $R; }
W9=runs_person/soup_v9x2/weights/best.pt
log "chain_q 시작 — Q1 추적 고치기 → Q2 YOLO26m ×2"
echo -e "# chain_q 자동 결과 (10-03~)\n\n| 모델 | test_obl | test_v2 | test_kr |\n|---|---|---|---|" > $R

# ── Q1 추적 고치기 ──────────────────────────────────────────────
run_q1(){ local tag=$1 trk=$2 extra=$3
  [ -f runs_state/samples_$tag.json ] || $S state_pipeline.py --weights=$W9 --posture=box --tracker=$trk $extra --tag=$tag --no-render --dump > logs/q_state_$tag.log 2>&1 \
    && log "   Q1 $tag 완료 ($trk $extra)" || log "   ❌ Q1 $tag 실패 (logs/q_state_$tag.log)"; }
run_q1 q1_v0 botsort.yaml ""
run_q1 q1_v1 configs/trackers/botsort_t015.yaml ""
run_q1 q1_v2 configs/trackers/botsort_t015.yaml --reset-jump
run_q1 q1_v3 configs/trackers/botsort_t015_reid.yaml --reset-jump
$S state_trackfix.py > logs/q_trackfix.log 2>&1 || log "   ❌ Q1 판정 실패"
log "   Q1 $(grep '→ A' logs/q_trackfix.log) · $(grep -o '→ 채택\|→ 기각' logs/q_trackfix.log | tail -1)"
{ echo; echo "## Q1 추적 고치기"; echo; grep -E "→|B 판정" logs/q_trackfix.log; echo; } >> $R
read TRK RST RULE < <($S -c "
import json; d=json.load(open('metrics/state_trackfix.json')); c=d['고른 것']
print((c['tracker'] if d['채택'] else 'botsort.yaml'), ('--reset-jump' if d['채택'] and c['reset'] else '-'), (c['규칙'] if d['채택'] else 'agg'))" 2>/dev/null || echo "botsort.yaml - agg")
[ "$RST" = "-" ] && RST=""

# ── Q2 YOLO26m ×2 ───────────────────────────────────────────────
RUNS=""
for s in 31 32; do
  name=y26_s$s
  A="--stage 1 --hyp configs/hyp/v6_oblique.yaml --weights yolo26m.pt --data configs/data_v9_s$s.yaml --name $name --epochs 30"
  if $P train_person.py $A --smoke >> logs/q_${name}_smoke.log 2>&1; then
    log "   $name 연기 통과 → 30 에폭"; $P train_person.py $A >> logs/q_$name.log 2>&1
    if [ -f runs_person/$name/weights/best.pt ]; then
      eval3 $name; RUNS="$RUNS $name"
      for t in test_obl test_v2 test_kr; do $P compare_ci.py --tag $t v9_s$s:$name 2>/dev/null | tail -1 | tee -a $Q >> $R; done
    else log "   ❌ $name 학습 실패"; fi
  else log "   ❌ $name 연기 실행 실패"; fi
done
BEST=""
if [ $(echo $RUNS | wc -w) = 2 ]; then
  $P soup.py soup_y26x2 $RUNS >> $Q 2>&1 && eval3 soup_y26x2
  for t in test_obl test_v2 test_kr; do $P compare_ci.py --tag $t soup_v9x2:soup_y26x2 2>/dev/null | tail -1 | tee -a $Q >> $R; done
  # 수프 무너짐 확인 (09-27 규칙): test_kr 가 두 단일보다 3.4 %p 넘게 낮으면 무너진 것 → 미리 정한 대체 = y26_s31
  BEST=$($P -c "
import csv
ap=lambda m: float(list(csv.reader(open(f'metrics/test_kr_{m}.csv')))[1][2])
print('soup_y26x2' if ap('soup_y26x2') >= min(ap('y26_s31'), ap('y26_s32')) - 0.034 else 'y26_s31')" 2>/dev/null || echo y26_s31)
  log "   파이프라인 · 속도에 쓸 YOLO26: $BEST"
elif [ -n "$RUNS" ]; then BEST=$(echo $RUNS | awk '{print $1}'); fi
if [ -n "$BEST" ]; then
  $P analyze_models.py soup_v9x2 $RUNS $([ "$BEST" = soup_y26x2 ] && echo soup_y26x2) --out=analyze_y26.json > logs/q_analyze_y26.log 2>&1 && $P - <<'PY' | tee -a $Q >> $R
import json
d = json.load(open("metrics/analyze_y26.json"))
for k, v in d.items():
    if k.endswith("test_obl"):
        s = v["층"]; print(f"   {k}: 가림 {s['가림:예'][0]} · 누움 {s['자세:Lying'][0]} · 앉음 {s['자세:Sitting'][0]} · 너비비 {v['너비비_25/50/75'][1]}")
PY
  $S state_pipeline.py --weights=runs_person/$BEST/weights/best.pt --posture=box --tracker=$TRK $RST --tag=y26_q1 --no-render --dump > logs/q_state_y26.log 2>&1 \
    && log "   파이프라인 $BEST ($TRK $RST · 규칙 $RULE): $($S state_trackfix.py eval y26_q1 $RULE 2>/dev/null | tail -1)" || log "   ❌ 파이프라인 $BEST 실패"
  $P bench_trt.py --weights runs_person/$BEST/weights/best.pt > logs/q_bench_$BEST.log 2>&1 && log "   NFR-V01 $BEST: $(tail -2 logs/q_bench_$BEST.log | tr '\n' ' ')" || log "   ❌ TensorRT 측정 실패 ($BEST)"
fi
log "chain_q 완료 → $R"
