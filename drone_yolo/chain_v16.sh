#!/bin/bash
# chain_v16.sh — 09-29~: v9 (v7 + NOMAD 가림·누움·숨음 부분집합 2,383) ×2 → 수프 · 층별(가림·누움·앉음) 분석
#   에폭 = chain_v15 결과 따라 — soup_e60x2 가 soup_v7r2 보다 고르기 통과면 60, 아니면 30 (바꾸는 것은 데이터 하나)
#   판정 (돌리기 전 · _학습 큐 「다음 계획」): 수프 대 수프 · 반복 폭보다 큼 + 세 평가셋 악화 없음 + 너비비 같음
#                                         + 층별 목표 test_obl 가림 0.24 · 누움 0.36 · 앉음 0.52 에서 오르는가
cd /home/se/JupyterLAB/Capstone/drone_yolo
P=/home/se/miniconda3/envs/drone/bin/python; Q=logs/QUEUE.md; R=metrics/AUTO_RESULT_v12.md
WAIT_PID=${WAIT_PID:?chain_v15 PID}
log(){ echo "[$(TZ=Asia/Seoul date '+%F %T') KST] $*" | tee -a $Q; }
ap(){ awk -F, 'NR==2{print $3}' "$1" 2>/dev/null; }
eval3(){ local m=$1 t Y
  for t in test_obl test_v2 test_kr; do
    case $t in test_obl) Y="--data data/test_obl --mode single --tag test_obl";; test_v2) Y="";; test_kr) Y="--data data/test_kr --tag test_kr";; esac
    [ -f metrics/${t}_$m.csv ] || $P eval_test_v2.py --weights runs_person/$m/weights/best.pt $Y >> logs/q_${m}_$t.log 2>&1
  done
  log "   $m AP50 obl $(ap metrics/test_obl_$m.csv) · v2 $(ap metrics/test_v2_$m.csv) · kr $(ap metrics/test_kr_$m.csv)"
  echo "| $m | $(ap metrics/test_obl_$m.csv) | $(ap metrics/test_v2_$m.csv) | $(ap metrics/test_kr_$m.csv) |" >> $R; }
while kill -0 $WAIT_PID 2>/dev/null; do sleep 120; done
log "chain_v16 시작 — v9 (NOMAD 가림·누움 부분집합)"
E=30; BASE_SOUP=soup_v7r2
if [ -f runs_person/soup_e60x2/weights/best.pt ]; then
  W=$($P chain_util.py pick soup_e60x2 soup_v7r2); log "   60 에폭 수프 고르기: $W"
  [ "${W%% *}" = cand ] && { E=60; BASE_SOUP=soup_e60x2; }
fi
log "   에폭 $E · 기준 수프 $BASE_SOUP"
[ -s configs/lists/nomad_v9.txt ] || { log "   ❌ nomad_v9.txt 없음"; exit 1; }
RUNS=""
for s in 31 32; do
  name=v9_s$s; [ $E = 60 ] && name=v9e60_s$s
  $P chain_util.py mklist $name $s train_v7.txt nomad_v9.txt | tee -a $Q
  A="--stage 1 --hyp configs/hyp/v6_oblique.yaml --weights yolo11m.pt --data configs/data_$name.yaml --name $name --epochs $E"
  if $P train_person.py $A --smoke >> logs/q_${name}_smoke.log 2>&1; then
    log "   $name 연기 통과 → $E 에폭"; $P train_person.py $A >> logs/q_$name.log 2>&1
    [ -f runs_person/$name/weights/best.pt ] && { eval3 $name; RUNS="$RUNS $name"; } || log "   ❌ $name 학습 실패"
  else log "   ❌ $name 연기 실행 실패"; fi
done
if [ $(echo $RUNS | wc -w) = 2 ]; then
  $P soup.py soup_v9x2 $RUNS >> $Q 2>&1 && eval3 soup_v9x2
  for t in test_obl test_v2 test_kr; do $P compare_ci.py --tag $t $BASE_SOUP:soup_v9x2 | tail -1 | tee -a $Q; done
  $P analyze_models.py $BASE_SOUP soup_v9x2 --out=analyze_v9.json > logs/q_analyze_v9.log 2>&1 \
    && log "   층별 · 너비비 → metrics/analyze_v9.json" \
    && $P - <<'PY' | tee -a $Q
import json
d = json.load(open("metrics/analyze_v9.json"))
for k, v in d.items():
    if k.endswith("test_obl"):
        s = v["층"]; print(f"   {k}: 가림 {s['가림:예'][0]} · 누움 {s['자세:Lying'][0]} · 앉음 {s['자세:Sitting'][0]} · 너비비 {v['너비비_25/50/75'][1]}")
PY
fi
log "chain_v16 완료"
