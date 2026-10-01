#!/bin/bash
# chain_v17.sh — 10-01: v9 (v7 + NOMAD 가림·누움 2,383) 반복 2회 더 → 두 번째 수프 · v9 채택 판정용
#   배경: soup_v9x2 (s31+s32) vs soup_v7r2 — test_obl +2.7 (반복 폭 2.8 안) · test_v2 −2.0 (구분 안 됨) · 누움 0.36 → 0.50 → 보류
#   판정 (돌리기 전): soup_v9x2b (s33+s34) 가 soup_v7r2 대비 같은 방향이면(test_obl ↑ · 누움 ↑ · test_v2 유의 악화 없음) v9 채택
cd /home/se/JupyterLAB/Capstone/drone_yolo
P=/home/se/miniconda3/envs/drone/bin/python; Q=logs/QUEUE.md; R=metrics/AUTO_RESULT_v12.md
log(){ echo "[$(TZ=Asia/Seoul date '+%F %T') KST] $*" | tee -a $Q; }
ap(){ awk -F, 'NR==2{print $3}' "$1" 2>/dev/null; }
eval3(){ local m=$1 t Y
  for t in test_obl test_v2 test_kr; do
    case $t in test_obl) Y="--data data/test_obl --mode single --tag test_obl";; test_v2) Y="";; test_kr) Y="--data data/test_kr --tag test_kr";; esac
    [ -f metrics/${t}_$m.csv ] || $P eval_test_v2.py --weights runs_person/$m/weights/best.pt $Y >> logs/q_${m}_$t.log 2>&1
  done
  log "   $m AP50 obl $(ap metrics/test_obl_$m.csv) · v2 $(ap metrics/test_v2_$m.csv) · kr $(ap metrics/test_kr_$m.csv)"
  echo "| $m | $(ap metrics/test_obl_$m.csv) | $(ap metrics/test_v2_$m.csv) | $(ap metrics/test_kr_$m.csv) |" >> $R; }
log "chain_v17 시작 — v9 반복 (s33 · s34)"
RUNS=""
for s in 33 34; do
  name=v9_s$s
  $P chain_util.py mklist $name $s train_v7.txt nomad_v9.txt | tee -a $Q
  A="--stage 1 --hyp configs/hyp/v6_oblique.yaml --weights yolo11m.pt --data configs/data_$name.yaml --name $name --epochs 30"
  if $P train_person.py $A --smoke >> logs/q_${name}_smoke.log 2>&1; then
    log "   $name 연기 통과 → 30 에폭"; $P train_person.py $A >> logs/q_$name.log 2>&1
    [ -f runs_person/$name/weights/best.pt ] && { eval3 $name; RUNS="$RUNS $name"; } || log "   ❌ $name 학습 실패"
  else log "   ❌ $name 연기 실행 실패"; fi
done
if [ $(echo $RUNS | wc -w) = 2 ]; then
  $P soup.py soup_v9x2b $RUNS >> $Q 2>&1 && eval3 soup_v9x2b
  for t in test_obl test_v2 test_kr; do $P compare_ci.py --tag $t soup_v7r2:soup_v9x2b | tail -1 | tee -a $Q; done
  $P analyze_models.py soup_v7r2 soup_v9x2 soup_v9x2b --out=analyze_v9b.json > logs/q_analyze_v9b.log 2>&1 && $P - <<'PY' | tee -a $Q
import json
d = json.load(open("metrics/analyze_v9b.json"))
for k, v in d.items():
    if k.endswith("test_obl"):
        s = v["층"]; print(f"   {k}: 가림 {s['가림:예'][0]} · 누움 {s['자세:Lying'][0]} · 앉음 {s['자세:Sitting'][0]} · 너비비 {v['너비비_25/50/75'][1]}")
PY
fi
log "chain_v17 완료"
