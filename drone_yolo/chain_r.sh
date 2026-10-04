#!/bin/bash
# chain_r.sh — 10-04 R2: 공통 출발 미세조정 수프 (판정 기준은 obsidian _학습 큐 「10-04 R」 에 미리 적음)
#   출발점 soup_v9x2 → 10 에폭 · lr0 0.002 · warmup 1 · v9 목록 순서 3개 (s41 · s42 · s43)
#   → soup_ft3 (3개) · soup_ft3b (3개 + soup_v9x2) vs soup_v9x2 · BN 거리 (붕괴 원인)
#   끝나면 chain_r3.sh (R3 v10 LADD → R5) 가 있으면 이어서 실행 — 이 파일은 실행 중 고치지 않는다
cd /home/se/JupyterLAB/Capstone/drone_yolo
P=/home/se/miniconda3/envs/drone/bin/python; Q=logs/QUEUE.md; R=metrics/AUTO_RESULT_r.md
log(){ echo "[$(TZ=Asia/Seoul date '+%F %T') KST] $*" | tee -a $Q; }
ap(){ awk -F, 'NR==2{print $3}' "$1" 2>/dev/null; }
eval3(){ local m=$1 t Y
  for t in test_obl test_v2 test_kr; do
    case $t in test_obl) Y="--data data/test_obl --mode single --tag test_obl";; test_v2) Y="";; test_kr) Y="--data data/test_kr --tag test_kr";; esac
    [ -f metrics/${t}_$m.csv ] || $P eval_test_v2.py --weights runs_person/$m/weights/best.pt $Y >> logs/r_${m}_$t.log 2>&1
  done
  log "   $m AP50 obl $(ap metrics/test_obl_$m.csv) · v2 $(ap metrics/test_v2_$m.csv) · kr $(ap metrics/test_kr_$m.csv)"
  echo "| $m | $(ap metrics/test_obl_$m.csv) | $(ap metrics/test_v2_$m.csv) | $(ap metrics/test_kr_$m.csv) |" >> $R; }
cmp3(){ for t in test_obl test_v2 test_kr; do $P compare_ci.py --tag $t $1 | tail -1 | tee -a $Q $R; done; }

log "chain_r 시작 — R2 공통 출발 미세조정 (soup_v9x2 → ft_s41 · s42 · s43)"
{ echo "# chain_r 자동 결과 (10-04~)"; echo; echo "| 모델 | test_obl | test_v2 | test_kr |"; echo "|---|---|---|---|"; } > $R
RUNS=""
for s in 41 42 43; do
  name=ft_s$s
  $P chain_util.py mklist $name $s train_v7.txt nomad_v9.txt | tee -a $Q
  A="--stage 1 --hyp configs/hyp/v6_oblique.yaml --weights runs_person/soup_v9x2/weights/best.pt --data configs/data_$name.yaml --name $name --epochs 10 --lr0 0.002 --warmup-epochs 1"
  if $P train_person.py $A --smoke >> logs/r_${name}_smoke.log 2>&1; then
    log "   $name 연기 통과 → 10 에폭"; $P train_person.py $A >> logs/r_$name.log 2>&1
    [ -f runs_person/$name/weights/best.pt ] && { eval3 $name; RUNS="$RUNS $name"; } || log "   ❌ $name 학습 실패"
  else log "   ❌ $name 연기 실행 실패"; fi
done
if [ $(echo $RUNS | wc -w) -ge 2 ]; then
  $P soup.py soup_ft3 $RUNS >> $Q 2>&1 && eval3 soup_ft3
  $P soup.py soup_ft3b $RUNS soup_v9x2 >> $Q 2>&1 && eval3 soup_ft3b
  for m in soup_ft3 soup_ft3b; do echo "**soup_v9x2 → $m**" >> $R; cmp3 soup_v9x2:$m; done
  $P analyze_models.py soup_v9x2 soup_ft3 soup_ft3b --out=analyze_r2.json > logs/r_analyze_r2.log 2>&1 && $P - <<'PY' | tee -a $Q $R
import json
d = json.load(open("metrics/analyze_r2.json"))
for k, v in d.items():
    if k.endswith("test_obl"):
        s = v["층"]; print(f"   {k}: 가림 {s['가림:예'][0]} · 누움 {s['자세:Lying'][0]} · 앉음 {s['자세:Sitting'][0]} · 너비비 {v['너비비_25/50/75'][1]}")
PY
  set -- $RUNS
  $P bn_dist.py v7:v7_r2 v7:v7_r3 v9_s31:v9_s32 v9_s33:v9_s34 y26_s31:y26_s32 $1:$2 ${3:+$1:$3} ${3:+$2:$3} soup_v9x2:$1 2>&1 | tee -a $Q $R
else
  log "   ❌ 미세조정이 2개 미만 — 수프 건너뜀"
fi
log "chain_r (R2) 완료 → $R"
if [ -f chain_r3.sh ]; then log "   → chain_r3.sh 이어서"; bash chain_r3.sh; else log "   chain_r3.sh 없음 — R3 대기"; fi
