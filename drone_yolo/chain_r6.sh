#!/bin/bash
# chain_r6.sh — 10-06 R6: v10 공통 출발 미세조정 수프 (R3 가 FAIL 이라 chain_r3 의 R5 가지가 안 돌았다 → 사용자 지시로 따로)
#   출발점 soup_v10x2 (v10_s31 + v10_s32 · BN 통계 어긋나 kr 0.9127 로 무너짐) → 미세조정이 BN 통계를 다시 잡는다
#   10 에폭 · lr0 0.002 · warmup 1 · v10 목록 순서 3개 (s51 · s52 · s53) = R2 · R5 와 같은 설정
#   → soup_ft10x3 (3개 평균) · 판정: soup_v7r2 대비 같은 규칙 (judge_pairs.py) · 참고: soup_v9x2 · soup_ft3 대비 · BN 거리 · 너비비
#   판정 기준은 obsidian _학습 큐 「10-06 R6」 에 미리 적음 · 이 파일은 실행 중 고치지 않는다
cd /home/se/JupyterLAB/Capstone/drone_yolo
P=/home/se/miniconda3/envs/drone/bin/python; Q=logs/QUEUE.md; R=metrics/AUTO_RESULT_r6.md; C=logs/r6_cmp.txt
log(){ echo "[$(TZ=Asia/Seoul date '+%F %T') KST] $*" | tee -a $Q; }
ap(){ awk -F, 'NR==2{print $3}' "$1" 2>/dev/null; }
eval3(){ local m=$1 t Y
  for t in test_obl test_v2 test_kr test_ladd_h; do
    case $t in test_obl) Y="--data data/test_obl --mode single --tag test_obl";; test_v2) Y="";;
               test_kr) Y="--data data/test_kr --tag test_kr";; test_ladd_h) Y="--data data/test_ladd_h --tag test_ladd_h --boot 0";; esac
    [ -f metrics/${t}_$m.csv ] || $P eval_test_v2.py --weights runs_person/$m/weights/best.pt $Y >> logs/r6_${m}_$t.log 2>&1
  done
  log "   $m AP50 obl $(ap metrics/test_obl_$m.csv) · v2 $(ap metrics/test_v2_$m.csv) · kr $(ap metrics/test_kr_$m.csv) · (참고 LADD 뒤 $(ap metrics/test_ladd_h_$m.csv))"
  echo "| $m | $(ap metrics/test_obl_$m.csv) | $(ap metrics/test_v2_$m.csv) | $(ap metrics/test_kr_$m.csv) | $(ap metrics/test_ladd_h_$m.csv) |" >> $R; }
cmp3(){ for t in test_obl test_v2 test_kr; do $P compare_ci.py --tag $t $1 | tail -1 | tee -a $Q $R $C; done; }

log "chain_r6 시작 — R6 v10 공통 출발 미세조정 (soup_v10x2 → ft10_s51 · s52 · s53)"
: > $C
{ echo "# chain_r6 자동 결과 (10-06~) — v10 공통 출발 미세조정 수프"; echo
  echo "| 모델 | test_obl | test_v2 | test_kr | 참고 LADD 뒤 |"; echo "|---|---|---|---|---|"; } > $R
RUNS=""
for s in 51 52 53; do
  name=ft10_s$s
  $P chain_util.py mklist $name $s train_v7.txt nomad_v9.txt ladd_v10.txt | tee -a $Q
  A="--stage 1 --hyp configs/hyp/v6_oblique.yaml --weights runs_person/soup_v10x2/weights/best.pt --data configs/data_$name.yaml --name $name --epochs 10 --lr0 0.002 --warmup-epochs 1"
  if $P train_person.py $A --smoke >> logs/r6_${name}_smoke.log 2>&1; then
    log "   $name 연기 통과 → 10 에폭"; $P train_person.py $A >> logs/r6_$name.log 2>&1
    [ -f runs_person/$name/weights/best.pt ] && { eval3 $name; RUNS="$RUNS $name"; } || log "   ❌ $name 학습 실패"
  else log "   ❌ $name 연기 실행 실패"; fi
done
if [ $(echo $RUNS | wc -w) -ge 2 ]; then
  $P soup.py soup_ft10x3 $RUNS >> $Q 2>&1 && eval3 soup_ft10x3
  echo >> $R; echo "## 판정 — soup_v7r2 대비 (같은 규칙)" >> $R
  cmp3 soup_v7r2:soup_ft10x3
  log "   soup_ft10x3 vs soup_v7r2: $($P judge_pairs.py $C soup_v7r2:soup_ft10x3 | tee -a $R | tail -1)"
  echo >> $R; echo "## 참고 — soup_v9x2 · soup_ft3 대비 (판정에 안 씀)" >> $R
  cmp3 soup_v9x2:soup_ft10x3; cmp3 soup_ft3:soup_ft10x3
  $P analyze_models.py soup_v7r2 soup_v9x2 soup_ft10x3 --out=analyze_r6.json > logs/r6_analyze.log 2>&1 && $P - <<'PY' | tee -a $Q $R
import json
d = json.load(open("metrics/analyze_r6.json"))
for k, v in d.items():
    if k.endswith("test_obl"):
        s = v["층"]; print(f"   {k}: 가림 {s['가림:예'][0]} · 누움 {s['자세:Lying'][0]} · 앉음 {s['자세:Sitting'][0]} · 너비비 {v['너비비_25/50/75'][1]}")
PY
  set -- $RUNS
  $P bn_dist.py v10_s31:v10_s32 $1:$2 ${3:+$1:$3} ${3:+$2:$3} soup_v10x2:$1 2>&1 | tee -a $Q $R
else
  log "   ❌ 미세조정이 2개 미만 — 수프 건너뜀"
fi
log "chain_r6 (R6) 완료 → $R — 교체 여부는 사람이 확인 (10-24 동결)"
