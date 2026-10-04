#!/bin/bash
# chain_r3.sh — 10-04 R3 → R5 (chain_r.sh 가 R2 를 끝내고 부른다 · 판정 기준은 _학습 큐 「10-04 R」)
#   R3  v10 = v9 + LADD (make_v10_data.py) · s31 · s32 · 30 에폭 → v9 짝 비교 · soup_v10x2 vs soup_v9x2 → judge_pairs.py
#   R5  R3 PASS → soup_v10x2 에서 공통 출발 미세조정 ×3 (R2 방식) → 수프 · 아니면 학습 없이 기존 후보만
#       후보 (soup_ft3 · soup_ft3b · soup_v10x2 · soup_ft10x3) 를 soup_v7r2 와 같은 규칙으로 → 통과한 것 기록 (교체는 사람이 확인)
cd /home/se/JupyterLAB/Capstone/drone_yolo
P=/home/se/miniconda3/envs/drone/bin/python; Q=logs/QUEUE.md; R=metrics/AUTO_RESULT_r.md; C=logs/r_cmp.txt
log(){ echo "[$(TZ=Asia/Seoul date '+%F %T') KST] $*" | tee -a $Q; }
ap(){ awk -F, 'NR==2{print $3}' "$1" 2>/dev/null; }
eval3(){ local m=$1 t Y
  for t in test_obl test_v2 test_kr test_ladd_h; do
    case $t in test_obl) Y="--data data/test_obl --mode single --tag test_obl";; test_v2) Y="";;
               test_kr) Y="--data data/test_kr --tag test_kr";; test_ladd_h) Y="--data data/test_ladd_h --tag test_ladd_h --boot 0";; esac
    [ -f metrics/${t}_$m.csv ] || $P eval_test_v2.py --weights runs_person/$m/weights/best.pt $Y >> logs/r_${m}_$t.log 2>&1
  done
  log "   $m AP50 obl $(ap metrics/test_obl_$m.csv) · v2 $(ap metrics/test_v2_$m.csv) · kr $(ap metrics/test_kr_$m.csv) · (참고 LADD 뒤 $(ap metrics/test_ladd_h_$m.csv))"
  echo "| $m | $(ap metrics/test_obl_$m.csv) | $(ap metrics/test_v2_$m.csv) | $(ap metrics/test_kr_$m.csv) | 참고 LADD 뒤 $(ap metrics/test_ladd_h_$m.csv) |" >> $R; }
cmp3(){ for t in test_obl test_v2 test_kr; do $P compare_ci.py --tag $t $1 | tail -1 | tee -a $Q $R $C; done; }
train_run(){ # 이름 · 시작 가중치 · 에폭 · 추가 인자
  local name=$1 w=$2 ep=$3; shift 3
  local A="--stage 1 --hyp configs/hyp/v6_oblique.yaml --weights $w --data configs/data_$name.yaml --name $name --epochs $ep $*"
  if $P train_person.py $A --smoke >> logs/r_${name}_smoke.log 2>&1; then
    log "   $name 연기 통과 → $ep 에폭"; $P train_person.py $A >> logs/r_$name.log 2>&1
    [ -f runs_person/$name/weights/best.pt ] && { eval3 $name; return 0; } || log "   ❌ $name 학습 실패"
  else log "   ❌ $name 연기 실행 실패"; fi
  return 1; }

log "chain_r3 시작 — R3 v10 (v9 + LADD)"
: > $C
[ -s configs/lists/ladd_v10.txt ] || $P make_v10_data.py >> $Q 2>&1
echo >> $R; echo "## R3 v10 = v9 + LADD" >> $R
R3=""
for s in 31 32; do
  name=v10_s$s
  $P chain_util.py mklist $name $s train_v7.txt nomad_v9.txt ladd_v10.txt | tee -a $Q
  train_run $name yolo11m.pt 30 && R3="$R3 $name"
done
for m in v9_s31 v9_s32 soup_v9x2; do eval3 $m > /dev/null; done        # 참고 LADD 뒤 (없으면 만든다)
R3OK=FAIL
if [ $(echo $R3 | wc -w) = 2 ]; then
  cmp3 v9_s31:v10_s31; cmp3 v9_s32:v10_s32
  $P soup.py soup_v10x2 $R3 >> $Q 2>&1 && eval3 soup_v10x2 && cmp3 soup_v9x2:soup_v10x2
  R3OK=$($P judge_pairs.py $C v9_s31:v10_s31 v9_s32:v10_s32 soup_v9x2:soup_v10x2 | tee -a $Q $R | tail -1)
  $P analyze_models.py soup_v9x2 soup_v10x2 --out=analyze_r3.json > logs/r_analyze_r3.log 2>&1
fi
log "   R3 판정: $R3OK"

echo >> $R; echo "## R5 중간발표 후보 (soup_v7r2 대비 같은 규칙)" >> $R
CANDS=""
for m in soup_ft3 soup_ft3b; do [ -f runs_person/$m/weights/best.pt ] && CANDS="$CANDS $m"; done
if [ "$R3OK" = PASS ]; then
  CANDS="$CANDS soup_v10x2"; R5=""
  for s in 51 52 53; do
    name=ft10_s$s
    $P chain_util.py mklist $name $s train_v7.txt nomad_v9.txt ladd_v10.txt | tee -a $Q
    train_run $name runs_person/soup_v10x2/weights/best.pt 10 --lr0 0.002 --warmup-epochs 1 && R5="$R5 $name"
  done
  if [ $(echo $R5 | wc -w) -ge 2 ]; then
    $P soup.py soup_ft10x3 $R5 >> $Q 2>&1 && eval3 soup_ft10x3 && CANDS="$CANDS soup_ft10x3"
  fi
fi
for m in $CANDS; do
  cmp3 soup_v7r2:$m
  log "   $m vs soup_v7r2: $($P judge_pairs.py $C soup_v7r2:$m | tail -1)"
done
log "chain_r3 (R3 · R5) 완료 → $R — 교체 여부는 사람이 확인 (10-24 동결)"
