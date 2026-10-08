#!/bin/bash
# chain_sd.sh — 10-08 계획 S1 · S2 · S3 → D0 → D1 → D2 (판정 기준은 _학습 큐 「10-08 계획」 · 돌리기 전에 적음)
#   S   state_worker_check.py --all (worker 진짜 추적 · soup_v9x2 · 비스듬 23편) → state_s123.py (S1 agg · S2 A4' · S3 관찰)
#   D0  v10 목록 95 % 무작위 뽑기 ×2 (chain_util mklistf · 시드 61 · 62) — 두 목록 집합이 달라야 함 · 연기 실행
#   D1  v10_f61 · v10_f62 (30 에폭 · v10_s31 과 같은 설정) → 세 평가셋 · 진짜 반복 폭 (f61 : f62) · 각각 soup_v7r2 대비 judge_pairs
#   D2  BN 거리 (bn_dist) 평균 mean_z < 0.25 일 때만 soup_v10f2 → 붕괴 확인 (kr 이 단일 평균보다 3.4 %p 넘게 낮으면 FAIL) → soup_v7r2 대비
#   교체는 사람이 확인 (10-24 동결) · 통과 없으면 soup_v7r2 유지
cd /home/se/JupyterLAB/Capstone/drone_yolo
P=/home/se/miniconda3/envs/drone/bin/python; Q=logs/QUEUE.md; R=metrics/AUTO_RESULT_d.md; C=logs/d_cmp.txt
POSTURE=/tmp/claude-1001/-home-se-JupyterLAB/027ea9bd-abdf-476f-ad0d-25f3e2ea7ffc/scratchpad/stage/models/posture.json
log(){ echo "[$(TZ=Asia/Seoul date '+%F %T') KST] $*" | tee -a $Q; }
ap(){ awk -F, 'NR==2{print $3}' "$1" 2>/dev/null; }
eval3(){ local m=$1 t Y
  for t in test_obl test_v2 test_kr test_ladd_h; do
    case $t in test_obl) Y="--data data/test_obl --mode single --tag test_obl";; test_v2) Y="";;
               test_kr) Y="--data data/test_kr --tag test_kr";; test_ladd_h) Y="--data data/test_ladd_h --tag test_ladd_h --boot 0";; esac
    [ -f metrics/${t}_$m.csv ] || $P eval_test_v2.py --weights runs_person/$m/weights/best.pt $Y >> logs/d_${m}_$t.log 2>&1
  done
  log "   $m AP50 obl $(ap metrics/test_obl_$m.csv) · v2 $(ap metrics/test_v2_$m.csv) · kr $(ap metrics/test_kr_$m.csv) · (참고 LADD 뒤 $(ap metrics/test_ladd_h_$m.csv))"
  echo "| $m | $(ap metrics/test_obl_$m.csv) | $(ap metrics/test_v2_$m.csv) | $(ap metrics/test_kr_$m.csv) | 참고 LADD 뒤 $(ap metrics/test_ladd_h_$m.csv) |" >> $R; }
cmp3(){ for t in test_obl test_v2 test_kr; do $P compare_ci.py --tag $t $1 | tail -1 | tee -a $Q $R $C; done; }
train_run(){ local name=$1 w=$2 ep=$3; shift 3
  local A="--stage 1 --hyp configs/hyp/v6_oblique.yaml --weights $w --data configs/data_$name.yaml --name $name --epochs $ep $*"
  if $P train_person.py $A --smoke >> logs/d_${name}_smoke.log 2>&1; then
    log "   $name 연기 통과 → $ep 에폭"; $P train_person.py $A >> logs/d_$name.log 2>&1
    [ -f runs_person/$name/weights/best.pt ] && { eval3 $name; return 0; } || log "   ❌ $name 학습 실패"
  else log "   ❌ $name 연기 실행 실패"; fi
  return 1; }

log "chain_sd 시작 — S1 · S2 · S3 → D0 → D1 → D2"
: > $C
echo -e "\n# chain_sd (10-08) — 상태 다시 재기 · v10 진짜 반복" >> $R

# ── S ──
if [ -f "$POSTURE" ]; then
  $P state_worker_check.py --weights=runs_person/soup_v9x2/weights/best.pt --tag=s123 --posture=$POSTURE --all > logs/d_s123.log 2>&1 \
    && $P state_s123.py >> logs/d_s123.log 2>&1 && log "   S 끝 → $R" || log "   ❌ S 실패 (logs/d_s123.log)"
else log "   ❌ S 건너뜀 — posture.json 없음 ($POSTURE)"; fi

# ── D0 ──
for s in 61 62; do $P chain_util.py mklistf v10_f$s $s 0.95 train_v7.txt nomad_v9.txt ladd_v10.txt | tee -a $Q; done
DIFF=$(comm -3 configs/lists/train_v10_f61.txt configs/lists/train_v10_f62.txt | wc -l)
log "   D0 두 목록 차이 $DIFF 줄"
if [ "$DIFF" -eq 0 ]; then log "   ❌ D0 FAIL — 목록이 같다 → 멈춤"; exit 1; fi

# ── D1 ──
echo -e "\n## D1 v10 진짜 반복 (95 % 뽑기)" >> $R
D1=""
for s in 61 62; do train_run v10_f$s yolo11m.pt 30 && D1="$D1 v10_f$s"; done
[ -f metrics/test_obl_soup_v7r2.csv ] || eval3 soup_v7r2
if [ $(echo $D1 | wc -w) = 2 ]; then
  echo "- 진짜 반복 폭 (f61 → f62):" >> $R; cmp3 v10_f61:v10_f62
  for m in $D1; do cmp3 soup_v7r2:$m; log "   $m vs soup_v7r2: $($P judge_pairs.py $C soup_v7r2:$m | tail -1)"; done
  # ── D2 ──
  $P bn_dist.py v10_f61:v10_f62 >> $Q 2>&1
  MZ=$($P -c "import json;print(json.load(open('metrics/bn_dist.json'))['v10_f61:v10_f62']['mean_z']['평균'])")
  log "   D2 BN 거리 mean_z $MZ (기준 < 0.25)"
  if $P -c "import sys;sys.exit(0 if float('$MZ') < 0.25 else 1)"; then
    $P soup.py soup_v10f2 v10_f61 v10_f62 >> $Q 2>&1 && eval3 soup_v10f2
    COL=$($P -c "
a=[float(open(f'metrics/test_kr_{m}.csv').read().splitlines()[1].split(',')[2]) for m in ('v10_f61','v10_f62','soup_v10f2')]
print('붕괴' if (a[0]+a[1])/2 - a[2] > 0.034 else '정상')")
    log "   soup_v10f2 붕괴 확인: $COL"
    if [ "$COL" = 정상 ]; then cmp3 soup_v7r2:soup_v10f2; log "   soup_v10f2 vs soup_v7r2: $($P judge_pairs.py $C soup_v7r2:soup_v10f2 | tail -1)"
    else log "   ❌ D2 FAIL — 수프 붕괴"; fi
  else log "   D2 건너뜀 — BN 거리 ≥ 0.25 (붕괴 위험)"; fi
else log "   ❌ D1 — 학습 2개가 안 끝남 · D2 건너뜀"; fi
log "chain_sd 완료 → $R — 교체 여부는 사람이 확인 (10-24 동결)"
