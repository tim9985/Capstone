#!/bin/bash
# chain_s.sh — 10-12 S 팔: 야생 지형의 작은 사람 더하기 (판정 기준은 _학습 큐 「10-11~12 V」 · 돌리기 전에 적음)
#   S0  v7 + small_v11 목록 95 % 무작위 뽑기 ×2 (시드 71 · 72) — 두 목록이 달라야 함
#   S1  v11s_f71 · v11s_f72 (v7 과 같은 설정 · 30 에폭) → test_obl · test_v2 · test_kr (+ _edge 합치기) · 참고 LADD 뒤
#   S2  진짜 반복 폭 (f71 : f72) · 각각 soup_v7r2 대비 (설명용)
#   S3  BN 거리 mean_z < 0.25 일 때만 soup_v11s → 붕괴 확인 → soup_v7r2 대비 judge_pairs (_edge 평가셋 · 반복 폭 obl 3.1 · v2 2.9 · kr 1.2)
#   S4  설명용 — 덤프 (장소별 · 크기별) · test_auair 사람 없는 장 오탐
#   교체는 사람이 확인 (10-24 동결)
cd /home/se/JupyterLAB/Capstone/drone_yolo
P=/home/se/miniconda3/envs/drone/bin/python; Q=logs/QUEUE.md; R=metrics/AUTO_RESULT_s.md; C=logs/s_cmp.txt
BAND="--band=test_obl:3.1,test_v2_edge:2.9,test_kr_edge:1.2"
log(){ echo "[$(TZ=Asia/Seoul date '+%F %T') KST] $*" | tee -a $Q; }
ap(){ awk -F, 'NR==2{print $3}' "$1" 2>/dev/null; }
eval4(){ local m=$1 t Y
  for t in test_obl test_v2 test_v2_edge test_kr test_kr_edge test_ladd_h; do
    case $t in test_obl) Y="--data data/test_obl --mode single --tag test_obl";; test_v2) Y="";; test_v2_edge) Y="--merge edge";;
               test_kr) Y="--data data/test_kr --tag test_kr";; test_kr_edge) Y="--data data/test_kr --tag test_kr --merge edge";;
               test_ladd_h) Y="--data data/test_ladd_h --tag test_ladd_h --boot 0";; esac
    [ -f metrics/${t}_$m.csv ] || $P eval_test_v2.py --weights runs_person/$m/weights/best.pt $Y >> logs/s_${m}_$t.log 2>&1
  done
  log "   $m AP50 obl $(ap metrics/test_obl_$m.csv) · v2 $(ap metrics/test_v2_$m.csv) → edge $(ap metrics/test_v2_edge_$m.csv) · kr $(ap metrics/test_kr_$m.csv) → edge $(ap metrics/test_kr_edge_$m.csv) · (참고 LADD 뒤 $(ap metrics/test_ladd_h_$m.csv))"
  echo "| $m | $(ap metrics/test_obl_$m.csv) | $(ap metrics/test_v2_$m.csv) → $(ap metrics/test_v2_edge_$m.csv) | $(ap metrics/test_kr_$m.csv) → $(ap metrics/test_kr_edge_$m.csv) | 참고 LADD 뒤 $(ap metrics/test_ladd_h_$m.csv) |" >> $R; }
cmp3(){ for t in test_obl test_v2_edge test_kr_edge; do $P compare_ci.py --tag $t $1 | tail -1 | tee -a $Q $R $C; done; }
train_run(){ local name=$1 w=$2 ep=$3; shift 3
  local A="--stage 1 --hyp configs/hyp/v6_oblique.yaml --weights $w --data configs/data_$name.yaml --name $name --epochs $ep $*"
  if $P train_person.py $A --smoke >> logs/s_${name}_smoke.log 2>&1; then
    log "   $name 연기 통과 → $ep 에폭"; $P train_person.py $A >> logs/s_$name.log 2>&1
    [ -f runs_person/$name/weights/best.pt ] && { eval4 $name; return 0; } || log "   ❌ $name 학습 실패"
  else log "   ❌ $name 연기 실행 실패"; fi
  return 1; }

log "chain_s 시작 — S0 → S1 → S2 → S3 → S4 (야생 지형 작은 사람)"
: > $C
echo -e "\n# chain_s (10-12) — S 야생 지형 작은 사람 더하기\n\n| 모델 | obl | v2 → edge | kr → edge | 참고 |\n|---|---|---|---|---|" >> $R
eval4 soup_v7r2

# ── S0 ──
for s in 71 72; do $P chain_util.py mklistf v11s_f$s $s 0.95 train_v7.txt small_v11.txt | tee -a $Q; done
DIFF=$(comm -3 configs/lists/train_v11s_f71.txt configs/lists/train_v11s_f72.txt | wc -l)
log "   S0 두 목록 차이 $DIFF 줄"
if [ "$DIFF" -eq 0 ]; then log "   ❌ S0 FAIL — 목록이 같다 → 멈춤"; exit 1; fi

# ── S1 ──
S1=""
for s in 71 72; do train_run v11s_f$s yolo11m.pt 30 && S1="$S1 v11s_f$s"; done

# ── S2 · S3 ──
if [ $(echo $S1 | wc -w) = 2 ]; then
  echo "- 진짜 반복 폭 (f71 → f72):" >> $R; cmp3 v11s_f71:v11s_f72
  for m in $S1; do cmp3 soup_v7r2:$m; log "   $m vs soup_v7r2 (설명용): $($P judge_pairs.py $BAND $C soup_v7r2:$m | tail -1)"; done
  $P bn_dist.py v11s_f71:v11s_f72 >> $Q 2>&1
  MZ=$($P -c "import json;print(json.load(open('metrics/bn_dist.json'))['v11s_f71:v11s_f72']['mean_z']['평균'])")
  log "   S3 BN 거리 mean_z $MZ (기준 < 0.25)"
  if $P -c "import sys;sys.exit(0 if float('$MZ') < 0.25 else 1)"; then
    $P soup.py soup_v11s v11s_f71 v11s_f72 >> $Q 2>&1 && eval4 soup_v11s
    COL=$($P -c "
a=[float(open(f'metrics/test_kr_{m}.csv').read().splitlines()[1].split(',')[2]) for m in ('v11s_f71','v11s_f72','soup_v11s')]
print('붕괴' if (a[0]+a[1])/2 - a[2] > 0.034 else '정상')")
    log "   soup_v11s 붕괴 확인: $COL"
    if [ "$COL" = 정상 ]; then cmp3 soup_v7r2:soup_v11s; log "   ★ soup_v11s vs soup_v7r2 (판정): $($P judge_pairs.py $BAND $C soup_v7r2:soup_v11s | tail -1)"
    else log "   ❌ S3 FAIL — 수프 붕괴 → 판정은 두 반복 각각"; fi
  else log "   S3 수프 건너뜀 — BN 거리 ≥ 0.25 → 판정은 두 반복 각각 (위 설명용 줄)"; fi
else log "   ❌ S1 — 학습 2개가 안 끝남 · S2 · S3 건너뜀"; fi

# ── S4 설명용 ──
for m in soup_v7r2 $( [ -d runs_person/soup_v11s ] && echo soup_v11s ) $S1; do
  [ -f runs_person/dumps/${m}_test_obl.npz ] || $P dump_dets.py --model $m --sets test_v2,test_obl,test_kr >> logs/s_dump.log 2>&1
  [ -f metrics/test_auair_$m.csv ] || $P eval_test_v2.py --weights runs_person/$m/weights/best.pt --data data/test_auair --tag test_auair --boot 0 >> logs/s_auair.log 2>&1
  log "   $m test_auair 사람 없는 장 오탐 $(awk -F, 'NR==2{print $6}' metrics/test_auair_$m.csv) 건/장"
done
$P diag_small.py $S1 $( [ -d runs_person/soup_v11s ] && echo soup_v11s ) >> $R 2>> logs/s_dump.log
log "chain_s 완료 → $R — 교체 여부는 사람이 확인 (10-24 동결)"
