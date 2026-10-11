#!/bin/bash
# chain_arch.sh — 10-12 S 다시 판정 (무시 영역) → A 팔: Archangel-Real 더하기 (판정 기준은 _학습 큐 「10-11~12 V」 · 돌리기 전에 적음 · 10-12 정정)
#   A-  chain_s.sh 가 끝날 때까지 기다림 (GPU 하나 · PID)
#   S'  S 모델을 test_v2 무시 영역 (data/test_v2/ignore_v1.json · edge 합치기) 으로 다시 재고 판정 — 10-12 정정 기준 (S 결과를 보기 전에 정함)
#   A0  v7 + arch_v11 목록 95 % 무작위 뽑기 ×2 (시드 81 · 82) — 두 목록이 달라야 함
#   A1  v11a_f81 · v11a_f82 (v7 과 같은 설정 · 30 에폭) → test_obl · test_v2 (edge · edge+무시) · test_kr (edge) · 참고 LADD 뒤
#   A2  진짜 반복 폭 (f81 : f82) · 각각 soup_v7r2 대비 (설명용)
#   A3  BN 거리 mean_z < 0.25 일 때만 soup_v11a → 붕괴 확인 → soup_v7r2 대비 judge_pairs (obl 3.1 · v2 무시 3.3 · kr 1.2)
#   A4  설명용 — 덤프 (장소별 · 크기별) · test_auair 사람 없는 장 오탐
#   교체는 사람이 확인 (10-24 동결)
cd /home/se/JupyterLAB/Capstone/drone_yolo
P=/home/se/miniconda3/envs/drone/bin/python; Q=logs/QUEUE.md; R=metrics/AUTO_RESULT_arch.md; C=logs/a_cmp.txt; C2=logs/s2_cmp.txt
IGN=data/test_v2/ignore_v1.json
BAND="--band=test_obl:3.1,test_v2_edge_ign:3.3,test_kr_edge:1.2"
log(){ echo "[$(TZ=Asia/Seoul date '+%F %T') KST] $*" | tee -a $Q; }
ap(){ awk -F, 'NR==2{print $3}' "$1" 2>/dev/null; }
eval5(){ local m=$1 pre=$2 t Y
  for t in test_obl test_v2 test_v2_edge test_v2_edge_ign test_kr test_kr_edge test_ladd_h; do
    case $t in test_obl) Y="--data data/test_obl --mode single --tag test_obl";; test_v2) Y="";; test_v2_edge) Y="--merge edge";;
               test_v2_edge_ign) Y="--merge edge --ignore $IGN";;
               test_kr) Y="--data data/test_kr --tag test_kr";; test_kr_edge) Y="--data data/test_kr --tag test_kr --merge edge";;
               test_ladd_h) Y="--data data/test_ladd_h --tag test_ladd_h --boot 0";; esac
    [ -f metrics/${t}_$m.csv ] || $P eval_test_v2.py --weights runs_person/$m/weights/best.pt $Y >> logs/${pre}_${m}_$t.log 2>&1
  done
  log "   $m AP50 obl $(ap metrics/test_obl_$m.csv) · v2 $(ap metrics/test_v2_$m.csv) → edge $(ap metrics/test_v2_edge_$m.csv) → +무시 $(ap metrics/test_v2_edge_ign_$m.csv) · kr $(ap metrics/test_kr_$m.csv) → edge $(ap metrics/test_kr_edge_$m.csv) · (참고 LADD 뒤 $(ap metrics/test_ladd_h_$m.csv))"
  echo "| $m | $(ap metrics/test_obl_$m.csv) | $(ap metrics/test_v2_$m.csv) → $(ap metrics/test_v2_edge_$m.csv) → **$(ap metrics/test_v2_edge_ign_$m.csv)** | $(ap metrics/test_kr_$m.csv) → $(ap metrics/test_kr_edge_$m.csv) | 참고 LADD 뒤 $(ap metrics/test_ladd_h_$m.csv) |" >> $R; }
cmp3(){ local f=$2; for t in test_obl test_v2_edge_ign test_kr_edge; do $P compare_ci.py --tag $t $1 | tail -1 | tee -a $Q $R $f; done; }
train_run(){ local name=$1 w=$2 ep=$3; shift 3
  local A="--stage 1 --hyp configs/hyp/v6_oblique.yaml --weights $w --data configs/data_$name.yaml --name $name --epochs $ep $*"
  if $P train_person.py $A --smoke >> logs/a_${name}_smoke.log 2>&1; then
    log "   $name 연기 통과 → $ep 에폭"; $P train_person.py $A >> logs/a_$name.log 2>&1
    [ -f runs_person/$name/weights/best.pt ] && { eval5 $name a; return 0; } || log "   ❌ $name 학습 실패"
  else log "   ❌ $name 연기 실행 실패"; fi
  return 1; }

WAIT=${WAIT_PID:-1841232}; while kill -0 $WAIT 2>/dev/null; do sleep 300; done          # S 체인 (GPU · PID) 이 끝날 때까지
log "chain_arch 시작 — S' (무시 영역 다시 판정) → A0 → A1 → A2 → A3 → A4 (Archangel-Real)"
: > $C; : > $C2
echo -e "\n# chain_arch (10-12) — S 다시 판정 (test_v2 무시 영역) · A Archangel-Real 더하기\n\n| 모델 | obl | v2 → edge → **edge+무시** | kr → edge | 참고 |\n|---|---|---|---|---|" >> $R
eval5 soup_v7r2 a

# ── S' ──
SM=""; for m in v11s_f71 v11s_f72 soup_v11s; do [ -f runs_person/$m/weights/best.pt ] && { eval5 $m s2; SM="$SM $m"; }; done
if [ -n "$SM" ]; then
  echo "- S 다시 판정 (10-12 정정 기준 · test_v2 무시 영역):" >> $R
  for m in $SM; do cmp3 soup_v7r2:$m $C2; done
  if echo "$SM" | grep -q soup_v11s; then log "   ★ S 판정 (정정 기준) soup_v11s vs soup_v7r2: $($P judge_pairs.py $BAND $C2 soup_v7r2:soup_v11s | tail -1)"
  else log "   ★ S 판정 (정정 기준 · 수프 없음 → 두 반복 각각): $($P judge_pairs.py $BAND $C2 soup_v7r2:v11s_f71 soup_v7r2:v11s_f72 | tail -1)"; fi
else log "   S' 건너뜀 — S 모델 없음"; fi

# ── A0 ──
for s in 81 82; do $P chain_util.py mklistf v11a_f$s $s 0.95 train_v7.txt arch_v11.txt | tee -a $Q; done
DIFF=$(comm -3 configs/lists/train_v11a_f81.txt configs/lists/train_v11a_f82.txt | wc -l)
log "   A0 두 목록 차이 $DIFF 줄"
if [ "$DIFF" -eq 0 ]; then log "   ❌ A0 FAIL — 목록이 같다 → 멈춤"; exit 1; fi

# ── A1 ──
A1=""
for s in 81 82; do train_run v11a_f$s yolo11m.pt 30 && A1="$A1 v11a_f$s"; done

# ── A2 · A3 ──
if [ $(echo $A1 | wc -w) = 2 ]; then
  echo "- 진짜 반복 폭 (f81 → f82):" >> $R; cmp3 v11a_f81:v11a_f82 $C
  for m in $A1; do cmp3 soup_v7r2:$m $C; log "   $m vs soup_v7r2 (설명용): $($P judge_pairs.py $BAND $C soup_v7r2:$m | tail -1)"; done
  $P bn_dist.py v11a_f81:v11a_f82 >> $Q 2>&1
  MZ=$($P -c "import json;print(json.load(open('metrics/bn_dist.json'))['v11a_f81:v11a_f82']['mean_z']['평균'])")
  log "   A3 BN 거리 mean_z $MZ (기준 < 0.25)"
  if $P -c "import sys;sys.exit(0 if float('$MZ') < 0.25 else 1)"; then
    $P soup.py soup_v11a v11a_f81 v11a_f82 >> $Q 2>&1 && eval5 soup_v11a a
    COL=$($P -c "
a=[float(open(f'metrics/test_kr_{m}.csv').read().splitlines()[1].split(',')[2]) for m in ('v11a_f81','v11a_f82','soup_v11a')]
print('붕괴' if (a[0]+a[1])/2 - a[2] > 0.034 else '정상')")
    log "   soup_v11a 붕괴 확인: $COL"
    if [ "$COL" = 정상 ]; then cmp3 soup_v7r2:soup_v11a $C; log "   ★ A 판정 soup_v11a vs soup_v7r2: $($P judge_pairs.py $BAND $C soup_v7r2:soup_v11a | tail -1)"
    else log "   ❌ A3 FAIL — 수프 붕괴 → 판정은 두 반복 각각"; fi
  else log "   A3 수프 건너뜀 — BN 거리 ≥ 0.25 → 판정은 두 반복 각각 (위 설명용 줄)"; fi
else log "   ❌ A1 — 학습 2개가 안 끝남 · A2 · A3 건너뜀"; fi

# ── A4 설명용 ──
for m in soup_v7r2 $( [ -d runs_person/soup_v11a ] && echo soup_v11a ) $A1; do
  [ -f runs_person/dumps/${m}_test_obl.npz ] || $P dump_dets.py --model $m --sets test_v2,test_obl,test_kr >> logs/a_dump.log 2>&1
  [ -f metrics/test_auair_$m.csv ] || $P eval_test_v2.py --weights runs_person/$m/weights/best.pt --data data/test_auair --tag test_auair --boot 0 >> logs/a_auair.log 2>&1
  log "   $m test_auair 사람 없는 장 오탐 $(awk -F, 'NR==2{print $6}' metrics/test_auair_$m.csv) 건/장"
done
$P diag_small.py $A1 $( [ -d runs_person/soup_v11a ] && echo soup_v11a ) >> $R 2>> logs/a_dump.log
log "chain_arch 완료 → $R — 교체 여부는 사람이 확인 (10-24 동결)"
