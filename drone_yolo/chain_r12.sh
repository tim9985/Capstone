#!/bin/bash
# chain_r12.sh — 10-12 v12 (원본부터 다시 짠 데이터) 반복 ×2 → 판정 (판정 기준은 _학습 큐 「10-12 v12」 · 돌리기 전에 적음)
#   R0  기준 v7 단일 4개 (v7 · v7_r2 · v7_r3 · v7_r4) 를 판정 태그로 재기 (test_v2 edge+무시 · test_kr edge)
#   R1  train_v12 95 % 무작위 뽑기 ×2 (시드 91 · 92 · val = val_v12) → v12_f91 · v12_f92 (v7 과 같은 설정 · 30 에폭 · 에폭마다 저장)
#   R2  ★ 주 판정: 평균[v12_f91+f92] vs 평균[v7 단일 4] (compare_means) → judge_pairs (반복 폭 절반: obl 1.6 · v2 1.7 · kr 0.6)
#   R3  보조: 학습 안 평균 (마지막 5 에폭 · swa_ckpt) · 반복 수프 (BN 거리 < 0.25 일 때) vs soup_v7r2 — 설명용
#   R4  설명용 — 덤프 · 크기별 · 장소별 · test_auair 사람 없는 장 오탐
#   교체는 사람이 확인 (10-24 동결)
cd /home/se/JupyterLAB/Capstone/drone_yolo
P=/home/se/miniconda3/envs/drone/bin/python; Q=logs/QUEUE.md; R=metrics/AUTO_RESULT_r12.md; C=logs/r12_cmp.txt
IGN=data/test_v2/ignore_v1.json
BASE4=v7,v7_r2,v7_r3,v7_r4
HALF="--band=test_obl:1.6,test_v2_edge_ign:1.7,test_kr_edge:0.6"
FULL="--band=test_obl:3.1,test_v2_edge_ign:3.3,test_kr_edge:1.2"
log(){ echo "[$(TZ=Asia/Seoul date '+%F %T') KST] $*" | tee -a $Q; }
ap(){ awk -F, 'NR==2{print $3}' "$1" 2>/dev/null; }
eval5(){ local m=$1 t Y
  for t in test_obl test_v2 test_v2_edge_ign test_kr test_kr_edge test_ladd_h; do
    case $t in test_obl) Y="--data data/test_obl --mode single --tag test_obl";; test_v2) Y="";;
               test_v2_edge_ign) Y="--merge edge --ignore $IGN";;
               test_kr) Y="--data data/test_kr --tag test_kr";; test_kr_edge) Y="--data data/test_kr --tag test_kr --merge edge";;
               test_ladd_h) Y="--data data/test_ladd_h --tag test_ladd_h --boot 0";; esac
    [ -f metrics/${t}_$m.csv ] || $P eval_test_v2.py --weights runs_person/$m/weights/best.pt $Y >> logs/r12_${m}_$t.log 2>&1
  done
  log "   $m AP50 obl $(ap metrics/test_obl_$m.csv) · v2 $(ap metrics/test_v2_$m.csv) → edge+무시 $(ap metrics/test_v2_edge_ign_$m.csv) · kr $(ap metrics/test_kr_$m.csv) → edge $(ap metrics/test_kr_edge_$m.csv) · (참고 LADD 뒤 $(ap metrics/test_ladd_h_$m.csv))"
  echo "| $m | $(ap metrics/test_obl_$m.csv) | $(ap metrics/test_v2_$m.csv) → **$(ap metrics/test_v2_edge_ign_$m.csv)** | $(ap metrics/test_kr_$m.csv) → **$(ap metrics/test_kr_edge_$m.csv)** | 참고 LADD 뒤 $(ap metrics/test_ladd_h_$m.csv) |" >> $R; }
cmp3(){ for t in test_obl test_v2_edge_ign test_kr_edge; do $P compare_ci.py --tag $t $1 | tail -1 | tee -a $Q $R $C; done; }
cmpm(){ for t in test_obl test_v2_edge_ign test_kr_edge; do $P compare_means.py --tag $t $1 $2 | tail -1 | tee -a $Q $R $C; done; }
train_run(){ local name=$1; shift
  local A="--stage 1 --hyp configs/hyp/v6_oblique.yaml --weights yolo11m.pt --data configs/data_$name.yaml --name $name --epochs 30 $*"
  if $P train_person.py $A --smoke >> logs/r12_${name}_smoke.log 2>&1; then
    log "   $name 연기 통과 → 30 에폭"; $P train_person.py $A --set save_period=1 >> logs/r12_$name.log 2>&1
    [ -f runs_person/$name/weights/best.pt ] && { eval5 $name; return 0; } || log "   ❌ $name 학습 실패"
  else log "   ❌ $name 연기 실행 실패 (logs/r12_${name}_smoke.log)"; fi
  return 1; }

log "chain_r12 시작 — R0 기준 다시 재기 → R1 v12 ×2 → R2 평균 판정 → R3 보조 → R4 설명"
: > $C
echo -e "\n# chain_r12 (10-12) — v12 원본부터 다시 짠 데이터\n\n| 모델 | obl | v2 → **edge+무시** | kr → **edge** | 참고 |\n|---|---|---|---|---|" >> $R

# ── R0 ──
for m in v7 v7_r2 v7_r3 v7_r4 soup_v7r2; do eval5 $m; done

# ── R1 ──
for s in 91 92; do
  $P chain_util.py mklistf v12_f$s $s 0.95 train_v12.txt | tee -a $Q
  printf "# v12_f$s (chain_r12 · 2026-10-12) — train_v12 95 %% 무작위 뽑기 시드 $s · val = val_v12\ntrain: %s\nval: %s\nnc: 1\nnames: ['person']\n" \
    "$PWD/configs/lists/train_v12_f$s.txt" "$PWD/configs/lists/val_v12.txt" > configs/data_v12_f$s.yaml
done
DIFF=$(comm -3 configs/lists/train_v12_f91.txt configs/lists/train_v12_f92.txt | wc -l)
log "   R1 두 목록 차이 $DIFF 줄"
if [ "$DIFF" -eq 0 ]; then log "   ❌ R1 FAIL — 목록이 같다 → 멈춤"; exit 1; fi
DONE=""
for s in 91 92; do train_run v12_f$s && DONE="$DONE v12_f$s"; done

# ── R2 ★ ──
if [ $(echo $DONE | wc -w) = 2 ]; then
  echo "- ★ 주 판정 — 평균[v12 단일 2] vs 평균[v7 단일 4]:" >> $R
  cmpm $BASE4 v12_f91,v12_f92
  log "   ★ v12 판정 (평균 · 반복 폭 절반): $($P judge_pairs.py $HALF $C 평균[v7+v7_r2+v7_r3+v7_r4]:평균[v12_f91+v12_f92] | tail -1)"
  echo "- 진짜 반복 폭 (f91 → f92):" >> $R; cmp3 v12_f91:v12_f92
  # ── R3 보조 ──
  for m in $DONE; do $P swa_ckpt.py $m >> $Q 2>&1 && eval5 ${m}_swa; done
  [ -d runs_person/v12_f91_swa ] && [ -d runs_person/v12_f92_swa ] && { echo "- 보조 — 학습 안 평균 (마지막 5 에폭):" >> $R; cmpm $BASE4 v12_f91_swa,v12_f92_swa; cmpm v12_f91,v12_f92 v12_f91_swa,v12_f92_swa; }
  $P bn_dist.py v12_f91:v12_f92 >> $Q 2>&1
  MZ=$($P -c "import json;print(json.load(open('metrics/bn_dist.json'))['v12_f91:v12_f92']['mean_z']['평균'])")
  log "   R3 BN 거리 mean_z $MZ (기준 < 0.25)"
  if $P -c "import sys;sys.exit(0 if float('$MZ') < 0.25 else 1)"; then
    $P soup.py soup_v12 v12_f91 v12_f92 >> $Q 2>&1 && eval5 soup_v12 && cmp3 soup_v7r2:soup_v12 \
      && log "   soup_v12 vs soup_v7r2 (보조 · 옛 규칙): $($P judge_pairs.py $FULL $C soup_v7r2:soup_v12 | tail -1)"
  else log "   R3 반복 수프 건너뜀 — BN 거리 ≥ 0.25"; fi
else log "   ❌ R1 — 학습 2개가 안 끝남 · 판정 건너뜀"; fi

# ── R4 설명용 ──
for m in $DONE; do
  [ -f runs_person/dumps/${m}_test_obl.npz ] || $P dump_dets.py --model $m --sets test_v2,test_obl,test_kr >> logs/r12_dump.log 2>&1
  [ -f metrics/test_auair_$m.csv ] || $P eval_test_v2.py --weights runs_person/$m/weights/best.pt --data data/test_auair --tag test_auair --boot 0 >> logs/r12_auair.log 2>&1
  log "   $m test_auair 사람 없는 장 오탐 $(awk -F, 'NR==2{print $6}' metrics/test_auair_$m.csv) 건/장"
done
for m in v7_r2 v7_r3 v7_r4; do [ -f runs_person/dumps/${m}_test_obl.npz ] || $P dump_dets.py --model $m --sets test_v2,test_obl,test_kr >> logs/r12_dump.log 2>&1; done
$P diag_small.py v7 v7_r2 v7_r3 v7_r4 $DONE >> $R 2>> logs/r12_dump.log
log "chain_r12 완료 → $R — 교체 여부는 사람이 확인 (10-24 동결)"
