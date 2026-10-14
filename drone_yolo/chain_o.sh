#!/bin/bash
# chain_o.sh — 10-14 O 나무 잎 가림 합성 증강 (v13occ) 반복 ×2 → 판정 (판정 기준은 _학습 큐 「10-14 O」 · 돌리기 전에 적음)
#   O1  v13o_f91 · v13o_f92 (v12 와 같은 95 % 목록 · 자연 출처 양성 창 30 % 잎 가림 · 설정 그대로 · 30 에폭 · 에폭마다 저장) → 학습 안 평균 (swa)
#   O2  ★ 평균[v13o_f91_swa+f92_swa] vs 평균[v12_f91_swa+f92_swa+f93_swa] (compare_means) → judge_pairs (반복 폭 절반)
#   O3  참고 — AU-AIR 사람 없는 장 오탐
cd /home/se/JupyterLAB/Capstone/drone_yolo
P=/home/se/miniconda3/envs/drone/bin/python; Q=logs/QUEUE.md; R=metrics/AUTO_RESULT_o.md; C=logs/o_cmp.txt
IGN=data/test_v2/ignore_v1.json
BASE=v12_f91_swa,v12_f92_swa,v12_f93_swa
HALF="--band=test_obl:1.6,test_v2_edge_ign:1.7,test_kr_edge:0.6"
log(){ echo "[$(TZ=Asia/Seoul date '+%F %T') KST] $*" | tee -a $Q; }
ap(){ awk -F, 'NR==2{print $3}' "$1" 2>/dev/null; }
eval3(){ local m=$1 t Y
  for t in test_obl test_v2_edge_ign test_kr_edge; do
    case $t in test_obl) Y="--data data/test_obl --mode single --tag test_obl";;
               test_v2_edge_ign) Y="--merge edge --ignore $IGN";;
               test_kr_edge) Y="--data data/test_kr --tag test_kr --merge edge";; esac
    [ -f metrics/${t}_$m.csv ] || $P eval_test_v2.py --weights runs_person/$m/weights/best.pt $Y >> logs/o_${m}_$t.log 2>&1
  done
  log "   $m AP50 obl $(ap metrics/test_obl_$m.csv) · v2 edge+무시 $(ap metrics/test_v2_edge_ign_$m.csv) · kr edge $(ap metrics/test_kr_edge_$m.csv)"
  echo "| $m | $(ap metrics/test_obl_$m.csv) | $(ap metrics/test_v2_edge_ign_$m.csv) | $(ap metrics/test_kr_edge_$m.csv) |" >> $R; }
cmpm(){ for t in test_obl test_v2_edge_ign test_kr_edge; do $P compare_means.py --tag $t $1 $2 | tail -1 | tee -a $Q $R $C; done; }
train_run(){ local name=$1
  local A="--stage 1 --hyp configs/hyp/v6_oblique.yaml --weights yolo11m.pt --data configs/data_$name.yaml --name $name --epochs 30"
  if $P train_person.py $A --smoke >> logs/o_${name}_smoke.log 2>&1; then
    log "   $name 연기 통과 → 30 에폭"; $P train_person.py $A --set save_period=1 >> logs/o_$name.log 2>&1
    [ -f runs_person/$name/weights/best.pt ] && { eval3 $name; return 0; } || log "   ❌ $name 학습 실패"
  else log "   ❌ $name 연기 실행 실패 (logs/o_${name}_smoke.log)"; fi
  return 1; }

log "chain_o 시작 — O1 v13o ×2 (잎 가림 증강) → O2 평균 판정 → O3 참고"
: > $C
echo -e "\n# chain_o (10-14) — 나무 잎 가림 합성 증강\n\n| 모델 | obl | v2 edge+무시 | kr edge |\n|---|---|---|---|" >> $R
DONE=""
for s in 91 92; do train_run v13o_f$s && $P swa_ckpt.py v13o_f$s >> $Q 2>&1 && eval3 v13o_f${s}_swa && DONE="$DONE v13o_f${s}_swa"; done
if [ $(echo $DONE | wc -w) = 2 ]; then
  echo "- ★ 평균[v13o swa 2] vs 평균[v12 swa 3]:" >> $R
  cmpm $BASE v13o_f91_swa,v13o_f92_swa
  log "   ★ 잎 가림 증강 판정 (평균 · 반복 폭 절반): $($P judge_pairs.py $HALF $C 평균[v12_f91_swa+v12_f92_swa+v12_f93_swa]:평균[v13o_f91_swa+v13o_f92_swa] | tail -1)"
else log "   ❌ O1 — 학습 2개가 안 끝남 · 판정 건너뜀"; fi
for m in $DONE; do
  [ -f metrics/test_auair_$m.csv ] || $P eval_test_v2.py --weights runs_person/$m/weights/best.pt --data data/test_auair --tag test_auair --boot 0 >> logs/o_auair.log 2>&1
  log "   $m test_auair 사람 없는 장 오탐 $(awk -F, 'NR==2{print $6}' metrics/test_auair_$m.csv) 건/장"
done
log "chain_o 끝"
