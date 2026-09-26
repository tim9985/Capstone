#!/bin/bash
# chain_v12.sh — 09-26 새 계획: 데이터 + 반복 + 수프 (→ obsidian/11 서버 학습 계획/_학습 큐.md)
#   ① v7 을 test_kr 까지 재고, v6 반복 둘(v6_obl · r2)과 비교해 이을 데이터(v6 / v7)를 고른다 (chain_util.py pick)
#   ② 교차 수프 soup_v67 (v6_obl + r2 + v7) — 데이터가 다른 모델도 섞이나 (공짜 · 참고)
#   ③ v8 = 고른 데이터 + NII-CU 45° (configs/lists/nii_train.txt) → 기준 실행들과 비교해 최종 데이터 설정을 고른다
#   ④ 최종 설정을 순서만 섞어 두 번 더 → 매번 그 설정의 모든 실행을 평균한 수프를 세 평가셋에서 잰다
cd /home/se/JupyterLAB/Capstone/drone_yolo
P=/home/se/miniconda3/envs/drone/bin/python
Q=logs/QUEUE.md; R=metrics/AUTO_RESULT_v12.md
WAIT_PID=${WAIT_PID:?chain_v11 PID}
HYP="--stage 1 --hyp configs/hyp/v6_oblique.yaml --weights yolo11m.pt"
log(){ echo "[$(TZ=Asia/Seoul date '+%F %T') KST] $*" | tee -a $Q; }
ap(){ awk -F, 'NR==2{print $3}' "$1" 2>/dev/null; }
eval3(){  # 모델 이름 → 세 평가셋 AP50
  local m=$1 t Y
  for t in test_obl test_v2 test_kr; do
    case $t in test_obl) Y="--data data/test_obl --mode single --tag test_obl";; test_v2) Y="";; test_kr) Y="--data data/test_kr --tag test_kr";; esac
    [ -f metrics/${t}_$m.csv ] || $P eval_test_v2.py --weights runs_person/$m/weights/best.pt $Y >> logs/q_${m}_$t.log 2>&1
  done
  log "   $m AP50 obl $(ap metrics/test_obl_$m.csv) · v2 $(ap metrics/test_v2_$m.csv) · kr $(ap metrics/test_kr_$m.csv)"
  echo "| $m | $(ap metrics/test_obl_$m.csv) | $(ap metrics/test_v2_$m.csv) | $(ap metrics/test_kr_$m.csv) |" >> $R
}
cmp3(){  # 기준들(쉼표) 후보
  local t b
  for t in test_obl test_v2 test_kr; do for b in ${1//,/ }; do $P compare_ci.py --tag $t $b:$2 | tail -1 | tee -a $Q; done; done
}
train(){  # 이름 → 연기 실행 후 30 에폭 (data_<이름>.yaml)
  local A="$HYP --data configs/data_$1.yaml --name $1"
  if $P train_person.py $A --smoke >> logs/q_$1_smoke.log 2>&1; then
    log "   $1 연기 통과 → 30 에폭"; $P train_person.py $A >> logs/q_$1.log 2>&1
    [ -f runs_person/$1/weights/best.pt ] && return 0
    log "   ❌ $1 학습 실패 — logs/q_$1.log"; return 1
  fi
  log "   ❌ $1 연기 실행 실패 — logs/q_$1_smoke.log"; return 1
}

while kill -0 $WAIT_PID 2>/dev/null; do sleep 120; done
log "chain_v12 시작 — 데이터 + 반복 + 수프"
echo -e "\n## chain_v12 ($(TZ=Asia/Seoul date '+%F %T'))\n| 모델 | test_obl | test_v2 | test_kr |\n|---|---|---|---|" >> $R

# ① v7 판정 → 이을 데이터
declare -A LIST=([v6]=train_v6.txt [v7]=train_v7.txt)
declare -A RUNS=([v6]="v6_obl v6_obl_r2" [v7]="v7")
FAM=v6
if [ -f runs_person/v7/weights/best.pt ]; then
  eval3 v7; cmp3 v6_obl,v6_obl_r2 v7
  W=$($P chain_util.py pick v7 v6_obl,v6_obl_r2); log "   v7 고르기: $W"
  [ "${W%% *}" = cand ] && FAM=v7
  $P soup.py soup_v67 v6_obl v6_obl_r2 v7 >> $Q 2>&1 && { eval3 soup_v67; cmp3 soup_v6x2 soup_v67; }
fi
log "① 이을 데이터: $FAM (${RUNS[$FAM]})"

# ③ v8 = FAM + NII-CU
if [ -s configs/lists/nii_train.txt ]; then
  $P chain_util.py mklist v8 9 ${LIST[$FAM]} nii_train.txt | tee -a $Q
  if train v8; then
    eval3 v8; B=$(echo ${RUNS[$FAM]} | tr ' ' ,); cmp3 $B v8
    W=$($P chain_util.py pick v8 $B); log "   v8 고르기 (기준 $B): $W"
    if [ "${W%% *}" = cand ]; then LIST[v8]=train_v8.txt; RUNS[v8]="v8"; FAM=v8; fi
  fi
else log "   ⚠ nii_train.txt 없음 — v8 건너뜀"; fi
log "③ 최종 데이터 설정: $FAM (${RUNS[$FAM]})"

# ④ 반복 두 번 + 수프
n=$(echo ${RUNS[$FAM]} | wc -w)
for seed in 21 22; do
  n=$((n + 1)); name=${FAM}_r$n; [ $FAM = v6 ] && name=v6_obl_r$n
  $P chain_util.py mklist $name $seed ${LIST[$FAM]} | tee -a $Q
  if train $name; then
    eval3 $name; RUNS[$FAM]="${RUNS[$FAM]} $name"
    S=soup_${FAM}r$(echo ${RUNS[$FAM]} | wc -w)
    if $P soup.py $S ${RUNS[$FAM]} >> $Q 2>&1; then
      eval3 $S; cmp3 $(echo ${RUNS[$FAM]} | tr ' ' ,) $S
    fi
  fi
done
log "chain_v12 완료 — 최종 후보: soup_${FAM}r$(echo ${RUNS[$FAM]} | wc -w) (${RUNS[$FAM]}) · 09-29 고정 전 확인"
