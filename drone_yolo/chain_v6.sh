#!/bin/bash
# chain_v6.sh — 60° 기준 대기열 (2026-09-24)
# chain_v2 의 rect_place 를 빼고 이어 건다. chain_v2 bash 는 PID 로 끄고, 돌고 있는 p2m_place 학습은 살려 둔다.
#   ① p2m_place 학습 끝날 때까지 대기 → 판정 (test_v2 · test_obl)
#   ② 기준선 재측정 — v3_place · fov_11m 을 test_obl 로
#   ③ v6_obl 연기 실행(GPU) → 통과하면 30 에폭 학습 → 판정 (test_obl · test_v2)
# 판정 기준 (돌리기 전에 적는다): test_obl AP50 이 기준선 v3_place 보다 +0.01 초과면 개선 · ±0.01 동등 · 그 아래 악화
cd /home/se/JupyterLAB/Capstone/drone_yolo
P=/home/se/miniconda3/envs/drone/bin/python
Q=logs/QUEUE.md; R=metrics/AUTO_RESULT_v6.md
P2M_PID=${P2M_PID:?p2m_place 학습 PID 를 환경변수로 준다}
log(){ echo "[$(TZ=Asia/Seoul date '+%F %T') KST] $*" | tee -a $Q; }
ap(){ awk -F, 'NR==2{print $3}' "$1" 2>/dev/null; }

evalset(){  # $1 run 이름 · $2 tag(test_v2|test_obl)
  W=runs_person/$1/weights/best.pt
  [ -s "$W" ] || { log "   $1 가중치 없음"; return 1; }
  if [ "$2" = test_obl ]; then A="--data data/test_obl --mode single --tag test_obl"; else A=""; fi
  $P eval_test_v2.py --weights $W $A >> logs/q_$1_$2.log 2>&1 || { log "   $1 $2 평가 실패"; return 1; }
  V=$(ap metrics/$2_$1.csv); log "   $1 $2 AP50 $V"; echo "| $1 | $2 | $V |" >> $R
}

: > $R
{ echo "# 60° 대기열 결과 (chain_v6) — AP50"; echo; echo "판정: test_obl 이 기준선 v3_place 대비 +0.01 초과 = 개선"; echo
  echo "| 모델 | 평가셋 | AP50 |"; echo "|---|---|---:|"; } >> $R
log "60° 대기열 시작 (chain_v6) — p2m_place(PID $P2M_PID) 끝나기 대기"

while kill -0 $P2M_PID 2>/dev/null; do sleep 120; done
log "① p2m_place 학습 끝 → 판정"
evalset p2m_place test_v2; evalset p2m_place test_obl

log "② 기준선 재측정 (test_obl)"
evalset v3_place test_obl; evalset fov_11m_1280_all test_obl
BASE_OBL=$(ap metrics/test_obl_v3_place.csv)

V6="--stage 1 --hyp configs/hyp/v6_oblique.yaml --weights yolo11m.pt --data configs/data_v6.yaml --name v6_obl"
log "③ v6_obl 연기 실행 (GPU)"
if $P train_person.py $V6 --smoke >> logs/q_v6_obl_smoke.log 2>&1; then
  log "   연기 실행 통과 → v6_obl 30 에폭 시작"
  $P train_person.py $V6 >> logs/q_v6_obl.log 2>&1
  evalset v6_obl test_obl; evalset v6_obl test_v2
  V=$(ap metrics/test_obl_v6_obl.csv)
  J=$($P -c "a=float('$V' or 0);b=float('$BASE_OBL' or 0);print(f'{(a-b)*100:+.1f} %p · ' + ('✅ 개선' if a>b+0.01 else ('➖ 동등' if a>=b-0.01 else '❌ 악화')))")
  log "   v6_obl test_obl 판정: 기준선 $BASE_OBL 대비 $J"
  { echo; echo "**v6_obl 판정 (test_obl)**: 기준선 v3_place $BASE_OBL → $V ($J)"; } >> $R
else
  log "   ❌ 연기 실행 실패 — logs/q_v6_obl_smoke.log 확인 · v6_obl 학습 안 함"
fi
log "60° 대기열 완료 → $R"
