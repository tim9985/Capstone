#!/bin/bash
# chain_v11.sh — v7 (2026-09-26): v6 + SARD + HERIDAL + 비스듬 음성 · NWD 는 v6_nwd 판정에 따라 자동
#   NWD 채택 규칙: v6_nwd − v6_obl_r2 짝 구간이 test_obl·test_v2 중 하나라도 0 을 넘고, 어느 쪽도 0 아래(hi<0)가 아니면 켠다
#   v7 판정: 기준 모델(NWD 채택이면 v6_nwd, 아니면 v6_obl_r2)과 짝 비교 — 구간이 0 을 넘으면 개선
cd /home/se/JupyterLAB/Capstone/drone_yolo
P=/home/se/miniconda3/envs/drone/bin/python
Q=logs/QUEUE.md; R=metrics/AUTO_RESULT_v6.md
WAIT_PID=${WAIT_PID:?chain_v10 PID}
log(){ echo "[$(TZ=Asia/Seoul date '+%F %T') KST] $*" | tee -a $Q; }
ap(){ awk -F, 'NR==2{print $3}' "$1" 2>/dev/null; }
while kill -0 $WAIT_PID 2>/dev/null; do sleep 120; done
NWD=$($P - <<'PY'
import csv
from pathlib import Path
lo_pos = hi_neg = False
for t in ("test_obl", "test_v2"):
    f = Path(f"metrics/compare_ci_{t}.csv")
    if not f.exists(): continue
    rows = [r for r in csv.DictReader(open(f)) if r["base"] == "v6_obl_r2" and r["model"] == "v6_nwd"]
    if rows:
        r = rows[-1]; lo_pos |= float(r["lo"]) > 0; hi_neg |= float(r["hi"]) < 0
print("on" if lo_pos and not hi_neg else "off")
PY
)
if [ "$NWD" = on ]; then X="--nwd 0.5 --nwd-c 32"; BASE=v6_nwd; else X=""; BASE=v6_obl_r2; fi
log "⑧ v7 — NWD $NWD (기준 $BASE)"
if [ ! -f configs/data_v7.yaml ]; then log "   ❌ configs/data_v7.yaml 없음 — v7 안 돌림"; exit 1; fi
A="--stage 1 --hyp configs/hyp/v6_oblique.yaml --weights yolo11m.pt --data configs/data_v7.yaml $X --name v7"
if $P train_person.py $A --smoke >> logs/q_v7_smoke.log 2>&1; then
  log "   연기 실행 통과 → 30 에폭"; $P train_person.py $A >> logs/q_v7.log 2>&1
  for t in test_obl test_v2; do
    if [ $t = test_obl ]; then Y="--data data/test_obl --mode single --tag test_obl"; else Y=""; fi
    $P eval_test_v2.py --weights runs_person/v7/weights/best.pt $Y >> logs/q_v7_$t.log 2>&1
    log "   v7 $t AP50 $(ap metrics/${t}_v7.csv) ($BASE $(ap metrics/${t}_${BASE}.csv))"; echo "| v7 | $t | $(ap metrics/${t}_v7.csv) |" >> $R
    $P compare_ci.py --tag $t $BASE:v7 | tee -a $Q
  done
  $P diag_misses.py --weights runs_person/v7/weights/best.pt > logs/q_v7_diag.log 2>&1 && log "   진단 → metrics/diag_misses_v7.json"
else log "   ❌ 연기 실행 실패 — logs/q_v7_smoke.log"; fi
log "chain_v11 완료"
