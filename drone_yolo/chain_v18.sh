#!/bin/bash
# chain_v18.sh — 10-01: 상태 인지 v1 을 탐지 모델별로 · v9 4개 수프
#   ① 지금: 상태 판정 (soup_v7r2 · soup_v9x2) — 누움을 얼마나 잡고 순위를 매기나 (v9 는 누움 재현율 0.36 → 0.50)
#   ② chain_v17 (v9_s33 · s34) 끝나면: 상태 판정 (soup_v9x2b) · soup_v9x4 (s31~s34 평균) 세 평가셋 + 너비비 (BN 붕괴 확인)
cd /home/se/JupyterLAB/Capstone/drone_yolo
P=/home/se/miniconda3/envs/drone/bin/python; Q=logs/QUEUE.md; R=metrics/AUTO_RESULT_v12.md
WAIT_PID=${WAIT_PID:?chain_v17 PID}
log(){ echo "[$(TZ=Asia/Seoul date '+%F %T') KST] $*" | tee -a $Q; }
ap(){ awk -F, 'NR==2{print $3}' "$1" 2>/dev/null; }
state(){  # 탐지 가중치 이름 → 상태 판정 요약
  local m=$1
  [ -f runs_person/$m/weights/best.pt ] || { log "   ⚠ $m 없음 — 상태 판정 건너뜀"; return; }
  $P state_pipeline.py --weights=runs_person/$m/weights/best.pt --tag=$m --no-render > logs/q_state_$m.log 2>&1 || { log "   ❌ 상태 판정 $m 실패"; return; }
  $P - "$m" <<'PY' | tee -a $Q
import json, sys
d = json.load(open(f"metrics/state_pipeline_{sys.argv[1]}.json"))
print(f"   상태 판정 {sys.argv[1]}: 누움 잡힌 비율 {d['누움 잡힌 비율']} ({d['누움 1초 표본 (탐지·추적으로 잡힌 것)']}/{d['누움 정답 1초 표본']}) · "
      f"누움 AUROC 점수 {d['누움 가려내기 AUROC (1초 표본)']['요구조 점수']} vs 확신도 {d['누움 가려내기 AUROC (1초 표본)']['탐지 확신도(기준선)']} · "
      f"추적 단위 {d['추적 단위 누운 적 있음 AUROC']['최고 점수']} · 누움 등급 {d['정답 자세 × 등급 (1초 표본)'].get('Lying')}")
PY
}
log "chain_v18 시작 — 상태 인지 v1 × 탐지 모델"
state soup_v7r2
state soup_v9x2
while kill -0 $WAIT_PID 2>/dev/null; do sleep 120; done
log "   chain_v17 끝 — v9 4개 수프 · 상태 판정 이어서"
state soup_v9x2b
if [ -f runs_person/v9_s33/weights/best.pt ] && [ -f runs_person/v9_s34/weights/best.pt ]; then
  $P soup.py soup_v9x4 v9_s31 v9_s32 v9_s33 v9_s34 >> $Q 2>&1
  for t in test_obl test_v2 test_kr; do
    case $t in test_obl) Y="--data data/test_obl --mode single --tag test_obl";; test_v2) Y="";; test_kr) Y="--data data/test_kr --tag test_kr";; esac
    $P eval_test_v2.py --weights runs_person/soup_v9x4/weights/best.pt $Y >> logs/q_soup_v9x4_$t.log 2>&1
    $P compare_ci.py --tag $t soup_v7r2:soup_v9x4 | tail -1 | tee -a $Q
  done
  log "   soup_v9x4 AP50 obl $(ap metrics/test_obl_soup_v9x4.csv) · v2 $(ap metrics/test_v2_soup_v9x4.csv) · kr $(ap metrics/test_kr_soup_v9x4.csv) (kr 이 크게 떨어지면 BN 붕괴)"
  echo "| soup_v9x4 | $(ap metrics/test_obl_soup_v9x4.csv) | $(ap metrics/test_v2_soup_v9x4.csv) | $(ap metrics/test_kr_soup_v9x4.csv) |" >> $R
  $P analyze_models.py soup_v7r2 soup_v9x4 --out=analyze_v9x4.json > logs/q_analyze_v9x4.log 2>&1 && $P - <<'PY' | tee -a $Q
import json
d = json.load(open("metrics/analyze_v9x4.json"))
for k, v in d.items():
    if k.endswith("test_obl"):
        s = v["층"]; print(f"   {k}: 가림 {s['가림:예'][0]} · 누움 {s['자세:Lying'][0]} · 앉음 {s['자세:Sitting'][0]} · 너비비 {v['너비비_25/50/75'][1]}")
PY
fi
log "chain_v18 완료"
