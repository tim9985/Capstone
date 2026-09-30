#!/bin/bash
# chain_s2.sh — 10-01: 자세 분류 2차 — s1_aihub_sard + NOMAD (비스듬 누움·서기 · 활동 구간 유도) · 바꾸는 것은 데이터 하나
#   판정 (돌리기 전): s1_aihub_sard 대비
#     ① okutama_obl 누움 AUROC (0.907) · 매크로 F1 (0.451) · 누움 재현율 (0.123) 이 오르는가 — 비스듬 약점
#     ② aihub_test (0.979 · F1 0.781) · okutama_nadir 에서 크게 나빠지지 않는가
#     ③ 박스 모양 기준선 (okutama_obl 0.97) 을 넘는가
cd /home/se/JupyterLAB/Capstone/drone_yolo
P=/home/se/miniconda3/envs/drone/bin/python; Q=logs/QUEUE.md
WAIT_PID=${WAIT_PID:?make_posture_crops.py nomad PID}
log(){ echo "[$(TZ=Asia/Seoul date '+%F %T') KST] $*" | tee -a $Q; }
summ(){ $P - "$1" <<'PY' | tee -a $Q
import json, sys
d = json.load(open(f"metrics/posture_{sys.argv[1]}.json"))
for s, r in d.items():
    ci = r.get("95%구간", {})
    print(f"   {sys.argv[1]} · {s}: 누움 AUROC 분류기 {r['누움AUROC_분류기']} {ci.get('누움AUROC_분류기')} vs 박스 {r['누움AUROC_박스모양']} {ci.get('누움AUROC_박스모양')}"
          f" · 매크로F1 {r['매크로F1']} · 재현율 {r['재현율']}")
PY
}
while kill -0 $WAIT_PID 2>/dev/null; do sleep 30; done
log "chain_s2 시작 — 자세 분류 2차 (+ NOMAD)"
n=$(find data/pose_cls/nomad/train -name '*.jpg' | wc -l)
[ "$n" -gt 500 ] || { log "   ❌ NOMAD 크롭 $n 장 — 중단"; exit 1; }
log "   NOMAD 크롭 $n 장 (누움 $(ls data/pose_cls/nomad/train/lying | wc -l) · 서기 $(ls data/pose_cls/nomad/train/standing | wc -l))"
S=s2_aihub_sard_nomad
if $P train_posture_cls.py --set $S --smoke >> logs/q_${S}_smoke.log 2>&1; then
  log "   $S 연기 통과 → 30 에폭"
  $P train_posture_cls.py --set $S >> logs/q_$S.log 2>&1
  if [ -f weights/$S.pt ]; then
    $P eval_posture.py weights/$S.pt >> logs/q_${S}_eval.log 2>&1 && summ $S || log "   ❌ $S 평가 실패"
  else log "   ❌ $S 학습 실패"; fi
else log "   ❌ $S 연기 실행 실패"; fi
log "chain_s2 완료"
