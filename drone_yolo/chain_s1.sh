#!/bin/bash
# chain_s1.sh — 09-30~: 상태 인지 A2 자세 분류 기준선 (탐지 뒤 크롭 분류 · 장소 분리 평가)
#   s1_aihub      AI-Hub 5곳 (수직 · 한국) → 평가 aihub_test(산악5) · okutama_obl · okutama_nadir
#   s1_aihub_sard + SARD (비스듬) — 바꾸는 것은 데이터 하나
#   판정 (돌리기 전 · 볼트 「06 요구조자 상태 인지 계획」 실험 1):
#     ① 분류기 누움 AUROC 가 박스 모양 기준선보다 높은가 (95 % 구간) — 평가셋마다
#     ② 매크로 F1 (앉음 포함) · 서기 오인 10 % 에서 누움 재현율
#     ③ SARD 를 넣으면 okutama_obl 이 오르는가 (비스듬 학습 데이터 효과)
cd /home/se/JupyterLAB/Capstone/drone_yolo
P=/home/se/miniconda3/envs/drone/bin/python; Q=logs/QUEUE.md
WAIT_PID=${WAIT_PID:?make_posture_crops.py aihub PID}
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
while kill -0 $WAIT_PID 2>/dev/null; do sleep 60; done
log "chain_s1 시작 — 자세 분류 기준선 (A2)"
n=$(find data/pose_cls/aihub/val data/pose_eval/aihub_test -name '*.jpg' | wc -l)
[ "$n" -gt 1000 ] || { log "   ❌ AI-Hub val·평가 크롭 $n 장 — 중단"; exit 1; }
for S in s1_aihub s1_aihub_sard; do
  if $P train_posture_cls.py --set $S --smoke >> logs/q_${S}_smoke.log 2>&1; then
    log "   $S 연기 통과 → 30 에폭"
    $P train_posture_cls.py --set $S >> logs/q_$S.log 2>&1
    if [ -f weights/$S.pt ]; then
      $P eval_posture.py weights/$S.pt >> logs/q_${S}_eval.log 2>&1 && summ $S || log "   ❌ $S 평가 실패"
    else log "   ❌ $S 학습 실패"; fi
  else log "   ❌ $S 연기 실행 실패"; fi
done
log "chain_s1 완료"
