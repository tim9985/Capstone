#!/usr/bin/env bash
# ue_watch.sh — UE 촬영 압축이 ue_flight/ 에 도착하면 풀고 비전 전 과정 평가 (10-14 U · 판정 기준은 _학습 큐 U)
#   도착 = <run>_q85.tar.gz 크기가 30 s 그대로 + gzip -t 통과 · 16 m 먼저 (사용자 전송 순서)
#   모델 v12_f91_swa · TensorRT · 운용 임계 0.20 · FiftyOne ue/<run>
#   S4 = 주 질의 green (Standing_04) + 보조 8색 각각 (--tag _q<색>)
#   사용: nohup bash ue_watch.sh > logs/ue_watch.log 2>&1 &
set -u
cd "$(dirname "$0")"
P=/home/se/miniconda3/envs/drone/bin/python
Q=logs/QUEUE.md
W=runs_person/v12_f91_swa/weights
RUNS=${RUNS:-"S3_a16 S2_a16 S4_a16 S1_a16 S5_a16 S3_a20 S2_a20 S4_a20 S1_a20 S5_a20"}
log(){ echo "[$(TZ=Asia/Seoul date '+%F %T') KST] $*" | tee -a $Q; }
ev(){ $P ue_vision_eval.py "ue_flight/$1" --weights $W/best.pt --engine $W/best.engine --conf 0.20 "${@:2}" > "logs/ue_eval_$1$TAG.log" 2>&1; }

log "ue_watch 시작 — $RUNS"
left=$RUNS; deadline=$(( $(date +%s) + 8 * 3600 ))
while [ -n "${left// /}" ] && [ "$(date +%s)" -lt $deadline ]; do
  for r in $left; do
    f=ue_flight/${r}_q85.tar.gz
    [ -f "$f" ] || continue
    s1=$(stat -c %s "$f"); sleep 30; s2=$(stat -c %s "$f")
    [ "$s1" = "$s2" ] && gzip -t "$f" 2>/dev/null || continue
    left=$(echo " $left " | sed "s/ $r / /"); left=$(echo $left)
    tar xzf "$f" -C ue_flight/ || { log "  $r 풀기 실패"; continue; }
    n=$(ls ue_flight/$r/frames 2>/dev/null | wc -l)
    z=$($P -c "import json;r=json.load(open('ue_flight/$r/run.json'));print(r.get('start_ned'), r.get('ground_ned_z'))" 2>&1)
    log "  $r 도착 · 프레임 $n · start_ned · ground_ned_z = $z"
    TAG=""
    case $r in
      S4_*) ev $r --query green --fo; log "  $r 평가 (질의 green) exit $?"
            for c in red orange yellow blue purple white black; do TAG=_q$c; ev $r --query $c --tag _q$c; done; TAG=""
            log "  $r 보조 8색 질의 끝" ;;
      *)    ev $r --fo; log "  $r 평가 exit $?" ;;
    esac
    grep -E "IDF1|오탐/장|다시 잇기|상의색|좌표 오차 안정|탐지 ms|인상착의" metrics/ue_vision_$r.md | sed 's/^/    /' | tee -a $Q
  done
  sleep 60
done
log "ue_watch 끝 — 남음: ${left:-없음}"
