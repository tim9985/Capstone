#!/bin/bash
# download_nomad_far.sh — NOMAD 50 · 70 m 원본 받기 (학습 배우 60명)
#
# 서버 학습 계획 3절: 거리 10 · 30 → 10 · 30 · 50 m (+70 m). 작은 사람 목표는 먼 거리 원본에서 만든다.
# 대상 배우 = 서버에 30 m 원본이 있는 배우 (지금 학습 구성 60명) → 분할(배우 단위)을 그대로 유지한다.
#
# 실행 (세션이 닫혀도 계속):
#   cd ~/JupyterLAB/Capstone/drone_yolo && setsid nohup ./download_nomad_far.sh > logs/nomad_far_dl.log 2>&1 < /dev/null &
# 끊기거나 재부팅되면 같은 명령을 다시 실행한다 — rclone copy 는 받은 파일을 건너뛴다.
# 진행 확인: tail -n 5 logs/nomad_far_dl.log   · 끝나면 data/raw/nomad_far_dl.done 이 생긴다
# rclone 원격 gdrive 설정은 DATASETS.md 1절 (토큰은 문서 · 커밋에 쓰지 않는다)
set -u
RAW=/home/se/JupyterLAB/Capstone/data/raw
cd "$RAW" || exit 1
rm -f nomad_far_dl.done

FILTER=nomad_filter_a50_a70.txt
for d in NOMAD/images/Actor*; do
  a=$(basename "$d")
  [ -d "$d/${a}_a30" ] && printf '+ %s/%s_a50/**\n+ %s/%s_a70/**\n' "$a" "$a" "$a" "$a"
done > "$FILTER"
echo "- **" >> "$FILTER"
N=$(( ($(wc -l < "$FILTER") - 1) / 2 ))
echo "[$(date -u '+%F %T') UTC] 시작 — 배우 ${N}명 · a50 · a70 → $RAW/NOMAD"
[ "$N" -gt 0 ] || { echo "대상 배우 없음 — 중단"; exit 1; }

PACE="--checkers 32 --drive-pacer-min-sleep 10ms --drive-pacer-burst 200 --stats 60s --stats-one-line --stats-log-level NOTICE --retries 10 --low-level-retries 20"
for kind in labels images; do
  T=16; [ "$kind" = labels ] && T=32
  ok=0
  for try in 1 2 3 4 5; do
    echo "[$(date -u '+%F %T') UTC] $kind 받기 (시도 $try)"
    if rclone copy "gdrive:$kind" "NOMAD/$kind" --filter-from "$FILTER" --transfers $T $PACE; then ok=1; break; fi
    echo "[$(date -u '+%F %T') UTC] $kind 실패 — 10분 뒤 다시"
    sleep 600
  done
  [ $ok = 1 ] || { echo "[$(date -u '+%F %T') UTC] $kind 5회 실패 — 중단 (같은 명령으로 다시 실행)"; exit 1; }
done

echo "[$(date -u '+%F %T') UTC] 완료"
for alt in a50 a70; do
  echo "  $alt: 이미지 $(find NOMAD/images -path "*_${alt}/*" -name '*.jpg' | wc -l) · 라벨 $(find NOMAD/labels -path "*_${alt}/*" -type f | wc -l) · $(du -sch NOMAD/images/*/*_${alt} 2>/dev/null | tail -1 | cut -f1)"
done
touch nomad_far_dl.done
