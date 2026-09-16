#!/bin/bash
# aihub182_dl.sh — AI-Hub 182 조난자 원천 영상 순차 다운로드 (2026-09-16)
#
# 라벨(45개 · 0.31 GB)은 이미 받아 풀었다 → data/raw/AIHub182/.../라벨링데이터/03_survivor
# 원천은 45개 936 GB 라 전부 받지 않는다. 속도 약 150 MB/분 (2.5 MB/s) 기준으로 고른 것만 순서대로 받는다.
#   · aihubshell 은 한 번에 하나씩 (Ctrl+C 면 받던 tar 를 지운다) → 순차 실행
#   · 받은 뒤 zip 을 풀고 zip 은 지운다 (같은 용량을 두 번 쓰지 않게)
#   · 키는 data/raw/AIHub182/source_list.tsv (filekey · 이름 · 크기 · 분할)
#
# 실행: cd ~/JupyterLAB/Capstone/drone_yolo && setsid nohup ./aihub182_dl.sh > logs/aihub182_dl.log 2>&1 < /dev/null &
# 확인: tail -f logs/aihub182_dl.log · du -sh ../data/raw/AIHub182
set -u
ROOT=/home/se/JupyterLAB/Capstone/data/raw/AIHub182
KEY_FILE=~/.config/aihub/api_key
# 이름:키 — 화성26(Val · 시험셋용 · 11 GB) → 안산01(새 지역 · 15 GB) → 의왕03(lying 많음 · 17 GB)
JOBS="조난자_화성26:47867 조난자_안산01:47900 조난자_의왕03:47905"
log() { echo "=== $* [$(date -u '+%F %T') UTC] ==="; }

log "시작 — 먼저 실행 중인 다운로드가 끝나기를 기다린다"
while pgrep -f 'aihubshell -mode d' > /dev/null; do sleep 60; done
log "선행 다운로드 종료 확인"

for j in $JOBS; do
  name=${j%%:*}; key=${j##*:}
  d=$ROOT/$name
  if [ -d "$d" ] && [ -z "$(find "$d" -name '*.zip' 2>/dev/null)" ] && [ -n "$(find "$d" -name '*.jpg' 2>/dev/null | head -1)" ]; then
    log "$name 이미 받아서 풀려 있음 — 건너뜀"; continue
  fi
  mkdir -p "$d" && cd "$d" || exit 1
  log "$name (key $key) 받기 시작"
  AIHUB_APIKEY=$(cat $KEY_FILE) ~/bin/aihubshell -mode d -datasetkey 182 -filekey "$key" || { log "$name 받기 실패 — 다음으로"; continue; }
  log "$name 압축 풀기"
  find . -name '*.zip' | while read -r z; do unzip -oq "$z" -d "$(dirname "$z")/$(basename "${z%.zip}")" 2>/dev/null && rm -f "$z"; done
  log "$name 완료 · 이미지 $(find . -name '*.jpg' | wc -l)장 · $(du -sh . | cut -f1)"
done
log "전체 완료 · 총 $(du -sh $ROOT | cut -f1)"
touch $ROOT/.dl_done
