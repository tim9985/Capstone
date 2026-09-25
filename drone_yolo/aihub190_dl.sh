#!/bin/bash
# aihub190_dl.sh — AI-Hub 190 (008.자율주행드론) 라벨 우선 받기 (2026-09-25)
#   전체 4.8 TB → 전부 받지 않는다. 1) 승인될 때까지 1시간마다 확인 2) 라벨만: 산림지 45° → 관광지·도심지 60° (약 32 GB)
#   3) 라벨 분석 후 원천 이미지는 골라서 따로 받는다
#   키는 파일에서만 읽고 출력하지 않는다
ROOT=/home/se/JupyterLAB/Capstone/data/raw/AIHub190
KEY_FILE=$HOME/.config/aihub/api_key
log(){ echo "=== $* [$(TZ=Asia/Seoul date '+%F %T') KST] ==="; }
dl(){ cd "$1" && AIHUB_APIKEY=$(cat $KEY_FILE) ~/bin/aihubshell -mode d -datasetkey 190 -filekey "$2" 2>&1 | grep -iv "apikey"; }
T=$(awk -F'\t' 'NR==1{print $2}' $ROOT/label_jobs.tsv)
while true; do
  mkdir -p $ROOT/_probe; dl $ROOT/_probe $T > $ROOT/_probe/out.txt
  if find $ROOT/_probe -name '*.zip' | grep -q .; then rm -rf $ROOT/_probe; log "승인 확인 — 받기 시작"; break; fi
  rm -rf $ROOT/_probe; log "아직 미승인 — 1시간 뒤 다시"; sleep 3600
done
while IFS=$'\t' read -r pri key name; do
  d=$ROOT/labels; mkdir -p $d
  [ -e "$d/.done_$key" ] && continue
  log "라벨 $name ($key)"
  dl $d $key > /dev/null && touch "$d/.done_$key"
done < $ROOT/label_jobs.tsv
log "라벨 완료 · $(du -sh $ROOT/labels | cut -f1)"
