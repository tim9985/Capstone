#!/bin/bash
# aihub182_dl2.sh — AI-Hub 182 추가 장소 (2026-09-26 · v8 용) — 산악 4곳 + 평지(수풀) 1곳
#   지금 쓰는 곳(저수지2·평지(흙)3·산악6 학습 / 저수지1 val / 산악5 test_kr)과 겹치지 않는다
#   받은 뒤 압축 풀고 zip 삭제 · 키는 파일에서만 읽고 출력하지 않는다
ROOT=/home/se/JupyterLAB/Capstone/data/raw/AIHub182
KEY_FILE=$HOME/.config/aihub/api_key
log(){ echo "=== $* [$(TZ=Asia/Seoul date '+%F %T') KST] ==="; }
JOBS="조난자_화성27:산악8 조난자_화성20:산악2 조난자_시흥02:산악4 조난자_화성22:산악3 조난자_화성01:평지(수풀)1"
for j in $JOBS; do
  name=${j%%:*}; terr=${j##*:}
  key=$(awk -F'\t' -v n="$name.zip" '$2==n{print $1}' $ROOT/source_list.tsv)
  d=$ROOT/$name
  if [ -n "$(find "$d" -name '*.jpg' 2>/dev/null | head -1)" ] && [ -z "$(find "$d" -name '*.zip' 2>/dev/null)" ]; then log "$name 이미 있음"; continue; fi
  mkdir -p "$d" && cd "$d" || exit 1
  log "$name ($terr · key $key) 받기"
  AIHUB_APIKEY=$(cat $KEY_FILE) ~/bin/aihubshell -mode d -datasetkey 182 -filekey "$key" 2>&1 | grep -iv apikey | tail -2
  find . -name '*.zip' | while read -r z; do unzip -oq "$z" -d "$(dirname "$z")/$(basename "${z%.zip}")" 2>/dev/null && rm -f "$z"; done
  log "$name 완료 · 이미지 $(find . -name '*.jpg' | wc -l)장 · $(du -sh . | cut -f1)"
done
log "전체 완료"; touch $ROOT/.dl2_done
