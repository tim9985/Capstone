#!/bin/bash
# extra_eval.sh — 외부 평가셋 3개 (AU-AIR · 키프로스 · LADD) 받기 → 평가셋 → FiftyOne (2026-10-04)
#   세션을 닫아도 이어지게 setsid 로 띄운다. 끊긴 다운로드는 aria2c -c 로 이어 받고,
#   fo_all_sets 가 죽었으면 캐시 없는 원천만 다시 돈다.
cd /home/se/JupyterLAB/Capstone/drone_yolo
P=/home/se/miniconda3/envs/drone/bin/python; Q=logs/QUEUE.md; L=logs/extra_eval.log; RAW=../data/raw
R=metrics/AUTO_RESULT_extra_eval.md
log(){ echo "[$(TZ=Asia/Seoul date '+%F %T') KST] $*" | tee -a $Q >> $L; }
dl(){ # 폴더 파일 URL — .aria2 가 남아 있거나 파일이 없으면, aria2c 가 안 돌 때 이어 받기
  while [ -f "$1/$2.aria2" ] || [ ! -f "$1/$2" ]; do
    pgrep -af '^aria2c' | grep -q -- "-o $2" || aria2c -q -x8 -s8 -c -d "$1" -o "$2" "$3" >> $L 2>&1
    sleep 30
  done; }
log "extra_eval 시작"
dl $RAW/cyprus_sop Images.zip "https://zenodo.org/api/records/7740081/files/Images.zip/content"
U=$(curl -sL "https://raw.githubusercontent.com/dataset-ninja/lacmus-drone-dataset/main/DOWNLOAD.md" | grep -oE 'https://assets\.supervisely\.com/remote/[^)]+' | head -1)
dl $RAW/ladd ladd_supervisely.tar "$U"
log "   다운로드 끝 — 키프로스 $(du -h $RAW/cyprus_sop/Images.zip | cut -f1) · LADD $(du -h $RAW/ladd/ladd_supervisely.tar | cut -f1)"
[ -d $RAW/cyprus_sop/Images ] || unzip -q -o $RAW/cyprus_sop/Images.zip -d $RAW/cyprus_sop >> $L 2>&1
[ -n "$(find $RAW/ladd -path '*ann/*.json' -print -quit)" ] || tar xf $RAW/ladd/ladd_supervisely.tar -C $RAW/ladd >> $L 2>&1
$P make_test_extra.py cy ladd 2>&1 | tee -a $L | grep "^test_" | while read l; do log "   $l"; done
# FiftyOne 원천 올리기가 끝날 때까지 (GPU 를 둘이 나눠 쓰지 않게) — 죽었으면 캐시 없는 원천만 다시
while pgrep -af 'python fo_all_sets.py' | grep -qv pgrep; do sleep 60; done
MISS=$(for s in test_nii ue_level01 sard nii nomad_v9 unicamp visdrone nadir_rot neg nomad c2a wisard aihub; do [ -f runs_person/soup_v7r2/preds_src_$s.npz ] || echo -n "$s "; done)
[ -n "$MISS" ] && { log "   fo_all_sets 다시 — $MISS"; $P fo_all_sets.py $MISS >> logs/fo_all_sets.log 2>&1; }
tr '\r' '\n' < logs/fo_all_sets.log | grep -E "^[a-z_0-9]+ +\|" | sort -u > metrics/fo_all_sets_summary.txt
$P fo_eval_sets.py soup_v7r2 test_auair test_cy test_ladd 2>&1 | tr '\r' '\n' | tee -a $L | grep "soup_v7r2:" | while read l; do log "   $l"; done
{ echo "# 외부 평가셋 · FiftyOne 원천 (soup_v7r2 · 운용 임계 0.15 · IoU 0.5) — $(TZ=Asia/Seoul date '+%F %T') KST"; echo
  echo '```'; grep -E "soup_v7r2:" $L; echo; cat metrics/fo_all_sets_summary.txt; echo '```'; } > $R
log "extra_eval 끝 → $R"
