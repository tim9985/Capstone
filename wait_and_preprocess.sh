#!/bin/bash
# NOMAD 신규 30명 다운로드 완료를 기다린 뒤 전처리·설정 재생성까지 자동으로 진행
cd /home/se/JupyterLAB/drone_dev/drone_yolo
PY=/home/se/miniconda3/envs/drone/bin/python
RAW=/home/se/JupyterLAB/drone_dev/data/raw/NOMAD

echo "=== NOMAD 신규 다운로드 완료 대기 시작 $(date) ==="
while kill -0 10487 2>/dev/null; do sleep 15; done
echo "=== 다운로드 완료 감지 $(date) ==="
tail -5 nomad_rclone_new30.log

echo "=== 신규 30명 배치용 하드링크 스테이징 구성 $(date) ==="
STAGE=/home/se/JupyterLAB/drone_dev/data/raw/NOMAD_sel31_100
rm -rf "$STAGE"
mkdir -p "$STAGE/images"
ln -sf "$RAW/annotations.json" "$STAGE/annotations.json"
ln -sf "$RAW/metadata.json" "$STAGE/metadata.json"
ln -sf "$RAW/activityLabels.json" "$STAGE/activityLabels.json"
# 버그 수정(2026-09-11): while + 프로세스치환으로 짰더니 매 반복마다 파일을
# 다시 열어 read 가 무한히 성공해서 무한루프가 됐다(배우당 하드링크가 계속
# 중복 생성됨). 파일이 한 줄뿐이라 while 자체가 불필요 — 한 번만 읽는다.
IFS=',' read -ra ARR < "$(dirname "$0")"/nomad_new_actors.txt
for a in "${ARR[@]}"; do
  src="$RAW/images/${a}"
  if [ -d "$src" ]; then
    cp -al "$src" "$STAGE/images/${a}"
  else
    echo "  경고: $a 폴더 없음 (다운로드 누락 가능)"
  fi
done
echo "스테이징 배우 수: $(ls "$STAGE/images" | wc -l)"

echo "=== nomad_prep.py 신규 30명 배치 전처리 시작 $(date) ==="
$PY nomad_prep.py --nomad "$STAGE" --out data/det/nomad_actor_sel31_100 --workers 16
echo "=== nomad_prep 완료 $(date) ==="

echo "=== make_configs.py 재생성 $(date) ==="
$PY make_configs.py
echo "=== 전체 완료 $(date) ==="
