#!/bin/bash
# overnight_m1280.sh — 회의(2026-09-12 14:00 KST 무렵) 전 밤사이 1회 학습
#
#   1) E1(e1_11s_960) 종료 대기
#   2) data/det 을 data/det_v1_fov60 으로 백업 이동
#      — 전처리 스크립트는 출력 폴더를 비우지 않는다. 옛 .npy 디스크 캐시가 남으면
#        새 라벨에 옛 이미지가 붙어 학습된다. 폴더째 옮겨서 원천 차단
#   3) 화각 지터 확대(0.65~1.45) 반영된 nomad_prep / wisard_prep 로 재생성 → make_configs
#   4) 검증(장수 · val 배우 분할 동일 · .npy 없음) — 실패하면 학습 안 함
#   5) yolo11m @1280, COCO 사전학습에서, 마감 시각까지 시간 제한 학습
#   6) 도메인별 평가 (새 모델 @1280, E1 @960 — 둘 다 새 val 크롭으로)
#
# 실행: nohup ./overnight_m1280.sh > overnight_m1280.log 2>&1 &
set -u
cd /home/se/JupyterLAB/drone_dev/drone_yolo
# 화면 없는 서버에서 matplotlib 이 tkagg 를 고르면 학습 끝의 best.pt 최종 검증(PR 곡선 그리기)에서
# ImportError 로 죽는다 — E1 이 그렇게 끝나 weights/ 복사가 빠졌다 (2026-09-12)
export MPLBACKEND=Agg
PY=/home/se/miniconda3/envs/drone/bin/python
DATA=/home/se/JupyterLAB/drone_dev/data
RAW=$DATA/raw
BK=$DATA/det_v1_fov60
NAME=m1_11m_1280
# 학습 마감 = 12:15 KST. 이후 최종 검증 + 평가 2건(~30분) 해도 회의 전 여유가 남는다
TRAIN_DEADLINE_UTC="2026-09-12 03:15 UTC"

log()  { echo "=== $* [$(date '+%F %T %Z')] ==="; }
fail() { echo "!! $*"; echo "[$(date)] $*" >> overnight_incident.log; log "중단 — 사람 확인 필요"; exit 1; }

log "E1 종료 대기 (PID 196738)"
# 명령줄 패턴(pgrep -f ".*--name ...")은 첫 실행에서 E1 을 못 잡고 바로 통과했다 → PID 로 기다린다
while kill -0 196738 2>/dev/null; do sleep 30; done
log "E1 종료 감지"
tail -3 runs_person/e1_11s_960/results.csv
[ -f runs_person/e1_11s_960/weights/best.pt ] || echo "경고: E1 best.pt 없음"
# 실행 파일이 python 인 것만 본다 — 그냥 pgrep -f 는 이 스크립트를 띄운 셸의 명령줄
# (문자열 "train_person.py" 포함)까지 잡아서 두 번이나 거짓 중단됐다 (2026-09-12)
pgrep -af "train_person\.py" | awk '$2 ~ /python/' | grep . && fail "다른 train_person.py 가 실행 중 — 중복 학습 방지"

# ── 2) 백업 ──
[ -e "$BK" ] && fail "백업 경로가 이미 있다: $BK"
mv "$DATA/det" "$BK" || fail "data/det 백업 이동 실패"
mkdir -p "$DATA/det"
log "백업 완료: $BK"

# ── 3) 재생성 ──
# 배치 스테이징의 json 은 심볼릭 링크다. 레포를 ~/drone_dev → ~/JupyterLAB 로 옮기면서
# 옛 경로를 가리킨 채 끊겨 있었다(2026-09-12 발견) → 매번 현재 원본으로 다시 건다.
for b in NOMAD_b1_10 NOMAD_b11_20 NOMAD_b21_30; do
  for f in activityLabels.json annotations.json metadata.json; do
    ln -sfn "$RAW/NOMAD/$f" "$RAW/$b/$f"
    readlink -e "$RAW/$b/$f" >/dev/null || fail "링크 복구 실패: $b/$f"
  done
done
STAGE=$RAW/NOMAD_sel31_100
if [ ! -d "$STAGE/images" ]; then
  mkdir -p "$STAGE/images"
  ln -sf "$RAW/NOMAD/annotations.json" "$STAGE/annotations.json"
  ln -sf "$RAW/NOMAD/metadata.json" "$STAGE/metadata.json"
  ln -sf "$RAW/NOMAD/activityLabels.json" "$STAGE/activityLabels.json"
  IFS=',' read -ra ARR < nomad_new_actors.txt
  for a in "${ARR[@]}"; do cp -al "$RAW/NOMAD/images/$a" "$STAGE/images/$a" || fail "스테이징 $a 실패"; done
fi
echo "sel31_100 스테이징 배우 수: $(ls "$STAGE/images" | wc -l)"

for spec in NOMAD_b1_10:nomad_actor01_10 NOMAD_b11_20:nomad_actor11_20 \
            NOMAD_b21_30:nomad_actor21_30 NOMAD_sel31_100:nomad_actor_sel31_100; do
  src=${spec%%:*}; dst=${spec##*:}
  log "nomad_prep $dst"
  $PY nomad_prep.py --nomad "$RAW/$src" --out "data/det/$dst" --workers 16 || fail "nomad_prep $dst 실패"
done
log "wisard_prep"
$PY wisard_prep.py --workers 16 || fail "wisard_prep 실패"
$PY make_configs.py || fail "make_configs 실패"

# ── 4) 검증 ──
log "검증"
for d in nomad_actor01_10 nomad_actor11_20 nomad_actor21_30 nomad_actor_sel31_100 wisard; do
  n=$(find data/det/$d/images/train -name '*.jpg' | wc -l)
  v=$(find data/det/$d/images/val -name '*.jpg' | wc -l)
  echo "  $d  train $n / val $v"
  [ "$n" -gt 0 ] && [ "$v" -gt 0 ] || fail "$d 비어 있음"
done
npy=$(find data/det -name '*.npy' | wc -l)
[ "$npy" -eq 0 ] || fail ".npy 캐시가 남아 있다 ($npy개)"
grep -q nomad_actor_sel31_100 configs/data_all.yaml || fail "data_all.yaml 에 신규 배치 없음"
$PY - "$BK" <<'PY' || fail "val 분할이 이전과 다르다"
import json, sys
from pathlib import Path
bk = Path(sys.argv[1])
bad = 0
for d in ["nomad_actor01_10", "nomad_actor11_20", "nomad_actor21_30", "nomad_actor_sel31_100"]:
    old = json.load(open(bk / d / "nomad_prep_stats.json"))["val_actors"]
    new = json.load(open(Path("data/det") / d / "nomad_prep_stats.json"))["val_actors"]
    print(f"  {d} val 배우 {'동일' if old == new else '다름!'} {new}")
    bad += old != new
old = json.load(open(bk / "wisard" / "wisard_prep_stats.json"))["val_flights"]
new = json.load(open("data/det/wisard/wisard_prep_stats.json"))["val_flights"]
print(f"  wisard val 비행 {'동일' if old == new else '다름!'}")
bad += old != new
sys.exit(1 if bad else 0)
PY

# ── 5) 학습 ──
HOURS=$($PY -c "import time,subprocess;e=int(subprocess.check_output(['date','-d','$TRAIN_DEADLINE_UTC','+%s']));print(round((e-time.time())/3600-0.15,2))")
echo "학습 시간 예산: ${HOURS}시간 (캐시 생성 여유 0.15h 제외, 마감 $TRAIN_DEADLINE_UTC)"
$PY -c "import sys; sys.exit(0 if float('$HOURS') >= 3 else 1)" || fail "남은 시간이 3시간 미만 (${HOURS}h)"
log "학습 시작 $NAME"
$PY train_person.py --stage 1 --data configs/data_all.yaml --weights yolo11m.pt \
  --imgsz 1280 --batch 0.85 --name $NAME --workers 8 --cache disk \
  --time "$HOURS" --close-mosaic 5
rc=$?
log "학습 종료 (rc=$rc)"
W=runs_person/$NAME/weights/best.pt
[ -f "$W" ] || fail "best.pt 없음"
tail -3 runs_person/$NAME/results.csv

# ── 6) 평가 ──
log "평가 $NAME @1280"
$PY eval_domain.py --weights "$W" --imgsz 1280 --out metrics/eval_domain_${NAME}.csv > eval_${NAME}.log 2>&1 \
  || echo "경고: $NAME 평가 실패 (eval_${NAME}.log)"
log "평가 e1_11s_960 @960 (새 val 크롭)"
$PY eval_domain.py --weights runs_person/e1_11s_960/weights/best.pt --imgsz 960 \
  --out metrics/eval_domain_e1_11s_960_newval.csv > eval_e1_newval.log 2>&1 \
  || echo "경고: E1 평가 실패 (eval_e1_newval.log)"

log "전체 완료"
