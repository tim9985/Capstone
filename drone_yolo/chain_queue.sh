#!/bin/bash
# chain_queue.sh — GPU 를 쉬게 두지 않는 장기 큐 (2026-09-22)
# · 완료 판정은 **결과 파일**로만 (pgrep -f 는 자기 셸을 잡는다 · CLAUDE.md §4-17)
# · 시각은 KST (서버는 UTC · §4-18)
# · 각 단계는 실패해도 다음으로 넘어간다 — GPU 가 멈추지 않게
cd /home/se/JupyterLAB/Capstone/drone_yolo
P=/home/se/miniconda3/envs/drone/bin/python
Q=logs/QUEUE.md
log(){ echo "[$(TZ=Asia/Seoul date '+%F %T') KST] $*" | tee -a $Q; }

done30(){ [ -f "runs_person/$1/results.csv" ] && [ "$(awk -F, 'END{print $1}' runs_person/$1/results.csv)" = "$2" ]; }
wait_done(){  # $1=run이름 $2=에폭 $3=최대분
  for i in $(seq 1 ${3:-1200}); do done30 "$1" "$2" && return 0; sleep 60; done; return 1; }

log "큐 시작"

# ── ① 선행: chain_auto(D 평가·층화) 가 끝날 때까지
log "① chain_auto 완료 대기 (metrics/AUTO_RESULT.md 의 '## Unicamp')"
for i in $(seq 1 400); do grep -q "^## Unicamp" metrics/AUTO_RESULT.md 2>/dev/null && break; sleep 60; done
log "   → 진행"

# ── ② P : 장소 분리 학습  (NFR-V03 을 처음으로 잴 수 있게 한다)
if ! done30 v3_place 30; then
  log "② P 시작 — 장소 분리 (Carnation·Karen 제외 30,520장)"
  $P train_person.py --stage 1 --data configs/data_v3_place_neg.yaml --weights yolo11m.pt \
     --imgsz 1280 --batch 8 --epochs 30 --patience 100 --close-mosaic 0 \
     --optimizer SGD --lr0 0.01 --scale 0.3 --translate 0.15 --name v3_place \
     >> logs/q_v3_place.log 2>&1
fi
log "② P 종료 — test_v2 로 평가"
$P eval_fullframe.py --weights runs_person/v3_place/weights/best.pt --imgsz 1920 \
   --testsets data/test_v2 >> logs/q_v3_place_eval.log 2>&1 || log "   평가 실패(계속)"

# ── ③ P2 헤드 : 16~25 px 양성 앵커 부족을 구조로 푼다
if ! done30 p2_11m 30; then
  log "③ P2 헤드 시작 (stride 4 추가 · 앵커 4배)"
  $P - <<'PYX' >> logs/q_p2.log 2>&1
from ultralytics import YOLO
m = YOLO("configs/models/yolo11-p2.yaml")   # scale 은 이름으로 못 주므로 직접 m 스케일 파일 사용
m.train(data="configs/data_fov.yaml", imgsz=1280, batch=6, epochs=30, patience=100,
        close_mosaic=0, optimizer="SGD", lr0=0.01, scale=0.3, translate=0.15,
        degrees=180, flipud=0.5, fliplr=0.5, cache="disk",
        project="/home/se/JupyterLAB/Capstone/drone_yolo/runs_person", name="p2_11m", exist_ok=True)
PYX
fi
log "③ P2 종료 — 가림 40칸 평가"
$P eval_fov.py --weights runs_person/p2_11m/weights/best.pt --tag fovbudget >> logs/q_p2_eval.log 2>&1 || log "   평가 실패(계속)"

# ── ④ B : 우리가 1280 으로 VisDrone 사전학습 → 우리 데이터
if [ ! -s runs_person/pre_visdrone_11m/weights/best.pt ]; then
  log "④-1 VisDrone 1280 사전학습 (약 3 h)"
  $P -m ultralytics.cfg.__init__ 2>/dev/null
  yolo detect train data=configs/visdrone.yaml model=yolo11m.pt imgsz=1280 epochs=30 \
      batch=8 optimizer=SGD lr0=0.01 close_mosaic=0 cache=disk \
      project=/home/se/JupyterLAB/Capstone/drone_yolo/runs_person name=pre_visdrone_11m exist_ok=True \
      >> logs/q_pre_vd.log 2>&1
fi
if [ -s runs_person/pre_visdrone_11m/weights/best.pt ] && ! done30 vd1280_11m 30; then
  log "④-2 B 학습 시작"
  $P train_person.py --stage 1 --data configs/data_fov.yaml \
     --weights runs_person/pre_visdrone_11m/weights/best.pt \
     --imgsz 1280 --batch 8 --epochs 30 --patience 100 --close-mosaic 0 \
     --optimizer SGD --lr0 0.01 --scale 0.3 --translate 0.15 --name vd1280_11m \
     >> logs/q_vd1280.log 2>&1
  $P eval_fov.py --weights runs_person/vd1280_11m/weights/best.pt --tag fovbudget >> logs/q_vd1280_eval.log 2>&1
fi

log "큐 완료 — 요약은 metrics/AUTO_RESULT.md · logs/QUEUE.md"
