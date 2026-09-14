#!/bin/bash
# wait_and_preprocess.sh(전처리 체인) 완료를 기다린 뒤, 결과가 정상인지 확인하고
# E1~E8 학습 체인을 자동으로 시작한다.
cd /home/se/JupyterLAB/drone_dev/drone_yolo

echo "=== 전처리 체인(PID 194309) 완료 대기 시작 $(date) ==="
while kill -0 194309 2>/dev/null; do sleep 20; done
echo "=== 전처리 체인 프로세스 종료 감지 $(date) ==="
echo "--- wait_and_preprocess.log 마지막 부분 ---"
tail -25 wait_and_preprocess.log

# ── 성공 여부 검증 (실패인데 그냥 학습 시작하는 사고 방지) ──
ok=1
if [ ! -d data/det/nomad_actor_sel31_100/images/train ] || \
   [ -z "$(find data/det/nomad_actor_sel31_100/images/train -name '*.jpg' -print -quit)" ]; then
  echo "!! 신규 배우 전처리 결과 없음/비어있음 — data/det/nomad_actor_sel31_100"
  ok=0
fi
if [ ! -f configs/data_all.yaml ]; then
  echo "!! configs/data_all.yaml 없음"
  ok=0
elif ! grep -q "nomad_actor_sel31_100" configs/data_all.yaml; then
  echo "!! configs/data_all.yaml 에 신규 배치가 반영 안 됨 (make_configs.py 확인 필요)"
  ok=0
fi

if [ "$ok" != "1" ]; then
  echo "=== 검증 실패 — 학습을 시작하지 않는다. 사람 확인 필요 $(date) ===" | tee -a wait_and_train_incident.log
  exit 1
fi

echo "=== 검증 통과. E1~E8 학습 체인 시작 $(date) ==="
n_train=$(find data/det/nomad_actor_sel31_100/images/train -name '*.jpg' | wc -l)
n_val=$(find data/det/nomad_actor_sel31_100/images/val -name '*.jpg' 2>/dev/null | wc -l)
echo "신규 배치: train ${n_train}장 / val ${n_val}장"

# 증강 프로필·데이터가 바뀌어 재실행하므로 기존 E1~E8 결과가 덮어써진다.
# 값을 잃지 않게 타임스탬프 붙여 백업(이미 _old/_crashed 붙은 건 과거 백업이라 건너뜀).
BACKUP_TAG=$(date +%Y%m%d_%H%M)
for n in e1_11s_960 e2_v8s_1280 e3_11s_1280 e7_11m_1280 e8_11l_1280; do
  if [ -d "runs_person/$n" ]; then
    mv "runs_person/$n" "runs_person/${n}_before_${BACKUP_TAG}"
    echo "백업: runs_person/$n -> runs_person/${n}_before_${BACKUP_TAG}"
  fi
done

nohup ./run_remaining_experiments.sh > train_chain_final.log 2>&1 &
disown
echo "학습 체인 PID $!"
echo "=== wait_and_train.sh 종료 $(date) ==="
