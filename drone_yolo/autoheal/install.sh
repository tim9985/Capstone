#!/bin/bash
# install.sh — GPU 자동 복구 서비스 설치 · 제거
#
#   설치 + 켜기 : sudo bash autoheal/install.sh
#   제거        : sudo bash autoheal/install.sh --uninstall
#   잠시 끄기   : rm autoheal/ENABLED        (서비스는 남아 있고 아무것도 안 함. 다시 켜기: touch autoheal/ENABLED)
#   멈춘 뒤 재개: rm autoheal/HALTED         (재부팅 3회 제한에 걸렸을 때, 사람이 확인한 뒤)
#   기록        : autoheal/logs/autoheal.log
#
# 레포의 gpu_autoheal.sh 를 고쳤으면 이 스크립트를 다시 실행해야 반영된다
# (root 로 도는 파일은 일반 계정이 못 고치게 /usr/local/sbin 에 복사해서 쓴다)
set -eu
[ "$(id -u)" -eq 0 ] || { echo "sudo 로 실행하세요: sudo bash $0"; exit 1; }
DIR=$(cd "$(dirname "$0")" && pwd)

if [ "${1:-}" = "--uninstall" ]; then
  systemctl disable --now gpu-autoheal 2>/dev/null || true
  rm -f /etc/systemd/system/gpu-autoheal.service /usr/local/sbin/gpu_autoheal.sh
  systemctl daemon-reload
  echo "제거 완료 (기록 $DIR/logs 와 /var/lib/gpu-autoheal 은 남겨 둠)"
  exit 0
fi

bash -n "$DIR/gpu_autoheal.sh"
install -m 0755 -o root -g root "$DIR/gpu_autoheal.sh" /usr/local/sbin/gpu_autoheal.sh
install -m 0644 -o root -g root "$DIR/gpu-autoheal.service" /etc/systemd/system/gpu-autoheal.service
install -d -m 0755 -o se -g se "$DIR/logs"
install -d -m 0700 /var/lib/gpu-autoheal
runuser -u se -- touch "$DIR/ENABLED"
systemctl daemon-reload
systemctl enable gpu-autoheal
# enable --now 는 이미 도는 서비스를 다시 띄우지 않는다 → restart 로 새 스크립트 반영
# (KillMode=process 라 학습 · 평가 프로세스는 유지된다)
systemctl restart gpu-autoheal
sleep 3
systemctl --no-pager --lines=0 status gpu-autoheal || true
echo
echo "설치 완료 — 기록: $DIR/logs/autoheal.log"
tail -n 20 "$DIR/logs/autoheal.log" 2>/dev/null || true
