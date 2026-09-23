#!/bin/bash
# teamdrive_sync.sh — 팀 공유 드라이브 ↔ 서버 (2026-09-23 · 크론 매시 정각)
#  ① 올리기: 서버 obsidian/ → 드라이브 비젼/obsidian   (볼트의 기준본은 서버 git 이다)
#     --update : 드라이브 쪽이 더 최신인 파일은 덮지 않는다 (팀원이 드라이브에서 고친 것 보호)
#     copy     : 드라이브에만 있는 파일은 지우지 않는다 (병합)
#  ② 받기:  드라이브 전체 → 서버 팀드라이브/   (비젼/obsidian 은 제외 — ①과 도는 고리를 막는다)
#  팀드라이브/ 는 .gitignore — 전자서명·공문서가 있어 공개 저장소에 올리지 않는다
set -u
F=1seoJ7f9Vk9BMP_XQiH3CYpvy62DUFDzE
C=/home/se/JupyterLAB/Capstone
L=$C/drone_yolo/logs/teamdrive_sync.log
exec 9>/tmp/teamdrive_sync.lock; flock -n 9 || exit 0          # 이전 실행이 안 끝났으면 건너뜀
ts(){ TZ=Asia/Seoul date '+%F %T'; }
echo "[$(ts) KST] ① 올리기 시작" >> $L
rclone copy $C/obsidian gdrive:비젼/obsidian --drive-root-folder-id=$F --update \
  --exclude ".obsidian/workspace*.json" --stats-one-line >> $L 2>&1
echo "[$(ts) KST] ② 받기 시작" >> $L
rclone copy gdrive: $C/팀드라이브 --drive-root-folder-id=$F \
  --exclude "비젼/obsidian/**" --transfers 8 --checkers 16 --stats-one-line >> $L 2>&1
echo "[$(ts) KST] 완료" >> $L
