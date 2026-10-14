#!/usr/bin/env bash
# ue_vision_batch.sh — VisionEval 시나리오 × 고도 일괄 촬영 (노트북 Git Bash · 2026-10-14)
#   시나리오마다 UE 를 그 레벨로 띄우고 (우리 settings · -game) → ue_vision_capture.py → UE 종료 → C:\work\ve\<run>.tar.gz
#   사용: bash ue_vision_batch.sh "S1 S2 S3 S4 S5" "16 20"
set -u
export MSYS_NO_PATHCONV=1      # Git Bash 가 /Game/... 맵 경로를 C:/Program Files/Git/Game/... 로 바꾸지 않게
SCENS=${1:-"S1 S2 S3 S4 S5"}
ALTS=${2:-"16 20"}
HERE=$(cd "$(dirname "$0")" && pwd)
UE="/c/Program Files/Epic Games/UE_5.5/Engine/Binaries/Win64/UnrealEditor.exe"
PROJ='C:\work\capstone1_project\drone_simulater\droneSimulator\droneSimulator.uproject'
PY=/c/work/capstone1_project/drone_simulater/.venv/Scripts/python.exe
CONDA_PY=/c/Users/timjj/miniconda3/envs/drone/python.exe          # 재압축 (cv2)
VE=/c/work/ve
mkdir -p "$VE"
cp "$HERE/configs/settings_vision_eval.json" "$VE/settings_vision_eval.json"   # 한글 경로를 UE 인자로 넘기지 않는다

for s in $SCENS; do
  "$UE" "$PROJ" "/Game/VisionEval/Maps/Vision_$s" -game -windowed -ResX=640 -ResY=360 -nosplash -unattended \
      '-settings=C:\work\ve\settings_vision_eval.json' "-abslog=C:\\work\\ve\\unreal_$s.log" &
  ue=$!
  winpid=$(cat /proc/$ue/winpid)
  for i in $(seq 1 150); do netstat -ano | grep -q ":41451 .*LISTEN" && break; sleep 2; done
  sleep 5
  for a in $ALTS; do
    run="${s}_a${a}"
    echo "=== $run $(date +%T)"
    (cd "$HERE" && "$PY" -u ue_vision_capture.py --spec "configs/vision_eval/$s.json" --out "ue_flight/$run" --alt "$a") \
        > "$VE/capture_$run.log" 2>&1
    echo "    capture exit $? · $(tail -n 1 "$VE/capture_$run.log")"
    (cd "$HERE" && "$CONDA_PY" -u ue_vision_repack.py "$run") && echo "    → $VE/${run}_q85.tar.gz"   # 전송용 JPEG 85 만 (디스크)
    echo "    disk free $(df -h /c | tail -1 | awk '{print $4}')"
  done
  taskkill /PID "$winpid" /F > /dev/null 2>&1                  # 이 스크립트가 띄운 UE 만 끈다
  wait $ue 2>/dev/null
  for i in $(seq 1 30); do netstat -ano | grep -q ":41451 .*LISTEN" || break; sleep 1; done
done
echo "=== done $(date +%T)"
