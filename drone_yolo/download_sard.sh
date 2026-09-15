#!/bin/bash
# download_sard.sh — NOMAD 50 · 70 m 받기가 끝나면 SARD (Roboflow 재배포본) 받기
#
# 서버 학습 계획 3절 5항: SARD 를 1클래스 탐지 학습에 넣는다 (누운 사람 1,096개 · 사람이 직접 붙인 라벨).
# Roboflow `rescuedby/sard-peykp-lxuf9` v1 · yolov8 형식 → data/raw/sard2/search-and-rescue-2
# (make_pose3_dataset.py · eval_pose3.py 가 보는 경로 그대로)
#
# Roboflow API 키가 필요하다. 키는 **파일로만** 넘긴다 — 명령줄 · 로그 · 문서 · 채팅에 남기지 않는다.
#   키 파일: ~/.config/roboflow/api_key (권한 600) — 없으면 생길 때까지 5분마다 다시 본다
#   만들기 (별도 터미널에서):
#     mkdir -p ~/.config/roboflow && (umask 077; read -rsp 'Roboflow key: ' K; echo; printf '%s' "$K" > ~/.config/roboflow/api_key)
#
# 실행 (세션이 닫혀도 계속):
#   cd ~/JupyterLAB/Capstone/drone_yolo && setsid nohup ./download_sard.sh > logs/sard_dl.log 2>&1 < /dev/null &
# 진행 확인: tail -n 5 logs/sard_dl.log   · 끝나면 data/raw/sard_dl.done 이 생긴다
# 실패하면 같은 명령을 다시 실행한다 — 이미 다 받았으면 건너뛴다.
set -u
RAW=/home/se/JupyterLAB/Capstone/data/raw
PY=/home/se/miniconda3/envs/drone/bin/python
KEY=${RF_KEY_FILE:-$HOME/.config/roboflow/api_key}
DEST=$RAW/sard2/search-and-rescue-2
TMP=$RAW/sard2/_dl
now() { date -u '+%F %T'; }
rm -f "$RAW/sard_dl.done"

# 1) NOMAD 받기가 끝날 때까지 기다린다 (성공 · 실패 무관 — SARD 는 따로 받아도 된다)
if pgrep -f 'download_nomad_far\.sh' > /dev/null; then
  echo "[$(now) UTC] NOMAD 50·70 m 받기 끝나기를 기다린다"
  while pgrep -f 'download_nomad_far\.sh' > /dev/null; do sleep 60; done
fi
if [ -f "$RAW/nomad_far_dl.done" ]; then
  echo "[$(now) UTC] NOMAD 받기 완료 확인 (nomad_far_dl.done)"
else
  echo "[$(now) UTC] 경고: NOMAD 받기가 완료 표시 없이 끝났다 — logs/nomad_far_dl.log 확인. SARD 는 계속 받는다"
fi

# 2) 이미 받았으면 건너뛴다
count() { find "$DEST/$1/images" -type f 2>/dev/null | wc -l; }
if [ "$(count train)" = 1386 ] && [ "$(count valid)" = 396 ] && [ "$(count test)" = 198 ]; then
  echo "[$(now) UTC] SARD 가 이미 있다 ($DEST) — 건너뜀"
  touch "$RAW/sard_dl.done"; exit 0
fi

# 3) 키 파일을 기다린다
n=0
while [ ! -s "$KEY" ]; do
  [ $((n % 12)) = 0 ] && echo "[$(now) UTC] Roboflow 키 파일 없음 ($KEY) — 5분마다 다시 본다"
  n=$((n + 1)); sleep 300
done
perm=$(stat -c %a "$KEY")
[ "$perm" = 600 ] || [ "$perm" = 400 ] || echo "[$(now) UTC] 경고: 키 파일 권한이 $perm — chmod 600 $KEY 권장"

# 4) 내보내기 링크를 받아 zip 을 내려받는다 (키는 파이썬 안에서만 읽는다)
mkdir -p "$TMP"
echo "[$(now) UTC] SARD 받기 시작"
RF_KEY_FILE="$KEY" "$PY" - "$TMP/sard.zip" <<'PY' || { echo "[$(now) UTC] SARD 받기 실패 — 같은 명령으로 다시 실행"; exit 1; }
import json, os, sys, time, urllib.error, urllib.request

out = sys.argv[1]
key = open(os.environ["RF_KEY_FILE"]).read().strip()
API = "https://api.roboflow.com/rescuedby/sard-peykp-lxuf9/1/yolov8"
stamp = lambda: time.strftime("%F %T", time.gmtime())

def api_get():
    req = urllib.request.Request(API, headers={"Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:   # 메시지에 키가 들어가지 않는다 (URL 에 키가 없다)
        try:
            return e.code, json.load(e)
        except Exception:
            return e.code, {}

link = None
for i in range(60):                        # 내보내기가 아직 준비 안 됐으면 최대 20분 기다린다
    code, body = api_get()
    link = (body.get("export") or {}).get("link")
    if code == 200 and link:
        break
    if code in (401, 403):
        sys.exit(f"[{stamp()} UTC] 키 거부 ({code}): {(body.get('error') or {}).get('message', '')} — 키 파일을 고친 뒤 다시 실행")
    if code not in (200, 202):
        sys.exit(f"[{stamp()} UTC] API 오류 {code}: {(body.get('error') or {}).get('message', body)}")
    print(f"[{stamp()} UTC] 내보내기 준비 중 ({code}) — 20초 뒤 다시", flush=True)
    time.sleep(20)
if not link:
    sys.exit(f"[{stamp()} UTC] 20분 안에 내보내기 링크를 못 받았다")

for attempt in range(1, 6):
    try:
        with urllib.request.urlopen(link, timeout=120) as r, open(out + ".part", "wb") as f:
            total = int(r.headers.get("Content-Length") or 0)
            got, last = 0, 0
            while chunk := r.read(1 << 20):
                f.write(chunk); got += len(chunk)
                if got - last >= 100 << 20:
                    print(f"[{stamp()} UTC]   {got / 2**20:.0f} / {total / 2**20:.0f} MiB", flush=True); last = got
        os.replace(out + ".part", out)
        print(f"[{stamp()} UTC] zip 받음 {got / 2**20:.1f} MiB", flush=True)
        break
    except Exception as e:
        print(f"[{stamp()} UTC] 내려받기 실패 (시도 {attempt}): {type(e).__name__} — 30초 뒤 다시", flush=True)
        time.sleep(30)
else:
    sys.exit(f"[{stamp()} UTC] zip 5회 실패")
PY

# 5) 풀고 검사한 뒤 제자리로 옮긴다
unzip -tq "$TMP/sard.zip" || { echo "[$(now) UTC] zip 손상 — 다시 실행"; rm -f "$TMP/sard.zip"; exit 1; }
rm -rf "$TMP/x" && mkdir -p "$TMP/x" && unzip -q "$TMP/sard.zip" -d "$TMP/x"
SRC=$TMP/x
[ -d "$SRC/train" ] || SRC=$(dirname "$(find "$TMP/x" -maxdepth 3 -type d -name train | head -1)")
tr=$(find "$SRC/train/images" -type f | wc -l); va=$(find "$SRC/valid/images" -type f | wc -l); te=$(find "$SRC/test/images" -type f | wc -l)
echo "[$(now) UTC] 장수 train $tr · valid $va · test $te (기대 1386 · 396 · 198)"
if [ "$tr" != 1386 ] || [ "$va" != 396 ] || [ "$te" != 198 ]; then
  echo "[$(now) UTC] 장수가 문서와 다르다 — $SRC 를 그대로 두고 멈춘다 (옮기지 않음)"; exit 1
fi
if [ -e "$DEST" ]; then
  echo "[$(now) UTC] $DEST 가 이미 있다 (불완전) — 덮어쓰지 않고 $SRC 에 둔 채 멈춘다"; exit 1
fi
mkdir -p "$(dirname "$DEST")" && mv "$SRC" "$DEST"

# 6) 서버 학습 계획 3절 확인 항목: 재배포본 해상도 · 클래스
"$PY" - "$DEST" <<'PY'
import sys, collections
from pathlib import Path
import cv2
root = Path(sys.argv[1])
print("  data.yaml:", " ".join(l.strip() for l in (root / "data.yaml").read_text().splitlines() if l.startswith(("nc", "names"))))
for split in ("train", "valid", "test"):
    sizes = collections.Counter()
    for p in (root / split / "images").iterdir():
        im = cv2.imread(str(p))
        if im is not None:
            sizes[f"{im.shape[1]}x{im.shape[0]}"] += 1
    boxes = sum(1 for t in (root / split / "labels").glob("*.txt") for l in t.read_text().splitlines() if l.strip())
    print(f"  {split}: 해상도 {dict(sizes.most_common(5))} · 박스 {boxes}")
PY

rm -rf "$TMP"
echo "[$(now) UTC] 완료 → $DEST"
touch "$RAW/sard_dl.done"
