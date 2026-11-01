"""
rf_get.py — Roboflow Universe 공개 데이터셋 받기 (11-01 · 누움 라벨 데이터 찾기 · 평가 먼저)
  키는 ~/.config/roboflow/api_key 에서만 읽음 — 키 · 키가 든 주소 · 내려받기 주소는 출력 안 함 · 받은 것은 git 밖 data/raw/rf/<이름>
실행  python rf_get.py <workspace> <project> <version> <이름> [형식 yolov8]
"""
import io, json, sys, time, urllib.request, zipfile
from pathlib import Path
KEY = (Path.home() / ".config" / "roboflow" / "api_key").read_text().strip()
ws, pr, ver, name = sys.argv[1:5]; fmt = sys.argv[5] if len(sys.argv) > 5 else "yolov8"
dest = Path(__file__).resolve().parents[1] / "data" / "raw" / "rf" / name
if dest.exists() and any(dest.iterdir()):
    sys.exit(f"이미 있음: {dest}")
for _ in range(40):
    e = json.load(urllib.request.urlopen(f"https://api.roboflow.com/{ws}/{pr}/{ver}/{fmt}?api_key={KEY}", timeout=60)).get("export", {})
    if e.get("link"):
        break
    time.sleep(10)
else:
    sys.exit("내보내기 링크 없음")
z = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(e["link"], timeout=900).read()))
dest.mkdir(parents=True); z.extractall(dest)
print(name, {s: len(list((dest / s / "images").glob("*"))) for s in ("train", "valid", "test") if (dest / s / "images").exists()})
