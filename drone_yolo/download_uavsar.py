"""
download_uavsar.py — UAV-SAR-Human-Detection-Dataset (Roboflow · hamzaghitri) 받기 (10-31 · 외부 평가셋 E · 평가 전용)

  DJI Air 3S 4K · 15 · 30 m × 45° · 90° · 조건마다 75장 (300장) · 서기 · 앉음 · 누움 · 무리 · 거리 섞임 · 움직임 · 의자 · 가방 (라벨 없음 = 오탐 시험)
  키는 ~/.config/roboflow/api_key 파일에서만 읽음 — 키 · 키가 든 주소 · 내려받기 주소는 출력하지 않음 (DATASETS.md 규칙)
  ⚠ 평가 전용 · 학습에 넣지 않음 · 원본과 파생물 (크롭 · 결과 그림) 은 git 에 올리지 않음
실행  python download_uavsar.py info            # 라이선스 · 버전 · 장수만
      python download_uavsar.py get <버전>       # yolov8 형식 → data/raw/uav_sar/ (이미 있으면 멈춤)
"""
import io, json, sys, time, urllib.request, zipfile
from pathlib import Path

WS, PROJ = "hamzaghitri", "uav-sar-human-detection-dataset"
KEY = (Path.home() / ".config" / "roboflow" / "api_key").read_text().strip()
DEST = Path(__file__).resolve().parents[1] / "data" / "raw" / "uav_sar"


def api(path):
    with urllib.request.urlopen(f"https://api.roboflow.com/{path}?api_key={KEY}", timeout=60) as r:
        return json.load(r)


def info():
    d = api(f"{WS}/{PROJ}")
    p = d.get("project", {})
    keep = {k: p.get(k) for k in ("name", "type", "license", "images", "annotation", "classes", "splits", "public", "created", "updated") if k in p}
    print(json.dumps(keep, ensure_ascii=False, indent=1))
    for v in d.get("versions", []):
        print("버전", v.get("id", "").split("/")[-1], "·", v.get("name"), "· 장", v.get("images"), "· 나눔", v.get("splits"), "· 내보내기", v.get("exports"))
    other = sorted(set(p) - set(keep))
    print("그 밖의 항목 이름:", other)


def get(ver):
    if DEST.exists() and any(DEST.iterdir()):
        sys.exit(f"이미 있음: {DEST} — 지우거나 옮긴 뒤 다시")
    for _ in range(30):                                  # 내보내기를 처음 만들면 준비될 때까지 기다림
        e = api(f"{WS}/{PROJ}/{ver}/yolov8").get("export", {})
        if e.get("link"):
            break
        time.sleep(10)
    else:
        sys.exit("내보내기 링크가 안 생김")
    with urllib.request.urlopen(e["link"], timeout=600) as r:
        z = zipfile.ZipFile(io.BytesIO(r.read()))
    DEST.mkdir(parents=True, exist_ok=True); z.extractall(DEST)
    n = {s: len(list((DEST / s / "images").glob("*"))) for s in ("train", "valid", "test") if (DEST / s / "images").exists()}
    print("받음", DEST, n, "· 합", sum(n.values()))


if __name__ == "__main__":
    info() if sys.argv[1] == "info" else get(sys.argv[2])
