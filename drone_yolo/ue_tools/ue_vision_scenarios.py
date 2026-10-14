"""
ue_vision_scenarios.py — 팀 금오공대 맵(KumohMinimal)에서 비전 전체 검증용 시나리오 스펙 생성 (노트북 · 2026-10-14)

배우마다 캐릭터 · 애니 · 상의 색 · 위치(NEU m) · 방위 · 지면 높이(terrain.tif 쌍선형 + 0.08 m) 와
걷는 배우의 경로(지면 높이 포함)를 JSON 으로 쓴다 → ue_vision_level.py(에디터) 와 ue_vision_capture.py 가 같이 읽는다.
UE 지면 = DEM + 0.08 m (10-05 시나리오 5곳 확인) · UE 월드 = NEU (UTM 52N 격자축 · cm)

실행 (팀 simulation_world 가상환경 — rasterio · pyproj)
  C:/work/capstone1_project/simulation_world/.venv/Scripts/python.exe ue_vision_scenarios.py
"""
import json
import math
import sys
from pathlib import Path

import pyproj
import rasterio

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

TEAM = Path("C:/work/capstone1_project")
OUT = Path(__file__).resolve().parents[1] / "configs" / "vision_eval"
GROUND_OFFSET_M = 0.08

# 상의 12색 (서버 판별 이름) → BaseColor 선형 값
COLORS = {"red": (0.80, 0.04, 0.04), "orange": (0.90, 0.30, 0.02), "yellow": (0.90, 0.78, 0.04),
          "green": (0.06, 0.45, 0.08), "blue": (0.04, 0.14, 0.75), "navy": (0.02, 0.03, 0.22),
          "purple": (0.33, 0.07, 0.55), "pink": (0.95, 0.42, 0.58), "white": (0.85, 0.85, 0.85),
          "gray": (0.33, 0.33, 0.33), "black": (0.02, 0.02, 0.02), "brown": (0.30, 0.14, 0.05)}

# S1 기본 — 통합 시나리오 지점(NEU 60, −15) 둘레 풀밭 · run01 에서 25 m 안 탐지 거의 100 % 확인한 곳
S1 = [
    # 이름                 캐릭터          애니                     색       N      E      방위°
    ("Person_Standing_01", "Ch02_nonPBR", "A_Walking", "red", 58.0, -18.0, 30, "standing"),
    ("Person_Standing_02", "X_Bot", "A_Walking", "blue", 68.0, -4.0, 200, "standing"),
    ("Person_Walking_01", "Ch31_nonPBR", "A_Walking", "yellow", 50.0, -24.0, 90, "walking"),
    ("Person_Sitting_01", "Ch41_nonPBR", "A_Sitting_Idle", "green", 63.0, -11.0, 140, "sitting"),
    ("Person_Lying_01", "Ch02_nonPBR", "A_Laying_Breathless", "white", 54.0, -8.0, 70, "lying"),
    ("Person_Lying_02", "Ch31_nonPBR", "A_Male_Laying_Pose", "black", 66.0, -16.0, 250, "lying"),
]
# 걷는 배우 경로 (NEU 꼭짓점 · 왕복) · 속도 m/s
WALK = {"Person_Walking_01": ([(50.0, -24.0), (72.0, -24.0)], 1.2)}
# 서 있는 배우: 받은 Mixamo 애니에 Idle(서기)이 없어 걷기 애니 0 초에 멈춘 자세로 대신한다 (freeze_t)
STAND_FREEZE_T = 0.0


def main():
    cfg = json.loads((TEAM / "simulation_world/runtime/config.json").read_text(encoding="utf-8"))
    ds = rasterio.open(Path(cfg["release"]) / "terrain.tif"); Z = ds.read(1).astype(float); inv = ~ds.transform
    lon0, lat0, z0 = cfg["frame"]["origin"]
    x0, y0 = pyproj.Transformer.from_crs(4326, 32652, always_xy=True).transform(lon0, lat0)

    def ground(n, e):
        c, r = inv * (x0 + e, y0 + n); c -= .5; r -= .5
        i, j = int(r), int(c); fr, fc = r - i, c - j
        d = Z[i, j] * (1 - fr) * (1 - fc) + Z[i, j + 1] * (1 - fr) * fc + Z[i + 1, j] * fr * (1 - fc) + Z[i + 1, j + 1] * fr * fc
        return round(d - z0 + GROUND_OFFSET_M, 3)

    actors = []
    for name, ch, anim, color, n, e, yaw, pose in S1:
        a = {"name": name, "character": ch, "anim": anim, "pose_raw": pose, "upper_color": color,
             "tint": COLORS[color], "neu": [n, e, ground(n, e)], "yaw_deg": yaw, "moving": name in WALK}
        if name in WALK:
            pts, speed = WALK[name]
            a["path_neu"] = [[pn, pe, ground(pn, pe)] for pn, pe in pts]
            a["speed_mps"] = speed
        if pose == "standing":
            a["freeze_t"] = STAND_FREEZE_T
        actors.append(a)
    sn = cfg["spawnNeu"]
    spec = {"scenario": "S1", "map_src": cfg["level"], "map_dst": "/Game/VisionEval/Maps/Vision_S1",
            "origin": cfg["frame"]["origin"], "spawnNeu": sn, "spawnGeographic": cfg["spawnGeographic"],
            "home_ground_neu_z": ground(sn[0], sn[1]), "ground_offset_m": GROUND_OFFSET_M, "actors": actors}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "S1.json").write_text(json.dumps(spec, indent=2, ensure_ascii=False), encoding="utf-8")
    for a in actors:
        print(f"{a['name']:20} {a['character']:12} {a['anim']:20} {a['upper_color']:7} NEU {a['neu']}")
    print(f"홈 지면 NEU z {spec['home_ground_neu_z']} → {OUT / 'S1.json'}")


if __name__ == "__main__":
    main()
