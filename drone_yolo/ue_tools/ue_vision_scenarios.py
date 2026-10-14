"""
ue_vision_scenarios.py — 팀 금오공대 맵(KumohMinimal)에서 비전 전체 검증용 시나리오 스펙 생성 (노트북 · 2026-10-14)

시나리오마다 배우(캐릭터 · 애니 · 상의 색 · 위치 NEU m · 방위 · 지면 높이) · 걷는 배우 경로 · 비행 경로를 JSON 으로 쓴다
→ ue_vision_level.py(에디터) 와 ue_vision_capture.py 가 같이 읽는다.
  지면 높이 = terrain.tif 쌍선형 + 0.08 m (UE 지면 = DEM + 0.08 · 10-05 시나리오 5곳 확인) · UE 월드 = NEU (UTM 52N 격자축)
  고도 기준 = 시나리오 중심(center) 지면 — 비행은 그 위 alt m 의 일정 고도 (지형을 따르지 않는다 · 실기체 기압 고도와 같게)

S1 기본 · S2 추적(엇갈림 + 건물 뒤 가림) · S3 요구조자(누움 · 앉음 · 웅크림 · 손 흔들기 · 기어가기 · 한 사람 둘레 선회) · S4 상의색 8명 · S5 빈 장면

실행 (팀 simulation_world 가상환경 — rasterio · pyproj)
  C:/work/capstone1_project/simulation_world/.venv/Scripts/python.exe ue_vision_scenarios.py
"""
import json
import sys
from pathlib import Path

import pyproj
import rasterio

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

TEAM = Path("C:/work/capstone1_project")
OUT = Path(__file__).resolve().parents[1] / "configs" / "vision_eval"
GROUND_OFFSET_M = 0.08

COLORS = {"red": (0.80, 0.04, 0.04), "orange": (0.90, 0.30, 0.02), "yellow": (0.90, 0.78, 0.04),
          "green": (0.06, 0.45, 0.08), "blue": (0.04, 0.14, 0.75), "navy": (0.02, 0.03, 0.22),
          "purple": (0.33, 0.07, 0.55), "pink": (0.95, 0.42, 0.58), "white": (0.85, 0.85, 0.85),
          "gray": (0.33, 0.33, 0.33), "black": (0.02, 0.02, 0.02), "brown": (0.30, 0.14, 0.05)}
ANIM = {"standing": "A_Idle", "standing2": "A_Breathing_Idle", "waving": "A_Waving", "walking": "A_Walking",
        "sitting": "A_Sitting_Idle", "lying": "A_Laying_Breathless", "lying2": "A_Male_Laying_Pose",
        "crouching": "A_Crouching_Idle", "crawling": "A_Crawling"}
# Ch09 · Ch43 은 만화풍(원래 옷 색이 상의 밖에도 강함 — 빨간 모자 · 보라 전신복) → 상의색 정답이 흐려져 쓰지 않는다 (10-14)
CHARS8 = ["Ch02_nonPBR", "Ch17_nonPBR", "Ch31_nonPBR", "Ch41_nonPBR", "Ch49_nonPBR", "claire", "Remy", "X_Bot"]
GRASS = (60.0, -14.0)      # S1 · S3 · S4 · S5 풀밭 중심 (10-05 좌표 비행에서 25 m 안 탐지 ~100 %)


def P(name, ch, pose, color, n, e, yaw, walk=None, anim=None, action=None):
    """pose = 서버 pose_raw (standing|walking|sitting|lying|crouching|crawling) · action = 손 흔들기 등 덧붙임"""
    return {"name": name, "character": ch, "pose": pose, "color": color, "n": n, "e": e, "yaw": yaw,
            "walk": walk, "anim": anim, "action": action}


def zigzag(cn, ce, spacing=18.0, before=55.0, after=20.0):
    """중심 둘레 북쪽 방향 줄 3개 (지그재그 1회 반)"""
    pts = []
    for i, de in enumerate((-spacing, 0.0, spacing)):
        a, b = (cn - before, cn + after) if i % 2 == 0 else (cn + after, cn - before)
        pts += [[a, ce + de], [b, ce + de]]
    return [{"line": pts}]


SCEN = {
    "S1": {"center": GRASS, "flight": zigzag(*GRASS), "actors": [
        P("Person_Standing_01", "Ch17_nonPBR", "standing", "red", 58.0, -18.0, 30),
        P("Person_Standing_02", "Remy", "standing", "blue", 68.0, -4.0, 200, anim=ANIM["standing2"]),
        P("Person_Walking_01", "Ch31_nonPBR", "walking", "yellow", 50.0, -24.0, 90, walk=[(50.0, -24.0), (72.0, -24.0)]),
        P("Person_Sitting_01", "Ch41_nonPBR", "sitting", "green", 63.0, -11.0, 140),
        P("Person_Lying_01", "Ch02_nonPBR", "lying", "white", 54.0, -8.0, 70),
        P("Person_Lying_02", "Ch49_nonPBR", "lying", "black", 66.0, -16.0, 250, anim=ANIM["lying2"])]},
    # S2 — 홀로 선 12 m 건물(NEU N 140.1~150.6 · E −281.4~−273.3 · 35 m 안 다른 건물 · 나무 없음 · 높은 건물 76 m 밖)
    #   기체는 건물 가운데 줄(E −277) 위로 남쪽에서 다가오고 · W1 은 북쪽 벽 7 m 밖(N 158)을 서→동으로 지나며 시선상 건물 뒤로 숨는다
    #   광선 × 건물 직육면체 모의(10-14)로 고른 값: 16 m 보임 2.7 s → 가림 5.8 s → 다시 1.4 s · 20 m 6.0 → 4.8 → 1.7 s
    #   W2 는 서쪽 벽 밖을 남북으로 (W1 · W3 와 엇갈림) · W3 는 남쪽 벽 밖을 동서로
    #   (10-14 첫 자리 NEU 192, 35 는 46.5 m 건물 단지 안뜰이라 벽만 보였다 → 옮김)
    "S2": {"center": (145.0, -278.0), "flight": [{"line": [[70.0, -277.0], [195.0, -277.0], [195.0, -255.0], [100.0, -255.0]]}],
           "actors": [
        P("Person_Walking_01", "Remy", "walking", "orange", 158.0, -315.0, 0, walk=[(158.0, -315.0), (158.0, -265.0)]),
        P("Person_Walking_02", "Ch17_nonPBR", "walking", "blue", 120.0, -290.0, 0, walk=[(120.0, -290.0), (170.0, -290.0)]),
        P("Person_Walking_03", "Ch41_nonPBR", "walking", "white", 134.0, -260.0, 0, walk=[(134.0, -260.0), (134.0, -297.0)])]},
    # S3 — 누움 2 (Lying_01 은 몸 축을 진입 방향(북)과 나란히) · 앉음 · 웅크림 · 걷기 → 진입 → Lying_01 둘레 한 바퀴 → 남쪽으로 다시 통과
    "S3": {"center": (60.0, -14.0), "orbit_target": "Person_Lying_01", "flight": [
        {"line": [[18.0, -14.0], [60.0, -14.0]]}, {"orbit": {"center_actor": "Person_Lying_01", "sweep_deg": 360}},
        {"line": [[60.0, -14.0], [85.0, -14.0], [85.0, 2.0], [25.0, 2.0]]}], "actors": [
        P("Person_Lying_01", "Ch31_nonPBR", "lying", "red", 60.0, -14.0, 90),   # 몸 축 = 북남 (진입 방향과 나란히)
        P("Person_Lying_02", "Ch02_nonPBR", "lying", "blue", 52.0, -22.0, 120, anim=ANIM["lying2"]),
        P("Person_Sitting_01", "Ch41_nonPBR", "sitting", "yellow", 66.0, -6.0, 200),
        P("Person_Crouching_01", "Ch49_nonPBR", "crouching", "green", 54.0, -6.0, 300),
        P("Person_Walking_01", "claire", "walking", "white", 70.0, -24.0, 0, walk=[(70.0, -24.0), (48.0, -24.0)]),
        P("Person_Standing_01", "Ch17_nonPBR", "standing", "purple", 74.0, -10.0, 180, anim=ANIM["waving"], action="waving"),
        P("Person_Crawling_01", "Remy", "crawling", "orange", 46.0, -12.0, 0, walk=[(46.0, -12.0), (46.0, -4.0)])]},
    # S4 — 8색 8명 서기 · 4×2 격자 간격 6 m
    "S4": {"center": GRASS, "flight": zigzag(*GRASS), "actors": [
        P(f"Person_Standing_{i + 1:02d}", CHARS8[i], "standing", c,
          GRASS[0] - 9.0 + 6.0 * (i % 4), GRASS[1] - 3.0 + 6.0 * (i // 4), 45 * i)
        for i, c in enumerate(["red", "orange", "yellow", "green", "blue", "purple", "white", "black"])]},
    "S5": {"center": GRASS, "flight": zigzag(*GRASS), "actors": []},
}


def main():
    cfg = json.loads((TEAM / "simulation_world/runtime/config.json").read_text(encoding="utf-8"))
    ds = rasterio.open(Path(cfg["release"]) / "terrain.tif"); Z = ds.read(1).astype(float); inv = ~ds.transform
    lon0, lat0, z0 = cfg["frame"]["origin"]
    x0, y0 = pyproj.Transformer.from_crs(4326, 32652, always_xy=True).transform(lon0, lat0)

    def ground(n, e):
        c, r = inv * (x0 + e, y0 + n); c -= .5; r -= .5
        i, j = int(r), int(c); fr, fc = r - i, c - j
        d = Z[i, j] * (1 - fr) * (1 - fc) + Z[i, j + 1] * (1 - fr) * fc + Z[i + 1, j] * fr * (1 - fc) + Z[i + 1, j + 1] * fr * fc
        return round(float(d - z0 + GROUND_OFFSET_M), 3)

    OUT.mkdir(parents=True, exist_ok=True)
    for sid, sc in SCEN.items():
        actors = []
        for p in sc["actors"]:
            pose = p["pose"]
            a = {"name": p["name"], "character": p["character"], "anim": p["anim"] or ANIM[pose], "pose_raw": pose,
                 "upper_color": p["color"], "tint": COLORS[p["color"]], "neu": [p["n"], p["e"], ground(p["n"], p["e"])],
                 "yaw_deg": p["yaw"], "moving": bool(p["walk"])}
            if p["walk"]:
                a["path_neu"] = [[n, e, ground(n, e)] for n, e in p["walk"]]
                a["speed_mps"] = 1.2
            if p["action"]:
                a["action"] = p["action"]
            actors.append(a)
        cn, ce = sc["center"]
        spec = {"scenario": sid, "map_src": cfg["level"], "map_dst": f"/Game/VisionEval/Maps/Vision_{sid}",
                "origin": cfg["frame"]["origin"], "spawnNeu": cfg["spawnNeu"], "spawnGeographic": cfg["spawnGeographic"],
                "center_neu": [cn, ce, ground(cn, ce)], "ground_offset_m": GROUND_OFFSET_M,
                "flight_neu": sc["flight"], "actors": actors}
        (OUT / f"{sid}.json").write_text(json.dumps(spec, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"{sid}: 배우 {len(actors)} · 중심 지면 NEU z {spec['center_neu'][2]} · "
              + ", ".join(f"{a['name'][7:]}({a['upper_color']})" for a in actors))


if __name__ == "__main__":
    main()
