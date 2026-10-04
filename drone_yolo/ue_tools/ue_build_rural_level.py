"""ue_build_rural_level.py — 비행 시험용 '농촌 · 산지 가장자리' 레벨을 만든다 (노트북 · 2026-10-05).

FlyingExampleMap 을 바탕으로 (원본은 저장하지 않음):
  · 바닥 → 잔디 재질 · 흙길(자갈) 띠 2줄
  · 나무 군락 (PCG 샘플 나무 3종 · 묘목) · 덤불 · 바위 · 비닐하우스 2동
  · 사람(Mixamo) — 트인 곳 / 나무 아래(가림) 를 섞어 배치 · 최저 뼈 바닥 맞춤
  · 사람 정답표 → drone_yolo/ue_posture/<LEVEL>_actors.csv (월드 cm · NED m · yaw · 자세 · 캐릭터 · 애니메이션 · 가림 여부)
결과: /Game/Rural/<LEVEL>  (환경변수 RL_LEVEL 기본 RuralSite01 · RL_SEED 기본 5)
"""
import csv
import math
import os
import random
import unreal

LEVEL = os.environ.get("RL_LEVEL", "RuralSite01")
SEED = int(os.environ.get("RL_SEED", "5"))
SRC = "/Game/FlyingCPP/Maps/FlyingExampleMap"
DST = f"/Game/Rural/{LEVEL}"
CSV_OUT = rf"C:\Users\timjj\Desktop\캡스톤\_capstone_repo\drone_yolo\ue_posture\{LEVEL}_actors.csv"
FLOOR_Z = 101.6
HALF = 3000.0                      # 배치 범위 ±30 m (바닥 레이 확인)

EAL = unreal.EditorAssetLibrary
TREES = ["/PCG/SampleContent/SimpleForest/Meshes/PCG_Tree_01", "/PCG/SampleContent/SimpleForest/Meshes/PCG_Tree_02",
         "/PCG/SampleContent/SimpleForest/Meshes/PCG_Tree_03"]
SEEDLING = "/PCG/SampleContent/SimpleForest/Meshes/PCG_Seedling_01"
BOULDER = "/PCG/SampleContent/SimpleForest/Meshes/PCG_Boulder_02"
BUSH = "/Game/StarterContent/Props/SM_Bush"
ROCK = "/Game/StarterContent/Props/SM_Rock"
CUBE = "/Game/Geometry/Meshes/1M_Cube"
M_GRASS = "/Game/Rural/Materials/MI_Ground_Grass"   # ue_make_ground_mats.py (월드 좌표 반복)
M_GRAVEL = "/Game/Rural/Materials/MI_Ground_Gravel"
M_GLASS = "/Game/StarterContent/Materials/M_Glass"

CHARS = ["Ch02_nonPBR", "Ch31_nonPBR", "Ch41_nonPBR", "X_Bot"]
POSES = {
    "Sitting": (["A_Sitting_Idle", "A_Sitting_Dazed"], 5),
    "Crouching": (["A_Crouching_Idle"], 3),
    "Lying": (["A_Sleeping_Idle", "A_Male_Laying_Pose", "A_Laying_Breathless", "A_Laying_Severe_Cough"], 8),
    "Walking": (["A_Walking"], 4),
}
UNDER_TREE_FRAC = 0.4              # 사람 중 나무 아래(가림) 비율

random.seed(SEED)
unreal.EditorLoadingAndSavingUtils.load_map(SRC)
asub = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
WORLD = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()
BASE_JUNK = ("1M_Cube_Chamfer", "Cylinder", "Sphere", "Cone", "1M_Cube")   # 원래 맵의 건물 블록·기둥 (바닥 TemplateFloor 는 남김)
removed = 0
for a in asub.get_all_level_actors():
    if a.get_actor_label().startswith(("Person_", "Test_", "Debris_", "Env_")):
        asub.destroy_actor(a)
        continue
    if isinstance(a, unreal.StaticMeshActor):
        m = a.static_mesh_component.static_mesh
        if m is not None and m.get_name() in BASE_JUNK:
            asub.destroy_actor(a)
            removed += 1
print(f"[rural] 원래 맵 블록·기둥 {removed}개 제거")

# ── 조명: 원래 맵은 파란 하늘 구를 스카이라이트(1.0)가 그대로 주변광으로 넣어 화면이 파랗다 ──
SKY_I = float(os.environ.get("RL_SKY", "1.0"))     # 원래 맵 값 (10-05: 파랗게 보인 건 확인 스크립트의 RGB/BGR 뒤집힘이었다)
SUN_I = float(os.environ.get("RL_SUN", "2.75"))
WHITE_TEMP = float(os.environ.get("RL_WTEMP", "6500"))   # 6500 = 보정 없음
for a in asub.get_all_level_actors():
    if isinstance(a, unreal.SkyLight):
        a.light_component.set_intensity(SKY_I)
    elif isinstance(a, unreal.DirectionalLight):
        a.light_component.set_intensity(SUN_I)
        a.light_component.set_light_color(unreal.LinearColor(1.0, 1.0, 1.0, 1.0))
    elif isinstance(a, unreal.PostProcessVolume):
        s = a.get_editor_property("settings")
        s.set_editor_property("override_white_temp", True)
        s.set_editor_property("white_temp", WHITE_TEMP)
        a.set_editor_property("settings", s)
print(f"[rural] 조명: 스카이라이트 {SKY_I} · 햇빛 {SUN_I} (따뜻한 색) · 화이트밸런스 {WHITE_TEMP:.0f}K")


def load(p):
    a = EAL.load_asset(p)
    if a is None:
        raise RuntimeError(f"에셋 없음: {p}")
    return a


def floor_ok(x, y):
    hit = unreal.SystemLibrary.line_trace_single(
        WORLD, unreal.Vector(x, y, FLOOR_Z + 3000.0), unreal.Vector(x, y, FLOOR_Z - 500.0),
        unreal.TraceTypeQuery.TRACE_TYPE_QUERY1, False, [], unreal.DrawDebugTrace.NONE, True)
    if hit is None:
        return False
    t = hit.to_tuple()
    return bool(t[0]) and abs(t[5].z - FLOOR_Z) < 30.0


def spawn_static(mesh, label, x, y, z, yaw=0.0, scale=(1, 1, 1), mat=None):
    act = asub.spawn_actor_from_object(mesh, unreal.Vector(x, y, z), unreal.Rotator(0, 0, yaw))
    if act is None:
        return None
    act.set_actor_label(label)
    act.rename(label)
    act.set_actor_scale3d(unreal.Vector(*scale))
    if mat is not None:
        act.static_mesh_component.set_material(0, mat)
    return act


# ── 바닥 · 길 ──
grass, gravel, glass = load(M_GRASS), load(M_GRAVEL), load(M_GLASS)
for a in asub.get_all_level_actors():
    if a.get_actor_label() == "Ground" and hasattr(a, "static_mesh_component"):
        n_mat = a.static_mesh_component.get_num_materials()
        for i in range(max(1, n_mat)):
            a.static_mesh_component.set_material(i, grass)
        print(f"[rural] 바닥 재질 → 잔디 ({n_mat}슬롯)")
cube = load(CUBE)
spawn_static(cube, "Env_Path_01", 0, -900, FLOOR_Z - 48, 8, (60, 3, 1), gravel)       # 동서 농로
spawn_static(cube, "Env_Path_02", 1300, 0, FLOOR_Z - 48, 95, (60, 2.5, 1), gravel)    # 남북 농로

# ── 비닐하우스 ──
for i, (x, y, yaw) in enumerate([(-1800, 1700, 10), (-1800, 2300, 10)], 1):
    spawn_static(cube, f"Env_Greenhouse_{i:02d}", x, y, FLOOR_Z + 125, yaw, (14, 4.5, 2.5), glass)

# ── 나무 군락 · 덤불 · 바위 ──
tree_meshes = [load(p) for p in TREES]
seed_m, boulder_m, bush_m, rock_m = load(SEEDLING), load(BOULDER), load(BUSH), load(ROCK)
trees = []
clusters = [(1800, 1800, 900, 28), (-2200, -1800, 800, 22), (2200, -2000, 600, 14), (-500, 2400, 500, 10)]
for ci, (cx, cy, r, cnt) in enumerate(clusters, 1):
    for j in range(cnt):
        for _ in range(50):
            ang, rr = random.uniform(0, 2 * math.pi), r * math.sqrt(random.random())
            x, y = cx + rr * math.cos(ang), cy + rr * math.sin(ang)
            if abs(x) < HALF and abs(y) < HALF and floor_ok(x, y) and all((x - tx) ** 2 + (y - ty) ** 2 > 180 ** 2 for tx, ty in trees):
                break
        else:
            continue
        s = random.uniform(0.8, 1.4)
        spawn_static(random.choice(tree_meshes), f"Env_Tree_{ci}_{j:02d}", x, y, FLOOR_Z, random.uniform(0, 360), (s, s, s))
        trees.append((x, y))
for j in range(40):
    x, y = random.uniform(-HALF, HALF), random.uniform(-HALF, HALF)
    if floor_ok(x, y):
        m = random.choice([bush_m, bush_m, seed_m])
        s = random.uniform(0.7, 1.5)
        spawn_static(m, f"Env_Bush_{j:02d}", x, y, FLOOR_Z, random.uniform(0, 360), (s, s, s))
for j in range(18):
    x, y = random.uniform(-HALF, HALF), random.uniform(-HALF, HALF)
    if floor_ok(x, y):
        s = random.uniform(0.4, 1.2)
        spawn_static(random.choice([rock_m, boulder_m]), f"Env_Rock_{j:02d}", x, y, FLOOR_Z - 10, random.uniform(0, 360), (s, s, s))
print(f"[rural] 나무 {len(trees)} 그루 배치")

# ── 사람 ──
ALIB = unreal.AnimationLibrary


def lowest_bone_z(comp, seq, t):
    names = [str(comp.get_bone_name(i)) for i in range(comp.get_num_bones())]
    local = {}
    for nm in names:
        try:
            local[nm] = ALIB.get_bone_pose_for_time(seq, nm, t, False)
        except Exception:
            local[nm] = None
    world = {}

    def solve(nm):
        if nm in world:
            return world[nm]
        lt, par = local.get(nm), str(comp.get_parent_bone(nm))
        if lt is None:
            world[nm] = None
        elif par in ("None", "") or par not in local:
            world[nm] = lt
        else:
            pw = solve(par)
            world[nm] = lt if pw is None else unreal.MathLibrary.compose_transforms(lt, pw)
        return world[nm]

    zs = [w.translation.z for w in (solve(n) for n in names) if w is not None]
    return min(zs) if zs else 0.0


people, used = [], []


def place_xy(under_tree):
    for _ in range(3000):
        if under_tree and trees:
            tx, ty = random.choice(trees)
            ang, rr = random.uniform(0, 2 * math.pi), random.uniform(60, 220)
            x, y = tx + rr * math.cos(ang), ty + rr * math.sin(ang)
        else:
            x, y = random.uniform(-HALF * 0.8, HALF * 0.8), random.uniform(-HALF * 0.8, HALF * 0.8)
            if any((x - tx) ** 2 + (y - ty) ** 2 < 400 ** 2 for tx, ty in trees):
                continue
        if abs(x) < HALF and abs(y) < HALF and floor_ok(x, y) and all((x - ux) ** 2 + (y - uy) ** 2 > 300 ** 2 for ux, uy in used):
            used.append((x, y))
            return x, y
    raise RuntimeError("사람 배치 공간 부족")


rows = []
for pose, (anims, count) in POSES.items():
    for i in range(1, count + 1):
        ch, an = random.choice(CHARS), random.choice(anims)
        mesh = EAL.load_asset(f"/Game/Mixamo/{ch}/{ch}.{ch}")
        seq = EAL.load_asset(f"/Game/Mixamo/{ch}/Anims/{an}.{an}")
        under = random.random() < UNDER_TREE_FRAC
        x, y = place_xy(under)
        yaw = random.uniform(0, 360)
        act = asub.spawn_actor_from_object(mesh, unreal.Vector(x, y, FLOOR_Z), unreal.Rotator(0, 0, yaw))
        label = f"Person_{pose}_{i:02d}"
        act.set_actor_label(label)
        act.rename(label)
        comp = act.skeletal_mesh_component
        t = seq.get_play_length() * random.uniform(0.3, 0.7)
        comp.set_editor_property("animation_mode", unreal.AnimationMode.ANIMATION_SINGLE_NODE)
        d = unreal.SingleAnimationPlayData()
        d.set_editor_property("anim_to_play", seq)
        d.set_editor_property("saved_looping", False)
        d.set_editor_property("saved_playing", False)
        d.set_editor_property("saved_position", t)
        d.set_editor_property("saved_play_rate", 0.0)
        comp.set_editor_property("animation_data", d)
        z = FLOOR_Z - lowest_bone_z(comp, seq, t) + 6.0
        act.set_actor_location(unreal.Vector(x, y, z), False, False)
        rows.append({"actor": label, "pose_raw": pose.lower(), "character": ch, "anim": an, "anim_t": round(t, 3),
                     "x_cm": round(x, 1), "y_cm": round(y, 1), "z_cm": round(z, 1), "yaw_deg": round(yaw, 1),
                     # AirSim NED (m): x=UE x, y=UE y, z=-(UE z) — PlayerStart 기준 원점 차이는 simGetObjectPose 로 다시 잴 것
                     "ned_x_m": round(x / 100, 3), "ned_y_m": round(y / 100, 3), "ned_z_m": round(-z / 100, 3),
                     "under_tree": int(under)})
        print(f"[rural] {label}: {ch} · {an} · {'나무 아래' if under else '트인 곳'} · ({x:.0f},{y:.0f})")

os.makedirs(os.path.dirname(CSV_OUT), exist_ok=True)
with open(CSV_OUT, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

ok = unreal.EditorLoadingAndSavingUtils.save_map(WORLD, DST)
print(f"[rural] 사람 {len(rows)} · 나무 {len(trees)} · 저장 {DST} -> {ok} · 정답표 {CSV_OUT}")
