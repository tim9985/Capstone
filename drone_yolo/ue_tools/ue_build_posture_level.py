"""ue_build_posture_level.py — Mixamo 배우를 자세별로 배치해 새 레벨로 저장.
원본 FlyingExampleMap 은 저장하지 않는다. 결과: /Game/Posture/<LEVEL>
환경변수 PL_LEVEL (기본 PostureLevel01) · PL_SEED (기본 1)
"""
import math
import os
import random
import unreal

LEVEL = os.environ.get("PL_LEVEL", "PostureLevel01")
SEED = int(os.environ.get("PL_SEED", "1"))
SRC = "/Game/FlyingCPP/Maps/FlyingExampleMap"
DST = f"/Game/Posture/{LEVEL}"
FLOOR_Z = 101.6
AREA_MIN, AREA_MAX = -3000.0, 3000.0
MIN_SEP = 300.0

CHARS = ["Ch02_nonPBR", "Ch31_nonPBR", "Ch41_nonPBR", "X_Bot"]
# 자세 이름 -> (애니메이션 후보, 인원)
POSES = {
    "Sitting": (["A_Sitting_Idle", "A_Sitting_Dazed"], 8),
    "Crouching": (["A_Crouching_Idle"], 6),
    "Lying": (["A_Laying_Breathless", "A_Laying_Severe_Cough", "A_Male_Laying_Pose", "A_Sleeping_Idle"], 10),
    "Walking": (["A_Walking"], 6),
}

random.seed(SEED)
unreal.EditorLoadingAndSavingUtils.load_map(SRC)
actor_sub = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
for a in actor_sub.get_all_level_actors():
    if a.get_actor_label().startswith(("Person_", "Test_", "Debris_")):
        actor_sub.destroy_actor(a)

used = []


WORLD = unreal.EditorLevelLibrary.get_editor_world()


def floor_ok(x, y):
    """위에서 아래로 레이를 쏴서 FLOOR_Z 근처에 평평한 바닥이 있는지 확인."""
    hit = unreal.SystemLibrary.line_trace_single(
        WORLD, unreal.Vector(x, y, FLOOR_Z + 2000.0), unreal.Vector(x, y, FLOOR_Z - 500.0),
        unreal.TraceTypeQuery.TRACE_TYPE_QUERY1, False, [], unreal.DrawDebugTrace.NONE, True)
    if hit is None:
        return False
    t = hit.to_tuple()
    blocking, impact = t[0], t[5]
    return bool(blocking) and abs(impact.z - FLOOR_Z) < 30.0


def rand_xy():
    for _ in range(4000):
        x, y = random.uniform(AREA_MIN, AREA_MAX), random.uniform(AREA_MIN, AREA_MAX)
        if all((x - ux) ** 2 + (y - uy) ** 2 > MIN_SEP ** 2 for ux, uy in used) and floor_ok(x, y):
            used.append((x, y))
            return x, y
    raise RuntimeError("배치 공간 부족 — AREA 를 넓히거나 인원을 줄일 것")


ALIB = unreal.AnimationLibrary
_lowest_cache = {}


def lowest_bone_z(comp, seq, t):
    """그 프레임에서 컴포넌트 공간 기준 가장 낮은 뼈의 높이(cm) — 바닥에 맞추는 데 쓴다."""
    key = (seq.get_path_name(), round(t, 3))
    if key in _lowest_cache:
        return _lowest_cache[key]
    n_bones = comp.get_num_bones()
    names = [comp.get_bone_name(i) for i in range(n_bones)]
    local = {}
    for nm in names:
        try:
            local[str(nm)] = ALIB.get_bone_pose_for_time(seq, nm, t, False)
        except Exception:
            local[str(nm)] = None
    world = {}

    def solve(nm):
        if nm in world:
            return world[nm]
        lt = local.get(nm)
        parent = str(comp.get_parent_bone(nm))
        if lt is None:
            world[nm] = None
            return None
        if parent in ("None", "") or parent not in local:
            world[nm] = lt
        else:
            pw = solve(parent)
            world[nm] = lt if pw is None else unreal.MathLibrary.compose_transforms(lt, pw)
        return world[nm]

    zs = [w.translation.z for w in (solve(str(nm)) for nm in names) if w is not None]
    low = min(zs) if zs else 0.0
    _lowest_cache[key] = low
    return low


n = 0
for pose, (anims, count) in POSES.items():
    for i in range(1, count + 1):
        ch = random.choice(CHARS)
        an = random.choice(anims)
        mesh = unreal.EditorAssetLibrary.load_asset(f"/Game/Mixamo/{ch}/{ch}.{ch}")
        seq = unreal.EditorAssetLibrary.load_asset(f"/Game/Mixamo/{ch}/Anims/{an}.{an}")
        if mesh is None or seq is None:
            print(f"[level] 에셋 없음: {ch} / {an}")
            continue
        x, y = rand_xy()
        rot = unreal.Rotator()
        rot.yaw = random.uniform(0.0, 360.0)
        actor = actor_sub.spawn_actor_from_object(mesh, unreal.Vector(x, y, FLOOR_Z), rot)
        if actor is None:
            print(f"[level] 스폰 실패 {pose} {i}")
            continue
        label = f"Person_{pose}_{i:02d}"
        actor.set_actor_label(label)
        actor.rename(label)
        comp = actor.skeletal_mesh_component
        length = seq.get_play_length()
        t = length * random.uniform(0.3, 0.7)
        comp.set_editor_property("animation_mode", unreal.AnimationMode.ANIMATION_SINGLE_NODE)
        data = unreal.SingleAnimationPlayData()
        data.set_editor_property("anim_to_play", seq)
        data.set_editor_property("saved_looping", False)
        data.set_editor_property("saved_playing", False)
        data.set_editor_property("saved_position", t)
        data.set_editor_property("saved_play_rate", 0.0)
        comp.set_editor_property("animation_data", data)
        # 가장 낮은 뼈가 바닥에서 FLESH cm 위에 오도록 높이 보정 (뼈는 살 안쪽에 있다)
        FLESH = 6.0
        low = lowest_bone_z(comp, seq, t)
        z = FLOOR_Z - low + FLESH
        actor.set_actor_location(unreal.Vector(x, y, z), False, False)
        n += 1
        print(f"[level] {label}: {ch} · {an} @ {t:.2f}s · yaw {rot.yaw:.0f} · 최저뼈 {low:.0f}cm → z {z:.0f}")

world = unreal.EditorLevelLibrary.get_editor_world()
ok = unreal.EditorLoadingAndSavingUtils.save_map(world, DST)
print(f"[level] {n}명 배치 · 저장 {DST} -> {ok}")
