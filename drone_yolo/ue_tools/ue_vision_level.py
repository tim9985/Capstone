"""
ue_vision_level.py — 팀 KumohMinimal 을 열어 시나리오 배우(Mixamo)를 놓고 우리 맵으로 따로 저장 (UE 에디터 Python · 2026-10-14)

팀 맵 · 팀 에셋은 고치지 않는다 — 결과는 /Game/VisionEval/... (팀 리포 .git/info/exclude 로 git 밖)
  ① 상의 재질 M_VE_Shirt: 디퓨즈 텍스처 + 상의 영역만 지정 색 (명암은 원래 텍스처 밝기로 유지)
     Mixamo nonPBR 은 옷 · 피부가 텍스처 한 장이라, 바인드 포즈(T 포즈) 위치로 상의를 가른다:
     허리(키 0.53) ~ 목(0.82) 사이 · 옆으로 0.24 키 안 (몸통 + 위팔 = 반팔 티)
  ② 캐릭터 × 색 마다 재질 인스턴스
  ③ 배우 스폰 → 이름 = 분할 객체 이름 (AirSim 분할은 actor 이름으로 맞춘다 · rename) · 애니 · 지면 맞춤(가장 낮은 뼈)
  ④ 걷기 애니는 루트 이동이 있으면 복제본에 force_root_lock (제자리 걸음 · 이동은 캡처가 맡음)
  ⑤ save_map(새 경로) · 빌드 보고 <scenario>_build.json (z 보정 · 자연 걷기 속도)

실행: UnrealEditor-Cmd.exe <팀 uproject> -ExecutePythonScript="<이 파일>" -unattended -nullrhi -nosplash -stdout
      시나리오는 환경변수 VE_SPEC (기본 configs/vision_eval/S1.json)
"""
import json
import os

import unreal

EAL, MEL = unreal.EditorAssetLibrary, unreal.MaterialEditingLibrary
ALIB = unreal.AnimationLibrary
tools = unreal.AssetToolsHelpers.get_asset_tools()
HERE = os.path.dirname(os.path.abspath(__file__))
SPEC = os.environ.get("VE_SPEC", os.path.join(HERE, "..", "configs", "vision_eval", "S1.json"))
MAT_DIR = "/Game/VisionEval/Materials"
ANIM_DIR = "/Game/VisionEval/Anims"
PARENT = f"{MAT_DIR}/M_VE_Shirt"
FLESH_CM = 6.0
spec = json.load(open(SPEC, encoding="utf-8"))


def shirt_material():
    if EAL.does_asset_exist(PARENT):                # 매번 새로 만든다 (실패한 실행이 남긴 반쪽 재질 방지)
        EAL.delete_asset(PARENT)
    m = tools.create_asset("M_VE_Shirt", MAT_DIR, unreal.Material, unreal.MaterialFactoryNew())
    E = lambda cls, x, y: MEL.create_material_expression(m, cls, x, y)
    tex = E(unreal.MaterialExpressionTextureSampleParameter2D, -1200, -300)
    tex.set_editor_property("parameter_name", "Diffuse")
    tex.set_editor_property("texture", EAL.load_asset("/Engine/EngineResources/WhiteSquareTexture"))
    pre = E(unreal.MaterialExpressionPreSkinnedPosition, -1600, 200)
    vi = E(unreal.MaterialExpressionVertexInterpolator, -1400, 200)
    MEL.connect_material_expressions(pre, "", vi, "VS")

    def scalar(name, val, y):
        s = E(unreal.MaterialExpressionScalarParameter, -1400, y)
        s.set_editor_property("parameter_name", name); s.set_editor_property("default_value", val)
        return s

    def comp(ch, y):
        c = E(unreal.MaterialExpressionComponentMask, -1200, y)
        for k in "rgba":
            c.set_editor_property(k, k == ch)
        MEL.connect_material_expressions(vi, "PS", c, "")
        return c

    def band(lo_node, val_node, hi_node, y):
        """saturate((v−lo)·0.5) · saturate((hi−v)·0.5) — 2 cm 경계"""
        out = []
        for a, b in ((val_node, lo_node), (hi_node, val_node)):
            sub = E(unreal.MaterialExpressionSubtract, -900, y); MEL.connect_material_expressions(a, "", sub, "A"); MEL.connect_material_expressions(b, "", sub, "B")
            mul = E(unreal.MaterialExpressionMultiply, -750, y); mul.set_editor_property("const_b", 0.5); MEL.connect_material_expressions(sub, "", mul, "A")
            sat = E(unreal.MaterialExpressionSaturate, -600, y); MEL.connect_material_expressions(mul, "", sat, "")
            out.append(sat); y += 60
        p = E(unreal.MaterialExpressionMultiply, -450, y); MEL.connect_material_expressions(out[0], "", p, "A"); MEL.connect_material_expressions(out[1], "", p, "B")
        return p

    zlo, zhi, half = scalar("ZLo", 95.0, 100), scalar("ZHi", 148.0, 160), scalar("XHalf", 43.0, 220)
    neg = E(unreal.MaterialExpressionMultiply, -1200, 280); neg.set_editor_property("const_b", -1.0); MEL.connect_material_expressions(half, "", neg, "A")
    mz = band(zlo, comp("b", 400), zhi, 400)
    mx = band(neg, comp("r", 600), half, 600)
    my = band(neg, comp("g", 800), half, 800)
    m1 = E(unreal.MaterialExpressionMultiply, -250, 600); MEL.connect_material_expressions(mz, "", m1, "A"); MEL.connect_material_expressions(mx, "", m1, "B")
    mask = E(unreal.MaterialExpressionMultiply, -100, 650); MEL.connect_material_expressions(m1, "", mask, "A"); MEL.connect_material_expressions(my, "", mask, "B")
    # 상의 색 = Tint × (0.7 + 0.5 × 밝기) — 주름 명암 유지
    lumw = E(unreal.MaterialExpressionConstant3Vector, -1000, -100); lumw.set_editor_property("constant", unreal.LinearColor(0.3, 0.59, 0.11, 0))
    dot = E(unreal.MaterialExpressionDotProduct, -800, -150); MEL.connect_material_expressions(tex, "RGB", dot, "A"); MEL.connect_material_expressions(lumw, "", dot, "B")
    shade = E(unreal.MaterialExpressionLinearInterpolate, -600, -150); shade.set_editor_property("const_a", 0.7); shade.set_editor_property("const_b", 1.2)
    MEL.connect_material_expressions(dot, "", shade, "Alpha")
    tint = E(unreal.MaterialExpressionVectorParameter, -600, -350); tint.set_editor_property("parameter_name", "Tint")
    tint.set_editor_property("default_value", unreal.LinearColor(0.8, 0.04, 0.04, 1))
    shirt = E(unreal.MaterialExpressionMultiply, -400, -250); MEL.connect_material_expressions(tint, "", shirt, "A"); MEL.connect_material_expressions(shade, "", shirt, "B")
    out = E(unreal.MaterialExpressionLinearInterpolate, -150, -250)
    MEL.connect_material_expressions(tex, "RGB", out, "A"); MEL.connect_material_expressions(shirt, "", out, "B"); MEL.connect_material_expressions(mask, "", out, "Alpha")
    MEL.connect_material_property(out, "", unreal.MaterialProperty.MP_BASE_COLOR)
    rough = E(unreal.MaterialExpressionConstant, -150, 0); rough.set_editor_property("r", 0.85)
    MEL.connect_material_property(rough, "", unreal.MaterialProperty.MP_ROUGHNESS)
    m.set_editor_property("used_with_skeletal_mesh", True)
    MEL.recompile_material(m)
    EAL.save_asset(PARENT)
    print("[ve] M_VE_Shirt 생성")
    return m


def body_slot(mesh):
    mats = mesh.get_editor_property("materials")
    for i, sm in enumerate(mats):
        n = str(sm.get_editor_property("material_slot_name")).lower()
        if "hair" not in n and "joint" not in n:
            return i, sm
    return 0, mats[0]


def shirt_mi(ch, mesh, color, tint, parent):
    path = f"{MAT_DIR}/MI_{ch}_{color}"
    mi = EAL.load_asset(path) if EAL.does_asset_exist(path) else tools.create_asset(
        f"MI_{ch}_{color}", MAT_DIR, unreal.MaterialInstanceConstant, unreal.MaterialInstanceConstantFactoryNew())
    MEL.set_material_instance_parent(mi, parent)
    _, sm = body_slot(mesh)
    src = sm.get_editor_property("material_interface")
    try:
        t = MEL.get_material_instance_texture_parameter_value(src, "Diffuse")
        if t:
            MEL.set_material_instance_texture_parameter_value(mi, "Diffuse", t)
    except Exception:
        pass                                     # X_Bot 등 텍스처 없는 원 재질 → 흰 텍스처 그대로
    b = mesh.get_bounds()
    hgt, bot = 2 * b.box_extent.z, b.origin.z - b.box_extent.z
    MEL.set_material_instance_scalar_parameter_value(mi, "ZLo", bot + 0.53 * hgt)
    MEL.set_material_instance_scalar_parameter_value(mi, "ZHi", bot + 0.82 * hgt)
    MEL.set_material_instance_scalar_parameter_value(mi, "XHalf", 0.24 * hgt)
    MEL.set_material_instance_vector_parameter_value(mi, "Tint", unreal.LinearColor(*tint, 1))
    EAL.save_asset(path)
    print(f"[ve] {path} · 키 {hgt:.0f} cm · 바닥 {bot:.0f}")
    return mi


_low = {}


def lowest_bone_z(comp, seq, t):
    key = (seq.get_path_name(), round(t, 3))
    if key in _low:
        return _low[key]
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
        lt, parent = local.get(nm), str(comp.get_parent_bone(nm))
        if lt is None:
            world[nm] = None
        elif parent in ("None", "") or parent not in local:
            world[nm] = lt
        else:
            pw = solve(parent)
            world[nm] = lt if pw is None else unreal.MathLibrary.compose_transforms(lt, pw)
        return world[nm]

    zs = [w.translation.z for w in (solve(nm) for nm in names) if w is not None]
    _low[key] = min(zs) if zs else 0.0
    return _low[key]


def root_travel(comp, seq):
    """애니 한 바퀴 동안 루트(Hips) 수평 이동 cm"""
    root = str(comp.get_bone_name(0))
    a = ALIB.get_bone_pose_for_time(seq, root, 0.0, False).translation
    b = ALIB.get_bone_pose_for_time(seq, root, seq.get_play_length() - 1e-3, False).translation
    return ((b.x - a.x) ** 2 + (b.y - a.y) ** 2) ** 0.5


def in_place(ch, seq, comp):
    travel = root_travel(comp, seq)
    speed = travel / 100.0 / max(seq.get_play_length(), 1e-3)
    if travel < 30:
        return seq, speed
    dst = f"{ANIM_DIR}/{ch}_{seq.get_name()}_InPlace"
    if not EAL.does_asset_exist(dst):
        EAL.duplicate_asset(seq.get_path_name().split(".")[0], dst)
        dup = EAL.load_asset(dst)
        dup.set_editor_property("force_root_lock", True)
        EAL.save_asset(dst)
    print(f"[ve] {seq.get_name()} 루트 이동 {travel:.0f} cm → 제자리 복제 {dst} · 자연 속도 {speed:.2f} m/s")
    return EAL.load_asset(dst), speed


def main():
    parent = shirt_material()
    unreal.EditorLoadingAndSavingUtils.load_map(spec["map_src"])
    asub = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    report = {"scenario": spec["scenario"], "map": spec["map_dst"], "actors": {}}
    for a in spec["actors"]:
        ch = a["character"]
        mesh = EAL.load_asset(f"/Game/Mixamo/{ch}/{ch}.{ch}")
        seq = EAL.load_asset(f"/Game/Mixamo/{ch}/Anims/{a['anim']}.{a['anim']}")
        if mesh is None or seq is None:
            raise RuntimeError(f"에셋 없음 {ch} / {a['anim']}")
        n, e, gz = a["neu"]
        rot = unreal.Rotator(); rot.yaw = float(a["yaw_deg"])
        actor = asub.spawn_actor_from_object(mesh, unreal.Vector(n * 100, e * 100, gz * 100), rot)
        actor.set_actor_label(a["name"]); actor.rename(a["name"])
        comp = actor.skeletal_mesh_component
        speed = None
        if a["pose_raw"] == "walking":
            seq, speed = in_place(ch, seq, comp)
        freeze = "freeze_t" in a
        t_ref = a.get("freeze_t", seq.get_play_length() * 0.5)
        comp.set_editor_property("animation_mode", unreal.AnimationMode.ANIMATION_SINGLE_NODE)
        data = unreal.SingleAnimationPlayData()
        data.set_editor_property("anim_to_play", seq)
        data.set_editor_property("saved_looping", True)
        data.set_editor_property("saved_playing", not freeze)
        data.set_editor_property("saved_position", t_ref)
        data.set_editor_property("saved_play_rate", 0.0 if freeze else 1.0)
        comp.set_editor_property("animation_data", data)
        idx, _ = body_slot(mesh)
        comp.set_material(idx, shirt_mi(ch, mesh, a["upper_color"], a["tint"], parent))
        dz = FLESH_CM - lowest_bone_z(comp, seq, t_ref)
        actor.set_actor_location(unreal.Vector(n * 100, e * 100, gz * 100 + dz), False, False)
        report["actors"][a["name"]] = {"z_offset_cm": round(dz, 1), "anim": seq.get_path_name(),
                                       "natural_speed_mps": speed, "body_slot": idx}
        print(f"[ve] {a['name']} {ch} {seq.get_name()} {a['upper_color']} z보정 {dz:+.0f} cm")
    world = unreal.EditorLevelLibrary.get_editor_world()
    ok = unreal.EditorLoadingAndSavingUtils.save_map(world, spec["map_dst"])
    out = os.path.splitext(SPEC)[0] + "_build.json"
    json.dump(report, open(out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"[ve] 저장 {spec['map_dst']} -> {ok} · 보고 {out}")


main()
