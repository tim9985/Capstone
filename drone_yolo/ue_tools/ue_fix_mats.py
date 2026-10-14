"""ue_fix_mats.py — Mixamo 캐릭터 재질을 '디퓨즈 텍스처 → BaseColor' 단순 재질 인스턴스로 교체 (10-14 일반화).

FBX 로 들어온 재질은 텍스처가 안 이어지거나 스켈레탈 플래그가 없어 회색으로 나온다 → 칸마다 MI_<캐릭터>_<칸>.
칸 → 텍스처 고르는 규칙 (캐릭터 폴더 안 *Diffuse* 텍스처 중)
  ① 칸 이름 낱말이 든 텍스처 (Remy: Topmat → Remy_Top_Diffuse)
  ② nonPBR UDIM: hair 칸 → *_1002_Diffuse · 아니면 칸 순서대로 1001, 1002 …
  ③ 디퓨즈가 한 장뿐이면 모든 칸에 (claire)
텍스처가 없는 캐릭터(X_Bot)는 원래 재질을 둔다. 뼈 기준 키(cm)도 찍는다 — 스케일이 이상한 캐릭터 확인용.
"""
import re

import unreal

MEL = unreal.MaterialEditingLibrary
tools = unreal.AssetToolsHelpers.get_asset_tools()
EAL = unreal.EditorAssetLibrary
ALIB = unreal.AnimationLibrary
ROOT = "/Game/Mixamo"
PARENT = f"{ROOT}/M_MixamoDiffuse"

if EAL.does_asset_exist(PARENT):
    parent = EAL.load_asset(PARENT)
else:
    parent = tools.create_asset("M_MixamoDiffuse", ROOT, unreal.Material, unreal.MaterialFactoryNew())
    tex = MEL.create_material_expression(parent, unreal.MaterialExpressionTextureSampleParameter2D, -400, 0)
    tex.set_editor_property("parameter_name", "Diffuse")
    tex.set_editor_property("texture", EAL.load_asset("/Engine/EngineResources/DefaultTexture.DefaultTexture"))
    MEL.connect_material_property(tex, "RGB", unreal.MaterialProperty.MP_BASE_COLOR)
    rough = MEL.create_material_expression(parent, unreal.MaterialExpressionConstant, -400, 300)
    rough.set_editor_property("r", 0.85)
    MEL.connect_material_property(rough, "", unreal.MaterialProperty.MP_ROUGHNESS)
    parent.set_editor_property("used_with_skeletal_mesh", True)
    MEL.recompile_material(parent)
    EAL.save_asset(PARENT)
    print("[fix] 부모 재질 생성")


def bone_height(mesh, folder):
    """Idle(없으면 아무 애니) 0초에서 뼈 위치 최고 z (cm) — bounds 가 부풀려진 캐릭터 확인용"""
    seqs = [EAL.load_asset(p) for p in EAL.list_assets(f"{folder}/Anims", recursive=False, include_folder=False)]
    seqs = [s for s in seqs if isinstance(s, unreal.AnimSequence)]
    if not seqs:
        return None
    seq = next((s for s in seqs if s.get_name() == "A_Idle"), seqs[0])
    skel_names = [str(n) for n in ALIB.get_animation_track_names(seq)]
    world, top = {}, 0.0
    comp = unreal.SkeletalMeshComponent()
    comp.set_skeletal_mesh_asset(mesh) if hasattr(comp, "set_skeletal_mesh_asset") else comp.set_skeletal_mesh(mesh)

    def solve(nm):
        if nm in world:
            return world[nm]
        lt = ALIB.get_bone_pose_for_time(seq, nm, 0.0, False)
        par = str(comp.get_parent_bone(nm))
        world[nm] = lt if par in ("None", "") else unreal.MathLibrary.compose_transforms(lt, solve(par))
        return world[nm]

    for nm in skel_names:
        try:
            top = max(top, solve(nm).translation.z)
        except Exception:
            pass
    return top


for folder in EAL.list_assets(ROOT, recursive=False, include_folder=True):
    folder = folder.rstrip("/")
    ch = folder.split("/")[-1]
    mesh_path = f"{folder}/{ch}"
    if not EAL.does_asset_exist(mesh_path):
        continue
    mesh = EAL.load_asset(mesh_path)
    if not isinstance(mesh, unreal.SkeletalMesh):
        continue
    texs = [p.split(".")[0] for p in EAL.list_assets(folder, recursive=False, include_folder=False)
            if re.search(r"diffuse", p.split("/")[-1], re.I)]
    h = bone_height(mesh, folder)
    print(f"[fix] {ch}: bounds 키 {mesh.get_bounds().box_extent.z * 2:.0f} cm · 뼈 최고 {h if h is None else round(h)} cm · 디퓨즈 {len(texs)}")
    if not texs:
        continue
    mats, new = mesh.get_editor_property("materials"), []
    for i, sm in enumerate(mats):
        slot = str(sm.get_editor_property("material_slot_name"))
        word = re.sub(r"(_?mat\d*$)|(^ch_?\d+_)|(\d+$)", "", slot, flags=re.I).lower()
        pick = next((t for t in texs if word and re.search(rf"[_/]{word}[_]", t.lower())), None)
        if pick is None and any("_100" in t for t in texs):
            udim = "1002" if "hair" in slot.lower() else f"{1001 + i}"
            pick = next((t for t in texs if f"_{udim}_" in t), None) or next((t for t in texs if "_1001_" in t), None)
        if pick is None:
            pick = texs[0] if len(texs) == 1 else None
        if pick is None:
            new.append(sm); print(f"[fix]   칸{i} {slot}: 텍스처 못 찾음 — 원래 재질"); continue
        mi_name = f"MI_{ch}_{re.sub(r'[^A-Za-z0-9]', '_', slot)}"
        mi_path = f"{folder}/{mi_name}"
        mi = EAL.load_asset(mi_path) if EAL.does_asset_exist(mi_path) else tools.create_asset(
            mi_name, folder, unreal.MaterialInstanceConstant, unreal.MaterialInstanceConstantFactoryNew())
        MEL.set_material_instance_parent(mi, parent)
        MEL.set_material_instance_texture_parameter_value(mi, "Diffuse", EAL.load_asset(pick))
        EAL.save_asset(mi_path)
        sm.set_editor_property("material_interface", mi)
        new.append(sm)
        print(f"[fix]   칸{i} {slot} <- {pick.split('/')[-1]}")
    mesh.set_editor_property("materials", new)
    EAL.save_asset(mesh_path)

# 남은 원래 재질(X_Bot 등)도 스켈레탈 메시에 쓸 수 있게
for path in EAL.list_assets(ROOT, recursive=True, include_folder=False):
    a = EAL.load_asset(path)
    if isinstance(a, unreal.Material) and not a.get_editor_property("used_with_skeletal_mesh"):
        a.set_editor_property("used_with_skeletal_mesh", True)
        MEL.recompile_material(a)
        EAL.save_asset(path.split(".")[0])
print("[fix] 끝")
