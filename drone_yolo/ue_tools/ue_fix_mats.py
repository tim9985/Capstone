"""ue_fix_mats.py — Mixamo 캐릭터 재질을 '디퓨즈 텍스처 → BaseColor' 단순 재질 인스턴스로 교체."""
import unreal

MEL = unreal.MaterialEditingLibrary
tools = unreal.AssetToolsHelpers.get_asset_tools()
EAL = unreal.EditorAssetLibrary
PARENT = "/Game/Mixamo/M_MixamoDiffuse"

if EAL.does_asset_exist(PARENT):
    parent = EAL.load_asset(PARENT)
else:
    parent = tools.create_asset("M_MixamoDiffuse", "/Game/Mixamo", unreal.Material, unreal.MaterialFactoryNew())
    tex = MEL.create_material_expression(parent, unreal.MaterialExpressionTextureSampleParameter2D, -400, 0)
    tex.set_editor_property("parameter_name", "Diffuse")
    tex.set_editor_property("texture", EAL.load_asset("/Engine/EngineResources/DefaultTexture.DefaultTexture"))
    MEL.connect_material_property(tex, "RGB", unreal.MaterialProperty.MP_BASE_COLOR)
    rough = MEL.create_material_expression(parent, unreal.MaterialExpressionConstant, -400, 300)
    rough.set_editor_property("r", 0.85)
    MEL.connect_material_property(rough, "", unreal.MaterialProperty.MP_ROUGHNESS)
    MEL.recompile_material(parent)
    EAL.save_asset(PARENT)
    print("[fix] 부모 재질 생성")

for ch in ["Ch02_nonPBR", "Ch31_nonPBR", "Ch41_nonPBR"]:
    code = ch.split("_")[0]
    mesh = EAL.load_asset(f"/Game/Mixamo/{ch}/{ch}.{ch}")
    mats = mesh.get_editor_property("materials")
    new = []
    for i, sm in enumerate(mats):
        slot_name = str(sm.get_editor_property("material_slot_name"))
        udim = "1002" if "hair" in slot_name.lower() or i == 1 else "1001"
        tpath = f"/Game/Mixamo/{ch}/{code}_{udim}_Diffuse"
        tex = EAL.load_asset(tpath)
        mi_name = f"MI_{code}_{'hair' if udim == '1002' else 'body'}"
        mi_path = f"/Game/Mixamo/{ch}/{mi_name}"
        mi = EAL.load_asset(mi_path) if EAL.does_asset_exist(mi_path) else tools.create_asset(
            mi_name, f"/Game/Mixamo/{ch}", unreal.MaterialInstanceConstant, unreal.MaterialInstanceConstantFactoryNew())
        MEL.set_material_instance_parent(mi, parent)
        if tex:
            MEL.set_material_instance_texture_parameter_value(mi, "Diffuse", tex)
        EAL.save_asset(mi_path)
        sm.set_editor_property("material_interface", mi)
        new.append(sm)
        print(f"[fix] {ch} slot{i}({slot_name}) <- {mi_name} tex={tpath if tex else 'MISSING'}")
    mesh.set_editor_property("materials", new)
    EAL.save_asset(f"/Game/Mixamo/{ch}/{ch}")
print("[fix] 끝")
