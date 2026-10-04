"""월드 좌표로 반복되는 바닥 재질 (메시 UV 무시) — 크게 늘린 바닥 메시에도 텍스처가 제대로 깔린다.
UV = WorldPosition.xy / TILE_CM · BaseColor 텍스처 + 노멀 · 거칠기 0.9 · 나나이트 사용 플래그.
결과: /Game/Rural/Materials/MI_Ground_Grass · MI_Ground_Gravel (부모 M_WorldGround)
"""
import unreal

EAL, MEL = unreal.EditorAssetLibrary, unreal.MaterialEditingLibrary
tools = unreal.AssetToolsHelpers.get_asset_tools()
DIR = "/Game/Rural/Materials"
PARENT = f"{DIR}/M_WorldGround"

if EAL.does_asset_exist(PARENT):
    EAL.delete_asset(PARENT)
m = tools.create_asset("M_WorldGround", DIR, unreal.Material, unreal.MaterialFactoryNew())
wp = MEL.create_material_expression(m, unreal.MaterialExpressionWorldPosition, -900, 0)
mask = MEL.create_material_expression(m, unreal.MaterialExpressionComponentMask, -700, 0)
mask.set_editor_property("r", True); mask.set_editor_property("g", True)
mask.set_editor_property("b", False); mask.set_editor_property("a", False)
MEL.connect_material_expressions(wp, "", mask, "")
tile = MEL.create_material_expression(m, unreal.MaterialExpressionScalarParameter, -700, 200)
tile.set_editor_property("parameter_name", "TileCm"); tile.set_editor_property("default_value", 250.0)
div = MEL.create_material_expression(m, unreal.MaterialExpressionDivide, -500, 0)
MEL.connect_material_expressions(mask, "", div, "A")
MEL.connect_material_expressions(tile, "", div, "B")

base = MEL.create_material_expression(m, unreal.MaterialExpressionTextureSampleParameter2D, -250, -150)
base.set_editor_property("parameter_name", "BaseTex")
base.set_editor_property("texture", EAL.load_asset("/Game/StarterContent/Textures/T_Ground_Grass_D"))
MEL.connect_material_expressions(div, "", base, "UVs")
tint = MEL.create_material_expression(m, unreal.MaterialExpressionVectorParameter, -250, -350)
tint.set_editor_property("parameter_name", "Tint"); tint.set_editor_property("default_value", unreal.LinearColor(1, 1, 1, 1))
mul = MEL.create_material_expression(m, unreal.MaterialExpressionMultiply, 0, -200)
MEL.connect_material_expressions(base, "RGB", mul, "A")
MEL.connect_material_expressions(tint, "", mul, "B")
MEL.connect_material_property(mul, "", unreal.MaterialProperty.MP_BASE_COLOR)

nrm = MEL.create_material_expression(m, unreal.MaterialExpressionTextureSampleParameter2D, -250, 150)
nrm.set_editor_property("parameter_name", "NormalTex")
nrm.set_editor_property("sampler_type", unreal.MaterialSamplerType.SAMPLERTYPE_NORMAL)
nrm.set_editor_property("texture", EAL.load_asset("/Game/StarterContent/Textures/T_Ground_Grass_N"))
MEL.connect_material_expressions(div, "", nrm, "UVs")
MEL.connect_material_property(nrm, "RGB", unreal.MaterialProperty.MP_NORMAL)
rough = MEL.create_material_expression(m, unreal.MaterialExpressionConstant, 0, 300)
rough.set_editor_property("r", 0.92)
MEL.connect_material_property(rough, "", unreal.MaterialProperty.MP_ROUGHNESS)
m.set_editor_property("used_with_nanite", True)
m.set_editor_property("used_with_static_lighting", True)
MEL.recompile_material(m)
EAL.save_asset(PARENT)

for name, d, n, tile_cm, tint_c in [
    ("MI_Ground_Grass", "T_Ground_Grass_D", "T_Ground_Grass_N", 250.0, unreal.LinearColor(0.85, 0.95, 0.75, 1)),
    ("MI_Ground_Gravel", "T_Ground_Gravel_D", "T_Ground_Gravel_N", 120.0, unreal.LinearColor(0.75, 0.7, 0.62, 1)),
]:
    path = f"{DIR}/{name}"
    mi = EAL.load_asset(path) if EAL.does_asset_exist(path) else tools.create_asset(
        name, DIR, unreal.MaterialInstanceConstant, unreal.MaterialInstanceConstantFactoryNew())
    MEL.set_material_instance_parent(mi, m)
    MEL.set_material_instance_texture_parameter_value(mi, "BaseTex", EAL.load_asset(f"/Game/StarterContent/Textures/{d}"))
    MEL.set_material_instance_texture_parameter_value(mi, "NormalTex", EAL.load_asset(f"/Game/StarterContent/Textures/{n}"))
    MEL.set_material_instance_scalar_parameter_value(mi, "TileCm", tile_cm)
    MEL.set_material_instance_vector_parameter_value(mi, "Tint", tint_c)
    EAL.save_asset(path)
    print(f"[ground] {name} 저장")
