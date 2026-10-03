"""/Game/Mixamo 아래 모든 Material 에 bUsedWithSkeletalMesh 를 켜고 재컴파일·저장."""
import unreal
EAL = unreal.EditorAssetLibrary
MEL = unreal.MaterialEditingLibrary
n = 0
for path in EAL.list_assets("/Game/Mixamo", recursive=True, include_folder=False):
    a = EAL.load_asset(path)
    if isinstance(a, unreal.Material):
        if not a.get_editor_property("used_with_skeletal_mesh"):
            a.set_editor_property("used_with_skeletal_mesh", True)
            MEL.recompile_material(a)
            EAL.save_asset(path.split(".")[0])
            n += 1
            print(f"[flag] {a.get_name()} -> used_with_skeletal_mesh")
print(f"[flag] {n}개 수정")
