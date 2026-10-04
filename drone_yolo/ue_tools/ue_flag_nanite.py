"""/Game/StarterContent 재질에 bUsedWithNanite 를 켠다 — 이 프로젝트의 바닥·큐브는 나나이트 메시라
플래그가 없으면 게임에서 기본 재질(청록·흰색)로 그려진다. 그리고 원래 맵의 정적 메시 목록을 찍는다."""
import unreal
EAL, MEL = unreal.EditorAssetLibrary, unreal.MaterialEditingLibrary
n = 0
for path in EAL.list_assets("/Game/StarterContent", recursive=True, include_folder=False):
    a = EAL.load_asset(path)
    if isinstance(a, unreal.Material) and not a.get_editor_property("used_with_nanite"):
        a.set_editor_property("used_with_nanite", True)
        MEL.recompile_material(a)
        EAL.save_asset(path.split(".")[0])
        n += 1
print(f"[nanite] 재질 {n}개 수정")

unreal.EditorLoadingAndSavingUtils.load_map("/Game/FlyingCPP/Maps/FlyingExampleMap")
asub = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
from collections import Counter
cnt = Counter()
for a in asub.get_all_level_actors():
    if isinstance(a, unreal.StaticMeshActor):
        m = a.static_mesh_component.static_mesh
        cnt[(m.get_path_name() if m else "None")] += 1
for k, v in cnt.most_common(20):
    print(f"[base] {v:3d} × {k}")
