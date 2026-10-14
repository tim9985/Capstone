"""ue_import_mixamo.py — Mixamo FBX 를 /Game/Mixamo 로 일괄 임포트 (이미 있는 캐릭터 · 애니는 건너뜀 · 10-14).
캐릭터마다 자기 스켈레톤으로 넣고, 애니메이션은 캐릭터 스켈레톤마다 따로 넣는다(호환 문제 회피).
"""
import os
import re
import unreal

ROOT = r"C:\Users\timjj\Desktop\캡스톤\mixamo"
CHAR_DIR = os.path.join(ROOT, "characters")
ANIM_DIR = os.path.join(ROOT, "anims")
DEST = "/Game/Mixamo"

tools = unreal.AssetToolsHelpers.get_asset_tools()


def clean(name):
    return re.sub(r"[^A-Za-z0-9_]", "_", os.path.splitext(name)[0]).strip("_")


def run(task):
    tools.import_asset_tasks([task])
    return list(task.get_editor_property("imported_object_paths") or [])


def char_task(fbx, dest):
    ui = unreal.FbxImportUI()
    ui.set_editor_property("import_mesh", True)
    ui.set_editor_property("import_as_skeletal", True)
    ui.set_editor_property("import_animations", False)
    ui.set_editor_property("import_materials", True)
    ui.set_editor_property("import_textures", True)
    ui.set_editor_property("create_physics_asset", False)
    ui.set_editor_property("mesh_type_to_import", unreal.FBXImportType.FBXIT_SKELETAL_MESH)
    t = unreal.AssetImportTask()
    t.set_editor_property("filename", fbx)
    t.set_editor_property("destination_path", dest)
    t.set_editor_property("automated", True)
    t.set_editor_property("save", True)
    t.set_editor_property("replace_existing", True)
    t.set_editor_property("options", ui)
    return t


def anim_task(fbx, dest, skeleton, name):
    ui = unreal.FbxImportUI()
    ui.set_editor_property("import_mesh", False)
    ui.set_editor_property("import_as_skeletal", True)
    ui.set_editor_property("import_animations", True)
    ui.set_editor_property("import_materials", False)
    ui.set_editor_property("import_textures", False)
    ui.set_editor_property("skeleton", skeleton)
    ui.set_editor_property("mesh_type_to_import", unreal.FBXImportType.FBXIT_ANIMATION)
    t = unreal.AssetImportTask()
    t.set_editor_property("filename", fbx)
    t.set_editor_property("destination_path", dest)
    t.set_editor_property("destination_name", name)
    t.set_editor_property("automated", True)
    t.set_editor_property("save", True)
    t.set_editor_property("replace_existing", True)
    t.set_editor_property("options", ui)
    return t


chars = sorted(f for f in os.listdir(CHAR_DIR) if f.lower().endswith(".fbx"))
anims = sorted(f for f in os.listdir(ANIM_DIR) if f.lower().endswith(".fbx"))
print(f"[mixamo] 캐릭터 {len(chars)} · 애니메이션 {len(anims)}")

EAL = unreal.EditorAssetLibrary
for c in chars:
    cname = clean(c)
    cdest = f"{DEST}/{cname}"
    mesh_path = f"{cdest}/{cname}.{cname}"
    if EAL.does_asset_exist(mesh_path):                      # 이미 있는 캐릭터는 다시 넣지 않는다 (고친 재질 보존)
        mesh = EAL.load_asset(mesh_path)
    else:
        mesh = None
        for p in run(char_task(os.path.join(CHAR_DIR, c), cdest)):
            a = EAL.load_asset(p)
            if isinstance(a, unreal.SkeletalMesh):
                mesh = a
    if mesh is None:
        print(f"[mixamo] 캐릭터 임포트 실패: {c}")
        continue
    skel = mesh.get_editor_property("skeleton")
    b = mesh.get_bounds()
    slots = [str(m.get_editor_property("material_slot_name")) for m in mesh.get_editor_property("materials")]
    print(f"[mixamo] 캐릭터 {cname}: 스켈레톤 {skel.get_name()} · 높이(cm)≈{b.box_extent.z * 2:.0f} · 재질칸 {slots}")
    for an in anims:
        aname = f"A_{clean(an)}"
        if EAL.does_asset_exist(f"{cdest}/Anims/{aname}"):
            continue
        got = run(anim_task(os.path.join(ANIM_DIR, an), f"{cdest}/Anims", skel, aname))
        seqs = [p for p in got if isinstance(EAL.load_asset(p), unreal.AnimSequence)]
        print(f"[mixamo]   {aname}: {'OK' if seqs else 'FAIL'}")

print("[mixamo] 임포트 끝")
