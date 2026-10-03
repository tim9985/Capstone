# ue_tools — UE(Cosys-AirSim) 자세 합성 데이터 준비 (노트북 · 2026-10-03)

`ue_posture_capture.py` 를 돌리기 전에 UE 프로젝트(`Blocks_editor_project_55_33`)에 사람 배우를 준비하는 스크립트 모음.
전부 **UE 에디터를 헤드리스로 띄워** 실행한다. 원본 맵 `FlyingExampleMap.umap` 은 저장하지 않는다.

## 순서

| # | 스크립트 | 하는 일 |
|---|---|---|
| 1 | `ue_import_mixamo.py` | `캡스톤/mixamo/characters/*.fbx` · `anims/*.fbx` → `/Game/Mixamo/<캐릭터>/` (애니메이션은 캐릭터 스켈레톤마다 따로) |
| 2 | `ue_fix_mats.py` | 재질을 "디퓨즈 텍스처 → BaseColor" 단순 재질 인스턴스로 교체 (FBX 재질에 텍스처가 안 붙어 옴) |
| 3 | `ue_flag_skel.py` | `/Game/Mixamo` 재질 전부 `bUsedWithSkeletalMesh=True` |
| 4 | `ue_build_posture_level.py` | 자세별 배우 배치 → `/Game/Posture/<PL_LEVEL>` 저장 (환경변수 `PL_LEVEL` · `PL_SEED`) |
| 5 | (게임 실행 후) `ue_posture_capture.py` | 캡처 — `drone_yolo/` 에서 실행 |
| 검증 | `ue_probe_lying.py <out.jpg>` | 누운 배우를 하나씩 가까이 찍은 모음판 (떠 있는지 확인) |

```powershell
$UE = "C:\Program Files\Epic Games\UE_5.5\Engine\Binaries\Win64"
$P  = "C:\Users\timjj\Desktop\Blocks_editor_project_55_33\Blocks.uproject"
& "$UE\UnrealEditor-Cmd.exe" $P -ExecutePythonScript="<스크립트 절대경로>" -unattended -nullrhi -nosplash
# 캡처: settings 교체 → 게임 모드로 레벨 실행 → 포트 41451 열리면 캡처
Copy-Item drone_yolo\configs\settings_geo.json $HOME\Documents\AirSim\settings.json   # 원본은 settings_backup_before_posture.json
& "$UE\UnrealEditor.exe" $P /Game/Posture/PostureLevel01 -game -windowed -ResX=640 -ResY=360
python ue_posture_capture.py --place level01 --out ue_posture/level01 --tod
# 끝나면 settings 원복 (팀원 비행 시뮬이 쓴다)
```

## 함정 (10-03 실제로 겪은 것)

| 증상 | 원인 | 해결 |
|---|---|---|
| Play 직후 에디터가 꺼지고 포트가 내려감 | 에디터 PIE + ComputerVision 모드에서 `QUIT_EDITOR` | **에디터 Play 대신 `-game` 스탠드얼론**으로 실행 |
| 배우를 못 찾음 ("배우를 찾지 못함") | AirSim 세그멘테이션은 액터 **라벨이 아니라 오브젝트 이름(GetName)** 을 쓴다 | `actor.set_actor_label()` + `actor.rename(label)` |
| 캐릭터가 전부 회색 | 재질에 `bUsedWithSkeletalMesh` 없음 → 게임에서 기본 재질 | `ue_flag_skel.py` (로그: `missing bUsedWithSkeletalMesh=True! Default Material will be used`) |
| 텍스처가 안 붙음 | Mixamo FBX 재질에 텍스처 연결 없음 | `ue_fix_mats.py` |
| 누운 사람이 공중에 뜸 | `Laying Breathless` · `Laying Severe Cough` 는 최저 뼈가 98~104 cm | 배치 때 그 프레임의 **최저 뼈를 바닥에 맞춤** (`lowest_bone_z`) |
| `UnrealEditor-Cmd` exit 3 · "Old world not cleaned up" | 복제한 맵을 바로 `load_map` | 원본 맵을 열어 배우를 놓고 **`save_map(world, 새경로)`** 로 다른 이름 저장 |
| 30명 배치 실패 | 1,800 cm 정사각 · 300 cm 간격엔 안 들어감 | 범위 ±3,000 cm + 위에서 레이로 바닥 확인 |

## 에셋 (git 밖)

- Mixamo FBX: `캡스톤/mixamo/` — 프로젝트 안 사용은 자유, **원본 FBX 재배포 금지**
- 현재: 캐릭터 4 (Ch02 · Ch31 · Ch41 · X Bot) × 애니메이션 8 — 무릎 꿇기 · 서 있기(Idle) 없음
- 캡처 산출물 `drone_yolo/ue_posture/` 는 `.gitignore` — 서버(`drone_yolo/data/pose_cls/ue_<place>/`)로만 보낸다
