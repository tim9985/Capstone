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

## 농촌 비행 시험 레벨 · 실제 비행 (10-05)

| # | 스크립트 | 하는 일 |
|---|---|---|
| 1 | `ue_flag_nanite.py` | StarterContent 재질에 `bUsedWithNanite` (바닥·큐브가 나나이트라 없으면 기본 재질) |
| 2 | `ue_make_ground_mats.py` | 월드 좌표로 반복되는 지면 재질 (크게 늘린 바닥 메시에도 텍스처가 깔린다) |
| 3 | `ue_build_rural_level.py` | `/Game/Rural/RuralSite01` — 원래 맵 블록·기둥 제거 · 잔디 · 흙길 · 나무 74 · 덤불 · 바위 · 비닐하우스 · 사람 20 (나무 아래 약 40 %) + 정답표 `ue_posture/RuralSite01_actors.csv` |
| 4 | `../ue_flight_sequence.py` | SimpleFlight 로 실제 비행 — 지그재그 왕복 · 프레임 · 라벨 · `flight.csv` (시각 · NED · 위경도 · roll/pitch/yaw) |

```powershell
Copy-Item drone_yolo\configs\settings_flight.json $HOME\Documents\AirSim\settings.json
& "$UE\UnrealEditor.exe" $P /Game/Rural/RuralSite01 -game -windowed -ResX=640 -ResY=360
python ue_flight_sequence.py --out ue_flight/rural01 --alt 20 --mount 45 --speed 4 --fps 5
```

- 결과 (10-05): 134 s · 341 프레임 · **2.5 fps** (1080p 장면+분할 두 장 · RTX 3050) · 사람 20/20 이 한 번 이상 · 사람별 2~93 프레임 (중앙 44) · 17/20 은 두 방향 이상에서 다시 봄
- ⚠ 카메라는 **기체에 고정** — 선회 때 |roll| 최대 25.8° · |pitch| 29.3° → 좌표 계산은 기체 자세를 합쳐서 · 안정 프레임은 roll/pitch 로 골라서
- ⚠ 확인용 사진: AirSim 이미지는 **RGB** — PIL 로 저장할 때 뒤집지 않는다 (뒤집으면 잔디가 청록으로 보인다 · 10-05 헛짚음)
- 나무는 PCG 샘플(저폴리) — 실제 산림처럼 보이려면 Megascans 등 필요

## MCP (ue-mcp · Remote Control) — 10-05 확인

- `C:\Users\timjj\tools\ue-mcp` (빌드됨) · Blocks.uproject 에 RemoteControl · PCG 활성 (원본 `Blocks.uproject.bak_before_mcp`)
- 에디터를 **GUI 로** 띄우면 HTTP 30010 (127.0.0.1) · WS 30020 (**0.0.0.0**) 이 열린다
- 됨: `ue_ping` · `ue_call_function` 으로 5.5 서브시스템 호출 (예: `/Script/UnrealEd.Default__EditorActorSubsystem` · `GetAllLevelActors`)
- 안 됨: `ue_get_current_level` · `ue_get_all_actors` · `ue_execute_console_command` — 5.5 에서 막힌 `EditorLevelLibrary` 를 부른다
- 원격 파이썬 실행(`bEnableRemotePythonExecution`)은 **켜지 않음** — WS 가 0.0.0.0 이라 같은 네트워크에서 에디터 코드 실행이 열린다. 대량 작업은 헤드리스 파이썬으로
