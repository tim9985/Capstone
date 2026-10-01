# SDD v1.5 — 비전 클래스 다이어그램 (객체지향 계층판 · 10-01)

> 9.30 발표 피드백 2번 ("설계는 객체지향 · DTO 패키지 · DAO · Service 인터페이스 · 패키지 간 호출은 인터페이스") 반영
> HWPX: `3. 설계명세서_…_v1.5.hwpx` (v1.4 (1) 에서 **그림 8장만 교체** · 표·본문·시퀀스는 그대로 · git 밖)

## 규칙 (모든 CD 공통으로 쓸 수 있음)

| # | 규칙 | 그림에서 |
|---|---|---|
| 1 | 클래스 다이어그램 = **소스 구성** — 패키지 = 폴더 · 클래스 = 파일 | 패키지 틀 (`vision.controller` · `vision.service` · `vision.service.impl` · `vision.dao` · `vision.entity` · `vision.dto`) |
| 2 | **HTTP 는 관계로 그리지 않는다** — 웹 요청의 입구는 `«controller»` 클래스 | `CandidateController` · `VideoController` |
| 3 | Controller 는 **Service 인터페이스만 import** · 구현은 `I…` 를 실현 | `ITargetService` ◁┄ `TargetService` |
| 4 | **Entity = DB 테이블 1:1** · Entity 를 읽고 쓰는 것은 **DAO 만** | `PersonCandidateDAO` → `PersonCandidate (DB-14)` |
| 5 | 계층 사이는 **DTO** 로 주고받는다 · Entity 는 Service 밖으로 나가지 않는다 (`toDTO()`) | `CandidateDTO` · `JudgementDTO` … |
| 6 | **다른 패키지 호출은 그 패키지의 인터페이스로만** | `IMediaStore` · `IFrameSource` · `IAlertService` · `ICoordinateTransform` … |

기존 클래스 (C-0601~0610 · C-0701~0720) 는 **구현 클래스로 그대로** — 이름·명세·시퀀스 73개가 바뀌지 않는다. 인터페이스·DTO·DAO·Entity·Controller 만 새로 추가.

## 그림 ↔ HWPX

| 파일 | 도식 | HWPX 그림 |
|---|---|---|
| CD-06.png | 영상·저장 — media 패키지 | image56 |
| CD-07_1.png | 후보 관리 — Controller · Service · DAO · Entity · DTO | image68 |
| CD-07_2.png | 탐지·위치·추적 — 인터페이스 · 구현 · DTO | image69 |
| CD-15_1~3.png | 타일 탐지 · 좌표 · 후보 등록 | image57 · 70 · 71 |
| CD-17_1~2.png | 오프라인 학습 · 모델 등록 (IModelRegistry → DAO → config_version) | image72 · 73 |

## 새 클래스 (표 갱신용 · 아직 HWPX 표에는 없음)

| 구분 | 06 영상 | 07 비전 |
|---|---|---|
| Controller | C-0611 VideoController | C-0721 CandidateController |
| Interface | C-0612 IVideoIngestService · C-0613 IMediaStore · C-0614 IFrameSource | C-0722 ITargetService · C-0723 IPersonDetectionService · C-0724 ITargetGeoLocator · C-0725 ITargetTrackingService · C-0726 IModelRegistry · (C-0807 ICameraModel) |
| DAO | C-0615 VideoFrameDAO · C-0616 MediaAssetDAO | C-0727 PersonCandidateDAO · C-0728 DetectionObservationDAO · C-0729 ReobservationRequestDAO · C-0730 ModelConfigDAO |
| Entity | C-0617 VideoFrame (DB-12) · C-0618 MediaAsset (DB-13) | C-0731 PersonCandidate (DB-14) · C-0732 DetectionObservation (DB-15) · C-0733 ReobservationRequest (DB-10) · C-0734 ModelConfig (DB-06) |
| DTO | C-0619 FrameEnvelopeDTO · C-0620 SyncedFrameDTO · C-0621 MediaRefDTO · C-0622 MediaViewDTO · C-0623 VideoStatusDTO | C-0735 DetectionDTO · C-0736 DetectionBatchDTO · C-0737 GeoResultDTO · C-0738 CandidateDTO · C-0739 CandidateDetailDTO · C-0740 JudgementDTO · C-0741 ReobserveRequestDTO · C-0742 TrackingStatusDTO · C-0743 ModelArtifactDTO · C-0744 EvaluationReportDTO · C-0745 ComparisonReportDTO · C-0746 TrainingManifestDTO |

## 다른 담당이 만들어야 할 인터페이스 (비전이 호출)

`IPermissionPolicy` (C-0103) · `IVehicleStateService` (C-0201) · `ICoordinateTransform` (C-0302) · `IViewpointPlanner` (C-0503) · `ICommandService` (C-0901) · `IAlertService` (C-1101) · `IFrameAnalysisLedger` (C-1206)
— 팀 CD 의 `…Repository` (MissionRepository · PlanRepository · AlertRepository) 는 역할상 DAO → `…DAO` 로 이름을 맞추고 Entity 를 따로 두면 같은 규칙이 된다
