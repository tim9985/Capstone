"""탐지 및 추적 대상 관리 — services.vision · vision_train"""
from mdl import K, I, DTO, ENT, DAO, unit
from seq import S, call, alt, opt, note, loop
from cdkit import CD, layered, box

unit("07", "탐지 및 추적 대상 관리", "탐지·대상 관리")
V, T = "services.vision", "vision_train"

I("C-0721", "IPersonDetectionService", V, "사람 탐지의 공개 계약", [
    ("detect", "SyncedFrameDTO", "DetectionBatchDTO", "사람 단일 클래스의 경계 상자·신뢰도를 산출하고 초기 후보를 즉시 보존한다."),
    ("getMetrics", "", "InferenceMetricsDTO", "추론 시간·분석 FPS·입력 크기·전후처리 포함 범위를 반환한다."),
], impl="C-0701")
I("C-0722", "ITargetService", V, "사람 후보 관리의 공개 계약 (Controller·계획·이력 패키지가 의존)", [
    ("queryCandidates", "missionId, filter", "List<CandidateDTO>", "초기·반복 관측·좌표 보류·판단 상태를 구분해 조회한다."),
    ("getCandidate", "candidateId", "CandidateDetailDTO", "좌표 보류 여부와 마지막 탐지 근거·스냅샷을 읽는다."),
    ("recordJudgement", "JudgementDTO, PrincipalDTO", "CandidateDTO", "운영자의 대응 필요·오탐 판단과 증거 시각을 보존한다."),
    ("requestReobserve", "ReobserveRequestDTO, PrincipalDTO", "CandidateDTO", "중복 요청을 억제하고 정밀 관측을 요청한다."),
    ("upsertInitial", "DetectionBatchDTO, SyncedFrameDTO", "List<CandidateDTO>", "단일 탐지·좌표 미산출 후보도 저장하고 알린다."),
    ("attachGeoEvidence", "candidateId, GeoResultDTO", "CandidateDTO", "유효 좌표 또는 보류 사유를 후보 이력에 더한다."),
    ("attachAppearance", "candidateId, AppearanceResultDTO", "CandidateDTO", "상의 색상 비교 결과를 후보에 더한다."),
], impl="C-0704")
I("C-0723", "ITargetGeoLocator", V, "대상 위치 산출의 공개 계약", [
    ("locate", "DetectionDTO, SyncedFrameDTO", "GeoResultDTO", "시각·보정·자세·지형 품질을 통과한 위치만 산출하고 아니면 보류 사유를 낸다."),
], impl="C-0703")
I("C-0724", "ITargetTrackingService", V, "개별 대상 추적의 공개 계약", [
    ("updateTracking", "TrackingActionDTO", "TrackingStatusDTO", "선택 대상의 추적·유실 후 주변 재탐색·종료를 처리한다."),
    ("stopTracking", "candidateId, reason", "TrackingStatusDTO", "추적 명령을 무효화하고 허용 시에만 남은 영역 탐색으로 돌아간다."),
], impl="C-0706")
I("C-0725", "IModelRegistry", V, "운용 모델 버전 관리 계약 (학습 도구·탐지기가 의존)", [
    ("register", "ModelArtifactDTO, EvaluationReportDTO", "UUID", "내용 해시·부모 모델·평가 상태를 검사해 등록한다."),
    ("activate", "modelId, DeploymentReport", "ResultDTO", "대상 GPU·입력 크기·회귀 시험 통과 시만 활성화한다."),
    ("getActive", "", "ModelArtifactDTO", "승인된 운용 모델과 엔진 경로를 반환한다."),
], impl="C-0714")

K("C-0701", "PersonDetectionService", "service", V, "YOLO11m 단일 person 추론·타일 병합을 조정 (IPersonDetectionService 구현)", [
    ("splitter", "TileSplitter", ""), ("detector", "PersonDetector", ""), ("merger", "DetectionMerger", ""),
    ("targets", "ITargetService", "초기 후보 보존"), ("geoLocator", "ITargetGeoLocator", "위치 산출"),
    ("threshold", "float", "초기 임계값 0.15 (검증 후 변경 허용)"),
], [], impl="C-0721", old="C-0701")
K("C-0704", "TargetService", "service", V, "초기 후보 보존·조회·운영자 판단·재관측 요청 관리 (ITargetService 구현)", [
    ("registry", "CandidateRegistry", "후보 단일 저장 진입점"), ("snapshots", "SnapshotWriter", "근거 영상"),
    ("candidateDao", "PersonCandidateDAO", "후보 조회"), ("permission", "IPermissionPolicy", "판단·명령 권한"),
    ("planner", "IViewpointPlanner", "정밀 관측 요청"), ("mediaStore", "IMediaStore", "스냅샷 조회"),
    ("alerts", "IAlertService", "첫 탐지 알림"), ("history", "IHistoryService", "판단 이력"),
], [("-toDTO", "PersonCandidate", "CandidateDTO", "엔티티를 화면용 DTO 로 바꾼다.")], impl="C-0722", old="C-0704")
K("C-0703", "TargetGeoLocator", "service", V, "유효 촬영 시점 자세·보정값·지형으로 대상 위치 산출 (ITargetGeoLocator 구현)", [
    ("resolver", "GeoResolver", "광선 × 지면"), ("transform", "ICoordinateTransform", "WGS84 변환"),
    ("targets", "ITargetService", "좌표 근거 저장"), ("qualityGate", "GeoQualityGate", "기울기 ≤ 6° · 자세 변화 ≤ 1.5° · 시각 · GNSS"),
], [("-checkQuality", "pose", "Decision", "좌표 품질 조건(기울기·자세 변화·시각·GNSS)을 검사한다.")], impl="C-0723", old="C-0703")
K("C-0706", "TargetTrackingService", "service", V, "선택 후보 관찰을 위한 제한된 추적 목표 생성 (ITargetTrackingService 구현)", [
    ("candidateDao", "PersonCandidateDAO", "추적 상태 (person_candidate.tracking)"), ("planner", "IViewpointPlanner", "추적 관측 목표"),
    ("commands", "ICommandService", "추적 이동 명령"), ("lostTimeout", "Duration", "유실 후 재탐색 제한시간 (현장 시험 후 확정)"),
], [], impl="C-0724", old="C-0706")
K("C-0714", "ModelRegistry", "service", V, "모델·엔진·평가셋·런타임 버전을 일관되게 제공 (IModelRegistry 구현)", [
    ("configDao", "ConfigVersionDAO", "MODEL 설정 버전"), ("artifactStore", "IMediaStore", "PT·ENGINE·평가 파일"),
    ("activeModelId", "UUID", "승인된 운용 모델"),
], [], impl="C-0725", old="C-0714")
K("C-0605", "FrameIngestor", "component", V, "최신 프레임을 받아 분석을 배정하고 탐지를 요청하는 비전 작업자", [
    ("frameSource", "IFrameSource", "최신 프레임"), ("ledger", "IFrameAnalysisLedger", "분석 배정·완료"),
    ("detection", "IPersonDetectionService", "사람 탐지"), ("missionMap", "IMissionMapService", "관측 반영"),
], [
    ("accept", "SyncedFrameDTO", "AdmissionResult", "연속 구간·영상 종류·디코딩 상태를 검사하고 대기열·처리 이력에 등록한다."),
    ("dropStale", "FrameAgePolicy", "DropReport", "오래된·손상 프레임을 제외하고 원인을 기록한다."),
], old="C-0605")
K("C-0707", "TileSplitter", "component", V, "1920×1080 을 겹치는 1280×720 타일 4장으로 나누고 역변환을 보존", [
    ("origins", "Point2D[4]", "(0,0) (640,0) (0,360) (640,360)"), ("tileSize", "Size", "1280×720"),
], [("split", "SyncedFrameDTO", "Tile[4]", "원본 픽셀 좌표·유효 crop·padding 을 함께 반환한다.")], old="C-0707")
K("C-0708", "PersonDetector", "component", V, "soup_v7r2 단일 TensorRT FP16 엔진으로 person 추론", [
    ("engine", "EngineHandle", "운용 GPU 에서 빌드한 엔진 · 입력 [736,1280] · batch 4"), ("thresholds", "Thresholds", "conf 0.15"),
    ("registry", "IModelRegistry", "활성 모델"),
], [("detect", "Tile[4]", "TileDetection[]", "실제 입력 크기·배치·시간 경계를 기록한다.")], old="C-0708")
K("C-0709", "DetectionMerger", "component", V, "타일 박스를 원본 좌표로 복원하고 중복 검출 제거", [
    ("iouThreshold", "float", "NMS 0.6"), ("transformChain", "Transform[]", "crop·padding·resize 역변환"),
], [("merge", "TileDetection[], Tile[4]", "List<DetectionDTO>", "padding 을 제거하고 원본 범위로 자른 뒤 전체 NMS 를 한다.")], old="C-0709")
K("C-0710", "GeoResolver", "component", V, "영상 접지점 광선과 지형·기준면의 교차 및 불확실도 계산", [
    ("camera", "ICameraModel", "렌즈·장착 보정"), ("qualityPolicy", "GeoPolicy", "시각·자세·지형·거리 제한"),
], [("toWorld", "DetectionDTO, FramePoseBinding", "GeoResultDTO", "유효 좌표와 오차 타원 또는 PENDING/INVALID 와 사유를 반환한다.")], old="C-0710")
K("C-0711", "ColorProfiler", "component", V, "상의 ROI 의 배경 제외 화소에서 Lab 12색 top2 추정", [
    ("palette", "ColorPalette", "정의·버전 있는 12색"), ("roiPolicy", "RoiPolicy", "박스 상단 30~60%"),
], [("profile", "CandidateCrop, FrameQuality", "ColorProfile", "색상 1·2순위와 판정불가·조명/가림 품질을 반환한다.")], old="C-0711")
K("C-0705", "AppearanceComparator", "component", V, "상의 색상 top1/top2 와 비교 신뢰도 산출", [
    ("profiler", "ColorProfiler", ""), ("qualityThreshold", "float?", "가림·작은 영역 판정 보류 기준"),
], [("compare", "crop, AppearanceQuery", "AppearanceResultDTO", "일치·불일치·판정불가와 신뢰도를 산출한다.")], old="C-0705")
K("C-0702", "TargetAssociationService", "component", V, "시간·위치·상의 색상 근거로 동일 후보 연결", [
    ("associationPolicy", "AssociationPolicy", "결합 기준과 미확정 정책"), ("trackIndex", "Map<TrackId, CandidateId>", "추적 ID ↔ 후보 ID"),
], [("associateTargets", "List<DetectionDTO>, history", "AssociationResult", "추적 ID 와 후보 ID 를 근거와 함께 연결한다.")], old="C-0702")
K("C-0713", "CandidateRegistry", "component", V, "검출·연결·운영자 판정·좌표보류 후보의 단일 저장 진입점", [
    ("association", "TargetAssociationService", ""), ("comparator", "AppearanceComparator", ""),
    ("candidateDao", "PersonCandidateDAO", ""), ("observationDao", "DetectionObservationDAO", ""),
], [
    ("upsert", "DetectionDTO, GeoResultDTO?", "PersonCandidate", "초기 후보와 후속 관측을 중복 없이 반영한다."),
    ("applyJudgement", "candidateId, JudgementDTO", "PersonCandidate", "운영자 판정을 자동 탐지 상태와 분리해 저장한다."),
    ("countValidObservations", "candidateId", "int", "서로 다른 촬영 시각 관측만 세어 3회 이상을 반복 확인으로 구분한다."),
], old="C-0713")
K("C-0712", "SnapshotWriter", "component", V, "원본·대상 크롭 파일을 확정하고 후보와 연결", [
    ("mediaStore", "IMediaStore", "원자적 파일 확정·DB 기록"), ("history", "IHistoryService", "실패 기록"),
], [("save", "SyncedFrameDTO, DetectionDTO, candidateId", "MediaRefDTO", "쓰기 실패에도 후보를 남기고 상태·해시·출처를 기록한다.")], old="C-0712")
# 오프라인 학습
for cid, nm, resp, attrs, ops in [
    ("C-0716", "DatasetBuilder", "장소별 분리와 원천 → 크롭 데이터 출처 관리", [("datasetManifest", "DatasetManifest", "원천·장소·시퀀스·라벨·크롭 버전")],
     [("build", "SourceDataset[]", "TrainingManifestDTO", "장소당 상한과 사람 크기·시점 조건을 적용한다."),
      ("verifySplit", "TrainingManifestDTO, EvaluationManifest", "Decision", "학습·평가를 장소와 연속 구간으로 분리한다.")]),
    ("C-0717", "Trainer", "같은 설정에서 순서만 바꾼 모델 반복 생성", [("trainingSpec", "TrainingSpec", "SGD 0.01 · 30 에폭 · batch 8")],
     [("train", "TrainingManifestDTO, seed", "ModelArtifactDTO", "설정·순서·로그·가중치 해시를 함께 보존한다.")]),
    ("C-0718", "Evaluator", "장소 분리 평가셋 3개의 person 성능과 운영 지표 측정", [("evaluationManifest", "EvaluationManifest", "test_obl · test_v2 · test_kr 고정")],
     [("evaluate", "ModelArtifactDTO", "EvaluationReportDTO", "AP50·재현율·정밀도·음성 오탐·bbox 기준을 기록한다.")]),
    ("C-0719", "PairedComparator", "짝 부트스트랩과 반복 흔들림으로 채택 판단", [("resamplingSpec", "BootstrapSpec", "연속 20장 묶음 · 95% 구간")],
     [("compare", "EvaluationReportDTO, EvaluationReportDTO", "ComparisonReportDTO", "세 평가셋 악화와 반복 변동을 함께 판정한다.")]),
    ("C-0720", "WeightSouper", "같은 기반 모델들의 검증된 가중치 평균 생성", [("sourceModels", "List<ModelArtifactDTO>", "같은 구조·출발점·설정")],
     [("average", "List<ModelArtifactDTO>", "ModelArtifactDTO", "가중치를 평균하고 BN 통계·평가 붕괴를 검사한다.")]),
    ("C-0715", "EngineBuilder", "운용 GPU 에서 원본 PT 를 TensorRT FP16 엔진으로 재빌드", [("buildSpec", "EngineBuildSpec", "GPU·런타임·[736,1280]·배치·FP16"), ("registry", "IModelRegistry", "엔진 등록")],
     [("build", "ModelArtifactDTO, TargetEnvironment", "ModelArtifactDTO", "빌드 로그·출력 해시·부모 PT 를 기록한다."),
      ("validate", "ModelArtifactDTO, EvaluationManifest", "ComparisonReportDTO", "기존 엔진 대비 출력·지연·정확도 회귀를 확인한다.")]),
]:
    K(cid, nm, "tool", T, resp, attrs, ops, old=cid)

K("C-0726", "CandidateController", "controller", "api", "후보 조회·판단·재관측·추적 요청의 입구 (REST)", [
    ("targetService", "ITargetService", ""), ("trackingService", "ITargetTrackingService", ""),
], [
    ("getCandidates", "missionId, filter", "List<CandidateDTO>", "GET /api/missions/{id}/candidates"),
    ("getCandidate", "candidateId", "CandidateDetailDTO", "GET /api/candidates/{id}"),
    ("postJudgement", "candidateId, JudgementDTO, PrincipalDTO", "CandidateDTO", "POST /api/candidates/{id}/judgements"),
    ("postReobserve", "candidateId, ReobserveRequestDTO, PrincipalDTO", "CandidateDTO", "POST /api/candidates/{id}/reobserve"),
    ("postTracking", "candidateId, TrackingActionDTO", "TrackingStatusDTO", "POST /api/candidates/{id}/tracking-actions"),
])

DTO("C-0727", "DetectionDTO", "탐지 한 건", [("frameId", "UUID", "프레임"), ("bbox", "int[4]", "원본 px"), ("confidence", "float", "신뢰도"),
    ("tileIndex", "int", "출처 타일"), ("trackId", "String?", "영상 ID")])
DTO("C-0728", "DetectionBatchDTO", "프레임 단위 탐지 결과", [("frameId", "UUID", "프레임"), ("detections", "List<DetectionDTO>", "탐지 (0건도 정상)"),
    ("modelConfigId", "UUID", "모델"), ("inferMs", "float", "추론 시간")])
DTO("C-0729", "GeoResultDTO", "좌표 산출 결과", [("status", "VALID | PENDING | INVALID", "상태"), ("position", "GeoPoint?", "WGS84"),
    ("ellipse", "Ellipse?", "오차 타원"), ("reason", "ReasonCode?", "보류 사유"), ("calibrationVersion", "UUID", "보정 버전")])
DTO("C-0730", "CandidateDTO", "후보 요약", [("candidateId", "UUID", "후보"), ("geoStatus", "VALID | PENDING | INVALID", "좌표 상태"),
    ("position", "GeoPoint?", "위치"), ("validGroupCount", "int", "서로 다른 시각 유효 관측"), ("colorTop2", "String[2]", "상의 색상"),
    ("judgement", "String", "운영자 판단"), ("tracking", "TrackingStatusDTO?", "추적 상태"), ("lastSeen", "Time", "마지막 관측")])
DTO("C-0731", "CandidateDetailDTO", "후보 상세", [("candidate", "CandidateDTO", "요약"), ("snapshots", "List<MediaViewDTO>", "근거 영상"),
    ("observations", "List<DetectionDTO>", "관측 이력"), ("geoReason", "ReasonCode?", "좌표 보류 사유")])
DTO("C-0732", "JudgementDTO", "운영자 판단", [("candidateId", "UUID", "후보"), ("decision", "RESCUE_NEEDED | FALLEN_SUSPECT | FALSE_POSITIVE | UNSURE", "판단"),
    ("note", "String", "메모")])
DTO("C-0733", "ReobserveRequestDTO", "재관측 요청", [("candidateId", "UUID", "후보"), ("reason", "String", "사유"), ("desiredView", "ViewHint?", "원하는 시점")])
DTO("C-0734", "TrackingActionDTO", "추적 시작·종료·새 관측", [("candidateId", "UUID", "대상"), ("action", "START | STOP | OBSERVED | LOST", "동작"),
    ("evidence", "DetectionDTO?", "새 관측")])
DTO("C-0735", "TrackingStatusDTO", "추적 상태", [("candidateId", "UUID", "대상"), ("state", "TRACKING | LOST | ENDED", "상태"),
    ("lastSeen", "Time", "마지막 관측"), ("position", "GeoPoint?", "위치")])
DTO("C-0736", "AppearanceResultDTO", "상의 색상 비교", [("top1", "ColorLabel", "1순위"), ("top2", "ColorLabel", "2순위"), ("confidence", "float", "확신도"),
    ("match", "MATCH | MISMATCH | UNDECIDABLE", "조건 일치")])
DTO("C-0737", "InferenceMetricsDTO", "추론 지표", [("inferMsMean", "float", "평균"), ("inferMsP95", "float", "P95"), ("analyzedFps", "float", "분석 FPS"),
    ("inputSize", "int[2]", "[736,1280]")])
DTO("C-0738", "ModelArtifactDTO", "모델 산출물", [("modelId", "UUID", "모델"), ("weightsUri", "String", "가중치"), ("engineUri", "String?", "엔진"),
    ("parentIds", "List<UUID>", "부모 (평균 원본)"), ("contentHash", "String", "해시")])
DTO("C-0739", "EvaluationReportDTO", "평가 결과", [("modelId", "UUID", "모델"), ("ap50", "Map<평가셋, float>", "AP50"), ("recall", "Map", "재현율"),
    ("precision", "Map", "정밀도"), ("negFpPerFrame", "float", "음성 오탐")])
DTO("C-0740", "ComparisonReportDTO", "짝 비교 결과", [("baseId", "UUID", "기준"), ("candidateId", "UUID", "비교"), ("diff", "Map<평가셋, CI95>", "차이 구간"),
    ("decision", "ADOPT | REJECT", "판정")])
DTO("C-0741", "TrainingManifestDTO", "학습 목록", [("trainList", "Path", "학습"), ("valList", "Path", "검증"), ("places", "List<String>", "장소"), ("imgsz", "int", "1280")])

DAO("C-0742", "PersonCandidateDAO", "person_candidate", [
    ("insert", "PersonCandidate", "UUID", "새 후보를 저장한다."),
    ("findById", "candidateId", "PersonCandidate?", "후보를 읽는다."),
    ("findByMission", "missionId, filter", "List<PersonCandidate>", "임무의 후보를 읽는다."),
    ("update", "PersonCandidate, rowVersion", "bool", "동시성 버전이 같을 때만 저장한다."),
])
DAO("C-0743", "DetectionObservationDAO", "detection_observation", [
    ("insert", "DetectionObservation", "UUID", "탐지 관측을 저장한다 (프레임·방식·번호 유일)."),
    ("findByCandidate", "candidateId", "List<DetectionObservation>", "후보의 관측 이력을 읽는다."),
])
ENT("C-0744", "PersonCandidate", "person_candidate", "사람 후보 (DB-14 한 행)",
    keys=["candidateId {PK}", "missionId {FK}", "geoStatus · geoReason · ellipse", "appearance · judgement · tracking : JSON", "rowVersion"])
ENT("C-0745", "DetectionObservation", "detection_observation", "프레임별 탐지 관측 (DB-15 한 행)",
    keys=["observationId {PK}", "candidateId · frameId {FK}", "bbox · confidence", "trackId", "geoResult : JSON"])


def _cd():
    a = layered("cd07a", "(1/4) 후보·추적 — 계층 구조", svc_pkg=V, ctl=["C-0726"],
                dto=["C-0730", "C-0731", "C-0732", "C-0733", "C-0734", "C-0735", "C-0736", "C-0729"],
                pairs=[("C-0722", "C-0704"), ("C-0724", "C-0706")], comps=["C-0713", "C-0712"],
                ext=[("C-0108", "services.auth"), ("C-0516", "services.mission.local"), ("C-0612", "services.media"),
                     ("C-1105", "services.alert"), ("C-1207", "services.history"), ("C-0908", "services.command")],
                daos=[("C-0742", False), ("C-0743", False)], ents=["C-0744", "C-0745"], api_w=0.4, dto_cols=4, ext_w=0.25,
                extra=[("k0713", "k0743", "dep", "", {"elbow": 1})])
    b = layered("cd07b", "(2/4) 탐지·좌표·모델 — 계층 구조", svc_pkg=V, dto=["C-0727", "C-0728", "C-0729", "C-0737"],
                pairs=[("C-0721", "C-0701"), ("C-0723", "C-0703"), ("C-0725", "C-0714")],
                comps=["C-0605", "C-0707", "C-0708", "C-0709", "C-0710"],
                ext=[("C-0613", "services.media"), ("C-0627", "services.media"), ("C-0808", "services.mission_map"),
                     ("C-0306", "services.spatial"), ("C-0807", "services.mission_map")],
                daos=[("C-0313", True)], ents=[], ext_w=0.25, drop=[("k0714", "k0313")],
                extra=[("k0605", "k0721", "dep", "", {"elbow": 1}), ("k0708", "k0725", "dep", "", {"elbow": 1}),
                       ("k0714", "k0313", "dep", "", {"pts": lambda g: [(g["k0714"][0] + g["k0714"][2], g["k0714"][1] + g["k0714"][3] * 0.6),
                           (g["svc"][0] + g["svc"][2] + 14, g["k0714"][1] + g["k0714"][3] * 0.6),
                           (g["svc"][0] + g["svc"][2] + 14, g["k0313"][1] - 34), (g["k0313"][0] + g["k0313"][2] / 2, g["k0313"][1] - 34),
                           (g["k0313"][0] + g["k0313"][2] / 2, g["k0313"][1])]})])
    B = {"ts": box("C-0704", ref=True)}
    for c in ["C-0713", "C-0702", "C-0705", "C-0711", "C-0712"]:
        B[c] = box(c)
    B["pc"] = box("C-0742", ref=True); B["do"] = box("C-0743", ref=True); B["ms"] = box("C-0612", ref=True); B["hs"] = box("C-1207", ref=True)
    P = [dict(name=V, row=0, x=0.3, w=0.4, rows=[[("ts", 1.0)]]),
         dict(name=V, row=1, x=0.005, w=0.99, rows=[[("C-0713", .36), ("C-0702", .32), ("C-0705", .32)], [("C-0712", .5), ("C-0711", .5)]]),
         dict(name="storage.dao", row=2, x=0.005, w=0.49, rows=[[("pc", .5), ("do", .5)]]),
         dict(name="services.media · services.history", row=2, x=0.505, w=0.49, rows=[[("ms", .5), ("hs", .5)]])]
    R = [("ts", "C-0713", "assoc", "registry", {"elbow": 1}), ("ts", "C-0712", "assoc", "", {"elbow": 1, "ax": 0.2, "bx": 0.5,
          "pts": lambda g: [(g["ts"][0], g["ts"][1] + g["ts"][3] / 2), (g["C-0713"][0] - 14, g["ts"][1] + g["ts"][3] / 2),
                            (g["C-0713"][0] - 14, g["C-0712"][1] + 40), (g["C-0712"][0], g["C-0712"][1] + 40)]}),
         ("C-0713", "C-0702", "comp", "", {}), ("C-0713", "C-0705", "assoc", "", {"elbow": 1}), ("C-0705", "C-0711", "assoc", "", {"elbow": 1}),
         ("C-0713", "pc", "dep", "", {"elbow": 1}), ("C-0713", "do", "dep", "", {"elbow": 1}),
         ("C-0712", "ms", "dep", "", {"elbow": 1}), ("C-0712", "hs", "dep", "", {"elbow": 1})]
    c = ("cd07c", "(3/4) 후보 구성요소 — 연결·색상·스냅샷", P, B, R)
    B2 = {k: box(k) for k in ["C-0716", "C-0717", "C-0718", "C-0719", "C-0720", "C-0715"]}
    for d in ["C-0741", "C-0738", "C-0739", "C-0740"]: B2[d] = box(d)
    B2["mr"] = box("C-0725", ref=True)
    P2 = [dict(name=T, row=0, x=0.005, w=0.99, rows=[[("C-0716", .333), ("C-0717", .333), ("C-0720", .334)],
                                                     [("C-0718", .333), ("C-0719", .333), ("C-0715", .334)]]),
          dict(key="ct", name="contracts", row=1, x=0.005, w=0.7, rows=[[("C-0741", .25), ("C-0738", .25), ("C-0739", .25), ("C-0740", .25)]]),
          dict(name=V, row=1, x=0.72, w=0.275, rows=[[("mr", 1.0)]])]
    R2 = [("C-0716", "C-0717", "dep", "", {}), ("C-0717", "C-0720", "dep", "", {}), ("C-0717", "C-0718", "dep", "", {"elbow": 1}),
          ("C-0720", "C-0718", "dep", "", {"elbow": 1}), ("C-0718", "C-0719", "dep", "", {}), ("C-0719", "C-0715", "dep", "채택", {}),
          ("C-0715", "mr", "dep", "register()", {"elbow": 1})]
    d = ("cd07d", "(4/4) 오프라인 학습 (학습 서버)", P2, B2, R2)
    return [a, b, c, d]


CD("CD-07", "탐지·대상 관리", "07", _cd)

OP = ("op", "관제 운영자", "actor")
CTL = ("ctl", "CandidateController", "controller")
TS = ("ts", "ITargetService", "interface")
S("SD-0701", "07", [("fi", "FrameIngestor", "component"), ("pd", "IPersonDetectionService", "interface"), ("sp", "TileSplitter", "component"),
                    ("de", "PersonDetector", "component"), ("mg", "DetectionMerger", "component"), TS, ("cr", "CandidateRegistry", "component"),
                    ("sw", "SnapshotWriter", "component"), ("al", "IAlertService", "interface")], [
    call("fi", "pd", "detect(SyncedFrameDTO)", "DetectionBatchDTO", "person 한 클래스 탐지를 시작한다.", [
        call("pd", "sp", "split(SyncedFrameDTO)", "Tile[4]", "1920×1080 에서 겹치는 1280×720 타일 4장을 만든다."),
        call("pd", "de", "detect(Tile[4])", "TileDetection[]", "soup_v7r2 TensorRT FP16 · [736,1280] · conf 0.15 로 추론한다."),
        call("pd", "mg", "merge(TileDetection[], Tile[4])", "List<DetectionDTO>", "원본 좌표 복원 후 IoU 0.6 NMS 를 적용한다."),
        alt([("탐지 1건 이상", [
                call("pd", "ts", "upsertInitial(DetectionBatchDTO, SyncedFrameDTO)", "List<CandidateDTO>", "좌표 산출 상태와 함께 최초 후보를 저장한다.", [
                    call("ts", "cr", "upsert(DetectionDTO, null)", "PersonCandidate", "후보와 관측을 중복 없이 저장한다."),
                    call("ts", "sw", "save(SyncedFrameDTO, DetectionDTO, candidateId)", "MediaRefDTO", "증거 프레임·크롭을 저장한다."),
                    call("ts", "al", "publish(AlertEventDTO)", "AlertDTO", "후보 저장 뒤 중복 억제한 초기 알림을 보낸다."),
                ])]),
             ("탐지 0건 (2a)", [note("pd", "cr", "정상 결과로 기록 (DetectionBatchDTO.detections = [])")])]),
    ]),
], entry="IPersonDetectionService.detect", 시작="최신 유효 분석 프레임 입력 (FrameIngestor)")
S("SD-0702", "07", [OP, CTL, TS, ("pc", "PersonCandidateDAO", "dao"), ("ms", "IMediaStore", "interface")], [
    call("op", "ctl", "getCandidates(missionId, filter)", "List<CandidateDTO>", "후보 목록을 조회한다.", [
        call("ctl", "ts", "queryCandidates(missionId, filter)", "List<CandidateDTO>", "초기·반복 관측·좌표 보류·판단 상태를 구분한다.", [
            call("ts", "pc", "findByMission(missionId, filter)", "List<PersonCandidate>", "후보를 읽는다."),
        ]),
    ]),
    call("op", "ctl", "getCandidate(candidateId)", "CandidateDetailDTO", "선택 후보 상세를 연다.", [
        call("ctl", "ts", "getCandidate(candidateId)", "CandidateDetailDTO", "근거 영상과 관측 이력을 묶는다.", [
            call("ts", "pc", "findById(candidateId)", "PersonCandidate", "후보를 읽는다."),
            call("ts", "ms", "getAsset(assetId)", "MediaViewDTO", "연결된 스냅샷과 보존 상태를 조회한다."),
        ]),
    ]),
    note("op", "ms", "좌표 보류 후보는 사유와 스냅샷으로 표시 — 임의의 확정 위치를 부여하지 않음 (2a)"),
], entry="CandidateController.getCandidates · getCandidate")
S("SD-0703", "07", [("cr", "CandidateRegistry", "component"), ("ta", "TargetAssociationService", "component"), ("ac", "AppearanceComparator", "component"),
                    ("pc", "PersonCandidateDAO", "dao"), ("do", "DetectionObservationDAO", "dao")], [
    call("cr", "ta", "associateTargets(List<DetectionDTO>, history)", "AssociationResult", "추적 ID 와 후보 ID 를 시간·위치·색상 근거로 연결한다."),
    call("cr", "ac", "compare(crop, AppearanceQuery)", "AppearanceResultDTO", "색상 근거와 판정 신뢰도를 얻는다."),
    alt([("연결 확정", [call("cr", "do", "insert(DetectionObservation)", "observationId", "유효 탐지만 관측 횟수에 반영한다."),
                        call("cr", "pc", "update(PersonCandidate, rowVersion)", "bool", "병합 위치·관측 횟수·반복 관측 상태를 갱신한다.")]),
         ("불명확 · 유실·만료 (1a)", [note("cr", "do", "연결 보류 · 이전 후보 이력 유지 · 유실/만료 표시")])]),
], entry="CandidateRegistry (내부)", 시작="새 탐지 관측 또는 추적 ID 유실·재연결 (CandidateRegistry)")
S("SD-0704", "07", [("pd", "PersonDetectionService", "service"), ("gl", "ITargetGeoLocator", "interface"), ("gr", "GeoResolver", "component"),
                    ("cm", "ICameraModel", "interface"), ("ct", "ICoordinateTransform", "interface"), TS], [
    call("pd", "gl", "locate(DetectionDTO, SyncedFrameDTO)", "GeoResultDTO", "원 프레임 픽셀과 시각·보정·자세를 검사한다.", [
        call("gl", "gl", "checkQuality(pose)", "Decision", "기울기 ≤ 6° · 자세 변화 ≤ 1.5° · 시각 · GNSS 를 검사한다."),
        alt([("품질 통과", [
                call("gl", "gr", "toWorld(DetectionDTO, FramePoseBinding)", "GeoResultDTO", "접지점 광선을 지면과 교차하고 오차 타원을 얻는다.", [
                    call("gr", "cm", "ray(pixel, pose)", "Ray3D", "발끝 픽셀(박스 아래 가운데) 광선을 만든다."),
                    call("gr", "cm", "projectToTerrain(Ray3D, terrain)", "IntersectionResult", "평면·DEM 과 교차한다."),
                ]),
                call("gl", "ct", "transform(GeometryDTO, contextId)", "GeometryDTO", "미터 내부 좌표를 WGS84·고도 기준으로 변환한다."),
             ]),
             ("품질 미달 (2a · EX-13)", [note("gl", "ct", "GeoResultDTO(PENDING, 사유) — 후보는 보존")])]),
        call("gl", "ts", "attachGeoEvidence(candidateId, GeoResultDTO)", "CandidateDTO", "VALID/PENDING/INVALID 와 보정 버전을 저장한다."),
    ]),
    note("pd", "ts", "추적만 갱신한 프레임은 새 좌표 보고에 쓰지 않는다 (2b 는 새 탐지 결과로만)"),
], entry="ITargetGeoLocator.locate", 시작="사람 탐지와 영상·비행정보 연결 완료 (PersonDetectionService)")
S("SD-0705", "07", [OP, CTL, TS, ("pp", "IPermissionPolicy", "interface"), ("cr", "CandidateRegistry", "component"), ("hs", "IHistoryService", "interface")], [
    call("op", "ctl", "postJudgement(candidateId, JudgementDTO, PrincipalDTO)", "CandidateDTO", "후보 판단을 기록한다.", [
        call("ctl", "ts", "recordJudgement(JudgementDTO, PrincipalDTO)", "CandidateDTO", "판단과 증거 시각을 보존한다.", [
            call("ts", "pp", "checkAccess(PrincipalDTO, CANDIDATE_JUDGE)", "AccessDecisionDTO", "판단 기록 권한을 확인한다."),
            call("ts", "cr", "applyJudgement(candidateId, JudgementDTO)", "PersonCandidate", "자동 탐지 상태와 분리해 저장한다."),
            call("ts", "hs", "append(MissionEventDTO)", "EventRefDTO", "판단자·시각·사유·이전 판단과 잇는다."),
        ]),
    ]),
    note("op", "hs", "쓰러짐 여부·대응 필요성은 운영자가 판단 (2a) · 오탐 사례는 모델 개선 자료로 보존"),
], entry="CandidateController.postJudgement")
S("SD-0706", "07", [OP, CTL, TS, ("pp", "IPermissionPolicy", "interface"), ("vp", "IViewpointPlanner", "interface")], [
    call("op", "ctl", "postReobserve(candidateId, ReobserveRequestDTO, PrincipalDTO)", "CandidateDTO", "재관측을 요청한다.", [
        call("ctl", "ts", "requestReobserve(ReobserveRequestDTO, PrincipalDTO)", "CandidateDTO", "중복 요청을 억제한다.", [
            call("ts", "pp", "authorizeCommand(PrincipalDTO, CommandContextDTO)", "AccessDecisionDTO", "현재 기체·임무의 명령 요청 권한을 확인한다."),
            call("ts", "vp", "requestPrecision(PrecisionRequestDTO)", "ViewpointPlanDTO?", "후보 근거에 맞는 정밀 관측 경로를 만든다 (SD-0504)."),
            opt("재관측 불가 (2a)", [note("ts", "vp", "불가 사유 표시 · 미탐지여도 기존 후보·이력 유지 (3a)")]),
        ]),
    ]),
], entry="CandidateController.postReobserve")
S("SD-0707", "07", [("cr", "CandidateRegistry", "component"), ("ac", "AppearanceComparator", "component"), ("cp", "ColorProfiler", "component"), TS], [
    call("cr", "ac", "compare(crop, AppearanceQuery)", "AppearanceResultDTO", "사용자 상의 조건과 비교할 품질을 확인한다.", [
        call("ac", "cp", "profile(CandidateCrop, FrameQuality)", "ColorProfile", "상의 ROI 의 배경 제외 화소를 Lab 12색 top2 로 요약한다."),
        opt("크기·가림·흐림 (2a)", [note("ac", "cp", "UNDECIDABLE (판정불가)")]),
    ]),
    call("cr", "ts", "attachAppearance(candidateId, AppearanceResultDTO)", "CandidateDTO", "일치·불일치·판정불가와 보류 사유를 저장한다."),
    note("cr", "ts", "모든 비교 결과의 후보를 목록에 유지 (4a) — 사람 검출 신뢰도와 구분해 제공"),
], entry="AppearanceComparator.compare", 시작="인상착의 조건이 있는 새 탐지 또는 비교 요청 (CandidateRegistry)")
S("SD-0708", "07", [OP, CTL, ("tt", "ITargetTrackingService", "interface"), ("pc", "PersonCandidateDAO", "dao"),
                    ("vp", "IViewpointPlanner", "interface"), ("cs", "ICommandService", "interface")], [
    call("op", "ctl", "postTracking(candidateId, TrackingActionDTO)", "TrackingStatusDTO", "개별 추적을 시작한다.", [
        call("ctl", "tt", "updateTracking(TrackingActionDTO)", "TrackingStatusDTO", "추적·유실 후 재탐색·종료를 처리한다.", [
            call("tt", "pc", "findById(candidateId)", "PersonCandidate", "최근 위치·색상·유효 시각을 읽는다."),
            alt([("관측 중", [call("tt", "vp", "buildTrackingGoal(TrackingTargetDTO)", "ViewpointPlanDTO?", "운용 범위 안 추적 목표를 만든다."),
                              call("tt", "cs", "submitAction(ActionRequestDTO)", "CommandRefDTO", "추적 이동을 공통 명령 검증 경로로 요청한다.")]),
                 ("유실 · 재탐색 기한 초과 (3a)", [note("tt", "cs", "마지막 관측 저장 · 추적 종료 → 구역 정찰 재개")])]),
            call("tt", "pc", "update(PersonCandidate, rowVersion)", "bool", "추적 진행·위치·유실·종료를 기록한다."),
        ]),
    ]),
    note("op", "cs", "장애·RC 수동·복귀·착륙 시 해당 안전 동작을 유지 (4a) · 다른 후보 탐지·알림은 계속"),
], entry="CandidateController.postTracking")
S("SD-X03", "07", [("fs", "FrameSynchronizer", "component"), ("fi", "FrameIngestor", "component"), ("pd", "IPersonDetectionService", "interface"),
                   ("gl", "ITargetGeoLocator", "interface"), TS, ("mm", "IMissionMapService", "interface")], [
    note("fs", "fi", "(SD-0602) 시각 결합 결과 SyncedFrameDTO 가 IFrameSource 로 들어온다"),
    call("fi", "pd", "detect(SyncedFrameDTO)", "DetectionBatchDTO", "탐지 가능성은 좌표 품질과 무관하게 판단한다."),
    alt([("시각·자세 VALID", [call("pd", "gl", "locate(DetectionDTO, SyncedFrameDTO)", "GeoResultDTO", "정밀 좌표를 산출한다.")]),
         ("시각·접지점·지형·짐벌 미확인", [call("pd", "ts", "attachGeoEvidence(candidateId, PENDING)", "CandidateDTO", "좌표 보류로 보존한다.")])]),
    call("fi", "mm", "applyFrameEvidence(SyncedFrameDTO, TaskQualityDTO)", "ObservationUpdateDTO", "품질 통과 프레임만 관측에 반영한다."),
    note("fi", "mm", "자동 이동에는 VALID · 최신 시각·버전 · 허용 오차를 충족한 현재 관측만 쓴다"),
], title="시각 결합·타일 탐지·좌표보류", uc="UC-0602, UC-0603, UC-0607, UC-0701~0704, UC-0707",
   개요="탐지 가능성과 정밀 좌표 가능성을 분리한다.", 시작="새 분석 프레임 (FrameIngestor)", 선행="프레임이 수신되고 비행정보 창이 있다.",
   사후="탐지 결과와 좌표 상태(VALID/PENDING/INVALID)가 분리되어 저장된다.", 예외="좌표 확인이 필요한 후보는 좌표 보류로 보존한다.",
   경계="내부 이벤트 처리 | 결과 화면: UI-05 | 연결 시험: AT-F05, AT-F06")
S("SD-X08", "07", [("db", "DatasetBuilder", "tool"), ("tr", "Trainer", "tool"), ("ev", "Evaluator", "tool"), ("pc", "PairedComparator", "tool"),
                   ("ws", "WeightSouper", "tool"), ("eb", "EngineBuilder", "tool"), ("mr", "IModelRegistry", "interface")], [
    call("db", "db", "build(SourceDataset[])", "TrainingManifestDTO", "장소 분리 · 사람 31~94 px 크롭 목록을 만든다."),
    call("db", "db", "verifySplit(TrainingManifestDTO, EvaluationManifest)", "Decision", "학습·평가 장소 분리를 확인한다."),
    loop("같은 설정 · 순서만 바꿔 2회 이상", [
        call("db", "tr", "train(TrainingManifestDTO, seed)", "ModelArtifactDTO", "반복 모델을 만든다."),
        call("tr", "ev", "evaluate(ModelArtifactDTO)", "EvaluationReportDTO", "test_obl · test_v2 · test_kr 을 평가한다."),
    ]),
    call("ev", "pc", "compare(EvaluationReportDTO, EvaluationReportDTO)", "ComparisonReportDTO", "짝 부트스트랩 95% · 반복 흔들림으로 판정한다."),
    alt([("채택", [call("pc", "ws", "average(List<ModelArtifactDTO>)", "ModelArtifactDTO", "가중치 평균 후 BN·붕괴를 검사한다."),
                   call("ws", "eb", "build(ModelArtifactDTO, 운용 GPU)", "ModelArtifactDTO", "TensorRT FP16 [736,1280] 엔진 · 18.9 ms 검증"),
                   call("eb", "mr", "register(ModelArtifactDTO, EvaluationReportDTO)", "UUID", "운용 모델로 등록·활성화한다.")]),
         ("기각", [note("pc", "mr", "기존 활성 모델 유지")])]),
], title="학습·가중치 평균·운용 엔진", uc="UC-0701 설계 지원 / NFR-V01~V06",
   개요="오프라인 학습과 실시간 운용 경로를 분리한다.", 시작="학습 서버 작업 (DatasetBuilder)", 선행="장소 분리 평가셋 3개가 고정되어 있다.",
   사후="채택 모델의 가중치·평가·엔진이 버전으로 등록된다.", 예외="효과가 반복 흔들림 안이면 기각하고 기존 모델을 유지한다.",
   경계="학습 서버 (운용 경로 밖) | 결과 화면: — | 연결 시험: AT-P (NFR-V01·V03)")
