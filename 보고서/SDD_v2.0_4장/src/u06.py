"""영상 관리 — services.media · gateway.media · research.grace"""
from mdl import K, I, DTO, ENT, DAO, unit
from seq import S, call, alt, opt, note, loop
from mdl import SEQS
from cdkit import CD, layered, box

unit("06", "영상 관리", "영상·저장")
M = "services.media"

I("C-0611", "IVideoIngestService", M, "영상 수신·처리 상태의 공개 계약", [
    ("receive", "packet, arrivalTime", "FrameEnvelopeDTO?", "수신 프레임의 스트림·순서·해상도·손상을 검사하고 분석 경로에 넘긴다."),
    ("getProcessingStatus", "streamId", "VideoStatusDTO", "수신 FPS·실제 분석 FPS·지연·폐기·손실·최근 분석 시각을 나눠 제공한다."),
], impl="C-0601")
I("C-0612", "IMediaStore", M, "녹화본·스냅샷 저장·조회의 공개 계약 (비전·이력·학습이 의존)", [
    ("saveRecording", "SegmentManifestDTO", "MediaRefDTO", "임무·시각·기체·결손 구간을 녹화 구간과 연결해 등록한다."),
    ("persistBundle", "SegmentManifestDTO", "ResultDTO", "영상·sidecar·DB 가 모두 확정된 뒤에만 저장 완료를 응답한다."),
    ("queryRecording", "missionId, timeRange, PrincipalDTO", "List<MediaViewDTO>", "권한 검사 후 임무·시각별 녹화 구간을 제공한다."),
    ("createSnapshot", "frameId, bbox, candidateId", "MediaRefDTO", "초기 탐지의 전체 프레임·대상 부분·시각·신뢰도를 보존한다."),
    ("getAsset", "assetId", "MediaViewDTO", "연결된 스냅샷과 보존 상태를 조회한다."),
], impl="C-0604")
I("C-0613", "IFrameSource", M, "최신 분석 프레임 제공 계약 (비전이 의존)", [
    ("takeLatest", "now", "SyncedFrameDTO?", "처리 시점의 최신 유효 프레임만 꺼내고 오래된 대기 프레임을 버린다."),
    ("getMetrics", "", "BufferMetrics", "현재 대기·폐기 수와 최신 프레임 상태를 반환한다."),
], impl="C-0603")
I("C-0627", "IFrameAnalysisLedger", M, "프레임 분석 배정·완료 기록 계약 (비전·복구가 의존)", [
    ("schedule", "frameId, AnalysisMode", "AnalysisJobDTO", "유일키로 중복 배정을 막는다."),
    ("complete", "jobId, DetectionBatchDTO", "ResultDTO", "실제 추론 완료 시간·품질·산출 참조를 기록한다."),
    ("markSkipped", "frameId, ReasonCode", "ResultDTO", "전송 누락과 다른 선택 제외 사유를 집계한다."),
], impl="C-1206")

K("C-0601", "VideoIngestService", "service", M, "영상 입력 어댑터·실제 수신 상태 관리 — SRT 수신 경계 (IVideoIngestService 구현)", [
    ("synchronizer", "FrameSynchronizer", "촬영 시각·자세 결합"), ("buffer", "LatestFrameBuffer", "최신 프레임 대기"),
    ("frameDao", "VideoFrameDAO", "프레임 기록"), ("linkMonitor", "ILinkMonitor", "영상 링크 수신 시각"),
    ("streamId", "UUID", "기체별 영상 스트림"), ("receivedSeq", "int64", "수신 순서 (송신 순서와 구분)"),
], [("-decode", "packet", "DecodedFrame", "H.264 를 디코딩하고 손상 여부를 표시한다.")], impl="C-0611", old="C-0601")
K("C-0602", "FrameSynchronizer", "component", M, "촬영 시점과 비행정보를 연결하고 시간 불확실성을 판정", [
    ("vehicleState", "IVehicleStateService", "비행정보 창"), ("transform", "ICoordinateTransform", "자세 기준 확인"),
    ("maxSkew", "Duration", "허용 시각 오차 (측정 후 확정)"),
], [
    ("associate", "FrameEnvelopeDTO", "SyncedFrameDTO", "촬영 시각과 비행정보 시각을 맞추고 측정/추정·오차를 명시한다."),
    ("bindPose", "FrameEnvelopeDTO, TelemetryWindowDTO", "FramePoseBinding", "촬영 시각 앞뒤 GNSS·기체·짐벌 표본과 불확실도를 잇는다."),
    ("qualityDecision", "frame", "TimeQuality", "동기 오차·보간 근거로 위치·관측 사용 여부를 판정한다."),
], old="C-0602")
K("C-0603", "LatestFrameBuffer", "component", M, "추론 대기 프레임을 제한하고 오래된 프레임을 폐기 (IFrameSource 구현)", [
    ("qualityGate", "FrameQualityGate", "업무별 사용 조건"), ("latestFrame", "SyncedFrameDTO?", "최신 처리 가능 프레임 (최대 2칸)"),
    ("droppedCount", "int64", "폐기 누적수"),
], [("offer", "SyncedFrameDTO", "BufferResult", "최신 유효 프레임을 대기열에 넣는다.")], impl="C-0613", old="C-0603")
K("C-0608", "FrameQualityGate", "component", M, "원본 프레임의 업무별 사용 조건(표시·분석·좌표·관측 완료·자동 이동) 분리 + GRACE 크롭 표시 조건을 별도 정의", [
    ("policyVersion", "VersionId", "원본 업무별 승인 조건"), ("cropDisplayPolicy", "CropDisplayPolicy", "GRACE 크롭 표시 조건 (원본 업무 기준과 별도)"),
], [("evaluate", "SyncedFrameDTO", "TaskQualityDTO", "원본 프레임의 시각·참조·품질·나이·버전을 검사해 업무별 허용을 반환한다."),
    ("cropPolicy", "", "CropDisplayPolicy", "FULL 표시 · PARTIAL 은 '복원 영상' 표지 · NONE 은 표시하지 않음 · 허용 나이 초과 시 오래된 영상 표지."),
    ("evaluateCropDisplay", "CropFeedbackDTO, CropFrameDTO", "CropDisplayDecision", "크롭 표시 상태를 판정·기록한다. 탐지·좌표·관측 완료 판정에는 쓰지 않는다.")], old="C-0608")
K("C-0604", "MediaStore", "service", M, "녹화 구간·스냅샷·위치 보류 영상의 저장 및 조회 (IMediaStore 구현)", [
    ("assetDao", "MediaAssetDAO", "파일 참조 기록"), ("permissionPolicy", "IPermissionPolicy", "조회 권한"),
    ("rootPath", "Path", "중앙 미디어 저장 루트"), ("retentionPolicy", "RetentionPolicy", "보존 기간·용량 (현장 용량 확인 후 확정)"),
], [("-toDTO", "MediaAsset", "MediaViewDTO", "엔티티를 조회용 DTO 로 바꾼다.")], impl="C-0612", old="C-0604")
K("C-1206", "FrameAnalysisLedger", "service", M, "분석 배정·완료·선택 제외·과거 영상 재분석 이력 (IFrameAnalysisLedger 구현)", [
    ("frameDao", "VideoFrameDAO", "video_frame 분석 상태·시간"),
], [], impl="C-0627", old="C-1206")
K("C-0607", "ArchiveRecoveryService", "service", M, "잔여 대역폭으로 Pi 원본 영상 조각을 회수하고 사후 분석을 요청", [
    ("gatewayLink", "IGatewayLink", "Pi 조각 이어받기"), ("mediaStore", "IMediaStore", "저장 확정"),
    ("ledger", "IFrameAnalysisLedger", "사후 분석 배정"), ("trafficPolicy", "TrafficPolicy", "P4 잔여 용량 제한"),
], [
    ("recover", "SegmentManifestDTO", "ResultDTO", "서버 확인 위치부터 이어 받고 최종 해시·저장 완료를 확인한다."),
    ("queryPersisted", "segmentId, hash", "ResultDTO", "ACK 유실 시 중복 저장 없이 상태를 조회한다."),
    ("queueHistoricalAnalysis", "SegmentManifestDTO", "List<AnalysisJobDTO>", "원관측 시각을 유지한 사후 분석으로만 배정한다."),
], old="C-0607")
K("C-0628", "VideoRelay", "component", "gateway.media", "Pi 의 영상 수신·현장 기록·서버 중계 (SRT 송신)", [
    ("recorder", "FieldRecorder", "현장 순환 기록"), ("srtSession", "SrtSession", "서버 송신 세션"),
], [("relay", "packet", "RelayResult", "수신 영상을 현장에 기록하고 서버로 보낸다.")])
K("C-0606", "VideoProfileManager", "component", "services.media.crop_grace", "서버→관제 단말 크롭 제공 프로파일(CROP_GRACE · CROP_BASELINE) 협상·전환 — 드론→게이트웨이→서버 상향 SRT 와 별개", [
    ("activeProfile", "CropProfile", "초기 CROP_BASELINE (기존 크롭 제공)"), ("capabilities", "TerminalCapabilityDTO", "단말 디코더 가용·modelVersion·패킷 형식·성능 측정값"),
    ("approvedPolicy", "ApprovedPolicy", "CROP_GRACE 허용 조건 (실험 초기값 기준 · 미검증)"),
], [
    ("negotiate", "TerminalCapabilityDTO, ApprovedPolicy", "ProfileDecision", "디코더 가용·모델/패킷 호환·성능 조건을 모두 만족할 때만 CROP_GRACE, 아니면 CROP_BASELINE 을 고른다."),
    ("switchProfile", "sessionId, CropProfile, ReasonCode", "CropSessionDTO", "새 epoch·코덱 세션을 발급하고 참조 재동기화를 요구한다. 상향 SRT 경로는 바꾸지 않는다."),
], old="C-0606")
K("C-0609", "GraceCodecAdapter", "component", "services.media.crop_grace", "서버 측 GRACE 크롭 인코딩·패킷화 — 단말의 부분 패킷 디코딩은 GraceCropDecoder(C-0633)가 맡는다 (같은 modelVersion 필요)", [
    ("session", "CodecSession", "modelVersion·패킷 형식·참조 세대·epoch"), ("limits", "CodecLimits", "600 ms·1200 B (실험 초기값 · 운용 성능 미검증)"),
], [
    ("encodeCrop", "CropImage, CodecSession", "LatentFrame", "서버 참조 상태로 크롭을 잠재 표현으로 바꾼다."),
    ("packetize", "LatentFrame, CodecLimits", "EncodedPackets", "독립 디코딩 가능한 패킷으로 나누고 frameId·epoch·순번을 붙인다."),
    ("requestBaseReference", "ReasonCode", "ResyncRequest", "참조 복구 불가 시 새 기준 프레임으로 다시 시작한다."),
], old="C-0609")
K("C-0610", "CodecStateSync", "component", "services.media.crop_grace", "단말이 실제 디코딩에 쓴 패킷 bitmap 으로 서버·단말 참조 상태를 일치", [
    ("referenceGeneration", "int64", "참조 상태 세대"), ("cacheLimit", "CacheBudget", "32프레임 또는 256 MiB (실험 초기값)"),
    ("cropKey", "targetId, cropSize", "바뀌면 참조 초기화"),
], [
    ("applyFeedback", "CropFeedbackDTO", "ResyncTag", "usedPacketBitmap 대로 서버 참조를 갱신해 단말과 같은 참조를 쓴다."),
    ("checkReset", "targetId, cropBox", "bool", "대상 변경·크롭 크기 변경·epoch 불일치·캐시 만료면 참조를 초기화한다."),
    ("reset", "ReasonCode", "CodecSession", "새 세션(epoch + 1)을 발급한다. 복구가 다시 실패하면 CROP_BASELINE 대체를 요청한다."),
], old="C-0610")
I("C-0630", "ICropDeliveryService", "services.media.crop_grace", "관제 단말 대상 크롭 제공 계약 (비전·api 가 의존)", [
    ("openSession", "targetId, TerminalCapabilityDTO", "CropSessionDTO", "단말 조건으로 제공 프로파일을 협상하고 세션·epoch·표시 조건을 발급한다."),
    ("publishCrop", "SyncedFrameDTO, targetId, cropBox", "CropFrameDTO", "원본 프레임에서 대상 크롭을 만들어 원본 크롭을 보존하고 협상된 프로파일로 보낸다. 좌표 보류 후보도 대상이다."),
    ("acceptFeedback", "CropFeedbackDTO", "ResyncTag", "단말이 실제 사용한 패킷 bitmap 과 수신 상태를 받아 참조·표시 상태를 갱신한다."),
    ("closeSession", "sessionId, ReasonCode", "ResultDTO", "대상 해제·화면 이탈 시 세션과 참조를 정리한다."),
], impl="C-0631")
K("C-0631", "CropDeliveryService", "service", "services.media.crop_grace", "대상 크롭 생성·GRACE 제공·기존 방식 대체 (ICropDeliveryService 구현) — 원본 분석 경로와 분리", [
    ("profileManager", "VideoProfileManager", "제공 프로파일"), ("codec", "GraceCodecAdapter", "서버 인코딩·패킷화"),
    ("stateSync", "CodecStateSync", "참조 동기화"), ("qualityGate", "FrameQualityGate", "크롭 표시 조건"),
    ("mediaStore", "IMediaStore", "원본 크롭 보존 · 기존 크롭 제공"), ("endpoint", "CropStreamEndpoint", "단말 송신 경계"),
], [
    ("-cropFrame", "SyncedFrameDTO, cropBox", "CropImage", "원본 해상도에서 대상 영역을 잘라 크롭 frameId 를 붙인다 (sourceFrameId 유지)."),
    ("-deliverFallback", "CropFrameDTO, ReasonCode", "SendResult", "보존한 원본 크롭을 기존 방식으로 보낸다."),
], impl="C-0630")
K("C-0632", "CropStreamEndpoint", "boundary", "api", "서버 쪽 크롭 스트림 끝점 — 단말 세션 열기 · 서버→관제 단말 크롭 패킷 송신 · 단말→서버 디코딩 결과 피드백 수신 (전송 기술은 구현·호환 검증 후 확정)", [
    ("cropService", "ICropDeliveryService", "세션·피드백 전달"),
], [
    ("open", "targetId, TerminalCapabilityDTO", "CropSessionDTO", "단말의 세션 요청을 ICropDeliveryService.openSession 으로 넘긴다."),
    ("sendPackets", "CropFrameDTO, EncodedPackets", "SendResult", "크롭 메타데이터와 패킷을 해당 단말 세션으로 보낸다."),
    ("sendFallback", "CropFrameDTO, MediaRefDTO", "SendResult", "기존 방식 크롭(보존한 원본 크롭 참조)을 보낸다."),
    ("onFeedback", "CropFeedbackDTO", "void", "받은 피드백을 ICropDeliveryService.acceptFeedback 으로 넘긴다."),
])
K("C-0633", "GraceCropDecoder", "component", "web", "관제 단말 전용 GRACE 디코더 — 부분 패킷 디코딩·FULL/PARTIAL/NONE 판정·사용 패킷 보고 (PWA·WebCodecs 기본 지원을 가정하지 않음)", [
    ("runtime", "DecoderRuntime", "전용 디코더 실행 환경 (배포 조건 · 미검증)"), ("modelVersion", "String", "서버 인코더와 같은 버전"),
    ("reference", "ReferenceState", "참조 세대·epoch"),
], [
    ("probe", "", "TerminalCapabilityDTO", "디코더 가용·modelVersion·패킷 형식·디코딩 시간을 보고한다."),
    ("decodePartial", "PacketSubset, Deadline", "DecodedCrop", "기한 안에 받은 패킷만으로 복원하고 FULL/PARTIAL/NONE·usedPacketBitmap 을 기록한다. NONE 은 복원하지 않는다."),
    ("reportUsed", "frameId", "CropFeedbackDTO", "실제 사용한 패킷 bitmap 과 수신 상태를 만든다."),
])
K("C-0634", "CropViewModel", "viewmodel", "web", "후보 화면(UI-05)의 크롭 표시 상태 — 복원 영상과 원본 크롭을 구분해 표시 (클라이언트)", [
    ("decoder", "GraceCropDecoder", "단말 디코더"), ("session", "CropSessionDTO", "프로파일·epoch·표시 조건"),
    ("lastShown", "DecodedCrop?", "마지막 표시 영상과 captureTime"),
], [
    ("openCandidate", "targetId", "void", "후보 상세에서 크롭 표시를 시작한다 — 디코더를 확인하고 세션을 연다."),
    ("onCropPackets", "CropFrameDTO, PacketSubset", "void", "받은 크롭 메타데이터·패킷을 디코더에 넘기고 표시를 갱신한다."),
    ("show", "DecodedCrop, CropDisplayPolicy", "void", "표시 조건을 만족하면 '복원 영상' 표지와 원본 촬영 시각을 함께 표시한다."),
    ("holdLast", "captureTime, ReasonCode", "void", "NONE·대체 지연 시 마지막 영상을 유지하고 관측 시각·오래된 영상 표지를 붙인다."),
    ("showFallback", "MediaViewDTO", "void", "기존 방식으로 받은 원본 크롭을 표시한다."),
])
K("C-0614", "VideoController", "controller", "api", "영상 처리 상태·녹화본 조회 요청의 입구 (REST)", [
    ("videoIngest", "IVideoIngestService", ""), ("mediaStore", "IMediaStore", ""),
], [
    ("getStatus", "streamId", "VideoStatusDTO", "GET /api/video/{id}/status"),
    ("getRecordings", "missionId, timeRange, PrincipalDTO", "List<MediaViewDTO>", "GET /api/missions/{id}/recordings"),
    ("getAsset", "assetId", "MediaViewDTO", "GET /api/media/{id}"),
])

DTO("C-0615", "FrameEnvelopeDTO", "수신 프레임 봉투", [("streamId", "String", "스트림"), ("streamEpoch", "int", "연속 구간"), ("frameSeq", "int64", "순번"),
    ("captureAt", "TimeEvidence", "촬영 시각과 근거"), ("frame", "DecodedFrame", "1920×1080 영상"), ("damaged", "bool", "손상")])
DTO("C-0616", "SyncedFrameDTO", "비행정보와 결합된 분석용 프레임", [("frameId", "UUID", "프레임"), ("captureAt", "Time", "촬영 시각"),
    ("image", "Image", "1920×1080"), ("pose", "FramePoseBinding?", "보간 자세·불확실도"), ("timeQuality", "TimeQuality", "시각 품질"),
    ("taskQuality", "TaskQualityDTO", "업무별 사용 조건")])
DTO("C-0617", "MediaRefDTO", "저장 파일 참조", [("assetId", "UUID", "파일"), ("state", "WRITING | STORED | FAILED", "상태"), ("storageUri", "String", "경로")])
DTO("C-0618", "MediaViewDTO", "녹화본·스냅샷 조회 결과", [("assetId", "UUID", "파일"), ("assetType", "AssetType", "종류"), ("url", "String", "재생 주소"),
    ("timeRange", "TimeRange", "구간"), ("gaps", "List<TimeRange>", "결손")])
DTO("C-0619", "VideoStatusDTO", "영상 처리 상태", [("receiveFps", "float", "수신 FPS"), ("analyzeFps", "float", "실제 분석 FPS"),
    ("latencyMs", "float", "지연"), ("dropped", "int64", "폐기"), ("lossRate", "float", "손실"), ("lastAnalyzedAt", "Time", "최근 분석"),
    ("cropStats", "CropDisplayStats?", "관제 단말 크롭 FULL·PARTIAL·NONE 비율·대체 횟수")])
DTO("C-0620", "SegmentManifestDTO", "영상 조각 묶음", [("segmentId", "UUID", "조각"), ("files", "List<FileEntry>", "영상·sidecar 크기·sha256"),
    ("timeRange", "TimeRange", "구간"), ("missionId", "UUID?", "임무")])
DTO("C-0621", "TaskQualityDTO", "업무별 사용 조건 판정", [("display", "bool", "표시"), ("detect", "bool", "분석"), ("geo", "bool", "좌표"),
    ("coverage", "bool", "관측 완료"), ("autonomy", "bool", "자동 이동"), ("reasons", "List<ReasonCode>", "제외 사유")])
DTO("C-0629", "AnalysisJobDTO", "분석 배정", [("jobId", "UUID", "작업"), ("frameId", "UUID", "프레임"), ("mode", "LIVE_DETECT | LIVE_TRACK_ONLY | HISTORICAL", "방식"),
    ("modelConfigId", "UUID", "모델")])

DTO("C-0635", "CropFrameDTO", "크롭 프레임 연결 — sourceFrameId(원본) 1 : N frameId(크롭) · cropBox 는 원본 좌표계 · captureTime 은 원본 촬영 시각 · (frameId, epoch, modelVersion) 이 맞을 때만 디코딩·피드백 적용", [("sourceFrameId", "UUID", "원본 분석 프레임 (video_frame)"), ("frameId", "UUID", "크롭 프레임"),
    ("targetId", "UUID", "대상 후보"), ("cropBox", "Box", "원본 좌표계 크롭 영역"), ("captureTime", "Time", "원본 촬영 시각"),
    ("epoch", "int", "코덱 세션 세대"), ("modelVersion", "String", "코덱 모델"), ("originalRef", "MediaRefDTO", "보존한 원본 크롭")])
DTO("C-0636", "CropFeedbackDTO", "단말 디코딩 결과 — (frameId, epoch) 로 CropFrameDTO 와 연결 · usedPacketBitmap 은 그 크롭 프레임 패킷 순번 중 실제 사용분", [("frameId", "UUID", "크롭 프레임"), ("epoch", "int", "세션 세대"),
    ("reception", "FULL | PARTIAL | NONE", "수신 상태"), ("usedPacketBitmap", "Bitmap", "디코딩에 실제 사용한 패킷"), ("decodeMs", "float", "디코딩 시간"),
    ("displayed", "bool", "표시 여부")])
DTO("C-0637", "CropSessionDTO", "크롭 제공 세션", [("sessionId", "UUID", "세션"), ("targetId", "UUID", "대상"), ("profile", "CROP_GRACE | CROP_BASELINE", "제공 방식"),
    ("epoch", "int", "세션 세대"), ("modelVersion", "String", "코덱 모델"), ("cropSize", "Size", "크롭 크기"), ("displayPolicy", "CropDisplayPolicy", "표시 조건")])
DTO("C-0638", "TerminalCapabilityDTO", "관제 단말 디코더 조건", [("decoderAvailable", "bool", "전용 디코더 가용"), ("modelVersion", "String", "디코더 모델"),
    ("packetFormat", "String", "패킷 형식"), ("decodeMs", "float", "측정 디코딩 시간"), ("runtime", "String", "실행 환경")])
DAO("C-0622", "VideoFrameDAO", "video_frame", [
    ("insert", "VideoFrame", "UUID", "수신 프레임을 기록한다 (스트림·epoch·순번 유일)."),
    ("updatePose", "frameId, pose, poseStatus", "bool", "촬영 자세 결합 결과를 저장한다."),
    ("updateAnalysis", "frameId, state, times", "bool", "분석 상태·시간을 저장한다."),
    ("updateCoverage", "frameId, footprint, coverage", "bool", "촬영 범위·셀별 관측 판정을 저장한다."),
    ("aggregateMetrics", "streamId, window", "StreamMetrics", "수신·분석 FPS 와 추론 시간을 집계한다."),
])
DAO("C-0623", "MediaAssetDAO", "media_asset", [
    ("insert", "MediaAsset", "UUID", "파일 참조를 WRITING 으로 등록한다."),
    ("updateState", "assetId, state, hash", "bool", "확정·실패 상태와 해시를 저장한다."),
    ("findByMission", "missionId, timeRange", "List<MediaAsset>", "녹화 구간을 읽는다."),
    ("findById", "assetId", "MediaAsset?", "파일 참조를 읽는다."),
])
ENT("C-0624", "VideoFrame", "video_frame", "수신 프레임·자세 결합·분석·관측 판정 (DB-12 한 행)",
    keys=["frameId {PK}", "missionId · droneId {FK}", "streamEpoch · frameSeq {UQ}", "captureAt · poseStatus", "analysisState · inferMs"])
ENT("C-0625", "MediaAsset", "media_asset", "녹화본·스냅샷·모델 등 파일 참조 (DB-13 한 행)",
    keys=["assetId {PK}", "missionId · frameId · candidateId {FK}", "assetType", "storageUri · contentHash", "state · retentionClass"])


def _cd():
    a = layered("cd06a", "(1/3) 계층 구조", svc_pkg=M, ctl=["C-0614"],
                dto=["C-0615", "C-0616", "C-0617", "C-0618", "C-0619", "C-0620", "C-0621", "C-0629"],
                pairs=[("C-0611", "C-0601"), ("C-0612", "C-0604"), ("C-0613", "C-0603"), ("C-0627", "C-1206")],
                comps=["C-0602", "C-0608"],
                ext=[("C-0206", "services.vehicle"), ("C-0205", "services.vehicle"), ("C-0306", "services.spatial"), ("C-0108", "services.auth")],
                daos=[("C-0622", False), ("C-0623", False)], ents=["C-0624", "C-0625"], api_w=0.33, dto_cols=4, ext_w=0.22)
    B = {k: box(k) for k in ["C-0628", "C-0607"]}
    B["fr"] = box("C-1202", ref=True); B["sm"] = box("C-1205", ref=True)
    B["gl"] = box("C-0909", ref=True); B["ms"] = box("C-0612", ref=True); B["al"] = box("C-0627", ref=True)
    P = [dict(name="gateway.media", row=0, x=0.005, w=0.49, rows=[[("C-0628", 1.0)], [("fr", .5), ("sm", .5)]]),
         dict(name=M, row=0, x=0.505, w=0.49, rows=[[("C-0607", 1.0)], [("ms", .5), ("al", .5)]]),
         dict(name="services.command", row=1, x=0.505, w=0.49, rows=[[("gl", 1.0)]])]
    R = [("C-0628", "fr", "assoc", "", {"elbow": 1}), ("C-0607", "ms", "dep", "", {"elbow": 1}), ("C-0607", "al", "dep", "", {"elbow": 1}),
         ("C-0607", "gl", "dep", "", {"elbow": 1, "ax": 0.85})]
    C3 = {k: box(k) for k in ["C-0632", "C-0633", "C-0634", "C-0630", "C-0631", "C-0606", "C-0609", "C-0610",
                              "C-0635", "C-0636", "C-0637", "C-0638"]}
    C3["gate"] = box("C-0608", ref=True); C3["ms2"] = box("C-0612", ref=True)
    P3 = [dict(name="api — 서버 경계", row=0, x=0.005, w=0.49, rows=[[("C-0632", 1.0)]]),
          dict(name="web — 관제 운용자 단말 (apps/web)", row=0, x=0.505, w=0.49, rows=[[("C-0634", .5), ("C-0633", .5)]]),
          dict(name="services.media.crop_grace (서버)", row=1, x=0.005, w=0.74,
               rows=[[("C-0630", .5), ("C-0631", .5)], [("C-0606", .333), ("C-0609", .333), ("C-0610", .334)]]),
          dict(name="services.media", row=1, x=0.755, w=0.24, rows=[[("gate", 1.0)], [("ms2", 1.0)]]),
          dict(name="contracts — 크롭 DTO", row=2, x=0.005, w=0.99, rows=[[("C-0635", .25), ("C-0636", .25), ("C-0637", .25), ("C-0638", .25)]])]
    R3 = [("C-0632", "C-0630", "dep", "세션·피드백", {"x": 0.29}), ("C-0631", "C-0630", "real", "", {}),
          ("C-0631", "C-0606", "assoc", "", {"elbow": 1}), ("C-0631", "C-0609", "assoc", "", {"elbow": 1}),
          ("C-0631", "C-0610", "assoc", "", {"elbow": 1}), ("C-0631", "C-0632", "assoc", "송신", {}),
          ("C-0631", "gate", "dep", "", {}), ("C-0631", "ms2", "dep", "", {}),
          ("C-0634", "C-0633", "assoc", "", {})]
    return [a, ("cd06b", "(2/3) 현장 중계·원본 회수", P, B, R), ("cd06c", "(3/3) GRACE 크롭 제공 — 서버 인코딩 → 관제 단말 디코딩", P3, C3, R3)]


CD("CD-06", "영상·저장", "06", _cd)

S("SD-0601", "06", [("cam", "카메라 영상 링크", "external"), ("vr", "VideoRelay", "component"), ("fr", "FieldRecorder", "component"),
                    ("vi", "IVideoIngestService", "interface"), ("vf", "VideoFrameDAO", "dao"), ("lm", "ILinkMonitor", "interface"),
                    ("fs", "FrameSynchronizer", "component")], [
    call("cam", "vr", "relay(packet)", "RelayResult", "Pi 가 H.264 영상을 받는다.", [
        call("vr", "fr", "appendVideo(frame)", "WriteResult", "압축 스트림과 시각·자세 sidecar 를 WAN 과 독립적으로 기록한다."),
        call("vr", "vi", "receive(packet, arrivalTime)", "FrameEnvelopeDTO", "SRT 로 서버에 보내고 형식·epoch·PTS·손상을 검사한다.", [
            call("vi", "vf", "insert(VideoFrame)", "frameId", "수신 프레임을 기록한다."),
            call("vi", "lm", "recordReception(droneId, VIDEO, time)", None, "실제 프레임 수신으로 마지막 정상 영상 시각을 갱신한다."),
            call("vi", "fs", "associate(FrameEnvelopeDTO)", "SyncedFrameDTO", "촬영 자세를 결합한다 (SD-0602)."),
        ]),
    ]),
    opt("영상 끊김 (1a)", [note("vi", "fs", "마지막 정상 시각 기록 → 영상 장애로 처리 (2초)")]),
], entry="VideoRelay.relay → IVideoIngestService.receive", 시작="현장 영상 수신 (카메라 → 게이트웨이 VideoRelay)")
S("SD-0602", "06", [("vi", "VideoIngestService", "service"), ("fs", "FrameSynchronizer", "component"), ("vs", "IVehicleStateService", "interface"),
                    ("ct", "ICoordinateTransform", "interface"), ("vf", "VideoFrameDAO", "dao")], [
    call("vi", "fs", "associate(FrameEnvelopeDTO)", "SyncedFrameDTO", "영상에 촬영 시점의 위치·고도·자세를 연결한다.", [
        call("fs", "vs", "getTelemetryWindow(droneId, captureAt, span)", "TelemetryWindowDTO", "촬영 시각 앞뒤 GNSS·기체·짐벌 표본을 받는다."),
        call("fs", "fs", "bindPose(FrameEnvelopeDTO, TelemetryWindowDTO)", "FramePoseBinding", "보간 자세와 불확실도를 계산한다."),
        call("fs", "ct", "validatePoseReference(Pose, contextId)", "ReferenceDecision", "ENU/NED·방위·고도 기준을 확인한다."),
        call("fs", "vf", "updatePose(frameId, pose, poseStatus)", "bool", "결합 결과와 시각 종류·불확실성을 저장한다."),
        opt("시간 대응 미확인 (2a)", [note("fs", "vf", "PENDING — 좌표 계산·완료도 반영 보류")]),
    ]),
], entry="FrameSynchronizer.associate", 시작="중앙 서버의 새 프레임 수신 (VideoIngestService)")
S("SD-0603", "06", [("fi", "FrameIngestor", "component"), ("src", "IFrameSource", "interface"), ("qg", "FrameQualityGate", "component"),
                    ("al", "IFrameAnalysisLedger", "interface"), ("vf", "VideoFrameDAO", "dao"), ("pd", "IPersonDetectionService", "interface")], [
    call("fi", "src", "takeLatest(now)", "SyncedFrameDTO?", "실행 중 프레임 외 최신 디코딩 프레임 2개 상한에서 고른다.", [
        call("src", "qg", "evaluate(SyncedFrameDTO)", "TaskQualityDTO", "탐지·좌표·관측 누적·자동 이동 허용을 따로 판정한다."),
    ]),
    alt([("분석 허용", [
            call("fi", "al", "schedule(frameId, LIVE_DETECT)", "AnalysisJobDTO", "원관측·모델·파이프라인의 중복 배정을 막는다.", [
                call("al", "vf", "updateAnalysis(frameId, SELECTED, times)", "bool", "배정 시각을 기록한다.")]),
            call("fi", "pd", "detect(SyncedFrameDTO)", "DetectionBatchDTO", "허용된 프레임을 분석한다."),
            call("fi", "al", "complete(jobId, DetectionBatchDTO)", "ResultDTO", "완료 시간을 기록한다."),
         ]),
         ("손상·시각 오류 (1a)", [call("fi", "al", "markSkipped(frameId, ReasonCode)", "ResultDTO", "제외 사유를 기록한다.")])]),
], entry="FrameIngestor (비전 작업자)", 시작="비전 작업자 처리 가능 이벤트 (FrameIngestor)")
S("SD-0604", "06", [("op", "관제 운영자", "actor"), ("ctl", "VideoController", "controller"), ("vi", "IVideoIngestService", "interface"),
                    ("vf", "VideoFrameDAO", "dao"), ("src", "IFrameSource", "interface")], [
    call("op", "ctl", "getStatus(streamId)", "VideoStatusDTO", "영상 처리 상태를 조회한다.", [
        call("ctl", "vi", "getProcessingStatus(streamId)", "VideoStatusDTO", "수신률과 실제 분석 처리율을 나눠 만든다.", [
            call("vi", "vf", "aggregateMetrics(streamId, window)", "StreamMetrics", "추론이 완료된 프레임 수로 분석 FPS 를 집계한다 (2a)."),
            call("vi", "src", "getMetrics()", "BufferMetrics", "현재 대기·폐기 수를 가져온다."),
        ]),
    ]),
], entry="VideoController.getStatus")
S("SD-0605", "06", [("sm", "SegmentManifestStore", "dao"), ("ar", "ArchiveRecoveryService", "service"), ("ms", "IMediaStore", "interface"),
                    ("ma", "MediaAssetDAO", "dao"), ("hs", "IHistoryService", "interface")], [
    call("sm", "ar", "recover(SegmentManifestDTO)", "ResultDTO", "Pi 가 확정한 조각을 서버로 이어 보낸다.", [
        call("ar", "ms", "saveRecording(SegmentManifestDTO)", "MediaRefDTO", "녹화 구간을 등록한다.", [
            call("ms", "ma", "insert(MediaAsset)", "assetId", "WRITING 으로 등록한다.")]),
        call("ar", "ms", "persistBundle(SegmentManifestDTO)", "ResultDTO", "파일·sidecar·DB 가 확정된 뒤에만 ACK 한다.", [
            call("ms", "ma", "updateState(assetId, STORED, hash)", "bool", "해시를 대조해 확정한다.")]),
        opt("저장·해시 실패 (2a)", [call("ar", "hs", "append(MissionEventDTO)", "EventRefDTO", "오류와 기록 가능한 구간을 남긴다.")]),
    ]),
], entry="ArchiveRecoveryService.recover", 시작="녹화 구간 확정 (게이트웨이 SegmentManifestStore → 서버)")
S("SD-0606", "06", [("op", "관제 운영자", "actor"), ("ctl", "VideoController", "controller"), ("ms", "IMediaStore", "interface"),
                    ("pp", "IPermissionPolicy", "interface"), ("ma", "MediaAssetDAO", "dao")], [
    call("op", "ctl", "getRecordings(missionId, timeRange, PrincipalDTO)", "List<MediaViewDTO>", "녹화본을 조회한다.", [
        call("ctl", "ms", "queryRecording(missionId, timeRange, PrincipalDTO)", "List<MediaViewDTO>", "녹화 구간을 조회한다.", [
            call("ms", "pp", "checkAccess(PrincipalDTO, VIDEO_VIEW)", "AccessDecisionDTO", "영상 조회 권한을 검사한다."),
            call("ms", "ma", "findByMission(missionId, timeRange)", "List<MediaAsset>", "임무·시각별 녹화 구간을 읽는다."),
            opt("녹화본 없음·조회 실패 (2a)", [note("ms", "ma", "사유와 조회 가능한 범위를 표시")]),
        ]),
    ]),
], entry="VideoController.getRecordings")
S("SD-0607", "06", [("ts", "TargetService", "service"), ("sw", "SnapshotWriter", "component"), ("ms", "IMediaStore", "interface"),
                    ("ma", "MediaAssetDAO", "dao"), ("hs", "IHistoryService", "interface")], [
    call("ts", "sw", "save(SyncedFrameDTO, DetectionDTO, candidateId)", "MediaRefDTO", "첫 탐지의 근거 영상을 보존한다.", [
        call("sw", "ms", "createSnapshot(frameId, bbox, candidateId)", "MediaRefDTO", "전체 프레임과 대상 크롭을 저장한다.", [
            call("ms", "ma", "insert(MediaAsset)", "assetId", "임시 쓰기 후 해시를 확정해 후보와 잇는다.")]),
        opt("저장 실패", [call("sw", "hs", "append(MissionEventDTO)", "EventRefDTO", "실패와 누락 상태를 기록한다 (후보는 유지).")]),
    ]),
], entry="SnapshotWriter.save", 시작="첫 탐지 또는 명시적 스냅샷 생성 (TargetService)")
# SD-X06 — 한 장에 담으면 생명선 11개·22단계로 글자가 너무 작아져 세 장으로 나눠 그린다 (번호·처리표는 하나로 잇는다)
X6_A = [
    call("op", "vm", "openCandidate(targetId)", None, "후보 상세(UI-05)에서 크롭 영상을 연다 — UC-0702 후보 조회 · UC-0705 판단 전 확인.", [
        call("vm", "dec", "probe()", "TerminalCapabilityDTO", "전용 디코더 실행 환경·modelVersion·패킷 형식·디코딩 시간을 확인한다."),
        call("vm", "ep", "open(targetId, TerminalCapabilityDTO)", "CropSessionDTO", "단말 → 서버 경계로 크롭 세션을 연다 (전송 기술 미정).", [
            call("ep", "cds", "openSession(targetId, TerminalCapabilityDTO)", "CropSessionDTO", "세션·epoch·표시 조건을 발급한다.", [
                call("cds", "vp", "negotiate(TerminalCapabilityDTO, ApprovedPolicy)", "ProfileDecision", "디코더 가용·모델/패킷 호환·성능 조건을 모두 만족하면 CROP_GRACE, 아니면 CROP_BASELINE 을 고른다 (상향 SRT 와 무관)."),
                call("cds", "qg", "cropPolicy()", "CropDisplayPolicy", "원본 업무 기준과 별도인 크롭 표시 조건을 세션에 싣는다."),
            ]),
        ]),
    ]),
]
X6_B = [
    call("ts", "cds", "publishCrop(SyncedFrameDTO, targetId, cropBox)", "CropFrameDTO", "원본 분석 경로의 탐지 결과로 크롭을 요청한다 — 탐지·좌표·관측 완료 판정은 원본으로 이미 수행했다.", [
        call("cds", "ms", "createSnapshot(frameId, cropBox, targetId)", "MediaRefDTO", "원본 크롭을 보존한다 — 복원 영상과 구분 (UC-0607 스냅샷)."),
        call("cds", "ss", "checkReset(targetId, cropBox)", "bool", "대상·크롭 크기가 바뀌면 참조를 초기화하고 epoch 를 올린다."),
        alt([("CROP_GRACE", [
                call("cds", "gc", "encodeCrop(CropImage, CodecSession)", "LatentFrame", "서버 참조 상태로 크롭을 인코딩한다."),
                call("cds", "gc", "packetize(LatentFrame, CodecLimits)", "EncodedPackets", "frameId·epoch·순번을 붙인다 (1200 B 는 실험 초기값)."),
                call("cds", "ep", "sendPackets(CropFrameDTO, EncodedPackets)", "SendResult", "서버 → 단말 경계로 보낸다 (전송 기술 미정).", [
                    call("ep", "vm", "onCropPackets(CropFrameDTO, PacketSubset)", None, "단말은 기한 안에 받은 패킷만 넘긴다.", [
                        call("vm", "dec", "decodePartial(PacketSubset, Deadline)", "DecodedCrop", "FULL·PARTIAL 은 받은 패킷으로 복원하고 NONE 은 복원하지 않는다."),
                        alt([("FULL · PARTIAL", [call("vm", "vm", "show(DecodedCrop, CropDisplayPolicy)", None, "'복원 영상' 표지와 원본 촬영 시각을 함께 표시한다.")]),
                             ("NONE", [call("vm", "vm", "holdLast(captureTime, ReasonCode)", None, "마지막 영상을 유지하고 관측 시각·오래된 영상 표지를 붙인다.")])]),
                    ]),
                ]),
             ]),
             ("CROP_BASELINE", [
                call("cds", "ep", "sendFallback(CropFrameDTO, MediaRefDTO)", "SendResult", "기존 크롭 제공 방식으로 보낸다.", [
                    call("ep", "vm", "showFallback(MediaViewDTO)", None, "보존한 원본 크롭을 기존 방식으로 표시한다."),
                ]),
             ])]),
    ]),
]
X6_C = [
    call("vm", "dec", "reportUsed(frameId)", "CropFeedbackDTO", "수신 상태(FULL·PARTIAL·NONE)와 실제 사용 패킷 bitmap 을 만든다."),
    call("vm", "ep", "onFeedback(CropFeedbackDTO)", None, "단말 → 서버 경계로 피드백을 보낸다.", [
        call("ep", "cds", "acceptFeedback(CropFeedbackDTO)", "ResyncTag", "피드백을 넘긴다.", [
            call("cds", "ss", "applyFeedback(CropFeedbackDTO)", "ResyncTag", "단말이 실제 사용한 패킷으로 서버 참조를 맞춘다."),
            call("cds", "qg", "evaluateCropDisplay(CropFeedbackDTO, CropFrameDTO)", "CropDisplayDecision", "표시 상태를 기록한다 — UC-0604 처리 상태에 집계, 업무 판정에는 쓰지 않는다."),
            alt([("참조 불일치 · epoch 불일치 · 캐시 만료", [
                    call("cds", "ss", "reset(ReasonCode)", "CodecSession", "새 세션(epoch + 1)을 발급한다."),
                    call("cds", "gc", "requestBaseReference(ReasonCode)", "ResyncRequest", "다음 크롭부터 새 기준 프레임으로 다시 보낸다."),
                 ]),
                 ("디코더 미가용 · 호환 불일치 · 복구 반복 실패", [
                    call("cds", "vp", "switchProfile(sessionId, CROP_BASELINE, ReasonCode)", "CropSessionDTO", "이후 크롭은 기존 제공 방식으로 대체한다 — 원본 분석 경로는 영향 없음."),
                 ])]),
        ]),
    ]),
]
X6_LOOP = "대상이 관측된 원본 프레임마다 (좌표 보류 후보 포함)"
X6_P = {"op": ("op", "관제 운영자", "actor"), "vm": ("vm", "CropViewModel", "viewmodel"), "dec": ("dec", "GraceCropDecoder", "component"),
        "ep": ("ep", "CropStreamEndpoint", "boundary"), "ts": ("ts", "TargetService", "service"), "cds": ("cds", "ICropDeliveryService", "interface"),
        "vp": ("vp", "VideoProfileManager", "component"), "qg": ("qg", "FrameQualityGate", "component"), "ms": ("ms", "IMediaStore", "interface"),
        "ss": ("ss", "CodecStateSync", "component"), "gc": ("gc", "GraceCodecAdapter", "component")}
S("SD-X06", "06", [X6_P[k] for k in ["op", "vm", "dec", "ep", "ts", "cds", "vp", "qg", "ms", "ss", "gc"]],
  X6_A + [loop(X6_LOOP, X6_B + X6_C)],
  title="GRACE 대상 크롭 제공 — 서버 → 관제 운용자 단말", uc="UC-0702 · UC-0705 (중심), UC-0604 · UC-0607 (연계)",
  개요="서버가 원본 영상으로 만든 대상 크롭을 GRACE 로 관제 단말에 제공한다. 복원 크롭은 운용자 확인용이며 탐지·좌표·관측 완료 판정은 원본 분석 경로를 쓴다.",
  시작="관제 운영자 (웹/PWA · 후보 상세 크롭 확인)",
  선행="후보(좌표 보류 포함)가 있고 원본 프레임에서 대상 상자를 얻었다. 단말 디코더 가용성은 세션 협상으로 확인한다.",
  사후="크롭 프레임별 FULL/PARTIAL/NONE·사용 패킷·표시 상태가 기록되고, 원본 크롭은 복원 영상과 구분해 보존된다.",
  예외="NONE 프레임은 복원을 보장하지 않는다 — 마지막 영상에 관측 시각·오래된 영상 표지를 붙인다. 디코더 미가용·호환 불일치·참조 복구 실패 시 CROP_BASELINE(기존 크롭 제공)으로 대체한다.",
  경계="CropStreamEndpoint — 단말 세션 · 서버→단말 크롭 패킷 · 단말→서버 피드백 (전송 기술 미정) | 결과 화면: UI-05 | 연결 시험: — (미검증)")
SEQS["SD-X06"]["split"] = [
    ("(1/3) 세션 협상 · 제공 프로파일 선택", ["op", "vm", "dec", "ep", "cds", "vp", "qg"], X6_A),
    ("(2/3) 크롭 생성 · 인코딩 · 전송 · 단말 표시", ["vm", "dec", "ep", "ts", "cds", "gc", "ss", "ms"], [loop(X6_LOOP, X6_B)]),
    ("(3/3) 사용 패킷 피드백 · 참조 복구 · 대체 제공", ["vm", "dec", "ep", "cds", "ss", "gc", "qg", "vp"], [loop(X6_LOOP + " — 계속", X6_C)]),
]
SEQS["SD-X06"]["split_parts"] = X6_P
