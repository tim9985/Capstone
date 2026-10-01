"""임무 이력 및 결과 관리 — services.history · gateway.media · gateway.storage"""
from mdl import K, I, DTO, ENT, DAO, unit
from seq import S, call, alt, opt, note
from cdkit import CD, layered, box

unit("12", "임무 이력 및 결과 관리", "이력·결과")
HS = "services.history"

I("C-1207", "IHistoryService", HS, "임무 이력 기록·조회·결과의 공개 계약 (모든 패키지가 의존)", [
    ("append", "MissionEventDTO", "EventRefDTO", "발생·수신·저장 시각과 공통 식별자·누락 상태를 가진 이력을 더한다."),
    ("queryOperatorActions", "missionId, filter", "List<OperatorActionDTO>", "제어권·임무 조작·후보 판단·재관측·추적·알림 확인을 주체·시각으로 조회한다."),
    ("queryRelated", "eventId", "List<MissionEventDTO>", "원인 사건과 후속 조치 이력을 조회한다."),
    ("getMissionResult", "missionId", "MissionResultDTO", "실제 비행·완료도·후보·판단·종료 사유·누락 구간을 종합한다."),
    ("exportKml", "missionId, PrincipalDTO", "ExportManifestDTO", "유효 좌표 후보와 실제 비행경로만 KML 로 내보내고 보류·누락 목록을 따로 준다."),
], impl="C-1201")
I("C-1208", "IStateSyncService", HS, "화면 재접속 재동기화의 공개 계약", [
    ("resynchronize", "missionId, cursor", "SyncBundleDTO", "실제 상태와 서버가 보관한 누락 후보·스냅샷·결과를 전달한다."),
], impl="C-1204")

K("C-1201", "HistoryService", "service", HS, "임무·비행·관측·명령·운영자 행동을 공통 식별자로 연결 (IHistoryService 구현)", [
    ("eventDao", "MissionEventDAO", "순서가 있는 불변 이력"), ("exporter", "ResultExporter", "KML 내보내기"),
    ("missionMap", "IMissionMapService", "완료도"), ("targets", "ITargetService", "후보·판단"), ("missionDao", "MissionDAO", "종료 사유"),
    ("publisher", "IEventPublisher", "저장 후 전달"), ("writeStatus", "WriteStatus", "저장 정상·일부 누락·실패"),
], [], impl="C-1207", old="C-1201")
K("C-1203", "ResultExporter", "component", HS, "유효 좌표의 후보·실제 비행경로를 KML 로 내보냄", [
    ("transform", "ICoordinateTransform", "WGS84 경도·위도·고도"), ("mediaStore", "IMediaStore", "내보내기 파일 저장"),
    ("schemaVersion", "int", "결과 파일 스키마 버전"),
], [("export", "MissionResultDTO", "ExportManifestDTO", "VALID 좌표만 KML 에 쓰고 보류 좌표는 null 과 사유로 목록에 둔다.")], old="C-1203")
K("C-1204", "StateSyncService", "service", HS, "브라우저 재접속 시 실제 상태와 서버 보관 누락 결과 전달 (IStateSyncService 구현)", [
    ("missionService", "IMissionService", "실제 임무 상태"), ("controlAuthority", "IControlAuthority", "제어권"),
    ("publisher", "IEventPublisher", "누락 이벤트 재전송"), ("snapshotVersion", "int64", "일관된 동기화 스냅샷 버전"),
], [], impl="C-1208", old="C-1204")
K("C-1202", "FieldRecorder", "component", "gateway.media", "현장 영상·비행정보를 제한된 용량에서 순환 기록 (서버 단절과 무관)", [
    ("manifestStore", "SegmentManifestStore", "조각 확정"), ("ringCapacity", "Bytes", "현장 기록 용량 (실제 가용량 기준)"),
], [
    ("record", "frame, telemetry", "FieldRecordResult", "Pi 수신 압축 영상과 시각 결합용 sidecar 를 함께 기록한다."),
    ("appendVideo", "frame", "WriteResult", "현장 수신 구간을 순환 기록한다."),
    ("enforceCapacity", "policy", "EvictionReport", "가용 용량·보존 정책에 따라 오래된 구간을 정리한다."),
], old="C-1202")
K("C-1205", "SegmentManifestStore", "dao", "gateway.storage", "Pi 로컬 영상·sidecar 조각 목록·해시·동기 상태 저장 (파일 · 중앙 DB 아님)", [
    ("retentionPolicy", "RetentionPolicy", "RING_BEST_EFFORT / RETAIN_UNTIL_PERSISTED"),
], [
    ("seal", "SegmentFiles", "SegmentManifestDTO", "파일 동기화·해시·재생 의존성을 기록하고 불변으로 확정한다."),
    ("markPersisted", "ResultDTO", "ArchiveStatus", "필수 파일 모두의 서버 저장 확인을 대조한다."),
    ("evict", "StoragePressure", "EvictionReport", "정책이 허용할 때만 지우고 미회수 손실 범위를 남긴다."),
], old="C-1205")
K("C-1209", "HistoryController", "controller", "api", "이력·결과·내보내기·재동기화 요청의 입구 (REST)", [
    ("history", "IHistoryService", ""), ("stateSync", "IStateSyncService", ""), ("permission", "IPermissionPolicy", ""),
], [
    ("getOperatorActions", "missionId, filter, PrincipalDTO", "List<OperatorActionDTO>", "GET /api/missions/{id}/operator-actions"),
    ("getResult", "missionId", "MissionResultDTO", "GET /api/missions/{id}/result"),
    ("export", "missionId, PrincipalDTO", "ExportManifestDTO", "POST /api/missions/{id}/exports"),
    ("synchronize", "missionId, cursor", "SyncBundleDTO", "POST /api/missions/{id}/synchronizations"),
])

DTO("C-1210", "MissionEventDTO", "임무 이력 사건", [("eventType", "EventType", "종류"), ("code", "String?", "EX·장애·사유 코드"), ("missionId", "UUID?", "임무"),
    ("actorUserId", "UUID?", "수행 운영자"), ("ref", "TableRef", "대상"), ("sourceTime", "Time", "발생"), ("correlationId", "String", "상관 ID"),
    ("payload", "JSON", "내용")])
DTO("C-1211", "OperatorActionDTO", "운영자 조치", [("actor", "UUID", "수행자"), ("action", "String", "조치"), ("target", "TableRef", "대상"), ("at", "Time", "시각"),
    ("result", "String", "결과")])
DTO("C-1212", "MissionResultDTO", "임무 결과", [("missionId", "UUID", "임무"), ("endReason", "String", "종료 사유"), ("track", "LineString", "실제 경로"),
    ("coverage", "CoverageDTO", "완료도"), ("candidates", "List<CandidateDTO>", "후보 (0건도 정상)"), ("gaps", "List<TimeRange>", "누락 구간")])
DTO("C-1213", "ExportManifestDTO", "내보내기 결과", [("assetId", "UUID", "KML 파일"), ("included", "int", "포함 후보"), ("pending", "List<UUID>", "좌표 보류 후보"),
    ("hash", "String", "파일 해시")])
DTO("C-1214", "SyncBundleDTO", "재동기화 묶음", [("progress", "MissionProgressDTO", "실제 임무"), ("control", "ControlStateDTO", "제어권"),
    ("missed", "EventBatchDTO", "누락 이벤트"), ("snapshotVersion", "int64", "스냅샷")])
DTO("C-1215", "EventRefDTO", "이력 참조", [("eventId", "UUID", "사건"), ("eventSeq", "int64", "재동기화 커서"), ("storedAt", "Time", "저장 시각")])

DAO("C-1216", "MissionEventDAO", "mission_event", [
    ("append", "MissionEvent", "int64", "사건을 저장하고 커서를 발급한다 (생산 순번 유일)."),
    ("findByMission", "missionId, filter", "List<MissionEvent>", "임무 이력을 읽는다."),
    ("findRelated", "eventId", "List<MissionEvent>", "원인·후속 사건을 상관 ID 로 읽는다."),
    ("findAfter", "cursor", "List<MissionEvent>", "커서 이후 사건을 읽는다."),
])
ENT("C-1217", "MissionEvent", "mission_event", "명령 상태·실행 보고·안전·운영자 조치 등 불변 사건 (DB-18 한 행)",
    keys=["eventId {PK}", "eventSeq {UQ}", "missionId · droneId · actorUserId {FK}", "eventType · code", "correlationId · payload"])


def _cd():
    a = layered("cd12a", "(1/2) 서버 — 계층 구조", svc_pkg=HS, ctl=["C-1209"],
                dto=["C-1210", "C-1211", "C-1212", "C-1213", "C-1214", "C-1215"],
                pairs=[("C-1207", "C-1201"), ("C-1208", "C-1204")], comps=["C-1203"],
                ext=[("C-0807", "services.mission_map"), ("C-0722", "services.vision"), ("C-1106", "api"), ("C-0405", "services.mission"),
                     ("C-0107", "services.auth"), ("C-0306", "services.spatial"), ("C-0612", "services.media")],
                daos=[("C-1216", False), ("C-0404", True)], ents=["C-1217"], api_w=0.4, ext_w=0.27)
    B = {"fr": box("C-1202"), "sm": box("C-1205"), "vr": box("C-0628", ref=True), "ar": box("C-0607", ref=True)}
    P = [dict(name="gateway.media", row=0, x=0.005, w=0.49, rows=[[("vr", 1.0)], [("fr", 1.0)]]),
         dict(name="gateway.storage", row=0, x=0.505, w=0.49, rows=[[("sm", 1.0)], [("ar", 1.0)]], tabs=["", "services.media (서버)"])]
    R = [("vr", "fr", "assoc", "", {}), ("fr", "sm", "assoc", "", {}), ("ar", "sm", "dep", "recover · markPersisted", {})]
    return [a, ("cd12b", "(2/2) 게이트웨이 (Pi) — 현장 기록", P, B, R)]


CD("CD-12", "이력·결과", "12", _cd)

OP = ("op", "관제 운영자", "actor")
CTL = ("ctl", "HistoryController", "controller")
HI = ("hs", "IHistoryService", "interface")
S("SD-1201", "12", [("src", "각 서비스 (명령·안전·비전·임무)", "service"), HI, ("ed", "MissionEventDAO", "dao"), ("ep", "IEventPublisher", "interface")], [
    call("src", "hs", "append(MissionEventDTO)", "EventRefDTO", "임무·비행·관측·명령·결과의 공통 식별자와 발생·저장 시각을 기록한다.", [
        call("hs", "ed", "append(MissionEvent)", "eventSeq", "불변 사건으로 저장하고 커서를 발급한다."),
        call("hs", "ep", "deliver(MissionEventDTO)", "DeliveryResult", "저장 성공한 사건만 웹 동기화에 제공한다."),
    ]),
    note("src", "ep", "미확인 상태와 누락 구간을 결과에 표시 (3a)"),
], entry="IHistoryService.append", 시작="임무 실행 중 각 모듈의 상태 변화")
S("SD-1202", "12", [OP, CTL, ("pp", "IPermissionPolicy", "interface"), HI, ("ed", "MissionEventDAO", "dao")], [
    call("op", "ctl", "getOperatorActions(missionId, filter, PrincipalDTO)", "List<OperatorActionDTO>", "운영자 조치 이력을 조회한다.", [
        call("ctl", "pp", "checkAccess(PrincipalDTO, MISSION_HISTORY)", "AccessDecisionDTO", "임무 이력 조회 권한을 확인한다."),
        call("ctl", "hs", "queryOperatorActions(missionId, filter)", "List<OperatorActionDTO>", "주체·시각으로 조치를 모은다.", [
            call("hs", "ed", "findByMission(missionId, OPERATOR_ACTION)", "List<MissionEvent>", "제어권·명령·판단·재관측 사건을 읽는다.")]),
    ]),
    note("op", "ed", "기록 미등록 항목은 확인 불가로 표시 (2a)"),
], entry="HistoryController.getOperatorActions")
S("SD-1203", "12", [("vr", "VideoRelay", "component"), ("fr", "FieldRecorder", "component"), ("sm", "SegmentManifestStore", "dao"),
                    ("sr", "ServerReporter", "component")], [
    call("vr", "fr", "record(frame, telemetry)", "FieldRecordResult", "Pi 수신 압축 영상과 시각 결합 sidecar 를 함께 기록한다.", [
        call("fr", "sm", "seal(SegmentFiles)", "SegmentManifestDTO", "조각을 확정하고 크기·해시·재생 의존성을 고정한다."),
        call("fr", "sm", "evict(StoragePressure)", "EvictionReport", "보존 프로파일이 허용할 때만 지운다."),
    ]),
    opt("현장 결손·미회수 삭제·공간 부족 (1a)", [call("fr", "sr", "reportFault(FaultDTO)", None, "서버 연결 시 결손 범위를 이력으로 보낸다.")]),
], entry="FieldRecorder.record (Pi)", 시작="현장 수신 영상·비행정보 (VideoRelay)")
S("SD-1204", "12", [OP, CTL, HI, ("md", "MissionDAO", "dao"), ("mm", "IMissionMapService", "interface"), ("ts", "ITargetService", "interface")], [
    call("op", "ctl", "getResult(missionId)", "MissionResultDTO", "임무 결과를 조회한다.", [
        call("ctl", "hs", "getMissionResult(missionId)", "MissionResultDTO", "결과를 종합한다.", [
            call("hs", "md", "findById(missionId)", "Mission", "종료 사유·실제 비행 상태를 읽는다."),
            call("hs", "mm", "getCoverage(missionId)", "CoverageDTO", "전체·관측가능 완료도를 구분한다."),
            call("hs", "ts", "queryCandidates(missionId, filter)", "List<CandidateDTO>", "좌표 보류·판단 미완료 후보를 포함한다."),
        ]),
    ]),
    note("op", "ts", "탐지 0건도 정상 결과 · 미관측 영역과 미완료 종료 사유 표시 (2a)"),
], entry="HistoryController.getResult")
S("SD-1205", "12", [OP, CTL, HI, ("rx", "ResultExporter", "component"), ("ct", "ICoordinateTransform", "interface"), ("ms", "IMediaStore", "interface")], [
    call("op", "ctl", "export(missionId, PrincipalDTO)", "ExportManifestDTO", "임무 결과를 내보낸다.", [
        call("ctl", "hs", "exportKml(missionId, PrincipalDTO)", "ExportManifestDTO", "일관된 스냅샷으로 내보낸다.", [
            call("hs", "hs", "getMissionResult(missionId)", "MissionResultDTO", "대상 상태와 증거를 얻는다."),
            call("hs", "rx", "export(MissionResultDTO)", "ExportManifestDTO", "VALID 좌표만 KML 에 쓴다.", [
                call("rx", "ct", "transform(GeometryDTO, contextId)", "GeometryDTO", "WGS84 경도·위도·고도로 변환한다."),
                call("rx", "ms", "saveRecording(SegmentManifestDTO EXPORT_KML)", "MediaRefDTO", "파일을 저장하고 해시를 남긴다."),
            ]),
        ]),
    ]),
    note("op", "ms", "탐지 0건도 경로·완료도·종료 사유 제공 · 보류 좌표는 null 과 사유 (2a)"),
], entry="HistoryController.export")
S("SD-1206", "12", [OP, CTL, ("ss", "IStateSyncService", "interface"), ("mi", "IMissionService", "interface"), ("ca", "IControlAuthority", "interface"),
                    ("ep", "IEventPublisher", "interface")], [
    call("op", "ctl", "synchronize(missionId, cursor)", "SyncBundleDTO", "재접속 후 상태를 맞춘다.", [
        call("ctl", "ss", "resynchronize(missionId, cursor)", "SyncBundleDTO", "실제 상태와 누락 결과를 묶는다.", [
            call("ss", "mi", "getProgress(missionId)", "MissionProgressDTO", "현재 실제 임무·기체 상태를 조회한다."),
            call("ss", "ca", "getControlState(droneId)", "ControlStateDTO", "보유자·RC·만료 상태를 다시 가져온다."),
            call("ss", "ep", "replayAfter(cursor, snapshotVersion)", "EventBatchDTO", "수신 커서 이후 보관 이벤트만 재전송한다."),
        ]),
    ]),
    note("op", "ep", "서버가 수신하지 못한 자료는 구분해 표시 · 비행 재개는 별도 승인 (3a)"),
], entry="HistoryController.synchronize")
S("SD-X05", "12", [("sm", "SegmentManifestStore (Pi)", "dao"), ("ar", "ArchiveRecoveryService", "service"), ("ms", "IMediaStore", "interface"),
                   ("al", "IFrameAnalysisLedger", "interface"), HI], [
    call("sm", "ar", "recover(SegmentManifestDTO)", "ResultDTO", "Pi 보관 원본 조각을 이어 보낸다 (전송 ACK).", [
        call("ar", "ms", "persistBundle(SegmentManifestDTO)", "ResultDTO", "영상·sidecar·DB 확정 뒤에만 저장 완료 (PERSISTED)."),
        alt([("영상 저장 · 메타데이터 대기", [note("ar", "ms", "VIDEO_PERSISTED_METADATA_PENDING 으로 기록")]),
             ("응답 유실", [call("ar", "ar", "queryPersisted(segmentId, hash)", "ResultDTO", "ID·해시로 저장 상태를 조회한다 (중복 저장 없음).")])]),
        call("ar", "al", "schedule(frameId, HISTORICAL)", "AnalysisJobDTO", "원관측 시각을 유지한 사후 분석을 배정한다 (ANALYZED 는 별도)."),
    ]),
    call("ar", "sm", "markPersisted(ResultDTO)", "ArchiveStatus", "필수 파일 모두의 저장 확인을 Pi 기록과 대조한다."),
    call("ar", "hs", "append(MissionEventDTO)", "EventRefDTO", "전송·저장·분석 상태를 각각 남긴다."),
], title="전송 ACK · 저장 확인 · 분석 완료 분리", uc="UC-0605, UC-0606, UC-1203~1206",
   개요="영상 전송 ACK, 파일 저장 확인, 분석 완료를 다른 상태로 기록한다.", 시작="Pi 조각 확정 후 회수 (SegmentManifestStore)",
   선행="Pi 에 확정된 조각이 있다.", 사후="전송·PERSISTED·ANALYZED 상태가 따로 기록된다.",
   예외="응답 유실 시 ID·해시로 저장 상태를 조회한다. 원본 회수는 Pi 에 보관된 파일만 대상으로 한다.",
   경계="내부 이벤트 처리 | 결과 화면: UI-05 | 연결 시험: AT-F04, AT-F10")
