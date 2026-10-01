"""임무 관리 — services.mission"""
from mdl import K, I, DTO, ENT, DAO, unit
from seq import S, call, alt, opt, note, loop
from cdkit import CD, layered

unit("04", "임무 관리", "임무 실행")

I("C-0405", "IMissionService", "services.mission", "임무 편집·실행 요청의 공개 계약", [
    ("setArea", "missionId?, AreaRequestDTO, PrincipalDTO", "MissionDraftDTO", "새 임무이면 ID·초기 편집본을 발급하고 구역 형상·운용 경계를 검사해 저장한다."),
    ("setPriorityArea", "missionId, AreaRequestDTO", "MissionDraftDTO", "우선구역이 임무 허용 범위 안인지 검사해 편집본에 저장한다."),
    ("editOrCancelArea", "missionId, DraftChangeDTO", "MissionDraftDTO", "편집본을 수정·취소하고 실행본은 유지한다."),
    ("setConditions", "missionId, MissionConditionsDTO", "MissionDraftDTO", "고도·속도·복귀 조건·임무 방식을 검증해 편집본에 저장한다."),
    ("previewPlan", "missionId, revision", "ExecutionPreviewDTO", "기체·장비 설정·구역·경로·제약을 실행 대상 버전으로 제시한다."),
    ("start", "missionId, revision, PrincipalDTO", "MissionRequestResultDTO", "제어권·준비·계획을 재검사해 실행 권한을 부여하고 시작을 요청한다."),
    ("getProgress", "missionId", "MissionProgressDTO", "서버·게이트웨이·기체 근거를 결합한 실제 상태와 실행본을 반환한다."),
    ("pause", "missionId, PrincipalDTO", "MissionRequestResultDTO", "새 목표 발행을 멈추고 진행 중 이동의 중단·위치 유지를 요청한다."),
    ("resume", "missionId, PrincipalDTO", "MissionRequestResultDTO", "기체·권한·안전 조건을 재검사하고 새 계획으로 재개한다."),
    ("end", "missionId, reason, PrincipalDTO", "MissionRequestResultDTO", "정찰 목표를 차단하고 설정된 복귀·착륙 종료 절차를 적용한다."),
], impl="C-0401")
I("C-0406", "IMissionRunner", "services.mission", "승인된 임무의 반복 실행 계약 (관측·안전·명령 패키지가 의존)", [
    ("start", "MissionGrantDTO, planId", "ResultDTO", "현재 lease 로 시작을 요청하고 실제 기체 수행을 별도로 확인한다."),
    ("blockTargets", "missionId, reason", "void", "대기 중인 이동 목표를 무효화하고 새 정찰 목표 생성을 막는다."),
    ("resume", "missionId, evidence", "ResultDTO", "누적 관측 이력으로 새 계획을 발행한다."),
    ("onObservation", "ObservationUpdateDTO", "ResultDTO", "실행 권한·안전 상태 확인 후 관측 → 후보 → 경로 → 명령 반복을 진행한다."),
    ("onRegionResult", "RegionResultDTO", "ResultDTO", "현재 영역 작업의 종료·실제 면적·남은 예산을 반영하고 다음 영역을 요청한다."),
], impl="C-0403")

K("C-0401", "MissionService", "service", "services.mission", "구역 편집·조건 설정·시작/정지/재개/종료 요청 처리 (IMissionService 구현)", [
    ("missionDao", "MissionDAO", "임무 상태·승인"), ("revisionDao", "MissionRevisionDAO", "편집본·실행본"),
    ("runner", "IMissionRunner", "실행 반복"), ("transform", "ICoordinateTransform", "구역 좌표 기준"),
    ("safetyPolicy", "SafetyPolicy", "조건 검증"), ("planner", "IViewpointPlanner", "실행 계획"),
    ("preflight", "IPreflightService", "시작 전 재점검"), ("controlAuthority", "IControlAuthority", "실행 권한 승인"),
    ("commands", "ICommandService", "정지·종료 명령"), ("safety", "ISafetySupervisor", "보호 동작 선택"),
    ("recovery", "IRecoveryCoordinator", "재개 전 상태 확인"), ("missionMap", "IMissionMapService", "완료도"),
], [("-checkArea", "GeometryDTO, limits", "Decision", "구역 형상·운용 경계를 검사한다.")], impl="C-0405", old="C-0401")
K("C-0403", "MissionRunner", "service", "services.mission", "승인된 임무 실행 권한으로 관측 → 계획 → 이동을 반복 (IMissionRunner 구현)", [
    ("missionDao", "MissionDAO", "실행 상태 전이"), ("planner", "IViewpointPlanner", "영역·관측 계획"),
    ("commands", "ICommandService", "이동 명령"), ("gatewayLink", "IGatewayLink", "Pi 실행 lease"),
    ("executionGrant", "MissionGrantDTO", "웹 세션과 독립된 실행 권한"), ("activePlanVersion", "int", "현재 유효 계획 버전"),
    ("blockedReason", "ReasonCode?", "수동·안전·복구 대기 중단 사유"),
], [("-renewExecutionLease", "LeaseContext", "LeaseDecisionDTO", "현재 Pi 세션·세대에서 heartbeat 를 보낸다.")], impl="C-0406", old="C-0403")

K("C-0407", "MissionController", "controller", "api", "임무 편집·실행 요청의 입구 (REST)", [("missionService", "IMissionService", "")], [
    ("createOrSetArea", "AreaRequestDTO, PrincipalDTO", "MissionDraftDTO", "POST /api/missions · PUT /api/missions/{id}/area"),
    ("setPriorityAreas", "missionId, AreaRequestDTO", "MissionDraftDTO", "PUT /api/missions/{id}/priority-areas"),
    ("patchDraft", "missionId, DraftChangeDTO", "MissionDraftDTO", "PATCH /api/missions/{id}/draft"),
    ("setConditions", "missionId, MissionConditionsDTO", "MissionDraftDTO", "PUT /api/missions/{id}/conditions"),
    ("previewPlan", "missionId, revision", "ExecutionPreviewDTO", "POST /api/missions/{id}/plan-preview"),
    ("start", "missionId, revision, PrincipalDTO", "MissionRequestResultDTO", "POST /api/missions/{id}/start"),
    ("getProgress", "missionId", "MissionProgressDTO", "GET /api/missions/{id}/progress"),
    ("pause", "missionId, PrincipalDTO", "MissionRequestResultDTO", "POST /api/missions/{id}/pause"),
    ("resume", "missionId, PrincipalDTO", "MissionRequestResultDTO", "POST /api/missions/{id}/resume"),
    ("end", "missionId, reason, PrincipalDTO", "MissionRequestResultDTO", "POST /api/missions/{id}/end"),
])

DTO("C-0408", "AreaRequestDTO", "임무구역·우선구역 지정 요청", [("geometry", "GeometryDTO", "구역 형상"), ("expectedRevision", "int?", "편집 충돌 검사 버전")])
DTO("C-0409", "DraftChangeDTO", "편집본 수정·취소 요청", [("kind", "EDIT | CANCEL", "종류"), ("target", "AREA | PRIORITY", "대상"),
    ("geometry", "GeometryDTO?", "수정 형상"), ("expectedRevision", "int", "편집 버전")])
DTO("C-0410", "MissionConditionsDTO", "임무 조건", [("altitudeM", "float", "고도"), ("speedMps", "float", "속도"),
    ("returnCondition", "ReturnRule", "복귀 조건"), ("missionMode", "MissionMode", "임무 방식"), ("expectedRevision", "int", "편집 버전")])
DTO("C-0411", "MissionDraftDTO", "임무 편집본", [("missionId", "UUID", "임무"), ("draftRevision", "int", "편집 개정"),
    ("activeRevision", "int?", "실행 개정"), ("area", "GeometryDTO", "구역"), ("priorityArea", "GeometryDTO?", "우선구역"),
    ("conditions", "MissionConditionsDTO", "조건"), ("issues", "List<ReasonCode>", "검사 결과")])
DTO("C-0412", "ExecutionPreviewDTO", "실행 전 확인 자료", [("missionId", "UUID", "임무"), ("revision", "int", "실행 대상 개정"),
    ("plan", "SearchPlanDTO", "계획"), ("drone", "VehicleStateDTO", "기체·장비 설정"), ("constraints", "List<Constraint>", "제약·보류 사유")])
DTO("C-0413", "MissionRequestResultDTO", "임무 요청 결과 — 요청 상태와 실제 상태를 구분",
    [("missionId", "UUID", "임무"), ("requestStatus", "ACCEPTED | REJECTED | PENDING", "요청 처리"),
     ("actualState", "MissionState", "기체 근거로 확인한 실제 상태"), ("reasonCode", "ReasonCode?", "사유"), ("commandId", "UUID?", "관련 명령")])
DTO("C-0414", "MissionProgressDTO", "임무 진행 상태", [("missionId", "UUID", "임무"), ("state", "MissionState", "실제 상태"),
    ("activeRevision", "int", "실행본"), ("coverage", "CoverageDTO", "완료도"), ("lastCommand", "CommandStatusDTO?", "최근 명령"),
    ("evidenceAt", "Time", "근거 시각")])

DAO("C-0404", "MissionDAO", "mission", [
    ("insert", "Mission", "UUID", "새 임무를 저장한다."),
    ("findById", "missionId", "Mission?", "임무 상태·승인·종료 사유를 읽는다."),
    ("update", "Mission, expectedState", "bool", "상태 전이와 근거를 조건부로 저장한다."),
    ("updateGrant", "missionId, MissionGrantDTO", "bool", "실행 승인 상태를 저장한다."),
], resp="DB-04 mission 테이블의 저장·조회 (구 MissionRepository · SQL 은 이 클래스에만 둔다)")
DAO("C-0415", "MissionRevisionDAO", "mission_revision", [
    ("insertDraft", "MissionRevision", "int", "초기 편집본을 저장한다."),
    ("findDraft", "missionId", "MissionRevision?", "편집 중 개정을 읽는다."),
    ("findActive", "missionId", "MissionRevision?", "실행 개정을 읽는다."),
    ("updateDraft", "MissionRevision, expectedRevision", "bool", "편집 버전이 같을 때만 저장한다."),
])
ENT("C-0402", "Mission", "mission", "임무의 편집본·실행본·실제 진행 상태를 분리 (DB-04 한 행)", ops=[
    ("transition", "event, evidence", "MissionState", "명령 요청과 실제 기체 근거를 구분해 상태를 바꾼다."),
], keys=["missionId {PK}", "droneId {FK}", "draftRevision · activeRevision", "state · stateEvidence", "grantStatus"])
ENT("C-0416", "MissionRevision", "mission_revision", "임무구역·조건의 개정본 (DB-05 한 행)",
    keys=["missionId {PK·FK}", "revision {PK}", "areaGeom · priorityGeom", "conditions : JSON", "areaDenominators"])


def _cd():
    return [layered("cd04", "", svc_pkg="services.mission", ctl=["C-0407"],
                    dto=["C-0408", "C-0409", "C-0410", "C-0411", "C-0412", "C-0413", "C-0414"],
                    pairs=[("C-0405", "C-0401"), ("C-0406", "C-0403")],
                    ext=[("C-0306", "services.spatial"), ("C-0516", "services.mission.local"), ("C-0207", "services.vehicle"),
                         ("C-0107", "services.auth"), ("C-0908", "services.command"), ("C-0909", "services.command"),
                         ("C-1005", "services.safety"), ("C-1006", "services.safety"), ("C-0807", "services.mission_map"),
                         ("C-1003", "policies")],
                    daos=[("C-0404", False), ("C-0415", False)], ents=["C-0402", "C-0416"], api_w=0.5, ext_w=0.27)]


CD("CD-04", "임무 실행", "04", _cd)

OP = ("op", "관제 운영자", "actor")
CTL = ("ctl", "MissionController", "controller")
SVC = ("svc", "IMissionService", "interface")
S("SD-0401", "04", [OP, CTL, SVC, ("ct", "ICoordinateTransform", "interface"), ("md", "MissionDAO", "dao"), ("rv", "MissionRevisionDAO", "dao")], [
    call("op", "ctl", "createOrSetArea(AreaRequestDTO, PrincipalDTO)", "MissionDraftDTO", "임무구역을 지정한다.", [
        call("ctl", "svc", "setArea(missionId?, AreaRequestDTO, PrincipalDTO)", "MissionDraftDTO", "구역을 검사해 편집본에 저장한다.", [
            call("svc", "ct", "transform(GeometryDTO, contextId)", "GeometryDTO", "구역의 공간 기준을 정규화한다."),
            call("svc", "svc", "checkArea(GeometryDTO, limits)", "Decision", "형상·운용 경계를 검사한다 (실패 시 2a 사유 표시)."),
            alt([("새 임무", [call("svc", "md", "insert(Mission)", "missionId", "임무 ID 를 발급한다."),
                              call("svc", "rv", "insertDraft(MissionRevision)", "revision", "초기 편집본을 저장한다.")]),
                 ("기존 임무 수정", [call("svc", "rv", "updateDraft(MissionRevision, expectedRevision)", "bool", "동시 편집 버전을 검사해 저장한다.")])]),
        ]),
    ]),
], entry="MissionController.createOrSetArea")
S("SD-0402", "04", [OP, CTL, SVC, ("rv", "MissionRevisionDAO", "dao")], [
    call("op", "ctl", "setPriorityAreas(missionId, AreaRequestDTO)", "MissionDraftDTO", "우선구역을 지정한다.", [
        call("ctl", "svc", "setPriorityArea(missionId, AreaRequestDTO)", "MissionDraftDTO", "허용 범위 안인지 검사한다.", [
            call("svc", "rv", "findDraft(missionId)", "MissionRevision", "확정 임무구역과 편집 버전을 읽는다."),
            call("svc", "svc", "checkArea(GeometryDTO, area)", "Decision", "우선구역이 임무구역 안인지 검사한다."),
            alt([("허용 범위 안", [call("svc", "rv", "updateDraft(MissionRevision, expectedRevision)", "bool", "편집본에 저장한다.")]),
                 ("범위 밖 (2a)", [note("svc", "rv", "사유 표시 · 수정 요청")])]),
        ]),
    ]),
], entry="MissionController.setPriorityAreas")
S("SD-0403", "04", [OP, CTL, SVC, ("rv", "MissionRevisionDAO", "dao")], [
    call("op", "ctl", "patchDraft(missionId, DraftChangeDTO)", "MissionDraftDTO", "구역을 수정하거나 취소한다.", [
        call("ctl", "svc", "editOrCancelArea(missionId, DraftChangeDTO)", "MissionDraftDTO", "편집본만 바꾸고 실행본은 유지한다.", [
            call("svc", "rv", "findDraft(missionId)", "MissionRevision", "편집본을 읽는다."),
            call("svc", "rv", "findActive(missionId)", "MissionRevision?", "실행본을 따로 읽는다."),
            alt([("검사 통과", [call("svc", "rv", "updateDraft(MissionRevision, expectedRevision)", "bool", "새 편집 버전을 저장한다.")]),
                 ("검사 실패 (2a)", [note("svc", "rv", "기존 확정 구역 유지")])]),
        ]),
    ]),
], entry="MissionController.patchDraft")
S("SD-0404", "04", [OP, CTL, SVC, ("sp", "SafetyPolicy", "component"), ("rv", "MissionRevisionDAO", "dao")], [
    call("op", "ctl", "setConditions(missionId, MissionConditionsDTO)", "MissionDraftDTO", "임무 조건을 설정한다.", [
        call("ctl", "svc", "setConditions(missionId, MissionConditionsDTO)", "MissionDraftDTO", "조건을 검증해 편집본에 저장한다.", [
            call("svc", "sp", "validateConditions(MissionConditionsDTO)", "ConditionDecision", "비행 범위·안전 설정과의 충돌을 검사한다."),
            alt([("허용 범위", [call("svc", "rv", "updateDraft(MissionRevision, expectedRevision)", "bool", "검사한 조건을 저장한다.")]),
                 ("범위 밖 (2a)", [note("svc", "rv", "사유 표시 · 수정 요청")])]),
        ]),
    ]),
], entry="MissionController.setConditions")
S("SD-0405", "04", [OP, CTL, SVC, ("rv", "MissionRevisionDAO", "dao"), ("vp", "IViewpointPlanner", "interface")], [
    call("op", "ctl", "previewPlan(missionId, revision)", "ExecutionPreviewDTO", "실행 계획을 확인한다.", [
        call("ctl", "svc", "previewPlan(missionId, revision)", "ExecutionPreviewDTO", "실행 대상 버전으로 제시한다.", [
            call("svc", "rv", "findDraft(missionId)", "MissionRevision", "구역·조건을 읽는다."),
            call("svc", "vp", "buildPlan(PlanningInputDTO)", "SearchPlanDTO", "좌표·관측·안전 조건을 확인해 계획을 만들고 저장한다."),
            opt("경로 검증 보류 (2a)", [note("svc", "vp", "사유 표시 · 시작 제한")]),
        ]),
    ]),
], entry="MissionController.previewPlan")
S("SD-0406", "04", [OP, CTL, SVC, ("pf", "IPreflightService", "interface"), ("ca", "IControlAuthority", "interface"),
                    ("rn", "IMissionRunner", "interface"), ("gw", "IGatewayLink", "interface"), ("md", "MissionDAO", "dao")], [
    call("op", "ctl", "start(missionId, revision, PrincipalDTO)", "MissionRequestResultDTO", "임무 시작을 요청한다.", [
        call("ctl", "svc", "start(missionId, revision, PrincipalDTO)", "MissionRequestResultDTO", "실행본·정책·필수 설정을 고정한다.", [
            call("svc", "pf", "runChecks(droneId, fieldChecks)", "PreflightReportDTO", "장비·시각·영상·RC·복귀 예산을 재확인한다."),
            call("svc", "ca", "approveExecution(missionId, revision, PrincipalDTO)", "MissionGrantDTO", "세션과 독립된 실행 권한을 만든다."),
            call("svc", "rn", "start(MissionGrantDTO, planId)", "ResultDTO", "첫 영역 작업으로 실행을 시작한다.", [
                call("rn", "gw", "acquireLease(LeaseRequestDTO)", "LeaseDecisionDTO", "Pi 가 세션·세대·현장 허용에 맞는 lease 를 발급한다."),
                call("rn", "md", "update(Mission, READY)", "bool", "STARTING 으로 바꾼다."),
            ]),
            note("svc", "md", "실제 수행 상태는 기체 근거(명령 결과·비행정보)로 따로 확인 (2a)"),
        ]),
    ]),
], entry="MissionController.start")
S("SD-0407", "04", [OP, CTL, SVC, ("md", "MissionDAO", "dao"), ("cs", "ICommandService", "interface"), ("mm", "IMissionMapService", "interface")], [
    call("op", "ctl", "getProgress(missionId)", "MissionProgressDTO", "진행 상태를 조회한다.", [
        call("ctl", "svc", "getProgress(missionId)", "MissionProgressDTO", "근거를 결합해 실제 상태를 만든다.", [
            call("svc", "md", "findById(missionId)", "Mission", "실행 버전과 상태를 읽는다."),
            call("svc", "cs", "getStatus(commandId)", "CommandStatusDTO", "최근 명령의 접수·실행·완료를 가져온다."),
            call("svc", "mm", "getCoverage(missionId)", "CoverageDTO", "고정 분모 완료도를 가져온다."),
            opt("상태 확인 지연 (2a)", [note("svc", "mm", "미확인으로 표시하고 기체 상태를 조회")]),
        ]),
    ]),
], entry="MissionController.getProgress")
S("SD-0408", "04", [OP, CTL, SVC, ("rn", "IMissionRunner", "interface"), ("sf", "ISafetySupervisor", "interface"),
                    ("cs", "ICommandService", "interface"), ("md", "MissionDAO", "dao")], [
    call("op", "ctl", "pause(missionId, PrincipalDTO)", "MissionRequestResultDTO", "일시정지를 요청한다.", [
        call("ctl", "svc", "pause(missionId, PrincipalDTO)", "MissionRequestResultDTO", "새 목표 발행을 멈춘다.", [
            call("svc", "rn", "blockTargets(missionId, PAUSE)", None, "대기 목표를 무효화한다."),
            call("svc", "sf", "selectProtection(droneId)", "ProtectionDecisionDTO", "위치 유지 가능성에 따라 동작을 고른다 (3a)."),
            call("svc", "cs", "submitAction(ActionRequestDTO)", "CommandRefDTO", "중단·보호 동작을 공통 검증 경로로 보낸다."),
            call("svc", "md", "update(Mission, RUNNING)", "bool", "PAUSING 으로 바꾼다 (실제 정지는 기체 근거로 확인)."),
        ]),
    ]),
], entry="MissionController.pause")
S("SD-0409", "04", [OP, CTL, SVC, ("rc", "IRecoveryCoordinator", "interface"), ("ca", "IControlAuthority", "interface"), ("rn", "IMissionRunner", "interface")], [
    call("op", "ctl", "resume(missionId, PrincipalDTO)", "MissionRequestResultDTO", "재개를 요청한다.", [
        call("ctl", "svc", "resume(missionId, PrincipalDTO)", "MissionRequestResultDTO", "조건을 재검사하고 재개한다.", [
            call("svc", "rc", "verifyForResume(missionId)", "RecoveryReportDTO", "수동·복귀·착륙·미확인 상태이면 거부한다."),
            call("svc", "ca", "validateAuthority(PrincipalDTO, CommandContextDTO)", "AuthorityDecisionDTO", "현재 세대와 권한을 검사한다."),
            alt([("재개 가능", [call("svc", "rn", "resume(missionId, evidence)", "ResultDTO", "누적 관측 이력으로 새 계획을 발행한다.")]),
                 ("보류 (2a)", [note("svc", "rn", "거부 사유 표시")])]),
        ]),
    ]),
], entry="MissionController.resume")
S("SD-0410", "04", [OP, CTL, SVC, ("rn", "IMissionRunner", "interface"), ("cs", "ICommandService", "interface"), ("md", "MissionDAO", "dao")], [
    call("op", "ctl", "end(missionId, reason, PrincipalDTO)", "MissionRequestResultDTO", "임무 종료를 요청한다.", [
        call("ctl", "svc", "end(missionId, reason, PrincipalDTO)", "MissionRequestResultDTO", "정찰을 멈추고 종료 절차를 적용한다.", [
            call("svc", "rn", "blockTargets(missionId, END)", None, "새 정찰 목표 생성을 막는다."),
            call("svc", "cs", "submitAction(ActionRequestDTO)", "CommandRefDTO", "가능한 복귀·착륙을 요청한다."),
            call("svc", "cs", "getStatus(commandId)", "CommandStatusDTO", "실제 종료를 확인한다."),
            call("svc", "md", "update(Mission, state)", "bool", "확인 전에는 결과 미확인 (3a), 확인 후 ENDED 와 종료 사유를 저장한다."),
        ]),
    ]),
], entry="MissionController.end")
S("SD-X01", "04", [("rn", "MissionRunner", "service"), ("vp", "IViewpointPlanner", "interface"), ("cs", "ICommandService", "interface"),
                   ("vd", "IVideoIngestService", "interface"), ("pd", "IPersonDetectionService", "interface"), ("mm", "IMissionMapService", "interface")], [
    call("rn", "vp", "planRegion(RegionTaskDTO)", "ObservationPlanDTO", "영역 작업의 구역 관측 계획과 첫 실행 구간을 받는다."),
    call("rn", "cs", "submitAction(ActionRequestDTO)", "CommandRefDTO", "첫 실행 구간 이동을 명령 검증 경로로 보낸다."),
    note("rn", "mm", "기체 이동 · 촬영 · 영상 수신은 비동기 — 결과는 명령 ID·프레임 ID·계획 버전으로 연결"),
    loop("실행 구간 동안 · 프레임마다", [
        call("vd", "pd", "detect(SyncedFrameDTO)", "DetectionBatchDTO", "최신 유효 프레임을 분석한다."),
        call("pd", "mm", "applyFrameEvidence(SyncedFrameDTO, TaskQualityDTO)", "ObservationUpdateDTO", "유효 관측만 셀에 누적한다."),
    ]),
    call("mm", "rn", "onObservation(ObservationUpdateDTO)", "ResultDTO", "관측 반영 결과를 실행기에 알린다."),
    call("rn", "vp", "applyRegionResult(RegionResultDTO)", "TaskUpdateDTO", "영역 처리 결과를 전역 상태에 반영한다."),
    alt([("시간·에너지 여유", [call("rn", "vp", "planRegion(RegionTaskDTO)", "ObservationPlanDTO", "다음 영역을 요청한다.")]),
         ("예산 부족 · 실행 제한", [note("rn", "mm", "미완료 영역 저장 → 종료·보호 판단으로 전달")])]),
], title="임무 시작부터 실제 관측 반영까지", uc="UC-0406, UC-0501~0505, UC-0601~0603, UC-0701, UC-0802~0806, UC-0901~0904",
   개요="전역 작업과 구역 실행, 영상 분석의 비동기 결과를 버전/ID로 연결한다.", 시작="MissionRunner (임무 시작 이후 반복)",
   선행="실행 권한(grant)과 Pi lease 가 유효하다.", 사후="관측 반영 결과와 영역 처리 결과가 계획 버전과 함께 저장된다.",
   예외="과거 요청의 결과는 해당 이력에만 반영한다. 시간·에너지 부족 또는 실행 제한 시 미완료 영역을 저장하고 종료·보호 판단으로 전달한다.",
   경계="내부 이벤트 처리 | 결과 화면: UI-04 | 연결 시험: AT-F03, AT-F07")
