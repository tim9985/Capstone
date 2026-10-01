"""비행 명령 관리 — services.command · gateway.* · policies"""
from mdl import K, I, DTO, ENT, DAO, unit
from seq import S, call, alt, opt, note, loop
from cdkit import CD, layered, box

unit("09", "비행 명령 관리", "비행 명령")
CM = "services.command"

I("C-0908", "ICommandService", CM, "비행 명령 구성·전달·결과 확인의 공개 계약", [
    ("submitAction", "ActionRequestDTO", "CommandRefDTO", "compose → 서버 검증 → 전달을 한 경로로 처리한다."),
    ("compose", "ActionRequestDTO, CommandContextDTO", "CommandEnvelopeDTO", "기체·임무·요청자·순서·유효시간·좌표 기준·계획 버전·제어 세대를 붙인다."),
    ("dispatch", "CommandEnvelopeDTO", "CommandStatusDTO", "검증된 고수준 목표를 게이트웨이로 전달한다."),
    ("requestReturn", "missionId, PrincipalDTO", "CommandRefDTO", "정찰 목표를 차단하고 복귀 가능 상태에서 복귀를 요청한다."),
    ("requestLand", "missionId, PrincipalDTO", "CommandRefDTO", "현장 조건을 확인한 착륙을 요청하고 실제 결과를 추적한다."),
    ("getStatus", "commandId", "CommandStatusDTO", "접수·실행·완료 조건을 반환한다."),
    ("getStatusByPlan", "planId", "List<CommandStatusDTO>", "계획에 연결된 명령 결과를 반환한다."),
    ("observeResult", "CommandEvidenceDTO", "CommandStatusDTO", "응답과 실제 비행 상태를 결합해 결과를 갱신한다."),
], impl="C-0901")
I("C-0909", "IGatewayLink", CM, "서버 → 게이트웨이(Pi) 통신 계약 — 네트워크(WSS)를 감추는 어댑터 인터페이스", [
    ("sendCommand", "CommandEnvelopeDTO", "CommandStatusDTO", "P0 전용 채널로 Pi 가드에 명령을 보낸다."),
    ("acquireLease", "LeaseRequestDTO", "LeaseDecisionDTO", "Pi 의 단일 실행 lease 를 요청한다."),
    ("renewLease", "LeaseHeartbeat", "LeaseDecisionDTO", "lease heartbeat 를 보낸다."),
    ("readSafetySettings", "droneId", "SafetySettingsDTO", "FC 의 읽을 수 있는 안전 설정·모드를 읽는다."),
    ("requestSegment", "segmentId, offset", "SegmentChunk", "보관된 원본 영상 조각을 이어 받는다."),
], impl="C-0910")

K("C-0901", "CommandService", "service", CM, "고수준 비행 명령 구성·전달과 복귀·착륙 요청 처리 (ICommandService 구현)", [
    ("ledger", "CommandLedger", "명령 상태 전이"), ("validator", "CommandValidator", "공유 검증 규칙"),
    ("permission", "IPermissionPolicy", "실행 권한"), ("runner", "IMissionRunner", "목표 차단"), ("safety", "ISafetySupervisor", "보호 동작"),
    ("vehicleState", "IVehicleStateService", "완료 판정 상태"), ("gatewayLink", "IGatewayLink", "Pi 전달"), ("history", "IHistoryService", "처리 이력"),
    ("sequence", "int64", "기체·임무 실행별 증가 순서"),
], [("-attachIdentity", "action, context", "CommandEnvelopeDTO", "식별자·순서·TTL·좌표 기준·계획 버전·세대를 붙인다.")], impl="C-0908", old="C-0901")
K("C-0904", "CommandLedger", "component", CM, "접수·실행·완료·미확인을 서로 다른 상태로 기록 (서버 측)", [
    ("commandDao", "FlightCommandDAO", "flight_command 테이블"),
], [
    ("create", "CommandEnvelopeDTO", "CommandRefDTO", "확정된 명령 원문과 요청 상태를 기록한다."),
    ("findDuplicate", "requestHash", "CommandStatusDTO?", "같은 ID·해시는 기존 결과, 다른 해시는 ID_CONFLICT 로 판단한다."),
    ("appendTransition", "commandId, CommandEvidenceDTO", "CommandStatusDTO", "SENT·FC_ACCEPTED 등 실제 근거만 기록한다."),
    ("getStatus", "commandId", "CommandStatusDTO", "현재 명령의 접수·실행·완료 조건을 읽는다."),
    ("applyRevision", "CommandStatusDTO", "StatusDecision", "원래 세션의 명령 ID 와 증가 revision 만 반영한다."),
    ("reconcilePending", "evidence", "ReconcileResult", "미확인 명령을 기체 근거와 대조하고 명시적 재개를 기다린다."),
], old="C-0904")
K("C-0910", "GatewayClient", "component", CM, "IGatewayLink 의 WSS 구현 — Pi 세션·응답 대기·재접속 관리", [
    ("session", "WssSession", "Pi 연결"), ("pendingAcks", "Map<UUID, Future>", "응답 대기"),
], [], impl="C-0909")
K("C-0902", "CommandValidator", "component", "policies", "공유 명령 계약의 순수 검증 규칙 — 서버와 Pi 가 같은 규칙을 쓴다", [
    ("policyVersion", "UUID", "권한·시각·영역 규칙 버전"),
], [
    ("validate", "CommandEnvelopeDTO, currentState", "ValidationResult", "만료·중복·버전·세대·좌표·안전 제한을 검사한다."),
    ("validateDispatch", "CommandEnvelopeDTO, GatewayContext", "Decision", "세션·epoch·lease·TTL·현재 상태를 송신 직전에 재검사한다."),
], old="C-0902")
# 게이트웨이 (Pi)
K("C-0912", "ServerChannelController", "controller", "gateway.api", "Pi 의 서버 명령 수신 입구 (WSS)", [
    ("guard", "GatewayCommandGuard", ""), ("leaseGuard", "ExecutionLeaseGuard", ""), ("flight", "FlightAdapter", ""),
], [
    ("onCommand", "CommandEnvelopeDTO", "CommandStatusDTO", "명령을 받아 가드에 넘긴다."),
    ("onLeaseRequest", "LeaseRequestDTO", "LeaseDecisionDTO", "lease 요청·heartbeat 를 받는다."),
    ("onSettingsQuery", "droneId", "SafetySettingsDTO", "FC 안전 설정 조회를 받는다."),
])
K("C-0906", "GatewayCommandGuard", "component", "gateway.command", "Pi 원격 명령의 검증 진입점 — 송신 직전 검사와 명령 기록", [
    ("validator", "CommandValidator", "공유 규칙"), ("leaseGuard", "ExecutionLeaseGuard", "현장 권한"),
    ("journal", "CommandJournalDAO", "Pi 로컬 명령 기록"), ("flight", "FlightAdapter", "MAVLink"),
], [
    ("accept", "CommandEnvelopeDTO", "CommandStatusDTO", "같은 ID·해시는 기존 결과, 다른 해시는 ID_CONFLICT 로 거부한다."),
    ("dispatch", "commandId", "DispatchResult", "TTL·lease·epoch·상태를 재검사하고 DISPATCHING 을 기록한 뒤 FC 로 보낸다."),
    ("requestPause", "LocalOrRemoteRequest", "CommandStatusDTO", "정지 전용 경로에서 동작 조건을 확인한다."),
], old="C-0906")
K("C-0905", "ExecutionLeaseGuard", "component", "gateway.command", "Pi 에서 단일 실행 주체·세대·lease 와 현장 잠금 관리", [
    ("gatewaySession", "UUID", "재시작 시 새로 발급"), ("controlEpoch", "int64", "Pi 가 소유하는 세대"),
    ("state", "AuthorityState", "LOCKED / REMOTE_READY / RECOVERY_LOCKED / LOCAL_OVERRIDE"),
], [
    ("acquire", "LeaseRequestDTO", "LeaseDecisionDTO", "세션·시각·현장 허용을 검사해 lease 를 발급한다."),
    ("renew", "LeaseHeartbeat", "LeaseDecisionDTO", "증가 순번과 유효 기간을 검사한다."),
    ("invalidate", "LocalEvent", "AuthorityState", "RC·failsafe·상태 상실 시 이전 권한을 즉시 잠근다."),
    ("check", "CommandEnvelopeDTO", "LeaseDecisionDTO", "세션·epoch·lease·현장 잠금 상태를 대조한다."),
], old="C-0905")
K("C-0907", "CommandRecoveryReconciler", "component", "gateway.command", "송신·ACK 경계 장애와 과거 세션 결과를 실제 FC 상태로 조정", [
    ("journal", "CommandJournalDAO", "송신 전후 기록"), ("flight", "FlightAdapter", "실제 모드·위치·착륙 조회"),
    ("reporter", "ServerReporter", "결과 보고"),
], [
    ("reconcile", "commandId, VehicleState", "CommandStatusDTO", "송신·완료가 불명확하면 UNCONFIRMED 를 유지한다."),
    ("query", "commandId, originalSessionId", "CommandStatusDTO", "TTL 이 지난 완료 기록도 원래 결과로 조회한다."),
], old="C-0907")
K("C-0903", "FlightAdapter", "component", "gateway.flight", "게이트웨이의 MAVLink 경계 — SITL/실기체별 접속만 교체", [
    ("endpoint", "MavlinkEndpoint", "활성 FC 접속"), ("targetSystem", "MavlinkTarget", "대상 FC"), ("adapterMode", "SIM | REAL", "접속 방식"),
], [
    ("send", "CommandEnvelopeDTO", "TransportReceipt", "지원되는 MAVLink 고수준 명령으로 바꿔 보낸다."),
    ("applyLocalProtection", "ProtectionDecisionDTO", "ProtectionReceipt", "현장 경로로 보호 동작을 보낸다. 기체 링크 단절 시 FC 자체 보호에 맡기고 미확인으로 기록한다."),
    ("readSafetySettings", "", "SafetySettingsDTO", "FC 의 읽을 수 있는 설정과 모드를 조회한다."),
    ("readMissionItems", "", "MissionItems", "업로드한 임무 항목을 FC 에서 다시 읽는다."),
], old="C-0903")
K("C-0913", "ServerReporter", "component", "gateway.api", "Pi → 서버 보고 어댑터 (명령 근거·현장 제어·비행정보·장애)", [
    ("session", "WssSession", "서버 연결"), ("outbox", "Queue<Report>", "단절 중 보관 · 재접속 시 순서대로 전송"),
], [
    ("reportEvidence", "CommandEvidenceDTO", "void", "명령 수락·실행·완료 근거를 보낸다."),
    ("reportLocalControl", "LocalControlReportDTO", "void", "RC 인수·제어 세대 변경을 보낸다."),
    ("reportFault", "FaultDTO", "void", "현장 장애와 적용한 보호 동작을 보낸다."),
])
K("C-0914", "CommandJournalDAO", "dao", "gateway.storage", "Pi 로컬 명령 기록 (SQLite·파일 · 중앙 DB 아님) — FC 송신 전 기록", [
    ("store", "LocalDb", "Pi 로컬 저장"),
], [
    ("appendDispatching", "CommandEnvelopeDTO", "bool", "송신 전 기록한다. 실패하면 새 실행 명령을 막는다."),
    ("appendTransition", "commandId, evidence", "bool", "응답·실제 상태 근거를 덧붙인다."),
    ("findById", "commandId", "JournalEntry?", "명령 기록을 읽는다."),
    ("findPending", "", "List<JournalEntry>", "결과 미확인 명령을 읽는다."),
])

K("C-0911", "CommandController", "controller", "api", "복귀·착륙 요청의 입구 (REST)", [("commandService", "ICommandService", "")], [
    ("requestReturn", "missionId, PrincipalDTO", "CommandRefDTO", "POST /api/missions/{id}/return"),
    ("requestLand", "missionId, PrincipalDTO", "CommandRefDTO", "POST /api/missions/{id}/land"),
])

DTO("C-0915", "ActionRequestDTO", "고수준 행동 요청", [("action", "GOTO | HOLD | RETURN | LAND | PAUSE | MISSION_UPLOAD", "동작"),
    ("target", "Pose?", "목표"), ("missionId", "UUID?", "임무"), ("planId", "UUID?", "계획"), ("requester", "PrincipalDTO", "요청 주체")])
DTO("C-0916", "CommandContextDTO", "실행 권한 판정 문맥", [("droneId", "String", "기체"), ("missionId", "UUID?", "임무"), ("action", "Action", "동작"),
    ("controlEpoch", "int64", "제어 세대")])
DTO("C-0917", "CommandEnvelopeDTO", "명령 원문 (서버·Pi 공유 계약 · schema 1.3)", [("commandId", "UUID", "명령"), ("requestHash", "String", "중복 키"),
    ("droneId", "String", "기체"), ("sequence", "int64", "순번"), ("issuedAt · expiresAt", "Time", "발행·TTL"), ("controlEpoch", "int64", "세대"),
    ("coordinateConfigId", "UUID", "좌표 기준"), ("planVersion", "int?", "계획 버전"), ("payload", "JSON", "목표")])
DTO("C-0918", "CommandRefDTO", "명령 참조", [("commandId", "UUID", "명령"), ("state", "CommandState", "현재 상태")])
DTO("C-0919", "CommandStatusDTO", "명령 상태", [("commandId", "UUID", "명령"), ("state", "REQUESTED | ACCEPTED | REJECTED | EXECUTING | COMPLETED | FAILED | UNCONFIRMED", "상태"),
    ("statusRevision", "int", "변경 번호"), ("reasonCode", "ReasonCode?", "사유"), ("lastEvidenceAt", "Time", "마지막 근거")])
DTO("C-0920", "CommandEvidenceDTO", "명령 결과 근거 (Pi → 서버)", [("commandId", "UUID", "명령"), ("kind", "SENT | FC_ACK | MODE | POSITION | LANDED", "근거"),
    ("observedAt", "Time", "시각"), ("originalSessionId", "UUID", "원 세션")])
DTO("C-0921", "LeaseRequestDTO", "실행 lease 요청", [("droneId", "String", "기체"), ("grant", "MissionGrantDTO", "실행 권한"), ("controlEpoch", "int64", "세대")])
DTO("C-0922", "LeaseDecisionDTO", "lease 판정", [("granted", "bool", "발급"), ("leaseId", "UUID?", "lease"), ("expiresAt", "Time?", "만료"),
    ("state", "AuthorityState", "Pi 권한 상태"), ("reasonCode", "ReasonCode?", "사유")])
DTO("C-0923", "SafetySettingsDTO", "FC 실제 안전 설정", [("rcLossAction", "Action", "RC 상실"), ("lowBattery", "Threshold", "저전압"),
    ("geofence", "Fence", "경계"), ("mode", "FlightMode", "현재 모드")])

DAO("C-0924", "FlightCommandDAO", "flight_command", [
    ("insert", "FlightCommand", "UUID", "명령 원문을 저장한다 (요청 해시 유일)."),
    ("findByHash", "requestHash", "FlightCommand?", "중복 요청을 찾는다."),
    ("findById", "commandId", "FlightCommand?", "명령을 읽는다."),
    ("updateState", "commandId, state, statusRevision", "bool", "증가 revision 일 때만 상태를 바꾼다."),
    ("findByPlan", "planId", "List<FlightCommand>", "계획별 명령을 읽는다."),
])
ENT("C-0925", "FlightCommand", "flight_command", "비행 명령과 상태 전이 (DB-16 한 행)",
    keys=["commandId {PK}", "requestHash {UQ}", "missionId · planId · droneId {FK}", "controlEpoch · expiresAt", "state · statusRevision"])


def _cd():
    a = layered("cd09a", "(1/2) 서버 — 계층 구조", svc_pkg=CM, ctl=["C-0911"],
                dto=["C-0915", "C-0916", "C-0917", "C-0918", "C-0919", "C-0920", "C-0921", "C-0922", "C-0923"],
                pairs=[("C-0908", "C-0901"), ("C-0909", "C-0910")], comps=["C-0904", "C-0902"],
                ext=[("C-0108", "services.auth"), ("C-0406", "services.mission"), ("C-1005", "services.safety"),
                     ("C-0205", "services.vehicle"), ("C-1207", "services.history")],
                daos=[("C-0924", False)], ents=["C-0925"], api_w=0.3, dto_cols=3, ext_w=0.26)
    B = {k: box(k) for k in ["C-0912", "C-0906", "C-0905", "C-0907", "C-0903", "C-0913", "C-0914"]}
    B["v"] = box("C-0902", ref=True); B["sp"] = box("C-1003", ref=True); B["lc"] = box("C-1004", ref=True)
    P = [dict(name="gateway.api", row=0, x=0.005, w=0.99, rows=[[("C-0912", .5), ("C-0913", .5)]]),
         dict(name="gateway.command", row=1, x=0.005, w=0.99, rows=[[("C-0906", .333), ("C-0905", .333), ("C-0907", .334)]]),
         dict(name="gateway.flight", row=2, x=0.005, w=0.49, rows=[[("C-0903", .6), ("lc", .4)]]),
         dict(name="gateway.storage · policies", row=2, x=0.505, w=0.49, rows=[[("C-0914", 1.0)], [("v", .5), ("sp", .5)]])]
    R = [("C-0912", "C-0906", "assoc", "", {"elbow": 1}), ("C-0912", "C-0905", "assoc", "", {"elbow": 1}),
         ("C-0906", "C-0905", "assoc", "", {}), ("C-0906", "C-0914", "dep", "", {"elbow": 1}), ("C-0906", "C-0903", "assoc", "", {"elbow": 1}),
         ("C-0906", "v", "dep", "", {"elbow": 1}), ("C-0907", "C-0914", "dep", "", {"elbow": 1}), ("C-0907", "C-0903", "assoc", "", {"elbow": 1}),
         ("C-0907", "C-0913", "assoc", "", {"elbow": 1}), ("C-0903", "sp", "dep", "", {"elbow": 1})]
    return [a, ("cd09b", "(2/2) 게이트웨이 (Pi) — 명령 검증·전달·복구", P, B, R)]


CD("CD-09", "비행 명령", "09", _cd)

CS = ("cs", "ICommandService", "interface")
S("SD-0901", "09", [("rq", "MissionRunner · TargetTrackingService", "service"), CS, ("pp", "IPermissionPolicy", "interface"),
                    ("cl", "CommandLedger", "component"), ("fd", "FlightCommandDAO", "dao")], [
    call("rq", "cs", "compose(ActionRequestDTO, CommandContextDTO)", "CommandEnvelopeDTO", "식별 가능한 명령을 만든다.", [
        call("cs", "cs", "attachIdentity(action, context)", "CommandEnvelopeDTO", "기체·임무·요청자·순서·TTL·좌표 기준·계획 버전·세대를 붙인다."),
        call("cs", "pp", "authorizeCommand(PrincipalDTO, CommandContextDTO)", "AccessDecisionDTO", "주체에 맞는 실행 권한을 검사한다."),
        alt([("필수값·권한 충족", [call("cs", "cl", "create(CommandEnvelopeDTO)", "CommandRefDTO", "확정 원문과 요청 상태를 기록한다.", [
                                       call("cl", "fd", "insert(FlightCommand)", "commandId", "REQUESTED 로 저장한다.")])]),
             ("필수값·좌표 기준 누락 (1a)", [note("cs", "fd", "명령 보류 · 필요한 항목 반환")])]),
    ]),
], entry="ICommandService.compose", 시작="임무·정밀 관측·개별 추적·안전 동작 요청 (MissionRunner · TargetTrackingService)")
S("SD-0902", "09", [("cs", "CommandService", "service"), ("v", "CommandValidator", "component"), ("cl", "CommandLedger", "component"),
                    ("gw", "IGatewayLink", "interface"), ("g", "GatewayCommandGuard", "component"), ("lg", "ExecutionLeaseGuard", "component"),
                    ("sp", "SafetyPolicy", "component")], [
    call("cs", "v", "validate(CommandEnvelopeDTO, currentState)", "ValidationResult", "스키마·정책·좌표·요청 해시·현재 상태를 검사한다 (서버)."),
    call("cs", "cl", "findDuplicate(requestHash)", "CommandStatusDTO?", "같은 ID·해시는 기존 결과, 다른 해시는 거부한다."),
    call("cs", "gw", "sendCommand(CommandEnvelopeDTO)", "CommandStatusDTO", "Pi 로 보낸다.", [
        call("gw", "g", "accept(CommandEnvelopeDTO)", "CommandStatusDTO", "Pi 수신 직후 검증한다 (ServerChannelController 경유).", [
            call("g", "lg", "check(CommandEnvelopeDTO)", "LeaseDecisionDTO", "Pi 세션·세대·lease·현장 인수·시각 신뢰도를 확인한다."),
            call("g", "sp", "checkCommand(CommandEnvelopeDTO, state)", "SafetyDecisionDTO", "동작별 경계·항법·속도·배터리 조건을 검사한다."),
        ]),
    ]),
    opt("중단·RC 인수 전 세대 (2a)", [note("cs", "sp", "거부 · 사유 기록")]),
], entry="CommandValidator.validate · GatewayCommandGuard.accept", 시작="중앙 명령 발행 전 및 게이트웨이 수신 직후 (CommandService)")
S("SD-0903", "09", [("cs", "CommandService", "service"), ("gw", "IGatewayLink", "interface"), ("sc", "ServerChannelController", "controller"),
                    ("g", "GatewayCommandGuard", "component"), ("jd", "CommandJournalDAO", "dao"), ("fa", "FlightAdapter", "component"),
                    ("fc", "비행제어기 (FC)", "external")], [
    call("cs", "gw", "sendCommand(CommandEnvelopeDTO)", "CommandStatusDTO", "P0 전용 WSS 로 Pi 에 전달한다.", [
        call("gw", "sc", "onCommand(CommandEnvelopeDTO)", "CommandStatusDTO", "Pi 입구가 받는다.", [
            call("sc", "g", "accept(CommandEnvelopeDTO)", "CommandStatusDTO", "중복·충돌·권한·TTL 을 확인해 이동 큐에 넣는다."),
            call("sc", "g", "dispatch(commandId)", "DispatchResult", "송신 직전 조건을 다시 검사한다.", [
                call("g", "jd", "appendDispatching(CommandEnvelopeDTO)", "bool", "파일 기록 실패면 새 FC 송신을 거부한다."),
                call("g", "fa", "send(CommandEnvelopeDTO)", "TransportReceipt", "MAVLink 고수준 명령으로 바꿔 보낸다.", [
                    call("fa", "fc", "MAVLink 명령", "COMMAND_ACK", "COMMAND_LONG · SET_POSITION_TARGET 으로 FC 에 보낸다.")]),
                call("g", "jd", "appendTransition(commandId, evidence)", "bool", "SENT·FC_ACCEPTED 등 실제 근거만 기록한다."),
            ]),
        ]),
    ]),
    opt("응답 대기 초과 (2a)", [note("cs", "fc", "UNCONFIRMED 기록 · 중복 위험·재시도 한도 확인 후 같은 ID 로 재전송")]),
], entry="IGatewayLink.sendCommand → ServerChannelController.onCommand", 시작="유효 명령 구성 완료 (CommandService)")
S("SD-0904", "09", [("sr", "ServerReporter (Pi)", "component"), ("gc", "GatewayController", "controller"), CS, ("cl", "CommandLedger", "component"),
                    ("vs", "IVehicleStateService", "interface"), ("hs", "IHistoryService", "interface")], [
    call("sr", "gc", "onCommandEvidence(CommandEvidenceDTO)", "ResultDTO", "Pi 가 응답·실제 상태 근거를 보고한다.", [
        call("gc", "cs", "observeResult(CommandEvidenceDTO)", "CommandStatusDTO", "프로토콜 수락과 실제 수행·완료·불명을 분리한다.", [
            call("cs", "vs", "getState(droneId)", "VehicleStateDTO", "동작별 완료에 필요한 모드·속도·위치·착륙 표본을 얻는다."),
            call("cs", "cl", "applyRevision(CommandStatusDTO)", "StatusDecision", "증가 revision 의 결과만 반영한다."),
            call("cs", "hs", "append(MissionEventDTO)", "EventRefDTO", "처리 상태·원 세션·증거 시각·미확인 원인을 남긴다."),
        ]),
    ]),
    opt("단절로 확인 불가 (2a)", [note("sr", "hs", "UNCONFIRMED 로 기록 — 완료는 기체 동작으로만 확인")]),
], entry="GatewayController.onCommandEvidence", 시작="명령 응답 또는 실제 기체 상태 갱신 (ServerReporter)")
S("SD-0905", "09", [("op", "관제 운영자", "actor"), ("ctl", "CommandController", "controller"), CS, ("rn", "IMissionRunner", "interface"),
                    ("sf", "ISafetySupervisor", "interface")], [
    call("op", "ctl", "requestReturn(missionId, PrincipalDTO)", "CommandRefDTO", "복귀를 요청한다.", [
        call("ctl", "cs", "requestReturn(missionId, PrincipalDTO)", "CommandRefDTO", "정찰 목표를 차단하고 복귀를 요청한다.", [
            call("cs", "rn", "blockTargets(missionId, RETURN)", None, "이전 정찰·추적 목표를 무효화한다."),
            call("cs", "sf", "selectProtection(droneId)", "ProtectionDecisionDTO", "복귀에 필요한 위치·링크 유효성을 확인한다."),
            call("cs", "cs", "submitAction(ActionRequestDTO RETURN)", "CommandRefDTO", "공통 구성·검증·전달 절차로 복귀를 보낸다."),
        ]),
    ]),
    opt("위치 이상·통신 단절 (2a)", [note("ctl", "sf", "FC 보호 설정과 현장 RC 경로 적용")]),
], entry="CommandController.requestReturn")
S("SD-0906", "09", [("op", "관제 운영자", "actor"), ("ctl", "CommandController", "controller"), CS, ("rn", "IMissionRunner", "interface"),
                    ("cl", "CommandLedger", "component")], [
    call("op", "ctl", "requestLand(missionId, PrincipalDTO)", "CommandRefDTO", "착륙을 요청한다.", [
        call("ctl", "cs", "requestLand(missionId, PrincipalDTO)", "CommandRefDTO", "현장 조건을 확인하고 착륙을 보낸다.", [
            call("cs", "rn", "blockTargets(missionId, LAND)", None, "새 이동 목표를 멈춘다."),
            call("cs", "cs", "submitAction(ActionRequestDTO LAND)", "CommandRefDTO", "공통 검증 경로로 착륙 명령을 보낸다."),
            call("cs", "cl", "getStatus(commandId)", "CommandStatusDTO", "FC 착륙 상태로 완료를 확인한다."),
        ]),
    ]),
    opt("착륙 확인 지연 (3a)", [note("ctl", "cl", "UNCONFIRMED 표시")]),
], entry="CommandController.requestLand")
S("SD-X04", "09", [("lc", "LocalControlMonitor", "component"), ("lg", "ExecutionLeaseGuard", "component"), ("g", "GatewayCommandGuard", "component"),
                   ("sr", "ServerReporter", "component"), ("gc", "GatewayController", "controller"), ("ca", "IControlAuthority", "interface"),
                   ("rn", "IMissionRunner", "interface")], [
    call("lc", "lg", "invalidate(RC_TAKEOVER)", "AuthorityState", "RC 인수 즉시 LOCAL_OVERRIDE 로 잠그고 세대를 올린다."),
    call("lc", "sr", "reportLocalControl(LocalControlReportDTO)", None, "현장 세대·RC 상태를 서버에 보고한다."),
    call("sr", "gc", "onLocalControl(LocalControlReportDTO)", "ResultDTO", "서버가 후행 반영한다.", [
        call("gc", "ca", "observeLocalGeneration(LocalControlReportDTO)", "ControlStateDTO", "서버 제어 세대를 Pi 값으로 맞춘다."),
        call("gc", "rn", "blockTargets(missionId, RC_TAKEOVER)", None, "원격 정찰·추적의 새 목표를 막는다."),
    ]),
    call("g", "lg", "check(지연 도착 명령)", "LeaseDecisionDTO", "이전 세대 명령은 거부한다 (같은 ID 의 내용 변경도 거부)."),
], title="원격 권한·명령 실행·현장 인수", uc="UC-0104~0107, UC-0901~0904, UC-1002~1005",
   개요="원격 권한과 명령 실행 상태를 분리하고 현장 인수를 우선한다.", 시작="현장 RC 인수 (LocalControlMonitor)",
   선행="원격 실행 lease 가 유효하다.", 사후="이전 세대 명령은 FC 로 가지 않고, 서버는 현장 세대를 후행 반영한다.",
   예외="RC 인수 직후 지연 명령 도착을 FC 에서 시험한다. 명령 완료는 실제 기체 동작으로 확인한다.",
   경계="내부 이벤트 처리 | 결과 화면: UI-06 | 연결 시험: AT-F08, AT-F09")
S("SD-X09", "09", [("rn", "MissionRunner", "service"), CS, ("gw", "IGatewayLink", "interface"), ("g", "GatewayCommandGuard", "component"),
                   ("fa", "FlightAdapter", "component"), ("cl", "CommandLedger", "component")], [
    call("rn", "cs", "submitAction(ActionRequestDTO MISSION_UPLOAD)", "CommandRefDTO", "선택적 임무 파일 계약으로 업로드를 요청한다.", [
        call("cs", "gw", "sendCommand(CommandEnvelopeDTO)", "CommandStatusDTO", "Pi 로 보낸다.", [
            call("gw", "g", "dispatch(commandId)", "DispatchResult", "검증 후 업로드한다.", [
                call("g", "fa", "send(MISSION_ITEMS)", "TransportReceipt", "FC 에 임무 항목을 올린다."),
                call("g", "fa", "readMissionItems()", "MissionItems", "FC 의 전체 항목을 다시 읽어 비교한다."),
            ]),
        ]),
        alt([("일치", [call("cs", "cs", "submitAction(ActionRequestDTO ACTIVATE)", "CommandRefDTO", "승인된 활성화 명령으로 실행한다.")]),
             ("불일치·실패·취소", [call("cs", "cl", "appendTransition(commandId, MISSION_UNVERIFIED)", "CommandStatusDTO", "재검사 대상으로 기록한다.")])]),
    ]),
    note("rn", "cl", "FC plan ID · 파일 해시 · 임무 버전을 각각 보존"),
], title="선택적 임무 파일 업로드 계약", uc="UC-0405~0406, UC-0901~0904",
   개요="기본 목표점 방식과 구분한 선택적 임무 파일 계약이다.", 시작="임무 파일 업로드 요청 (MissionRunner)",
   선행="임무 파일 방식이 승인된 실행본에 포함되어 있다.", 사후="FC 임무 항목과 서버 임무 버전의 일치 여부가 기록된다.",
   예외="실패·취소 시 MISSION_UNVERIFIED 로 기록해 재검사한다.", 경계="내부 이벤트 처리 | 결과 화면: UI-06 | 연결 시험: AT-F08")
