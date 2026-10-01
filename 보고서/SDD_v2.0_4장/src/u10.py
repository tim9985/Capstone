"""안전 및 장애 관리 — services.safety · policies · gateway.flight"""
from mdl import K, I, DTO, unit
from seq import S, call, alt, opt, note, loop
from cdkit import CD, layered, box

unit("10", "안전 및 장애 관리", "안전·복구")
SF = "services.safety"

I("C-1005", "ISafetySupervisor", SF, "안전 상태 판정·장애 처리의 공개 계약", [
    ("evaluate", "droneId", "SafetyStateDTO", "링크·배터리·위치·경계·비행 모드의 유효성과 위험을 통합 판정한다."),
    ("detectFaults", "droneId, now", "List<FaultDTO>", "영상 2초·서버/기체 상태 3초 단절과 기타 장애를 구간별로 판정한다."),
    ("handleFault", "FaultDTO", "ProtectionDecisionDTO", "현장에서 적용된 보호 동작을 반영하고 새 자동 목표를 막는다."),
    ("selectProtection", "droneId", "ProtectionDecisionDTO", "현재 위치 유지 가능성에 따라 안전 동작을 고른다."),
], impl="C-1001")
I("C-1006", "IRecoveryCoordinator", SF, "연결 복구·재개 전 상태 대조의 공개 계약", [
    ("reconcile", "droneId, connectionEpoch", "RecoveryReportDTO", "재접속 기체의 모드·권한·전원·위치·명령을 재확인한다."),
    ("verifyForResume", "missionId", "RecoveryReportDTO", "수동·복귀·착륙·미확인 상태에서는 재개를 거부한다."),
], impl="C-1002")

K("C-1001", "SafetySupervisor", "service", SF, "링크·전원·위치·경계를 감시하고 새 정찰 목표를 제한 (ISafetySupervisor 구현)", [
    ("linkMonitor", "ILinkMonitor", "링크 상태"), ("vehicleState", "IVehicleStateService", "기체 상태"), ("safetyPolicy", "SafetyPolicy", "보호 동작 규칙"),
    ("runner", "IMissionRunner", "자동 목표 차단"), ("alerts", "IAlertService", "장애 알림"), ("history", "IHistoryService", "장애 이력"),
    ("activeFaults", "Set<Fault>", "현재 장애와 구간"),
], [], impl="C-1005", old="C-1001")
K("C-1002", "RecoveryCoordinator", "service", SF, "복구 후 기체·제어권·모드·명령을 대조하고 명시적 재개 대기 (IRecoveryCoordinator 구현)", [
    ("vehicleState", "IVehicleStateService", ""), ("controlAuthority", "IControlAuthority", ""), ("gatewayLink", "IGatewayLink", "미확인 명령 조회"),
    ("missionDao", "MissionDAO", "임무 실제 상태"), ("recoveryEpoch", "int64", "재동기화 세대"),
], [], impl="C-1006", old="C-1002")
K("C-1003", "SafetyPolicy", "component", "policies", "비행제어기와 게이트웨이의 사전 승인 안전 설정 규칙 — 서버·Pi 공유", [
    ("configuredActions", "Map<Fault, ActionRule>", "장애별 가능한 보호 동작"), ("settingsVersion", "UUID", "현장 검증된 설정 버전"),
], [
    ("verifySettings", "SafetySettingsDTO", "SafetyCheckReportDTO", "FC·게이트웨이의 RC 상실·저전압·경계 실제 설정을 대조한다."),
    ("validateConditions", "MissionConditionsDTO", "ConditionDecision", "비행 범위와 안전 설정의 충돌을 검사한다."),
    ("checkRoute", "route, state", "RouteDecision", "지도 장애물·운용 경계·고도·여유를 검사한다."),
    ("checkCommand", "CommandEnvelopeDTO, state", "SafetyDecision", "허용 모드·경계·고도·속도·안전 상태를 검사한다."),
    ("chooseAction", "FaultDTO, state", "ProtectionDecisionDTO", "위치·링크·기체 상태와 현장 설정으로 가능한 보호 동작을 고른다."),
], old="C-1003")
K("C-1004", "LocalControlMonitor", "component", "gateway.flight", "RC 인수·상실과 실제 비행 모드를 현장에서 감시 (watchdog)", [
    ("rcState", "RCState", "수동 인수·RC 연결"), ("lastFlightMode", "FlightMode", "관측한 비행 모드"),
    ("leaseGuard", "ExecutionLeaseGuard", "원격 권한 잠금"), ("reporter", "ServerReporter", "서버 보고"),
], [
    ("onRcTakeover", "rcEvidence", "ControlTransition", "현장에서 RC 인수·세대 갱신·이전 자동 목표 폐기를 처리한 뒤 서버에 알린다."),
    ("onFault", "LocalEvent", "ProtectionReceipt", "Pi 감시 타이머가 장애를 감지하면 보호 동작을 먼저 적용한다."),
    ("snapshot", "droneId", "LocalControlReportDTO", "RC 인수 여부와 실제 모드를 확인한다."),
], old="C-1004")
K("C-1007", "SafetyController", "controller", "api", "안전 상태·안전 설정 점검 요청의 입구 (REST)", [
    ("safety", "ISafetySupervisor", ""), ("preflight", "IPreflightService", ""),
], [
    ("getSafety", "droneId", "SafetyStateDTO", "GET /api/drones/{id}/safety"),
    ("checkSafetySettings", "droneId, List<FieldCheckDTO>", "PreflightReportDTO", "POST /api/drones/{id}/safety-check"),
])

DTO("C-1008", "SafetyStateDTO", "안전 상태", [("droneId", "String", "기체"), ("allowed", "bool", "새 원격 이동 허용"), ("faults", "List<FaultDTO>", "현재 장애"),
    ("reasons", "List<ReasonCode>", "미충족 사유"), ("evaluatedAt", "Time", "판정 시각")])
DTO("C-1009", "FaultDTO", "장애", [("kind", "VIDEO_LOSS | SERVER_LINK | VEHICLE_LINK | BATTERY | POSITION | GEOFENCE", "종류"), ("segment", "LinkId", "구간"),
    ("detectedAt", "Time", "감지"), ("lastOkAt", "Time", "마지막 정상"), ("appliedAction", "Action?", "현장 적용 동작")])
DTO("C-1010", "ProtectionDecisionDTO", "보호 동작 결정", [("action", "HOLD | RETURN | LAND | FC_FAILSAFE", "동작"), ("reason", "ReasonCode", "근거"),
    ("fallback", "Action?", "대체 동작")])
DTO("C-1011", "RecoveryReportDTO", "복구 점검 결과", [("droneId", "String", "기체"), ("mode", "FlightMode", "실제 모드"), ("controlEpoch", "int64", "세대"),
    ("pendingCommands", "List<CommandStatusDTO>", "미확인 명령"), ("resumable", "bool", "재개 가능"), ("reasons", "List<ReasonCode>", "거부 사유")])
DTO("C-1012", "SafetyCheckReportDTO", "안전 설정 대조 결과", [("items", "List<CheckResult>", "RC 상실·저전압·경계 항목"), ("approvedVersion", "UUID", "승인 설정"),
    ("passed", "bool", "통과")])


def _cd():
    a = layered("cd10", "", svc_pkg=SF, ctl=["C-1007"], dto=["C-1008", "C-1009", "C-1010", "C-1011", "C-1012"],
                pairs=[("C-1005", "C-1001"), ("C-1006", "C-1002")],
                ext=[("C-0206", "services.vehicle"), ("C-0205", "services.vehicle"), ("C-0406", "services.mission"),
                     ("C-1105", "services.alert"), ("C-1207", "services.history"), ("C-0107", "services.auth"), ("C-0909", "services.command")],
                daos=[("C-0404", True)], ents=[], api_w=0.35, ext_w=0.27)
    key, sub, P, B, R = a
    B["sp"] = box("C-1003"); B["lc"] = box("C-1004")
    B["lg"] = box("C-0905", ref=True); B["sr"] = box("C-0913", ref=True)
    P.append(dict(name="policies", row=3, x=0.005, w=0.5, rows=[[("sp", 1.0)]]))
    P.append(dict(name="gateway.flight", row=3, x=0.515, w=0.48, rows=[[("lc", 1.0)], [("lg", .5), ("sr", .5)]]))
    R += [("k1001", "sp", "dep", "", {"elbow": 1}), ("lc", "lg", "assoc", "", {"elbow": 1}), ("lc", "sr", "assoc", "", {"elbow": 1})]
    return [a]


CD("CD-10", "안전·복구", "10", _cd)

SFI = ("sf", "ISafetySupervisor", "interface")
S("SD-1001", "10", [("rn", "MissionRunner · CommandService", "service"), SFI, ("lm", "ILinkMonitor", "interface"), ("vs", "IVehicleStateService", "interface")], [
    call("rn", "sf", "evaluate(droneId)", "SafetyStateDTO", "현재 비행과 새 명령의 안전 조건을 확인한다.", [
        call("sf", "lm", "getLinkStates(droneId)", "LinkStateDTO", "각 구간의 정상·단절을 따로 조회한다."),
        call("sf", "vs", "getState(droneId)", "VehicleStateDTO", "보호 동작이 가능한 기체 상태를 확인한다."),
        opt("필수 상태 미확인 (2a)", [note("sf", "vs", "새 원격 이동 보류 (allowed=false)")]),
    ]),
], entry="ISafetySupervisor.evaluate", 시작="주기 상태 감시 또는 상태 변화 (MissionRunner · CommandService)")
S("SD-1002", "10", [("rc", "현장 조종자 (RC)", "actor"), ("lc", "LocalControlMonitor", "component"), ("lg", "ExecutionLeaseGuard", "component"),
                    ("sr", "ServerReporter", "component"), ("gc", "GatewayController", "controller"), ("ca", "IControlAuthority", "interface"),
                    ("rn", "IMissionRunner", "interface"), ("hs", "IHistoryService", "interface")], [
    call("rc", "lc", "onRcTakeover(rcEvidence)", "ControlTransition", "독립 RC 인수와 실제 FC 모드를 현장에서 확인한다.", [
        call("lc", "lg", "invalidate(RC_TAKEOVER)", "AuthorityState", "세대·lease 를 무효화하고 LOCAL_OVERRIDE 로 잠근다."),
        call("lc", "sr", "reportLocalControl(LocalControlReportDTO)", None, "현장 세대·상태를 서버에 보고한다."),
    ]),
    call("sr", "gc", "onLocalControl(LocalControlReportDTO)", "ResultDTO", "서버가 확인된 현장 상태를 후행 반영한다.", [
        call("gc", "ca", "observeLocalGeneration(LocalControlReportDTO)", "ControlStateDTO", "서버 제어 세대를 맞춘다."),
        call("gc", "rn", "blockTargets(missionId, RC_TAKEOVER)", None, "원격 정찰·추적의 새 목표를 막는다."),
        call("gc", "hs", "append(MissionEventDTO)", "EventRefDTO", "인수 전후 지연 명령·실제 결과를 기록한다."),
    ]),
    note("rc", "hs", "RC 는 서버·인터넷과 독립 · RC 상실 시 FC 보호 → 복구 후 현장 확인·재개 승인 (1a)"),
], entry="LocalControlMonitor.onRcTakeover", 시작="현장 조종자 → RC → 비행제어기 (서버와 독립)")
S("SD-1003", "10", [("tm", "감시 타이머 (현장·중앙)", "external"), SFI, ("lm", "ILinkMonitor", "interface"), ("al", "IAlertService", "interface")], [
    call("tm", "sf", "detectFaults(droneId, now)", "List<FaultDTO>", "영상 2초·서버/기체 상태 3초 단절과 기타 장애를 구간별로 판정한다.", [
        call("sf", "lm", "getLinkStates(droneId)", "LinkStateDTO", "마지막 정상 시각을 확인한다."),
        opt("장애 있음", [call("sf", "al", "publish(AlertEventDTO)", "AlertDTO", "장애 원인·영향·현재 보호 상태를 알린다.")]),
    ]),
    note("tm", "al", "통신 단절 시 마지막 값과 갱신 시각을 오래된 정보로 표시 (3a)"),
], entry="ISafetySupervisor.detectFaults", 시작="현장·중앙 독립 감시 타이머")
S("SD-1004", "10", [("lc", "LocalControlMonitor", "component"), ("sp", "SafetyPolicy", "component"), ("fa", "FlightAdapter", "component"),
                    ("sr", "ServerReporter", "component"), ("gc", "GatewayController", "controller"), SFI, ("rn", "IMissionRunner", "interface")], [
    call("lc", "lc", "onFault(LocalEvent)", "ProtectionReceipt", "Pi 가 장애를 감지하면 새 자동 이동을 먼저 막는다.", [
        call("lc", "sp", "chooseAction(FaultDTO, state)", "ProtectionDecisionDTO", "위치·링크·기체 상태와 현장 설정으로 보호 동작을 고른다."),
        call("lc", "fa", "applyLocalProtection(ProtectionDecisionDTO)", "ProtectionReceipt", "현장 경로로 보낸다 (기체 링크 단절이면 FC 자체 보호 · 미확인 기록)."),
        call("lc", "sr", "reportFault(FaultDTO)", None, "적용한 동작을 서버에 보고한다."),
    ]),
    opt("중앙 연결됨", [call("sr", "gc", "onFault(FaultDTO)", "ResultDTO", "서버가 후행 반영한다.", [
        call("gc", "sf", "handleFault(FaultDTO)", "ProtectionDecisionDTO", "장애 상태를 반영한다.", [
            call("sf", "rn", "blockTargets(missionId, FAULT)", None, "정찰·추적 중단을 후행 통지한다 (현장 보호 동작과 독립).")])])]),
    note("lc", "rn", "위치 유지·복귀가 어려우면 설정된 대체 보호 동작 (3a)"),
], entry="LocalControlMonitor.onFault → GatewayController.onFault", 시작="장애 감지 또는 비행제어기 보호 상태 변화 (Pi 감시)")
S("SD-1005", "10", [("gc", "GatewayController", "controller"), ("rc", "IRecoveryCoordinator", "interface"), ("vs", "IVehicleStateService", "interface"),
                    ("ca", "IControlAuthority", "interface"), ("gw", "IGatewayLink", "interface"), ("md", "MissionDAO", "dao")], [
    call("gc", "rc", "reconcile(droneId, connectionEpoch)", "RecoveryReportDTO", "복구된 연결에서 실제 상태를 다시 맞춘다.", [
        call("rc", "vs", "getState(droneId)", "VehicleStateDTO", "실제 FC/RC·항법·전원·최근 명령 근거를 조회한다."),
        call("rc", "ca", "observeLocalGeneration(LocalControlReportDTO)", "ControlStateDTO", "Pi 가 확인한 현재 세대를 반영한다."),
        call("rc", "gw", "sendCommand(RECONCILE_QUERY)", "CommandStatusDTO", "미확인 명령을 Pi 기록과 실제 상태로 대조한다 (CommandRecoveryReconciler)."),
        call("rc", "md", "update(Mission, state)", "bool", "오래된 명령을 무효화하고 재개 대기 상태로 둔다."),
    ]),
    note("gc", "md", "연결 복구 후 기체 상태·현장 허용 확인 → 운영자 승인으로만 재개 (3a)"),
], entry="IRecoveryCoordinator.reconcile", 시작="게이트웨이·서버·기체 링크 복구 이벤트 (GatewayController)")
S("SD-1006", "10", [("op", "관제 운영자 · 현장 조종자", "actor"), ("ctl", "SafetyController", "controller"), ("pf", "IPreflightService", "interface"),
                    ("gw", "IGatewayLink", "interface"), ("fa", "FlightAdapter", "component"), ("sp", "SafetyPolicy", "component")], [
    call("op", "ctl", "checkSafetySettings(droneId, List<FieldCheckDTO>)", "PreflightReportDTO", "보호 설정 점검을 요청한다.", [
        call("ctl", "pf", "recordFieldCheck(FieldCheckDTO)", "ResultDTO", "현장 조종자의 확인 항목·시각을 잇는다."),
        call("ctl", "pf", "runChecks(droneId, List<FieldCheckDTO>)", "PreflightReportDTO", "안전 설정을 포함해 점검한다.", [
            call("pf", "gw", "readSafetySettings(droneId)", "SafetySettingsDTO", "Pi 를 거쳐 FC 설정을 읽는다.", [
                call("gw", "fa", "readSafetySettings()", "SafetySettingsDTO", "읽을 수 있는 설정과 모드를 조회한다.")]),
            call("pf", "sp", "verifySettings(SafetySettingsDTO)", "SafetyCheckReportDTO", "RC 상실·저전압·경계 실제 설정을 대조한다."),
        ]),
    ]),
    note("op", "sp", "필수 안전 설정을 모두 확인한 뒤에만 임무 시작 허용 (3a)"),
], entry="SafetyController.checkSafetySettings", 시작="운영자·현장 조종자의 준비 점검")
