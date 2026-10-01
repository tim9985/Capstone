"""기체 및 게이트웨이 상태 관리 — services.vehicle"""
from mdl import K, I, DTO, ENT, DAO, unit
from seq import S, call, alt, opt, note
from cdkit import CD, layered

unit("02", "기체 및 게이트웨이 상태 관리", "기체·게이트웨이 상태")

I("C-0205", "IVehicleStateService", "services.vehicle", "기체 상태·비행정보 조회의 공개 계약", [
    ("getState", "droneId", "VehicleStateDTO", "위치·자세·고도·속도·배터리·모드를 필드 유효성과 함께 반환한다."),
    ("getTelemetryWindow", "droneId, captureAt, span", "TelemetryWindowDTO", "촬영 시각 앞뒤의 GNSS·자세·짐벌 표본을 반환한다."),
    ("recordTelemetry", "TelemetryReportDTO", "ResultDTO", "게이트웨이가 보낸 비행정보 표본을 순번 중복 없이 저장한다."),
], impl="C-0201")
I("C-0206", "ILinkMonitor", "services.vehicle", "링크 상태 감시의 공개 계약", [
    ("getLinkStates", "droneId", "LinkStateDTO", "웹–서버·서버–게이트웨이·기체·영상 링크를 구분하여 반환한다."),
    ("recordReception", "droneId, linkId, time", "void", "실제 프레임·표본 수신으로 마지막 정상 시각을 갱신한다."),
], impl="C-0202")
I("C-0207", "IPreflightService", "services.vehicle", "운용 준비 점검의 공개 계약", [
    ("runChecks", "droneId, List<FieldCheckDTO>", "PreflightReportDTO", "필수 점검과 현장 확인 결과로 운용 준비 여부를 반환한다."),
    ("recordFieldCheck", "FieldCheckDTO", "ResultDTO", "현장 조종자의 확인 항목·시각을 기록한다."),
], impl="C-0203")

K("C-0201", "VehicleStateService", "service", "services.vehicle", "기체 상태를 단위·기준·시각·유효성과 함께 제공 (IVehicleStateService 구현)", [
    ("telemetryDao", "TelemetrySampleDAO", "비행정보 표본 저장·조회"),
    ("droneDao", "DroneDAO", "기체 등록 정보"),
    ("freshnessLimit", "Duration", "상태 단절 판정값 3초"),
], [("-toStateDTO", "TelemetrySample, ValidityFlags", "VehicleStateDTO", "표본을 화면용 DTO 로 바꾼다.")], impl="C-0205", old="C-0201")
K("C-0202", "LinkMonitor", "service", "services.vehicle", "웹·서버·게이트웨이·비행정보·영상 링크를 각각 감시 (ILinkMonitor 구현)", [
    ("droneDao", "DroneDAO", "링크별 상태 (drone.link_state)"),
    ("videoTimeout", "Duration", "영상 단절 판정값 2초"),
    ("statusTimeout", "Duration", "서버·기체 상태 단절 판정값 3초"),
], [("-classify", "lastOkAt, now, timeout", "LinkState", "마지막 정상 시각으로 정상/지연/단절을 판정한다.")], impl="C-0206", old="C-0202")
K("C-0203", "PreflightService", "service", "services.vehicle", "운용 필수조건과 현장 확인 결과를 결합하여 시작 가능성 판정 (IPreflightService 구현)", [
    ("vehicleState", "IVehicleStateService", "위치·전원·모드 확인"),
    ("linkMonitor", "ILinkMonitor", "필수 통신·영상 확인"),
    ("safetyPolicy", "SafetyPolicy", "승인된 안전 설정 대조"),
    ("gatewayLink", "IGatewayLink", "FC 실제 안전 설정 읽기"),
    ("requiredChecks", "List<CheckSpec>", "연결·영상·위치·전원·모드·안전 설정"),
], [("-merge", "List<CheckResult>", "PreflightReportDTO", "항목별 결과와 근거를 합친다.")], impl="C-0207", old="C-0203")

K("C-0208", "DroneController", "controller", "api", "기체 상태·링크·운용 준비 점검 요청의 입구 (REST)", [
    ("vehicleState", "IVehicleStateService", ""), ("linkMonitor", "ILinkMonitor", ""), ("preflight", "IPreflightService", ""),
], [
    ("getState", "droneId", "VehicleStateDTO", "GET /api/drones/{id}/state"),
    ("getLinks", "droneId", "LinkStateDTO", "GET /api/drones/{id}/links"),
    ("runPreflight", "droneId, List<FieldCheckDTO>", "PreflightReportDTO", "POST /api/drones/{id}/preflight"),
])
K("C-0218", "GatewayController", "controller", "api", "게이트웨이(Pi)가 보낸 비행정보·명령 근거·현장 제어 보고의 입구 (WSS)", [
    ("vehicleState", "IVehicleStateService", ""), ("controlAuthority", "IControlAuthority", ""),
    ("commandService", "ICommandService", ""), ("safety", "ISafetySupervisor", ""),
], [
    ("onTelemetry", "TelemetryReportDTO", "ResultDTO", "비행정보 표본 보고를 받는다."),
    ("onLocalControl", "LocalControlReportDTO", "ResultDTO", "RC 인수·제어 세대 변경 보고를 받는다."),
    ("onCommandEvidence", "CommandEvidenceDTO", "ResultDTO", "명령 수락·실행·완료 근거를 받는다."),
    ("onFault", "FaultDTO", "ResultDTO", "현장에서 감지한 장애와 적용한 보호 동작을 받는다."),
])

DTO("C-0209", "VehicleStateDTO", "기체 상태 — 값마다 단위·기준·측정 시각·유효성",
    [("droneId", "String", "기체"), ("position", "GeoPoint", "WGS84 위치"), ("altitude", "Altitude", "고도와 기준"),
     ("attitude", "Attitude", "자세"), ("speed", "Velocity", "속도"), ("battery", "BatteryState", "전원"), ("mode", "FlightMode", "비행 모드"),
     ("validity", "ValidityFlags", "필드별 유효·오래됨·미확인"), ("measuredAt", "Time", "측정 시각")])
DTO("C-0210", "LinkStateDTO", "링크별 연결 상태",
    [("droneId", "String", "기체"), ("links", "List<LinkState>", "링크 ID·상태·사유·마지막 정상 시각")])
DTO("C-0211", "PreflightReportDTO", "운용 준비 점검 결과",
    [("droneId", "String", "기체"), ("items", "List<CheckResult>", "항목별 결과·근거"), ("ready", "bool", "시작 가능 여부"),
     ("blockingReasons", "List<ReasonCode>", "시작을 막는 사유")])
DTO("C-0212", "FieldCheckDTO", "현장 조종자 확인 항목", [("item", "CheckItem", "항목"), ("passed", "bool", "결과"), ("checkedBy", "UUID", "확인자"), ("checkedAt", "Time", "시각")])
DTO("C-0213", "TelemetryReportDTO", "게이트웨이 → 서버 비행정보 표본",
    [("droneId", "String", "기체"), ("producerSessionId", "String", "생산 세션"), ("sequence", "int64", "순번"), ("kind", "GNSS | ATTITUDE | GIMBAL | STATUS", "종류"),
     ("sourceTime", "TimeEvidence", "측정 시각과 근거"), ("pose", "Pose", "위치·고도·자세"), ("flightState", "FlightState", "속도·배터리·모드·RC")])
DTO("C-0214", "LocalControlReportDTO", "게이트웨이 → 서버 현장 제어 보고",
    [("droneId", "String", "기체"), ("controlEpoch", "int64", "Pi 제어 세대"), ("rcManual", "bool", "RC 수동 인수"), ("flightMode", "FlightMode", "실제 모드"),
     ("observedAt", "Time", "확인 시각")])
DTO("C-0219", "TelemetryWindowDTO", "촬영 시각 앞뒤의 비행정보 표본 묶음",
    [("droneId", "String", "기체"), ("samples", "List<TelemetrySample>", "GNSS·자세·짐벌 표본"), ("span", "TimeRange", "범위")])

DAO("C-0215", "TelemetrySampleDAO", "telemetry_sample", [
    ("insert", "TelemetrySample", "UUID", "표본을 저장한다 (생산 세션·순번 유일)."),
    ("findLatest", "droneId", "TelemetrySample?", "가장 최근 표본을 읽는다."),
    ("findWindow", "droneId, from, to", "List<TelemetrySample>", "시각 범위의 표본을 읽는다."),
])
DAO("C-0216", "DroneDAO", "drone", [
    ("findById", "droneId", "Drone?", "기체 등록·제어권·게이트웨이·링크 상태를 읽는다."),
    ("updateControlLease", "Drone, expectedRevision", "bool", "제어권 세대가 같을 때만 보유자를 바꾼다."),
    ("updateLinkState", "droneId, linkState", "bool", "링크별 상태를 저장한다."),
    ("updateGatewayState", "droneId, gatewayState, controlEpoch", "bool", "Pi 세션·제어 세대를 저장한다."),
])
ENT("C-0204", "TelemetrySample", "telemetry_sample", "획득 시점의 비행정보와 좌표·고도 기준·품질 (DB-11 한 행 · 불변)", ops=[
    ("validateFreshness", "now", "ValidityFlags", "3초 단절 기준과 필드별 측정 시각을 대조한다."),
], keys=["sampleId {PK}", "droneId {FK}", "sequence", "sourceTimeUtc", "pose : JSON", "validity : JSON"])
ENT("C-0217", "Drone", "drone", "운용 기체와 웹 제어권·게이트웨이·링크 상태 (DB-03 한 행)",
    keys=["droneId {PK}", "controlUserId {FK}", "controlRevision", "controlEpoch", "gatewayState : JSON", "linkState : JSON"])


def _cd():
    return [layered("cd02", "", svc_pkg="services.vehicle",
                    ctl=["C-0208", "C-0218"], dto=["C-0209", "C-0210", "C-0211", "C-0212", "C-0213", "C-0214", "C-0219"],
                    pairs=[("C-0205", "C-0201"), ("C-0206", "C-0202"), ("C-0207", "C-0203")],
                    ext=[("C-0909", "services.command"), ("C-1003", "policies")],
                    daos=[("C-0215", False), ("C-0216", False)], ents=["C-0204", "C-0217"], api_w=0.55)]


CD("CD-02", "기체·게이트웨이 상태", "02", _cd)

OP = ("op", "관제 운영자", "actor")
S("SD-0201", "02", [OP, ("ctl", "DroneController", "controller"), ("svc", "IVehicleStateService", "interface"), ("dao", "TelemetrySampleDAO", "dao")], [
    call("op", "ctl", "getState(droneId)", "VehicleStateDTO", "기체 상태를 조회한다.", [
        call("ctl", "svc", "getState(droneId)", "VehicleStateDTO", "상태를 필드 유효성과 함께 만든다.", [
            call("svc", "dao", "findLatest(droneId)", "TelemetrySample", "가장 최근 표본을 읽는다."),
            call("svc", "svc", "TelemetrySample.validateFreshness(now)", "ValidityFlags", "3초 단절 기준과 필드별 측정 시각을 대조한다."),
            opt("갱신 끊김 (2a)", [note("svc", "dao", "마지막 값을 '오래됨' 으로 표시")]),
        ]),
    ]),
], entry="DroneController.getState")
S("SD-0202", "02", [OP, ("ctl", "DroneController", "controller"), ("lm", "ILinkMonitor", "interface"), ("dr", "DroneDAO", "dao"),
                    ("vs", "IVehicleStateService", "interface")], [
    call("op", "ctl", "getLinks(droneId)", "LinkStateDTO", "통신 상태를 조회한다.", [
        call("ctl", "lm", "getLinkStates(droneId)", "LinkStateDTO", "링크별 상태를 구분해 만든다.", [
            call("lm", "dr", "findById(droneId)", "Drone", "링크별 마지막 정상 시각을 읽는다."),
            call("lm", "vs", "getState(droneId)", "VehicleStateDTO", "비행정보 갱신 시각을 링크 판정과 대조한다."),
        ]),
    ]),
], entry="DroneController.getLinks")
S("SD-0203", "02", [OP, ("ctl", "DroneController", "controller"), ("pf", "IPreflightService", "interface"), ("vs", "IVehicleStateService", "interface"),
                    ("lm", "ILinkMonitor", "interface"), ("gw", "IGatewayLink", "interface"), ("sp", "SafetyPolicy", "component")], [
    call("op", "ctl", "runPreflight(droneId, List<FieldCheckDTO>)", "PreflightReportDTO", "운용 준비 점검을 요청한다.", [
        call("ctl", "pf", "runChecks(droneId, List<FieldCheckDTO>)", "PreflightReportDTO", "필수 점검과 현장 확인을 결합한다.", [
            call("pf", "vs", "getState(droneId)", "VehicleStateDTO", "위치·전원·모드·측정 유효성을 검사한다."),
            call("pf", "lm", "getLinkStates(droneId)", "LinkStateDTO", "필수 통신과 영상 상태를 검사한다."),
            call("pf", "gw", "readSafetySettings(droneId)", "SafetySettingsDTO", "FC 의 실제 안전 설정을 읽는다."),
            call("pf", "sp", "verifySettings(SafetySettingsDTO)", "SafetyCheckReportDTO", "승인된 설정과 대조한다."),
            opt("필수 점검 미충족", [note("pf", "sp", "ready=false · 시작 차단 사유 반환")]),
        ]),
    ]),
], entry="DroneController.runPreflight", 예외="3a. 필수 점검을 통과한 뒤 운영자의 시작 요청으로 이륙 절차를 진행한다. 필수 조건 미충족 시 시작을 차단한다.")
