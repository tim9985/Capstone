"""4.x.1 사용자 및 제어권 관리 — services.auth"""
from mdl import K, I, DTO, ENT, DAO, unit
from seq import S, call, alt, opt, note

unit("01", "사용자 및 제어권 관리", "사용자 및 제어권")

# ── 서비스 인터페이스 ─────────────────────────────────────────────
I("C-0106", "IAuthService", "services.auth", "로그인·로그아웃 기능의 공개 계약 (AuthController 와 다른 패키지가 의존)", [
    ("login", "LoginRequestDTO", "SessionDTO", "등록 자격정보를 검증하고 세션과 접근 가능한 기능을 반환한다."),
    ("logout", "sessionId", "ResultDTO", "사용자 세션과 웹 제어권을 종료한다. 승인된 자동 임무 실행 권한은 유지한다."),
], impl="C-0101")
I("C-0107", "IControlAuthority", "services.auth", "웹 제어권과 자동 임무 실행 권한의 공개 계약", [
    ("acquire", "droneId, PrincipalDTO", "ControlStateDTO", "동일 기체의 웹 제어권을 조건부 갱신으로 원자적으로 확보한다."),
    ("release", "droneId, PrincipalDTO", "ControlStateDTO", "현재 보유자와 제어권 세대를 확인한 뒤 반납한다."),
    ("getControlState", "droneId", "ControlStateDTO", "보유·만료·RC 수동 상태와 제어 세대를 함께 반환한다."),
    ("releaseForSession", "sessionId", "ResultDTO", "로그아웃 세션의 웹 보유권만 해제한다. 임무 실행 권한은 별도이다."),
    ("validateAuthority", "PrincipalDTO, CommandContextDTO", "AuthorityDecisionDTO", "웹 제어권 또는 활성 임무 실행 권한과 현장 제어 세대를 검사한다."),
    ("approveExecution", "missionId, revision, PrincipalDTO", "MissionGrantDTO", "승인된 실행본에 브라우저 세션과 독립된 자동 임무 실행 권한을 부여한다."),
    ("observeLocalGeneration", "LocalControlReportDTO", "ControlStateDTO", "Pi 가 확인한 현장 제어 세대·RC 상태를 서버 상태에 후행 반영한다."),
], impl="C-0102")
I("C-0108", "IPermissionPolicy", "services.auth", "조회·편집·명령 권한 판정의 공개 계약", [
    ("checkAccess", "PrincipalDTO, feature", "AccessDecisionDTO", "인증 주체의 조회·편집·제어 요청 권한을 검사한다."),
    ("authorizeCommand", "PrincipalDTO, CommandContextDTO", "AccessDecisionDTO", "사용자 요청과 자동 임무 요청을 구분하여 실행 권한을 판정한다."),
    ("resolve", "userId", "PermissionSetDTO", "계정의 조회·제어 요청 권한 집합을 계산한다."),
], impl="C-0103")

# ── 구현 · 구성 요소 ──────────────────────────────────────────────
K("C-0101", "AuthService", "service", "services.auth", "등록 계정 인증·세션 종료와 접근 가능한 기능 반환 (IAuthService 구현)", [
    ("accountDao", "UserAccountDAO", "계정 조회"),
    ("sessionDao", "UserSessionDAO", "인증 세션 저장·폐기"),
    ("passwordVerifier", "PasswordVerifier", "자격정보 해시 검증"),
    ("permissionPolicy", "IPermissionPolicy", "접근 가능한 기능 계산"),
    ("controlAuthority", "IControlAuthority", "로그아웃 시 웹 보유권 해제"),
    ("sessionTtl", "Duration", "세션 유효 시간"),
], [
    ("-toSessionDTO", "UserSession, PermissionSetDTO", "SessionDTO", "엔티티를 화면용 DTO 로 바꾼다."),
], impl="C-0106", old="C-0101")
K("C-0102", "ControlAuthority", "service", "services.auth", "웹 제어권과 승인된 자동 임무 실행 권한을 분리하여 관리 (IControlAuthority 구현)", [
    ("droneDao", "DroneDAO", "기체별 웹 제어권·제어 세대 (drone 테이블)"),
    ("missionDao", "MissionDAO", "임무 실행 승인 상태 (mission 테이블)"),
    ("permissionPolicy", "IPermissionPolicy", "제어 요청 권한 확인"),
    ("leaseTtl", "Duration", "웹 제어권 유효 시간"),
], [
    ("-advanceGeneration", "droneId, reason", "int64", "RC 인수·복구 시 Pi 가 확인한 제어 세대로 서버 값을 올린다."),
    ("-toStateDTO", "Drone", "ControlStateDTO", "drone 엔티티를 제어권 상태 DTO 로 바꾼다."),
], impl="C-0107", old="C-0102")
K("C-0103", "PermissionPolicy", "service", "services.auth", "조회·편집·명령 요청의 주체별 권한 판정 (IPermissionPolicy 구현)", [
    ("accountDao", "UserAccountDAO", "계정 권한 집합 조회"),
    ("sessionDao", "UserSessionDAO", "세션 유효성 확인"),
    ("denyReason", "ReasonCode?", "최근 판정의 거부 사유"),
], [
    ("-isSessionValid", "sessionId, now", "bool", "세션 만료·폐기 여부를 확인한다."),
], impl="C-0108", old="C-0103")
K("C-0105", "PasswordVerifier", "component", "services.auth", "등록 자격정보 해시 검증 (평문 비밀번호는 저장하지 않는다)", [
    ("algorithm", "HashSpec", "해시 알고리즘·반복 횟수"),
], [
    ("verify", "password, credentialHash", "bool", "입력 비밀번호와 저장된 해시를 비교한다."),
])

# ── Controller ───────────────────────────────────────────────────
K("C-0109", "AuthController", "controller", "api", "로그인·로그아웃·권한 조회 요청의 입구 (REST)", [
    ("authService", "IAuthService", "인증 서비스"),
    ("permissionPolicy", "IPermissionPolicy", "권한 판정"),
], [
    ("login", "LoginRequestDTO", "SessionDTO", "POST /api/sessions — 로그인 요청을 받아 세션 DTO 를 응답한다."),
    ("logout", "PrincipalDTO", "ResultDTO", "DELETE /api/sessions/current — 현재 세션을 종료한다."),
    ("checkPermission", "PrincipalDTO, feature", "AccessDecisionDTO", "GET /api/me/permissions — 기능별 사용 가능 여부를 응답한다."),
])
K("C-0110", "ControlController", "controller", "api", "기체 웹 제어권 확보·반납·조회 요청의 입구 (REST)", [
    ("controlAuthority", "IControlAuthority", "제어권 서비스"),
], [
    ("acquireLease", "droneId, PrincipalDTO", "ControlStateDTO", "POST /api/drones/{id}/control-lease"),
    ("releaseLease", "droneId, PrincipalDTO", "ControlStateDTO", "DELETE /api/drones/{id}/control-lease"),
    ("getControlState", "droneId", "ControlStateDTO", "GET /api/drones/{id}/control-state"),
])

# ── DTO ─────────────────────────────────────────────────────────
DTO("C-0111", "LoginRequestDTO", "로그인 요청", [("loginId", "String", "로그인 ID"), ("password", "String", "비밀번호 (TLS 구간만 전송)")])
DTO("C-0112", "SessionDTO", "로그인 응답 — 세션과 접근 가능한 기능",
    [("sessionId", "UUID", "세션"), ("userId", "UUID", "계정"), ("expiresAt", "Time", "만료"), ("permissions", "PermissionSetDTO", "접근 가능한 기능")])
DTO("C-0113", "PermissionSetDTO", "계정의 조회·제어 요청 권한 집합",
    [("userId", "UUID", "계정"), ("viewFeatures", "List<Feature>", "조회 기능"), ("controlFeatures", "List<Feature>", "편집·제어 요청 기능")])
DTO("C-0114", "AccessDecisionDTO", "권한 판정 결과",
    [("allowed", "bool", "허용 여부"), ("feature", "Feature", "대상 기능"), ("reasonCode", "ReasonCode?", "거부 사유")])
DTO("C-0115", "ControlStateDTO", "기체 웹 제어권·현장 제어 상태",
    [("droneId", "String", "기체"), ("holderUserId", "UUID?", "보유자"), ("controlRevision", "int64", "웹 제어권 변경 세대"),
     ("expiresAt", "Time?", "만료"), ("controlEpoch", "int64", "Pi 제어 세대"), ("rcManual", "bool", "RC 수동 인수 여부"),
     ("reasonCode", "ReasonCode?", "거부 사유")])
DTO("C-0116", "AuthorityDecisionDTO", "실행 권한 판정 결과",
    [("allowed", "bool", "허용 여부"), ("source", "WEB_LEASE | MISSION_GRANT", "권한 근거"), ("controlEpoch", "int64", "판정 기준 세대"),
     ("reasonCode", "ReasonCode?", "거부 사유")])
DTO("C-0117", "MissionGrantDTO", "자동 임무 실행 권한",
    [("missionId", "UUID", "임무"), ("revision", "int", "실행 개정"), ("grantStatus", "GrantStatus", "승인 상태"),
     ("approvedBy", "UUID", "승인자"), ("approvedAt", "Time", "승인 시각")])
DTO("C-0118", "PrincipalDTO", "요청 주체 — 인증 미들웨어가 만든다",
    [("userId", "UUID?", "운영자 (자동 임무이면 비어 있음)"), ("sessionId", "UUID?", "세션"), ("kind", "USER | SYSTEM", "사용자/자동 임무 구분")])
DTO("C-0122", "ResultDTO", "공통 처리 결과 — 모든 실패는 사유 코드·현재 상태·관측 시각을 가진다",
    [("status", "OK | REJECTED | PENDING | UNKNOWN", "처리 상태"), ("reasonCode", "ReasonCode?", "사유"), ("currentState", "String?", "현재 상태"),
     ("observedAt", "Time", "관측 시각")])

# ── DAO · Entity ────────────────────────────────────────────────
DAO("C-0119", "UserAccountDAO", "user_account", [
    ("findByLoginId", "loginId", "UserAccount?", "로그인 ID 로 계정을 읽는다."),
    ("findById", "userId", "UserAccount?", "계정과 권한 집합을 읽는다."),
])
DAO("C-0120", "UserSessionDAO", "user_session", [
    ("insert", "UserSession", "UUID", "발급한 세션을 저장한다."),
    ("findById", "sessionId", "UserSession?", "세션을 읽는다."),
    ("update", "UserSession", "bool", "폐기 시각 등 변경을 저장한다."),
])
ENT("C-0121", "UserAccount", "user_account", "등록 계정과 권한 집합 (DB-01 한 행)",
    keys=["userId {PK}", "loginId {UQ}", "credentialHash", "permissions : JSON", "status"])
ENT("C-0104", "UserSession", "user_session", "인증 세션의 유효성 및 종료 상태 (DB-02 한 행)", ops=[
    ("issue", "userId, ttl", "UserSession", "인증된 계정의 세션을 만든다 (정적 생성)."),
    ("revoke", "now", "void", "세션을 무효화한다."),
    ("validate", "now", "SessionStatus", "만료·종료 여부를 확인한다."),
], keys=["sessionId {PK}", "userId {FK}", "tokenDigest {UQ}", "expiresAt", "revokedAt"])


# ── 클래스 다이어그램 CD-01 ─────────────────────────────────────
from cdkit import box, CD


def _cd01():
    B = {
        "ac": box("C-0109"), "cc": box("C-0110"),
        "d1": box("C-0111", compact=True), "d2": box("C-0112", compact=True), "d3": box("C-0113", compact=True),
        "d4": box("C-0114", compact=True), "d5": box("C-0115", compact=True), "d6": box("C-0116", compact=True),
        "d7": box("C-0117", compact=True), "d8": box("C-0118", compact=True), "d9": box("C-0122", compact=True),
        "ipp": box("C-0108"), "ias": box("C-0106"), "ica": box("C-0107"),
        "pp": box("C-0103", ops=[]), "as": box("C-0101", ops=[]), "ca": box("C-0102", ops=[]), "pv": box("C-0105"),
        "dacc": box("C-0119"), "dses": box("C-0120"), "ddr": box("C-0216", ref=True), "dms": box("C-0404", ref=True),
        "eacc": box("C-0121"), "eses": box("C-0104"),
    }
    P = [
        dict(key="api", name="api", row=0, x=0.005, w=0.43, rows=[[("ac", 0.5), ("cc", 0.5)]]),
        dict(key="ct", name="contracts", row=0, x=0.455, w=0.54, rows=[[("d1", .333), ("d2", .333), ("d3", .334)],
                                                                      [("d4", .333), ("d5", .333), ("d6", .334)],
                                                                      [("d7", .333), ("d8", .333), ("d9", .334)]]),
        dict(key="svc", name="services.auth", row=1, x=0.005, w=0.99, rows=[[(None, .25), ("ias", .25), ("ipp", .25), ("ica", .25)],
                [("pv", .25), ("as", .25), ("pp", .25), ("ca", .25)]]),
        dict(name="storage.dao", row=2, x=0.005, w=0.99, rows=[[(None, .2), ("dacc", .2), ("dses", .2), ("ddr", .2), ("dms", .2)]]),
        dict(name="storage.entity", row=3, x=0.2, w=0.42, rows=[[("eacc", .5), ("eses", .5)]]),
    ]
    R = [
        ("api", "ct", "dep", "«import»", {}),
        ("ac", "ias", "dep", "", {}), ("ac", "ipp", "dep", "", {"elbow": 1, "ax": 0.8, "bx": 0.5}),
        ("cc", "ica", "dep", "", {}),
        ("pp", "ipp", "real", "", {}), ("as", "ias", "real", "", {}), ("ca", "ica", "real", "", {}),
        ("as", "pv", "assoc", "", {}),
        ("pp", "dacc", "dep", "", {}), ("as", "dses", "dep", "", {}), ("as", "dacc", "dep", "", {"elbow": 1, "ax": 0.2, "bx": 0.8}),
        ("ca", "ddr", "dep", "", {"elbow": 1, "ax": 0.6, "bx": 0.4}), ("ca", "dms", "dep", "", {"elbow": 1, "ax": 0.85, "bx": 0.5, "dy": 18}),
        ("dacc", "eacc", "dep", "CRUD", {}), ("dses", "eses", "dep", "CRUD", {}),
    ]
    return [("cd01", "", P, B, R)]


CD("CD-01", "사용자 및 제어권", "01", _cd01)

# ── 시퀀스 ───────────────────────────────────────────────────────
OP = ("op", "관제 운영자", "actor")
S("SD-0101", "01", [OP, ("ctl", "AuthController", "controller"), ("svc", "IAuthService", "interface"), ("acc", "UserAccountDAO", "dao"),
                    ("pv", "PasswordVerifier", "component"), ("ses", "UserSessionDAO", "dao"), ("pp", "IPermissionPolicy", "interface")], [
    call("op", "ctl", "login(LoginRequestDTO)", "SessionDTO", "로그인 요청을 받는다.", [
        call("ctl", "svc", "login(LoginRequestDTO)", "SessionDTO", "자격정보를 검증하고 세션을 발급한다.", [
            call("svc", "acc", "findByLoginId(loginId)", "UserAccount", "등록 계정을 읽는다."),
            call("svc", "pv", "verify(password, credentialHash)", "bool", "비밀번호 해시를 비교한다."),
            alt([("인증 성공", [
                    call("svc", "svc", "UserSession.issue(userId, ttl)", "UserSession", "추측 불가능한 세션을 만든다."),
                    call("svc", "ses", "insert(UserSession)", "sessionId", "세션을 저장한다."),
                    call("svc", "pp", "resolve(userId)", "PermissionSetDTO", "조회·제어 요청 권한을 계산한다."),
                 ]), ("인증 실패 (2a)", [note("svc", "ses", "ResultDTO(AUTH_FAILED) — 접속 거부 · 사유 표시")])]),
        ]),
    ]),
], entry="AuthController.login")
S("SD-0102", "01", [OP, ("ctl", "AuthController", "controller"), ("svc", "IAuthService", "interface"), ("ses", "UserSessionDAO", "dao"),
                    ("ca", "IControlAuthority", "interface")], [
    call("op", "ctl", "logout(PrincipalDTO)", "ResultDTO", "로그아웃 요청을 받는다.", [
        call("ctl", "svc", "logout(sessionId)", "ResultDTO", "세션과 웹 제어권을 종료한다.", [
            call("svc", "ses", "findById(sessionId)", "UserSession", "세션을 읽는다."),
            call("svc", "svc", "UserSession.revoke(now)", None, "세션을 무효화한다."),
            call("svc", "ses", "update(UserSession)", "bool", "폐기 시각을 저장한다."),
            call("svc", "ca", "releaseForSession(sessionId)", "ResultDTO", "웹 보유권만 해제한다."),
            note("svc", "ca", "승인된 자동 임무의 실행 권한(grant)은 유지한다 (2a)"),
        ]),
    ]),
], entry="AuthController.logout")
S("SD-0103", "01", [OP, ("ctl", "AuthController", "controller"), ("pp", "IPermissionPolicy", "interface"),
                    ("ses", "UserSessionDAO", "dao"), ("acc", "UserAccountDAO", "dao")], [
    call("op", "ctl", "checkPermission(PrincipalDTO, feature)", "AccessDecisionDTO", "기능 사용 가능 여부를 묻는다.", [
        call("ctl", "pp", "checkAccess(PrincipalDTO, feature)", "AccessDecisionDTO", "조회·편집·제어 요청 권한을 검사한다.", [
            call("pp", "ses", "findById(sessionId)", "UserSession", "세션 만료·종료 여부를 확인한다."),
            call("pp", "acc", "findById(userId)", "UserAccount", "계정 권한 집합을 읽는다."),
            opt("권한 미충족 (2a)", [note("pp", "acc", "allowed=false · reasonCode 반환 → 화면에 사유 표시")]),
        ]),
    ]),
], entry="AuthController.checkPermission")
S("SD-0104", "01", [OP, ("ctl", "ControlController", "controller"), ("ca", "IControlAuthority", "interface"),
                    ("pp", "IPermissionPolicy", "interface"), ("dr", "DroneDAO", "dao")], [
    call("op", "ctl", "acquireLease(droneId, PrincipalDTO)", "ControlStateDTO", "제어권 확보를 요청한다.", [
        call("ctl", "ca", "acquire(droneId, PrincipalDTO)", "ControlStateDTO", "웹 제어권을 원자적으로 확보한다.", [
            call("ca", "pp", "checkAccess(PrincipalDTO, DRONE_CONTROL)", "AccessDecisionDTO", "기체 제어 요청 권한을 검사한다."),
            call("ca", "dr", "findById(droneId)", "Drone", "현재 보유자·세대와 Pi 가 보고한 RC 상태를 읽는다."),
            alt([("보유자 없음 · RC 자동", [
                    call("ca", "dr", "updateControlLease(Drone, expectedRevision)", "bool", "세대가 같을 때만 보유자를 기록한다."),
                 ]), ("다른 보유자 · RC 수동 (2a·3a)", [note("ca", "dr", "거부 — 사유와 현재 보유자 반환")])]),
        ]),
    ]),
], entry="ControlController.acquireLease")
S("SD-0105", "01", [OP, ("ctl", "ControlController", "controller"), ("ca", "IControlAuthority", "interface"),
                    ("pp", "IPermissionPolicy", "interface"), ("dr", "DroneDAO", "dao")], [
    call("op", "ctl", "releaseLease(droneId, PrincipalDTO)", "ControlStateDTO", "제어권 반납을 요청한다.", [
        call("ctl", "ca", "release(droneId, PrincipalDTO)", "ControlStateDTO", "보유자와 세대를 확인한 뒤 반납한다.", [
            call("ca", "pp", "checkAccess(PrincipalDTO, DRONE_CONTROL)", "AccessDecisionDTO", "요청자와 유효 세션을 확인한다."),
            call("ca", "dr", "findById(droneId)", "Drone", "현재 보유자를 읽는다."),
            call("ca", "dr", "updateControlLease(Drone, expectedRevision)", "bool", "보유자를 비우고 세대를 올린다."),
        ]),
    ]),
], entry="ControlController.releaseLease")
S("SD-0106", "01", [OP, ("ctl", "ControlController", "controller"), ("ca", "IControlAuthority", "interface"), ("dr", "DroneDAO", "dao")], [
    call("op", "ctl", "getControlState(droneId)", "ControlStateDTO", "제어권 상태를 조회한다.", [
        call("ctl", "ca", "getControlState(droneId)", "ControlStateDTO", "보유·만료·RC 수동 상태를 함께 만든다.", [
            call("ca", "dr", "findById(droneId)", "Drone", "웹 제어권과 Pi 보고 RC 상태를 읽는다."),
            call("ca", "ca", "toStateDTO(Drone)", "ControlStateDTO", "만료된 제어권은 새 명령 차단으로 표시한다 (2a)."),
        ]),
    ]),
], entry="ControlController.getControlState")
S("SD-0107", "01", [("cs", "CommandService", "service"), ("pp", "IPermissionPolicy", "interface"), ("ca", "IControlAuthority", "interface"),
                    ("dr", "DroneDAO", "dao"), ("ms", "MissionDAO", "dao")], [
    call("cs", "pp", "authorizeCommand(PrincipalDTO, CommandContextDTO)", "AccessDecisionDTO", "사용자 요청과 자동 임무 요청을 구분해 판정한다.", [
        call("pp", "ca", "validateAuthority(PrincipalDTO, CommandContextDTO)", "AuthorityDecisionDTO", "웹 제어권 또는 임무 실행 권한을 검사한다.", [
            call("ca", "dr", "findById(droneId)", "Drone", "웹 제어권·제어 세대·RC 상태를 읽는다."),
            opt("자동 임무 요청 (SYSTEM)", [call("ca", "ms", "findById(missionId)", "Mission", "실행 승인(grant) 상태를 읽는다.")]),
        ]),
        opt("RC 수동 · 권한 없음 (2a)", [note("pp", "ms", "allowed=false — 자동 이동 요청 차단")]),
    ]),
], entry="IPermissionPolicy.authorizeCommand", 시작="명령 요청 (CommandService · MissionRunner 내부 호출)")
