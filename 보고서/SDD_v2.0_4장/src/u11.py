"""알림 관리 — services.alert · api(EventPublisher)"""
from mdl import K, I, DTO, ENT, DAO, unit
from seq import S, call, alt, opt, note
from cdkit import CD, layered, box

unit("11", "알림 관리", "알림·이벤트")
AL = "services.alert"

I("C-1105", "IAlertService", AL, "알림 생성·조회·확인의 공개 계약 (비전·안전·명령이 의존)", [
    ("publish", "AlertEventDTO", "AlertDTO", "초기 후보·안전·명령 결과의 중복을 억제하고 근거를 저장한 뒤 전달한다."),
    ("queryAlerts", "missionId, filter, PrincipalDTO", "List<AlertDTO>", "시각·종류·중요도·확인 여부별 목록을 반환한다."),
    ("getAlert", "alertId", "AlertDetailDTO", "발생 시각·원인·기체·임무·후보/명령·영상 관계를 반환한다."),
    ("acknowledge", "alertId, PrincipalDTO", "AlertDTO", "확인자와 확인 시각을 기록한다 (사건 해결·후보 판단과 별도)."),
], impl="C-1101")
I("C-1106", "IEventPublisher", "api", "서버 저장 이벤트의 웹 전달 계약 (WebSocket 경계를 감춘다)", [
    ("deliver", "MissionEventDTO", "DeliveryResult", "저장된 이벤트만 WebSocket 으로 보낸다."),
    ("replayAfter", "cursor, snapshotVersion", "EventBatchDTO", "수신 커서 이후 보관 이벤트만 재전송한다."),
], impl="C-1103")

K("C-1101", "AlertService", "service", AL, "탐지·안전·명령 실패 알림 생성과 운영자 확인 처리 (IAlertService 구현)", [
    ("alertDao", "AlertDAO", "알림 저장"), ("publisher", "IEventPublisher", "웹 전달"), ("history", "IHistoryService", "원인·후속 이력"),
    ("permission", "IPermissionPolicy", "조회 권한"), ("dedupWindow", "Duration", "같은 이벤트 중복 기준 (시험 후 확정)"),
    ("priorityPolicy", "AlertPriorityPolicy", "안전·탐지·일반 우선순위"),
], [("-toDTO", "Alert", "AlertDTO", "엔티티를 DTO 로 바꾼다.")], impl="C-1105", old="C-1101")
K("C-1103", "EventPublisher", "boundary", "api", "서버에 저장된 이벤트를 웹에 전달하고 누락 재전송 지원 — WebSocket 경계 (IEventPublisher 구현)", [
    ("eventDao", "MissionEventDAO", "재동기화 커서"), ("subscriberCursor", "Map<ConnectionId, int64>", "연결별 수신 위치"),
], [], impl="C-1106", old="C-1103")
K("C-1107", "AlertController", "controller", "api", "알림 조회·확인 요청의 입구 (REST)", [("alertService", "IAlertService", "")], [
    ("getAlerts", "missionId, filter, PrincipalDTO", "List<AlertDTO>", "GET /api/missions/{id}/alerts"),
    ("getAlert", "alertId", "AlertDetailDTO", "GET /api/alerts/{id}"),
    ("acknowledge", "alertId, PrincipalDTO", "AlertDTO", "POST /api/alerts/{id}/acknowledgements"),
])

DTO("C-1108", "AlertEventDTO", "알림 원인 사건", [("type", "NEW_CANDIDATE | COLOR_MATCH | FALLEN_SUSPECT | ID_LOST | FAULT | COMMAND_FAILED", "종류"),
    ("missionId", "UUID?", "임무"), ("sourceEventId", "UUID", "원인 사건"), ("evidenceRefs", "JSON", "후보·스냅샷·명령 참조"), ("dedupKey", "String", "억제 키")])
DTO("C-1109", "AlertDTO", "알림 요약", [("alertId", "UUID", "알림"), ("type", "AlertType", "종류"), ("priority", "int", "우선도"), ("raisedAt", "Time", "발생"),
    ("acknowledged", "bool", "확인 여부")])
DTO("C-1110", "AlertDetailDTO", "알림 상세", [("alert", "AlertDTO", "요약"), ("cause", "MissionEventDTO", "원인"), ("related", "List<MissionEventDTO>", "후속 조치"),
    ("snapshot", "MediaViewDTO?", "스냅샷"), ("geoReason", "ReasonCode?", "위치 보류 사유")])
DTO("C-1111", "EventBatchDTO", "재전송 이벤트 묶음", [("fromCursor", "int64", "시작 커서"), ("events", "List<MissionEventDTO>", "이벤트"), ("snapshotVersion", "int64", "스냅샷")])

DAO("C-1102", "AlertDAO", "alert", [
    ("upsertByDedupKey", "Alert", "Alert", "같은 원인 사건을 같은 알림으로 잇는다."),
    ("findByMission", "missionId, filter", "List<Alert>", "목록을 읽는다."),
    ("findById", "alertId", "Alert?", "알림을 읽는다."),
    ("update", "Alert", "bool", "확인 기록을 저장한다."),
], resp="DB-17 alert 테이블의 저장·조회 (구 AlertRepository)")
ENT("C-1104", "Alert", "alert", "발생 사건과 운영자 확인 여부를 각각 표현 (DB-17 한 행)", ops=[
    ("acknowledge", "userId, time", "void", "읽음·확인 상태만 바꾼다."),
], keys=["alertId {PK}", "missionId · sourceEventId {FK}", "dedupKey {UQ}", "type · priority", "ackUserId · acknowledgedAt"])


def _cd():
    return [layered("cd11", "", svc_pkg=AL, ctl=["C-1107"], dto=["C-1108", "C-1109", "C-1110", "C-1111"],
                    pairs=[("C-1105", "C-1101")],
                    ext=[("C-1106", "api"), ("C-1207", "services.history"), ("C-0108", "services.auth")],
                    daos=[("C-1102", False), ("C-1216", True)], ents=["C-1104"], api_w=0.4, ext_w=0.3,
                    extra=[])]


def _cd_full():
    a = _cd()[0]
    key, sub, P, B, R = a
    B["ep"] = box("C-1103")
    for p in P:
        if p["name"] == "storage.entity": p["w"] = 0.49; p["rows"] = [[(k, 0.6) for k, _ in p["rows"][0]]]
    P.append(dict(name="api", row=3, x=0.505, w=0.49, rows=[[("ep", 1.0)]]))
    R.append(("ep", "k1106", "real", "", {"elbow": 1}))
    return [a]


CD("CD-11", "알림·이벤트", "11", _cd_full)

OP = ("op", "관제 운영자", "actor")
AI = ("al", "IAlertService", "interface")
S("SD-1101", "11", [("src", "TargetService · SafetySupervisor", "service"), AI, ("ad", "AlertDAO", "dao"), ("ep", "IEventPublisher", "interface"),
                    ("web", "관제 웹", "external")], [
    call("src", "al", "publish(AlertEventDTO)", "AlertDTO", "새 후보와 운용 이상을 알린다.", [
        call("al", "ad", "upsertByDedupKey(Alert)", "Alert", "동일 원인 사건을 같은 알림으로 잇는다."),
        call("al", "ep", "deliver(MissionEventDTO)", "DeliveryResult", "저장된 알림만 연결된 관제 화면에 보낸다.", [
            call("ep", "web", "WebSocket push", None, "화면에 알림을 표시한다.")]),
    ]),
    note("src", "web", "좌표 보류 후보를 포함해 첫 탐지부터 알린다 (3a)"),
], entry="IAlertService.publish", 시작="탐지·상태·명령 처리 이벤트 (TargetService · SafetySupervisor · CommandService)")
S("SD-1102", "11", [OP, ("ctl", "AlertController", "controller"), AI, ("pp", "IPermissionPolicy", "interface"), ("ad", "AlertDAO", "dao")], [
    call("op", "ctl", "getAlerts(missionId, filter, PrincipalDTO)", "List<AlertDTO>", "알림 목록을 조회한다.", [
        call("ctl", "al", "queryAlerts(missionId, filter, PrincipalDTO)", "List<AlertDTO>", "확인 여부별 목록을 만든다.", [
            call("al", "pp", "checkAccess(PrincipalDTO, MISSION_VIEW)", "AccessDecisionDTO", "임무 결과 조회 권한을 확인한다."),
            call("al", "ad", "findByMission(missionId, filter)", "List<Alert>", "시각·종류·중요도별로 읽는다."),
        ]),
    ]),
    note("op", "ad", "알림 0건이면 빈 목록 (2a)"),
], entry="AlertController.getAlerts")
S("SD-1103", "11", [OP, ("ctl", "AlertController", "controller"), AI, ("ad", "AlertDAO", "dao"), ("hs", "IHistoryService", "interface")], [
    call("op", "ctl", "getAlert(alertId)", "AlertDetailDTO", "알림 상세를 조회한다.", [
        call("ctl", "al", "getAlert(alertId)", "AlertDetailDTO", "원인과 후속 조치를 묶는다.", [
            call("al", "ad", "findById(alertId)", "Alert", "발생 시각·원인·기체·임무 참조를 읽는다."),
            call("al", "hs", "queryRelated(eventId)", "List<MissionEventDTO>", "원인 사건과 후속 조치 이력을 조회한다."),
        ]),
    ]),
    note("op", "hs", "위치 보류·자료 미등록 시 사유 표시 (3a)"),
], entry="AlertController.getAlert")
S("SD-1104", "11", [OP, ("ctl", "AlertController", "controller"), AI, ("ad", "AlertDAO", "dao"), ("hs", "IHistoryService", "interface")], [
    call("op", "ctl", "acknowledge(alertId, PrincipalDTO)", "AlertDTO", "알림을 확인 처리한다.", [
        call("ctl", "al", "acknowledge(alertId, PrincipalDTO)", "AlertDTO", "확인자·시각을 기록한다.", [
            call("al", "ad", "findById(alertId)", "Alert", "알림을 읽는다."),
            call("al", "al", "Alert.acknowledge(userId, time)", None, "읽음·확인 상태만 바꾼다."),
            call("al", "ad", "update(Alert)", "bool", "확인 기록을 보존한다."),
            call("al", "hs", "append(MissionEventDTO)", "EventRefDTO", "확인 조치를 운영자 이력과 잇는다."),
        ]),
    ]),
    note("op", "hs", "알림 확인·닫기와 사건 해결·후보 처리 상태는 각각 기록 (2a)"),
], entry="AlertController.acknowledge")
