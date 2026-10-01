"""지도 및 공간정보 관리 — services.spatial · web"""
from mdl import K, I, DTO, ENT, DAO, unit
from seq import S, call, alt, opt, note
from cdkit import CD, layered, box

unit("03", "지도 및 공간정보 관리", "지도·공간 기준")

I("C-0305", "ISpatialDataService", "services.spatial", "기본지도·건물 높이·차폐 자료 조회의 공개 계약", [
    ("getMap", "bounds, layers", "MapSnapshotDTO", "기본지도와 건물 윤곽·높이를 출처·캐시 상태와 함께 제공한다."),
    ("getLayerStatus", "layerIds", "LayerStatusDTO", "레이어별 사용 가능 여부와 실패 사유를 조회한다."),
    ("getSceneGeometry", "bounds, version", "SceneGeometryDTO", "건물·지형 차폐 판정 자료를 가져온다."),
], impl="C-0301")
I("C-0306", "ICoordinateTransform", "services.spatial", "좌표 기준 변환의 공개 계약 (임무·비전·관측·이력이 의존)", [
    ("transform", "GeometryDTO, contextId", "GeometryDTO", "원본 → 내부 탐색 좌표 또는 WGS84 로 원점·축·단위·고도 기준을 고정해 변환한다."),
    ("validatePoseReference", "Pose, contextId", "ReferenceDecision", "자세·고도의 좌표 기준(ENU/NED·방위·고도)을 검사한다."),
], impl="C-0302")

K("C-0301", "SpatialDataService", "service", "services.spatial", "외부 기본지도·건물 높이를 조회하고 출처와 캐시 상태를 보존 (ISpatialDataService 구현)", [
    ("provider", "VWorldProvider", "외부 지도 어댑터"),
    ("configDao", "ConfigVersionDAO", "공간자료 스냅샷 버전"),
    ("cacheVersion", "UUID", "불변 공간자료 스냅샷 버전"),
], [("-fallbackToCache", "bounds, reason", "MapSnapshotDTO", "외부 조회 실패 시 저장 자료와 제약을 반환한다.")], impl="C-0305", old="C-0301")
K("C-0302", "CoordinateTransform", "service", "services.spatial", "WGS84·내부 탐색·시뮬레이션 좌표의 기준을 명시하여 변환 (ICoordinateTransform 구현)", [
    ("configDao", "ConfigVersionDAO", "공간 기준(SPATIAL·COORDINATE) 설정 버전"),
    ("contexts", "Map<UUID, SpatialContext>", "읽은 공간 기준 캐시"),
], [("-loadContext", "contextId", "SpatialContext", "설정 버전에서 공간 기준을 만든다."),
    ("-project", "geometry, SpatialContext", "GeometryDTO", "원점·축·단위·고도 기준을 고정해 좌표를 바꾼다.")], impl="C-0306", old="C-0302")
K("C-0304", "SpatialContext", "value", "services.spatial", "모든 지도·계획·명령이 참조하는 공간 기준의 불변 버전 (config_version payload 에서 생성)", [
    ("contextId", "UUID", "공간 기준 버전"), ("origin", "GeoPoint", "내부 좌표 원점 WGS84"),
    ("axisAndUnit", "AxisUnitSpec", "축 방향과 선형·각도 단위"), ("altitudeReference", "AltitudeReference", "지면·원점·해발 높이 기준"),
], [
    ("validate", "", "ReferenceDecision", "원점·단위·고도 기준이 확정됐는지 확인한다."),
    ("describe", "", "ReferenceSpec", "원점·축·단위·고도 기준·변환 버전을 반환한다."),
], old="C-0304")
K("C-0308", "VWorldProvider", "component", "services.spatial", "VWorld 지도·건물 높이 API 어댑터 (외부 서비스 경계)", [
    ("apiKeyRef", "SecretRef", "배포 설정의 키 참조"), ("timeout", "Duration", "외부 조회 제한 시간"),
], [
    ("fetchTiles", "bounds, layers", "MapTiles", "기본지도 타일을 가져온다."),
    ("fetchBuildings", "bounds", "BuildingSet", "건물 윤곽·높이를 가져온다."),
])
K("C-0303", "MapViewModel", "viewmodel", "web", "관제 웹의 지도 위치·축척·표시 레이어 상태 (클라이언트)", [
    ("viewport", "Viewport", "화면 중심·축척·범위"), ("visibleLayers", "Set<LayerId>", "보이기 선택 집합"),
], [
    ("setViewport", "viewport", "MapViewState", "이동·축척 변경 후 같은 기체·경로·후보를 새 범위에 표시한다."),
    ("setLayers", "layerIds", "MapViewState", "레이어 표시만 바꾸고 임무 구역·관측 기록은 유지한다."),
], old="C-0303")
K("C-0307", "MapController", "controller", "api", "지도·레이어 조회 요청의 입구 (REST)", [
    ("spatialData", "ISpatialDataService", ""),
], [
    ("getFeatures", "bounds, layers", "MapSnapshotDTO", "GET /api/map/features"),
    ("getLayerStatus", "layerIds", "LayerStatusDTO", "GET /api/map/layers"),
])

DTO("C-0309", "MapSnapshotDTO", "지도 스냅샷", [("bounds", "BBox", "범위"), ("layers", "List<Layer>", "기본지도·건물 윤곽·높이"),
    ("source", "SourceInfo", "출처·캐시 상태"), ("contextId", "UUID", "공간 기준"), ("limits", "List<ReasonCode>", "제약")])
DTO("C-0310", "LayerStatusDTO", "레이어 사용 가능 여부", [("layers", "List<LayerState>", "레이어 ID·가능 여부·실패 사유")])
DTO("C-0311", "GeometryDTO", "좌표 기준이 명시된 형상", [("geometry", "GeoJSON", "형상"), ("contextId", "UUID", "공간 기준"),
    ("crs", "WGS84 | LOCAL_ENU", "좌표계"), ("altitudeReference", "AltitudeReference", "고도 기준")])
DTO("C-0312", "SceneGeometryDTO", "차폐 판정 자료", [("buildings", "List<Prism>", "건물 윤곽·높이"), ("terrain", "DemRef", "지형"),
    ("resolution", "float", "해상도"), ("missing", "List<BBox>", "누락 범위")])

DAO("C-0313", "ConfigVersionDAO", "config_version", [
    ("findById", "configId", "ConfigVersion?", "설정 버전을 읽는다."),
    ("findActive", "kind", "ConfigVersion?", "종류별 승인된 활성 버전을 읽는다."),
    ("insert", "ConfigVersion", "UUID", "새 설정 버전을 저장한다."),
    ("setStatus", "configId, status", "bool", "검증·승인·폐기 상태를 바꾼다."),
])
ENT("C-0314", "ConfigVersion", "config_version", "공간 기준·카메라 보정·좌표·모델·정책의 불변 버전 (DB-06 한 행)",
    keys=["configId {PK}", "kind", "version {UQ}", "payload : JSON", "contentHash", "status"])


def _cd():
    P = layered("cd03", "", svc_pkg="services.spatial", ctl=["C-0307"], dto=["C-0309", "C-0310", "C-0311", "C-0312"],
                pairs=[("C-0305", "C-0301"), ("C-0306", "C-0302")], comps=["C-0308", "C-0304"],
                daos=[("C-0313", False)], ents=["C-0314"], api_w=0.45, dto_cols=2,
                extra=[("k0302", "k0304", "assoc", "", {"elbow": 1})])
    key, sub, pk, B, R = P
    B["k0303"] = box("C-0303")
    pk.append(dict(key="web", name="web", row=2, x=0.505, w=0.49, rows=[[("k0303", 1.0)]]))
    for p in pk:
        if p["name"] == "storage.dao": p["w"] = 0.49
        if p["name"] == "storage.entity": p["w"] = 0.49
    pass
    return [P]


CD("CD-03", "지도·공간 기준", "03", _cd)

OP = ("op", "관제 운영자", "actor")
S("SD-0301", "03", [OP, ("ctl", "MapController", "controller"), ("svc", "ISpatialDataService", "interface"), ("vw", "VWorldProvider", "component"),
                    ("cfg", "ConfigVersionDAO", "dao")], [
    call("op", "ctl", "getFeatures(bounds, layers)", "MapSnapshotDTO", "지도 자료를 요청한다.", [
        call("ctl", "svc", "getMap(bounds, layers)", "MapSnapshotDTO", "기본지도와 건물 정보를 출처와 함께 만든다.", [
            call("svc", "cfg", "findActive(SPATIAL)", "ConfigVersion", "공간 기준 버전을 읽는다."),
            call("svc", "svc", "SpatialContext.validate()", "ReferenceDecision", "좌표·고도 기준을 확인한다."),
            alt([("외부 조회 정상", [call("svc", "vw", "fetchTiles(bounds, layers)", "MapTiles", "기본지도를 가져온다."),
                                    call("svc", "vw", "fetchBuildings(bounds)", "BuildingSet", "건물 윤곽·높이를 가져온다.")]),
                 ("외부 조회 실패 (2a)", [call("svc", "svc", "fallbackToCache(bounds, reason)", "MapSnapshotDTO", "저장 자료와 제약을 표시한다.")])]),
        ]),
    ]),
], entry="MapController.getFeatures")
S("SD-0302", "03", [OP, ("vm", "MapViewModel", "viewmodel"), ("ctl", "MapController", "controller"), ("svc", "ISpatialDataService", "interface")], [
    call("op", "vm", "setViewport(viewport)", "MapViewState", "지도를 이동·확대·축소한다.", [
        call("vm", "ctl", "getFeatures(bounds, layers)", "MapSnapshotDTO", "필요한 범위의 자료만 요청한다.", [
            call("ctl", "svc", "getMap(bounds, layers)", "MapSnapshotDTO", "화면 범위의 자료를 조회한다."),
        ]),
        opt("자료 조회 불가 구간 (2a)", [note("vm", "svc", "조회 불가 상태로 구분 표시 — 기체·경로·후보 표시는 유지")]),
    ]),
], entry="MapViewModel.setViewport")
S("SD-0303", "03", [OP, ("vm", "MapViewModel", "viewmodel"), ("ctl", "MapController", "controller"), ("svc", "ISpatialDataService", "interface")], [
    call("op", "vm", "setLayers(layerIds)", "MapViewState", "표시할 레이어를 고른다.", [
        call("vm", "ctl", "getLayerStatus(layerIds)", "LayerStatusDTO", "레이어 사용 가능 여부를 묻는다.", [
            call("ctl", "svc", "getLayerStatus(layerIds)", "LayerStatusDTO", "실패 사유를 함께 조회한다."),
        ]),
        note("vm", "svc", "선택한 레이어만 표시 · 임무 구역과 관측 기록은 유지"),
    ]),
], entry="MapViewModel.setLayers")
S("SD-0304", "03", [("ms", "MissionService", "service"), ("ct", "ICoordinateTransform", "interface"), ("cfg", "ConfigVersionDAO", "dao")], [
    call("ms", "ct", "transform(GeometryDTO, contextId)", "GeometryDTO", "원본 형상을 탐색·표시용 기준으로 변환한다.", [
        call("ct", "cfg", "findById(contextId)", "ConfigVersion", "공간 기준 설정을 읽는다."),
        call("ct", "ct", "SpatialContext.validate()", "ReferenceDecision", "원점·단위·고도 기준을 확인한다."),
        alt([("기준 확정", [call("ct", "ct", "project(geometry, context)", "GeometryDTO", "원점·축·단위·고도 기준을 고정해 변환한다.")]),
             ("기준 불명확 (1a)", [note("ct", "cfg", "변환 보류 — ResultDTO(PENDING, 사유)")])]),
    ]),
], entry="ICoordinateTransform.transform", 시작="지도자료 수신 또는 표시·계획 변환 요청 (MissionService 등 내부 호출)")
