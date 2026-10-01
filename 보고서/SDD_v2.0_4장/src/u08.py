"""상황지도 및 관측정보 관리 — services.mission_map"""
from mdl import K, I, DTO, ENT, DAO, unit
from seq import S, call, alt, opt, note, loop
from cdkit import CD, layered, box

unit("08", "상황지도 및 관측정보 관리", "상황지도·관측")
MM = "services.mission_map"

I("C-0807", "IMissionMapService", MM, "상황지도·관측 상태·완료도의 공개 계약 (임무·계획·비전·이력이 의존)", [
    ("getSnapshot", "missionId, version", "MissionMapDTO", "기체·실제 경로·후보·관측 상태를 같은 공간 기준으로 제공한다."),
    ("getObservationStates", "missionId, filter", "ObservationStateDTO", "미관측·유효·품질 부족·가림/미확인·운용 제한을 나눠 반환한다."),
    ("getCoverage", "missionId", "CoverageDTO", "고정 분모의 전체·관측가능 완료도를 계산한다."),
    ("getObservationSnapshot", "missionId", "ObservationSnapshotDTO", "계획 입력용 유효 관측·가림·접근 불가 상태를 고정한다."),
    ("applyFrameEvidence", "SyncedFrameDTO, TaskQualityDTO", "ObservationUpdateDTO", "실제 분석 프레임의 유효 가시 영역만 셀에 누적한다."),
    ("updatePlanningEvidence", "ObservationUpdateDTO", "PlanningEvidenceDTO", "새 관측·존재 가능성·남은 셀을 다음 계획 입력으로 갱신한다."),
], impl="C-0801")
I("C-0808", "ICameraModel", MM, "카메라 광선·지면 투영의 공개 계약 (비전 좌표·관측 영역·계획이 공유)", [
    ("ray", "Pixel, CameraPose", "Ray3D", "왜곡·전처리를 역변환한 원본 픽셀의 광선을 만든다."),
    ("projectToTerrain", "Ray3D, TerrainModel", "IntersectionResult", "평면·DEM 교차와 가시성·미확정을 구분한다."),
], impl="C-0806")

K("C-0801", "MissionMapService", "service", MM, "기체·경로·후보·관측 상태를 일관된 지도 버전으로 제공 (IMissionMapService 구현)", [
    ("footprint", "FootprintCalculator", ""), ("ledger", "ObservationLedger", ""), ("coverage", "CoverageCalculator", ""),
    ("aggregator", "ObservationGroupAggregator", ""), ("belief", "BeliefUpdater", ""),
    ("transform", "ICoordinateTransform", "표시 좌표"), ("targets", "ITargetService", "후보 위치"),
    ("planner", "IViewpointPlanner", "재계획 사건"), ("runner", "IMissionRunner", "관측 반영 통지"),
    ("events", "IEventPublisher", "지도 갱신 전파"), ("mapVersion", "int64", "동적 지도 버전"),
], [], impl="C-0807", old="C-0801")
K("C-0802", "FootprintCalculator", "component", MM, "카메라 보정·촬영 자세·지형·가시성을 반영한 관측 영역 계산", [
    ("camera", "ICameraModel", "광선"), ("spatial", "ISpatialDataService", "건물·지형 차폐 자료"), ("frameDao", "VideoFrameDAO", "촬영 범위 저장"),
], [("calculate", "SyncedFrameDTO, SceneGeometryDTO", "FootprintDTO", "시각·장착각·화각·고도·자세·지형으로 가시 관측영역을 계산한다.")], old="C-0802")
K("C-0803", "ObservationLedger", "component", MM, "유효 관측의 근거·시각·면적을 중복 없이 누적", [
    ("cellDao", "ObservationCellDAO", "셀 상태"), ("appliedEvidence", "Set<EvidenceKey>", "중복 누적 방지 키"),
], [
    ("applyEvidence", "frameId, FootprintDTO, TaskQualityDTO", "ObservationUpdateDTO", "유효 가시 영역만 프레임·셀 키로 누적하고 제외 사유를 남긴다."),
    ("applyGroupEvidence", "ObservationGroup", "ObservationUpdateDTO", "중복 없는 묶음 근거로 면적·시간을 재집계한다."),
    ("queryStates", "missionId, filter", "ObservationStateDTO", "상태 5종을 나눠 반환한다."),
    ("snapshot", "missionId", "ObservationSnapshotDTO", "유효 관측과 가림·접근 불가 상태를 고정한다."),
], old="C-0803")
K("C-0804", "CoverageCalculator", "component", MM, "고정 분모의 전체·관측가능 영역 완료도 계산", [
    ("cellDao", "ObservationCellDAO", "유효 면적"), ("revisionDao", "MissionRevisionDAO", "시작 시 고정한 분모"),
], [("calculate", "missionId", "CoverageDTO", "유효 관측 합집합을 고정 분모로 나누고 가림·제한·품질 부족 면적을 함께 보고한다.")], old="C-0804")
K("C-0805", "ObservationGroupAggregator", "component", MM, "프레임 상관·실제 유효 시간·면적을 1초 묶음별로 집계", [
    ("groupPolicy", "ObservationGroupingPolicy", "약 1초 초기 구간·시선 차이·최대 공백"), ("activeGroups", "Map", "스트림 epoch 별 열린 묶음"),
], [
    ("add", "frameId, ObservationUpdateDTO", "ObservationGroupUpdate", "같은 원관측의 중복 표현을 지우고 유효 구간을 합집합한다."),
    ("seal", "groupId", "ObservationGroup", "묶음을 확정해 누적·존재 가능성 갱신으로 넘긴다."),
], old="C-0805")
K("C-0514", "BeliefUpdater", "component", MM, "유효 관측 묶음만으로 사람 존재 가능성 갱신", [
    ("cellDao", "ObservationCellDAO", "존재 가능성 현재값"), ("likelihood", "DetectionLikelihoodModel", "검증된 양성·음성 관측 모델"),
    ("modelKind", "TargetModel", "categorical / occupancy"),
], [
    ("update", "ObservationGroup", "BeliefRevision", "모델별 Bayes 갱신과 중복 방지를 원자적으로 한다."),
    ("replayCorrection", "EvidenceRevision", "BeliefRevision", "늦게 도착한 원본의 효과를 재생·대체한다."),
], old="C-0514")
K("C-0806", "CameraModel", "component", MM, "내부 보정·기체/짐벌 외부 자세·영상 변환을 적용하는 공용 모델 (ICameraModel 구현)", [
    ("calibration", "CameraCalibration", "최종 보정 K·왜곡·외부 변환 (f 1,141 px · 수평 80.2°)"),
    ("configDao", "ConfigVersionDAO", "CAMERA·COORDINATE 설정 버전"),
], [], impl="C-0808", old="C-0806")

K("C-0809", "MissionMapController", "controller", "api", "상황지도·관측 상태 조회 요청의 입구 (REST)", [("missionMap", "IMissionMapService", "")], [
    ("getMap", "missionId, version", "MissionMapDTO", "GET /api/missions/{id}/map"),
    ("getObservations", "missionId, filter", "ObservationStateDTO", "GET /api/missions/{id}/observations"),
])

DTO("C-0810", "MissionMapDTO", "상황지도", [("missionId", "UUID", "임무"), ("version", "int64", "지도 버전"), ("drone", "VehicleStateDTO", "기체"),
    ("track", "LineString", "실제 경로"), ("candidates", "List<CandidateDTO>", "후보 (좌표 보류는 목록만)"), ("cells", "List<CellState>", "관측 상태")])
DTO("C-0811", "ObservationStateDTO", "관측 상태", [("cells", "List<CellState>", "셀별 상태·횟수·최근 시각"), ("areas", "Map<State, float>", "상태별 면적")])
DTO("C-0812", "CoverageDTO", "완료도", [("total", "float", "전체 구역 완료도"), ("observable", "float", "관측가능 구역 완료도"),
    ("unobservedArea", "float", "미관측 면적"), ("excluded", "Map<Reason, float>", "가림·제한·품질 부족"), ("evaluable", "bool", "분모 0 이면 false")])
DTO("C-0813", "ObservationSnapshotDTO", "계획 입력용 관측 스냅샷", [("missionId", "UUID", "임무"), ("version", "int64", "스냅샷 버전"),
    ("cells", "List<CellSummary>", "유효·가림·접근 불가"), ("belief", "List<CellBelief>", "존재 가능성")])
DTO("C-0814", "FootprintDTO", "촬영 범위·가시 영역", [("frameId", "UUID", "프레임"), ("footprint", "Polygon", "촬영 범위"), ("visible", "MultiPolygon", "가시 영역"),
    ("cellParts", "List<CellPart>", "셀 부분면적")])
DTO("C-0815", "ObservationUpdateDTO", "관측 반영 결과", [("missionId", "UUID", "임무"), ("frameId", "UUID", "프레임"), ("cells", "List<CellId>", "갱신 셀"),
    ("validArea", "float", "유효 면적"), ("mapVersion", "int64", "새 지도 버전")])
DTO("C-0816", "PlanningEvidenceDTO", "다음 계획 입력", [("missionId", "UUID", "임무"), ("snapshot", "ObservationSnapshotDTO", "스냅샷"), ("changes", "List<ChangeEventDTO>", "재계획 사건")])

DAO("C-0817", "ObservationCellDAO", "observation_cell", [
    ("findByMission", "missionId, gridVersion", "List<ObservationCell>", "격자 셀을 읽는다."),
    ("applyEvidence", "List<CellDelta>, rowVersions", "int", "유효 면적·횟수·시각을 조건부로 누적한다."),
    ("updateBelief", "cellId, belief, detail", "bool", "존재 가능성 현재값을 저장한다."),
])
ENT("C-0818", "ObservationCell", "observation_cell", "관측 셀 (DB-08 한 행)",
    keys=["missionId · gridVersion · cellId {PK}", "regionId {FK}", "state", "validAreaM2 · validGroupCount", "belief · rowVersion"])


def _cd():
    a = layered("cd08a", "(1/2) 계층 구조", svc_pkg=MM, ctl=["C-0809"], dto=["C-0810", "C-0811", "C-0812", "C-0813", "C-0814", "C-0815", "C-0816"],
                pairs=[("C-0807", "C-0801"), ("C-0808", "C-0806")],
                ext=[("C-0306", "services.spatial"), ("C-0722", "services.vision"), ("C-0516", "services.mission.local"),
                     ("C-0406", "services.mission"), ("C-1106", "services.alert")],
                daos=[("C-0817", False), ("C-0622", True), ("C-0313", True)], ents=["C-0818"], api_w=0.33, dto_cols=4, ext_w=0.26)
    B = {"ms": box("C-0801", ref=True)}
    for c in ["C-0802", "C-0803", "C-0804", "C-0805", "C-0514"]: B[c] = box(c)
    B["cam"] = box("C-0808", ref=True); B["sp"] = box("C-0305", ref=True); B["lk"] = box("C-0511", ref=True)
    B["cd"] = box("C-0817", ref=True); B["vf"] = box("C-0622", ref=True); B["rv"] = box("C-0415", ref=True)
    P = [dict(name=MM, row=0, x=0.3, w=0.4, rows=[[("ms", 1.0)]]),
         dict(name=MM, row=1, x=0.005, w=0.99, rows=[[("C-0802", .333), ("C-0805", .333), ("C-0803", .334)], [(None, .333), ("C-0514", .333), ("C-0804", .334)]]),
         dict(name="다른 패키지", row=2, x=0.005, w=0.49, rows=[[("cam", .333), ("sp", .333), ("lk", .334)]], tabs=["services.mission_map · services.spatial · policies"]),
         dict(name="storage.dao", row=2, x=0.505, w=0.49, rows=[[("cd", .333), ("vf", .333), ("rv", .334)]])]
    R = [("ms", c, "assoc", "", {"elbow": 1}) for c in ["C-0802", "C-0805", "C-0803"]]
    R += [("C-0805", "C-0514", "dep", "seal → update", {}), ("C-0803", "C-0804", "dep", "", {}),
          ("C-0802", "cam", "dep", "", {"elbow": 1}), ("C-0802", "sp", "dep", "", {"elbow": 1}), ("C-0514", "lk", "dep", "", {"elbow": 1}),
          ("C-0803", "cd", "dep", "", {"elbow": 1}), ("C-0514", "cd", "dep", "", {"elbow": 1}), ("C-0802", "vf", "dep", "", {"elbow": 1, "ax": 0.8}),
          ("C-0804", "rv", "dep", "", {"elbow": 1})]
    return [a, ("cd08b", "(2/2) 관측 누적 구성요소", P, B, R)]


CD("CD-08", "상황지도·관측", "08", _cd)

OP = ("op", "관제 운영자", "actor")
MS = ("mm", "IMissionMapService", "interface")
S("SD-0801", "08", [OP, ("ctl", "MissionMapController", "controller"), MS, ("ol", "ObservationLedger", "component"),
                    ("ts", "ITargetService", "interface"), ("ct", "ICoordinateTransform", "interface")], [
    call("op", "ctl", "getMap(missionId, version)", "MissionMapDTO", "상황지도를 조회한다.", [
        call("ctl", "mm", "getSnapshot(missionId, version)", "MissionMapDTO", "같은 공간 기준으로 묶는다.", [
            call("mm", "ol", "snapshot(missionId)", "ObservationSnapshotDTO", "격자 관측 상태와 마지막 유효 시각을 읽는다."),
            call("mm", "ts", "queryCandidates(missionId, filter)", "List<CandidateDTO>", "후보와 좌표 상태를 읽는다."),
            call("mm", "ct", "transform(GeometryDTO, contextId)", "GeometryDTO", "표시 좌표계로 변환한다."),
        ]),
    ]),
    note("op", "ct", "좌표 보류 후보는 목록·스냅샷으로만 표시 — 확정 위치 마커로 표시하지 않음 (3a)"),
], entry="MissionMapController.getMap")
S("SD-0802", "08", [("ms", "MissionMapService", "service"), ("fc", "FootprintCalculator", "component"), ("sp", "ISpatialDataService", "interface"),
                    ("cm", "ICameraModel", "interface"), ("vf", "VideoFrameDAO", "dao")], [
    call("ms", "fc", "calculate(SyncedFrameDTO, SceneGeometryDTO)", "FootprintDTO", "영상에 대응하는 실제 촬영 영역을 계산한다.", [
        call("fc", "sp", "getSceneGeometry(bounds, version)", "SceneGeometryDTO", "등록 GIS/DEM 과 누락·해상도·높이 기준을 확인한다."),
        loop("화면 경계·셀 표본마다", [call("fc", "cm", "ray(pixel, pose)", "Ray3D", "촬영 시각 실제 위치·자세·보정으로 광선을 만든다."),
                                    call("fc", "cm", "projectToTerrain(Ray3D, terrain)", "IntersectionResult", "건물·지형 차폐를 뺀 교차점을 얻는다.")]),
        call("fc", "vf", "updateCoverage(frameId, footprint, coverage)", "bool", "촬영 범위와 가시 영역을 저장한다."),
    ]),
], entry="FootprintCalculator.calculate", 시작="영상·비행정보가 연결된 프레임 입력 (MissionMapService)")
S("SD-0803", "08", [("fi", "FrameIngestor", "component"), MS, ("ol", "ObservationLedger", "component"), ("ga", "ObservationGroupAggregator", "component"),
                    ("bu", "BeliefUpdater", "component"), ("cd", "ObservationCellDAO", "dao"), ("ev", "IEventPublisher", "interface")], [
    call("fi", "mm", "applyFrameEvidence(SyncedFrameDTO, TaskQualityDTO)", "ObservationUpdateDTO", "추론 완료·업무 품질 통과 프레임만 받는다.", [
        call("mm", "ol", "applyEvidence(frameId, FootprintDTO, TaskQualityDTO)", "ObservationUpdateDTO", "유효 가시 영역만 셀별로 누적한다.", [
            call("ol", "cd", "applyEvidence(List<CellDelta>, rowVersions)", "count", "면적·횟수·시각을 원자적으로 누적한다.")]),
        call("mm", "ga", "add(frameId, ObservationUpdateDTO)", "ObservationGroupUpdate", "같은 원관측·겹친 시간·면적 중복을 지운다."),
        opt("1초 묶음 확정", [
            call("mm", "bu", "update(ObservationGroup)", "BeliefRevision", "관측 조건별 탐지 모델로 존재 가능성을 갱신한다.", [
                call("bu", "cd", "updateBelief(cellId, belief, detail)", "bool", "현재값과 근거를 저장한다.")])]),
        call("mm", "ev", "deliver(MissionEventDTO)", "DeliveryResult", "저장 후 새 관측 버전을 전파한다."),
    ]),
], entry="IMissionMapService.applyFrameEvidence", 시작="프레임 관측영역·품질 평가 완료 (FrameIngestor)")
S("SD-0804", "08", [OP, ("ctl", "MissionMapController", "controller"), MS, ("ol", "ObservationLedger", "component"), ("cc", "CoverageCalculator", "component")], [
    call("op", "ctl", "getObservations(missionId, filter)", "ObservationStateDTO", "관측 상태를 조회한다.", [
        call("ctl", "mm", "getObservationStates(missionId, filter)", "ObservationStateDTO", "영역별 상태·횟수·최근 시각을 만든다.", [
            call("mm", "ol", "queryStates(missionId, filter)", "ObservationStateDTO", "미관측·유효·품질 부족·가림/미확인·운용 제한을 나눈다."),
            call("mm", "cc", "calculate(missionId)", "CoverageDTO", "전체·관측가능 분모와 상태별 면적을 함께 준다."),
        ]),
    ]),
    note("op", "cc", "가림·품질 부족 영역은 미확인 상태로 유지 (2a)"),
], entry="MissionMapController.getObservations")
S("SD-0805", "08", [("cl", "MissionService · HistoryService", "service"), MS, ("cc", "CoverageCalculator", "component"),
                    ("rv", "MissionRevisionDAO", "dao"), ("cd", "ObservationCellDAO", "dao")], [
    call("cl", "mm", "getCoverage(missionId)", "CoverageDTO", "정찰 완료도를 요청한다.", [
        call("mm", "cc", "calculate(missionId)", "CoverageDTO", "고정 분모와 실제 유효 합집합을 비교한다.", [
            call("cc", "rv", "findActive(missionId)", "MissionRevision", "시작 시 고정한 전체·관측가능 분모를 읽는다."),
            call("cc", "cd", "findByMission(missionId, gridVersion)", "List<ObservationCell>", "중복 없는 유효 관측·품질 부족·가림·제한 면적을 읽는다."),
            opt("분모 0", [note("cc", "cd", "평가불가 표시 · 미관측 면적은 별도 보고 (2a)")]),
        ]),
    ]),
], entry="IMissionMapService.getCoverage", 시작="유효 관측 누적 또는 완료도 조회 (MissionService · HistoryService)")
S("SD-0806", "08", [("ms", "MissionMapService", "service"), ("ol", "ObservationLedger", "component"), ("vp", "IViewpointPlanner", "interface")], [
    call("ms", "ms", "updatePlanningEvidence(ObservationUpdateDTO)", "PlanningEvidenceDTO", "관측 결과를 다음 탐색 입력으로 만든다.", [
        call("ms", "ol", "snapshot(missionId)", "ObservationSnapshotDTO", "현재 유효 관측과 미확인 사유를 고정한다."),
    ]),
    opt("전역 재계획 사건", [call("ms", "vp", "replan(missionId, ChangeEventDTO)", "SearchPlanDTO?", "잔여 영역 순서를 다시 계산한다 (SD-0505).")]),
    note("ms", "vp", "가려진 영역은 기존 사람 존재 가능성을 유지 (2a)"),
], entry="IMissionMapService.updatePlanningEvidence", 시작="상황지도·관측정보 갱신 이벤트 (MissionMapService)")
S("SD-X07", "08", [("fi", "FrameIngestor", "component"), MS, ("ga", "ObservationGroupAggregator", "component"), ("bu", "BeliefUpdater", "component"),
                   ("vp", "IViewpointPlanner", "interface")], [
    call("fi", "mm", "applyFrameEvidence(SyncedFrameDTO, TaskQualityDTO)", "ObservationUpdateDTO", "유효 분석 결과만 반영한다.", [
        call("mm", "ga", "add(frameId, ObservationUpdateDTO)", "ObservationGroupUpdate", "같은 관측 묶음은 한 번만 센다."),
        alt([("묶음 확정 · 유효", [call("mm", "bu", "update(ObservationGroup)", "BeliefRevision", "존재 가능성을 갱신한다.")]),
             ("관측 불가 · 미분석 · 품질 보류", [note("mm", "bu", "기존 값 유지 — 관측 완료 ≠ 사람 부재")])]),
    ]),
    call("vp", "vp", "LocalIPPPlanner.plan(가상 사본)", "ObservationPlan", "계획의 예상 이득은 실제 DB 와 분리된 사본에서 계산한다."),
], title="관측 묶음·존재 가능성·완료도 갱신", uc="UC-0802~0806, UC-0502~0505",
   개요="관측 완료와 사람 부재, 실제 존재 가능성과 계획 시뮬레이션을 분리한다.", 시작="분석 완료 프레임 (FrameIngestor)",
   선행="프레임 분석과 업무 품질 판정이 끝났다.", 사후="셀 관측·존재 가능성이 한 번씩만 반영된다.",
   예외="관측 불가·미분석·품질 보류 시 기존 값을 유지한다.", 경계="내부 이벤트 처리 | 결과 화면: UI-04 | 연결 시험: AT-F07")
