"""탐색 계획 및 비행 경로 관리 — services.mission.global · services.mission.local"""
from mdl import K, I, DTO, ENT, DAO, unit
from seq import S, call, alt, opt, note, loop
from cdkit import CD, layered, box

unit("05", "탐색 계획 및 비행 경로 관리", "탐색 계획")
G, L = "services.mission.global", "services.mission.local"

I("C-0516", "IViewpointPlanner", L, "탐색 계획 패키지의 단일 공개 계약 (임무·비전·관측 패키지가 의존 · Facade)", [
    ("buildPlan", "PlanningInputDTO", "SearchPlanDTO", "좌표·관측·안전 조건을 확인해 실행 계획을 만들고 저장한다."),
    ("generateRegions", "missionId", "RegionSetDTO", "임무 경계·공간자료·관측 이력으로 수색 후보영역을 만든다."),
    ("prioritize", "missionId", "GlobalPlanDTO", "전역 방문 순서와 복귀 가능한 prefix 를 계산한다."),
    ("selectNext", "missionId, VehicleStateDTO", "ViewpointPlanDTO", "다음 영역 작업과 구역 관측 계획의 첫 실행 구간을 정한다."),
    ("requestPrecision", "PrecisionRequestDTO", "ViewpointPlanDTO?", "재관측 요청을 대기열에 넣고 정밀 관측 경로를 만든다."),
    ("buildTrackingGoal", "TrackingTargetDTO", "ViewpointPlanDTO?", "운용 범위 안의 개별 추적 관측 목표를 만든다."),
    ("replan", "missionId, ChangeEventDTO", "SearchPlanDTO?", "변경 사건을 구역 재계획·전역 재순서로 분류해 다시 계산한다."),
    ("planRegion", "RegionTaskDTO", "ObservationPlanDTO", "영역 요청을 구역 IPP 또는 기준선에 위임하고 첫 실행 구간만 반환한다."),
    ("applyRegionResult", "RegionResultDTO", "TaskUpdateDTO", "실제 관측 증거와 실행 종료를 전역 작업 조정자에 반영한다."),
    ("queryHistory", "missionId, filter", "List<PlanHistoryDTO>", "후보·평가 근거·버전·실행 명령의 연결을 조회한다."),
], impl="C-0503")
K("C-0503", "ViewpointPlanner", "service", L, "전역·구역 계획 구성요소를 조정하는 탐색 계획 진입점 (IViewpointPlanner 구현)", [
    ("regionPlanner", "CandidateRegionPlanner", ""), ("scorer", "PriorityScorer", ""), ("globalPlanner", "GlobalVisitPlanner", ""),
    ("taskCoordinator", "RegionTaskCoordinator", ""), ("localPlanner", "LocalIPPPlanner", ""), ("baseline", "BaselineCoveragePlanner", ""),
    ("budget", "ReturnBudgetChecker", ""), ("planDao", "SearchPlanDAO", "계획 버전"), ("regionDao", "SearchRegionDAO", "수색 영역"),
    ("reobserveDao", "ReobservationRequestDAO", "정밀 관측 대기열"), ("missionMap", "IMissionMapService", "관측 스냅샷"),
    ("targets", "ITargetService", "후보 근거"), ("commands", "ICommandService", "계획별 실행 결과"),
    ("safetyMargin", "Distance", "지도 불확실성·장애물 여유 (현장 시험 후 확정)"),
], [("-toDTO", "SearchPlan", "SearchPlanDTO", "계획 엔티티를 DTO 로 바꾼다.")], impl="C-0516", old="C-0503")

# 전역 계획
K("C-0501", "CandidateRegionPlanner", "component", G, "시설·통로·건물 주변·미분류 표면을 전역 방문 영역으로 분할", [
    ("ruleVersion", "UUID", "후보 생성 규칙 버전"),
], [("generate", "PlanningInput, ObservationSnapshotDTO", "List<SearchRegion>", "지도·임무 경계로 영역을 만들고 셀 교차 할당의 누락·중복을 검사한다.")], old="C-0501")
K("C-0502", "PriorityScorer", "component", G, "전역 방문 비용 평가와 구역 발견 이득 평가를 구분", [
    ("weights", "ScoreWeights", "평가 가중치 (시험 전 확정)"), ("scoreVersion", "UUID", "정규화·가중치 버전"),
], [("score", "regions, state, policy", "ScoredRegions", "전역은 방문 비용, 구역은 추가 발견 기대값과 이동·관측 시간을 평가한다.")], old="C-0502")
K("C-0505", "GlobalVisitPlanner", "component", G, "미완료 영역 순서를 NN + 방향성 2-opt 로 개선하고 복귀 가능한 prefix 선택", [
    ("costProvider", "RegionCostProvider", "방향성 이동 비용"), ("planningDeadline", "Duration", "계산 시간 상한"),
    ("mode", "GlobalMode", "VISIT_ALL_BASELINE / OPERATOR_PRIORITY_FIRST"),
], [
    ("buildOrder", "RegionState[], VehicleState", "GlobalVisitPlan", "가까운 미방문 영역부터 전체 순서를 만든다."),
    ("improveOrder", "VisitOrder, CostMatrix", "VisitOrder", "2-opt 후보의 방향성 전체 비용을 재계산해 개선만 채택한다."),
    ("fitBudget", "VisitOrder, BudgetSnapshot", "BudgetedVisitPlan", "prefix 끝점마다 복귀·예비 비용을 재검사한다."),
], old="C-0505")
K("C-0506", "RegionTaskCoordinator", "component", G, "한 번에 하나의 영역 요청을 발행하고 실제 관측 결과를 전역 상태에 반영", [
    ("activeTask", "RegionTask", "현재 작업 (취소 대기 포함)"), ("passIndex", "int", "순회 회차"),
], [
    ("requestNext", "GlobalVisitPlan", "RegionTask", "남은 셀·할당 예산·불변 계획 버전을 전달한다."),
    ("applyResult", "RegionResultDTO", "RegionStateUpdate", "중복·과거 작업을 구분하고 실제 가시 합집합만 누적한다."),
    ("cancelActive", "ReasonCode", "CancelRequest", "종료·취소 확인 후 다음 작업을 허용한다."),
], old="C-0506")
I("C-0507", "RegionCostProvider", G, "전역 계획이 쓰는 전이·처리 시간 비용의 교체 가능한 입력 인터페이스", [
    ("transitionCost", "ExitState, EntryState", "CostResult", "방향성 이동 시간·가능 여부·오차·출처를 반환한다."),
    ("serviceCost", "RegionState, ObservationPolicy", "CostResult", "진입 뒤 처리 완료까지 비용을 추정한다."),
], impl="C-0517")
K("C-0517", "RouteCostProvider", "component", G, "경로 추정 기반 비용 (MOCK · ROUTE_ESTIMATE · MEASURED 출처 구분) — RegionCostProvider 구현", [
    ("sourceKind", "CostSource", "MOCK / ROUTE_ESTIMATE / MEASURED"), ("pathCost", "PathCostEstimator", "경로 비용"),
    ("costVersion", "VersionId", "상태·지도·경로 추정 버전"),
], [])
# 구역 계획
K("C-0508", "LocalIPPPlanner", "component", L, "구역 후보를 짧은 계획 범위에서 평가하고 첫 실행 구간만 선택", [
    ("candidateGenerator", "ViewCandidateGenerator", ""), ("visibility", "VisibilityEvaluator", ""), ("pathCost", "PathCostEstimator", ""),
    ("likelihood", "DetectionLikelihoodModel", "탐지 확률 보정"), ("deadline", "Duration", "처리 시간 측정 후 확정"),
], [
    ("plan", "RegionTask, PlanningSnapshot", "ObservationPlan", "강제 제약을 통과한 후보의 추가 이득·시간·중복을 비교한다."),
    ("replan", "ObservationPlan, ChangeEvent", "ObservationPlan", "가림·실행 편차·지도·예산 변화에 제한된 범위로 재계획한다."),
    ("firstExecutableSegment", "ObservationPlan", "PlanSegment", "만료·상태 조건을 확인한 짧은 구간만 반환한다."),
], old="C-0508")
K("C-0509", "ViewCandidateGenerator", "component", L, "기본 경로·새 표면·가림 해소·후보 재확인·복귀 행동 생성", [
    ("camera", "ICameraModel", "지원되는 시선 자유도"), ("candidateLimit", "int", "부하로 확정할 상한"),
], [
    ("generate", "RegionTask, SurfaceMap, VehicleState", "ViewCandidate[]", "위치·yaw·카메라 자세·체류를 포함한 후보를 만든다."),
    ("deduplicate", "ViewCandidate[]", "ViewCandidate[]", "유사 위치·시선을 합치고 제외 근거를 남긴다."),
], old="C-0509")
K("C-0510", "VisibilityEvaluator", "component", L, "영상 광선과 지도·수색 표면으로 실제 가시 부분을 평가", [
    ("mapVersion", "VersionId", "등록 GIS/DEM"), ("uncertaintyPolicy", "VisibilityPolicy", "누락 지형·가림의 보수 처리"),
], [
    ("evaluate", "ViewCandidate, SurfaceMap", "VisibilityResult", "지형·건물 차폐와 픽셀·거리 조건을 셀 부분면적별로 계산한다."),
    ("actualFootprint", "SyncedFrameDTO", "VisibilityResult", "촬영 시각의 실제 자세로 보이는 영역을 계산한다."),
], old="C-0510")
K("C-0512", "PathCostEstimator", "component", L, "이동·가감속·회전·관측·상승/하강과 예산 비용 계산", [
    ("motionProfile", "MotionLimits", "검증된 속도·가속도 한계"), ("costSource", "CostSource", "모의·추정·실측"),
], [
    ("connect", "VehicleState, ViewCandidate, MapSnapshot", "RouteResult", "등록 GIS + 안전 여유의 경로를 반환하거나 보류한다."),
    ("estimate", "Route, ObservationAction", "CostVector", "시간 s·에너지 Wh·예측 오차를 구분해 계산한다."),
], old="C-0512")
K("C-0513", "ReturnBudgetChecker", "component", L, "후보 끝점·예상 시각에서 복귀·착륙·예비 비용 검증", [
    ("returnPolicy", "ReturnPolicy", "복귀점·에너지/시간 모델·예비량"),
], [
    ("check", "Route, EndpointState, RemainingBudget", "Decision", "이동+관측+복귀+예비가 남은 시간·Wh 안인지 확인한다."),
    ("remaining", "ExecutionReport[]", "BudgetSnapshot", "중복 과금 없이 실제 소비량과 오차를 갱신한다."),
], old="C-0513")
K("C-0515", "BaselineCoveragePlanner", "component", L, "확률 보정 전 사용하는 구역 기본 관측 경로와 잔여 관측 후보", [
    ("camera", "ICameraModel", "실제 유효 관측 폭"), ("coveragePolicy", "CoveragePolicy", "중복률·격자·최소 유효 관측 시간"),
], [
    ("build", "RegionTask, SurfaceMap, VehicleState", "ObservationPlan", "개방 영역은 sweep, 복잡 영역은 등록 관측점을 잇는다."),
    ("remainingViews", "RegionState, ObservationSnapshotDTO", "ViewCandidate[]", "미관측·품질 부족·가림 부분의 후속 후보를 준다."),
], old="C-0515")
K("C-0511", "DetectionLikelihoodModel", "component", "policies", "보정된 탐지 이벤트 d/f 와 가시성 q 의 유효 적용 범위 (계획·존재 가능성 갱신이 공유)", [
    ("likelihoodId", "UUID", "미측정이면 확률 갱신 비활성"), ("validDomain", "ObservationDomain", "픽셀·거리·조도·압축·가림 범위"),
], [
    ("predict", "VisibleSurface, ImageQuality", "LikelihoodResult", "유효 범위에서 d/f 와 가시성 q 를 반환한다."),
    ("validateDomain", "ObservationContext", "Decision", "관측 조건별 시험으로 보정한 모델·버전인지 확인한다."),
], old="C-0511")

K("C-0518", "PlanController", "controller", "api", "탐색 계획 이력 조회 요청의 입구 (REST)", [("planner", "IViewpointPlanner", "")], [
    ("getPlans", "missionId, filter", "List<PlanHistoryDTO>", "GET /api/missions/{id}/plans"),
])

DTO("C-0519", "PlanningInputDTO", "계획 입력 버전 묶음", [("missionId", "UUID", "임무"), ("revision", "int", "실행 개정"),
    ("spatialVersion", "UUID", "공간자료"), ("observationVersion", "int64", "관측 스냅샷"), ("vehicle", "VehicleStateDTO", "기체 상태"),
    ("policyVersion", "UUID", "정책")])
DTO("C-0520", "SearchPlanDTO", "탐색 계획", [("planId", "UUID", "계획"), ("planVersion", "int", "임무별 번호"), ("kind", "GLOBAL | LOCAL | REOBSERVE | BASELINE", "종류"),
    ("route", "List<PlanSegment>", "실행 구간"), ("expiresAt", "Time", "만료"), ("status", "PlanStatus", "상태")])
DTO("C-0521", "RegionSetDTO", "수색 후보영역 집합", [("missionId", "UUID", "임무"), ("regions", "List<RegionSummary>", "영역·대표점·면적"),
    ("splitVersion", "UUID", "분할 버전"), ("reason", "ReasonCode?", "0건 사유")])
DTO("C-0522", "GlobalPlanDTO", "전역 방문 순서", [("planId", "UUID", "계획"), ("order", "List<UUID>", "방문 순서"), ("adoptedPrefix", "int", "채택 prefix"),
    ("deferred", "List<UUID>", "보류 영역"), ("costSource", "CostSource", "비용 출처")])
DTO("C-0523", "ViewpointPlanDTO", "다음 관측점·경로", [("planId", "UUID", "계획"), ("viewpoint", "Pose", "위치·고도·방향"),
    ("dwellSec", "float", "관측 시간"), ("segment", "PlanSegment", "첫 실행 구간"), ("reason", "ReasonCode?", "보류 사유")])
DTO("C-0524", "RegionTaskDTO", "영역 작업 계약", [("taskId", "UUID", "작업"), ("regionId", "UUID", "영역"), ("cells", "List<CellId>", "남은 셀"),
    ("budget", "BudgetSnapshot", "할당 예산"), ("planVersion", "int", "불변 계획 버전"), ("deadline", "Time", "마감")])
DTO("C-0525", "RegionResultDTO", "영역 작업 결과", [("taskId", "UUID", "작업"), ("status", "DONE | PARTIAL | CANCELLED | FAILED", "종료"),
    ("visibleArea", "float", "실제 가시 면적"), ("remainingBudget", "BudgetSnapshot", "남은 예산")])
DTO("C-0526", "ObservationPlanDTO", "구역 관측 계획", [("planId", "UUID", "계획"), ("candidates", "List<ViewCandidateSummary>", "선택·기각 후보"),
    ("firstSegment", "PlanSegment", "첫 실행 구간"), ("inputVersions", "VersionBundle", "입력 버전")])
DTO("C-0527", "PrecisionRequestDTO", "정밀 관측(재관측) 요청", [("candidateId", "UUID", "후보"), ("requesterKind", "SYSTEM | USER", "요청 주체"),
    ("requestedBy", "UUID?", "운영자"), ("hypothesis", "Hypothesis", "확인 가설"), ("limits", "Limits", "최대 시간·횟수")])
DTO("C-0528", "ChangeEventDTO", "재계획 사건", [("missionId", "UUID", "임무"), ("kind", "OBSERVATION | TARGET | STATE | OPERATOR", "종류"),
    ("evidenceRef", "String", "근거")])
DTO("C-0529", "TaskUpdateDTO", "영역 작업 반영 결과", [("regionId", "UUID", "영역"), ("state", "RegionState", "상태"), ("nextTask", "RegionTaskDTO?", "다음 작업")])
DTO("C-0530", "PlanHistoryDTO", "계획 이력", [("plan", "SearchPlanDTO", "계획"), ("decisionLog", "JSON", "우선순위·선택 근거"),
    ("commands", "List<CommandStatusDTO>", "실행 명령 결과")])
DTO("C-0531", "TrackingTargetDTO", "개별 추적 관측 목표 요청", [("candidateId", "UUID", "대상"), ("lastSeen", "GeoPoint", "마지막 위치"),
    ("speedLimit", "float", "보행 속도 이하 기준")])

DAO("C-0504", "SearchPlanDAO", "search_plan", [
    ("insert", "SearchPlan", "UUID", "계획과 입력 버전을 저장한다."),
    ("findActive", "missionId, kind", "SearchPlan?", "현재 유효 계획을 읽는다."),
    ("findByMission", "missionId, filter", "List<SearchPlan>", "계획 이력을 읽는다."),
    ("supersede", "oldPlanId, SearchPlan", "UUID", "이전 계획의 대기 목표를 폐기하고 새 버전과 잇는다."),
], resp="DB-09 search_plan 테이블의 저장·조회 (구 PlanRepository)")
DAO("C-0532", "SearchRegionDAO", "search_region", [
    ("insertAll", "List<SearchRegion>, inputVersions", "int", "분할 버전·출처와 함께 저장한다."),
    ("findOpen", "missionId", "List<SearchRegion>", "미완료 영역을 읽는다."),
    ("update", "SearchRegion, rowVersion", "bool", "남은 면적·상태를 조건부로 갱신한다."),
])
DAO("C-0533", "ReobservationRequestDAO", "reobservation_request", [
    ("insert", "ReobservationRequest", "UUID", "요청을 대기열에 넣는다 (후보당 진행 중 1건)."),
    ("findPending", "missionId", "List<ReobservationRequest>", "처리 순서대로 읽는다."),
    ("updateStatus", "requestId, status, planId?", "bool", "상태·연결 계획을 바꾼다."),
])
ENT("C-0534", "SearchPlan", "search_plan", "전역·구역·재관측 계획 버전 (DB-09 한 행)",
    keys=["planId {PK}", "missionId {FK}", "planVersion {UQ}", "planKind · status", "route · decisionLog : JSON"])
ENT("C-0535", "SearchRegion", "search_region", "전역 방문 단위 영역 (DB-07 한 행)",
    keys=["regionId {PK}", "missionId {FK}", "regionGeom", "remainingAreaM2", "state · passIndex"])
ENT("C-0536", "ReobservationRequest", "reobservation_request", "정밀 관측 요청 대기열 (DB-10 한 행)",
    keys=["requestId {PK}", "candidateId {FK}", "status · queueOrder", "planId {FK}", "limits · usage : JSON"])


def _cd():
    a = layered("cd05a", "(1/2) 계층 구조", svc_pkg=L, ctl=["C-0518"],
                dto=["C-0519", "C-0520", "C-0521", "C-0522", "C-0523", "C-0524", "C-0525", "C-0526", "C-0527", "C-0528", "C-0529", "C-0530", "C-0531"],
                pairs=[("C-0516", "C-0503")],
                ext=[("C-0807", "services.mission_map"), ("C-0722", "services.vision"), ("C-0908", "services.command")],
                daos=[("C-0504", False), ("C-0532", False), ("C-0533", False)], ents=["C-0534", "C-0535", "C-0536"], api_w=0.3, dto_cols=4,
                ext_w=0.3)
    # (2/2) 구성요소 — 전역 · 구역 · 공유 정책
    B = {"vp": box("C-0503", ref=True)}
    for c in ["C-0501", "C-0502", "C-0505", "C-0506", "C-0507", "C-0517", "C-0508", "C-0509", "C-0510", "C-0512", "C-0513", "C-0515"]:
        B[c] = box(c, attrs=[] if c in ("C-0508",) else None)
    B["C-0511"] = box("C-0511", ref=True); B["cam"] = box("C-0808", ref=True)
    P = [dict(name=L, row=0, x=0.3, w=0.4, rows=[[("vp", 1.0)]]),
         dict(name=G, row=1, x=0.005, w=0.99, rows=[[("C-0501", .25), ("C-0502", .25), ("C-0505", .25), ("C-0506", .25)],
                                                    [(None, .5), ("C-0507", .25), ("C-0517", .25)]]),
         dict(name=L, row=2, x=0.005, w=0.99, rows=[[(None, .375), ("C-0508", .25), (None, .375)],
                                                    [("C-0509", .25), ("C-0510", .25), ("C-0512", .25), ("C-0513", .25)],
                                                    [(None, .25), ("C-0515", .25), (None, .5)]]),
         dict(name="policies · services.mission_map", row=3, x=0.25, w=0.5, rows=[[("C-0511", .5), ("cam", .5)]])]
    R = [("vp", c, "assoc", "", {"elbow": 1}) for c in ["C-0501", "C-0502", "C-0505", "C-0506"]]
    R += [("vp", "C-0508", "assoc", "", {"pts": lambda g: [(g["vp"][0], g["vp"][1] + g["vp"][3] / 2),
          (g["C-0501"][0] - 18, g["vp"][1] + g["vp"][3] / 2), (g["C-0501"][0] - 18, g["C-0508"][1] + 40), (g["C-0508"][0], g["C-0508"][1] + 40)]})]
    R += [("C-0505", "C-0507", "assoc", "costProvider", {}), ("C-0517", "C-0507", "real", "", {}),
          ("C-0508", "C-0509", "assoc", "", {"elbow": 1}), ("C-0508", "C-0510", "assoc", "", {"elbow": 1}),
          ("C-0508", "C-0512", "assoc", "", {"elbow": 1}), ("C-0508", "C-0511", "dep", "", {"pts": lambda g: [
              (g["C-0508"][0], g["C-0508"][1] + g["C-0508"][3] / 2), (g["C-0509"][0] - 16, g["C-0508"][1] + g["C-0508"][3] / 2),
              (g["C-0509"][0] - 16, g["C-0511"][1] - 30), (g["C-0511"][0] + g["C-0511"][2] / 2, g["C-0511"][1] - 30),
              (g["C-0511"][0] + g["C-0511"][2] / 2, g["C-0511"][1])]}),
          ("C-0515", "cam", "dep", "", {"elbow": 1})]
    return [a, ("cd05b", "(2/2) 전역·구역 계획 구성요소", P, B, R)]


CD("CD-05", "탐색 계획", "05", _cd)

RN = ("rn", "MissionRunner", "service")
VP = ("vp", "IViewpointPlanner", "interface")
S("SD-0501", "05", [RN, VP, ("mm", "IMissionMapService", "interface"), ("cr", "CandidateRegionPlanner", "component"), ("rd", "SearchRegionDAO", "dao")], [
    call("rn", "vp", "generateRegions(missionId)", "RegionSetDTO", "수색 후보영역 생성을 요청한다.", [
        call("vp", "mm", "getObservationSnapshot(missionId)", "ObservationSnapshotDTO", "기존 격자와 실제 관측 잔여 영역을 읽는다."),
        call("vp", "cr", "generate(PlanningInput, ObservationSnapshotDTO)", "List<SearchRegion>", "미관측·오래된 관측·우선구역·후보 주변 영역을 만든다."),
        alt([("영역 1건 이상", [call("vp", "rd", "insertAll(List<SearchRegion>, inputVersions)", "count", "셀 교차 할당을 검증하고 분할 버전·출처를 저장한다.")]),
             ("유효 후보 0건 (2a)", [note("vp", "rd", "사유 기록 · 새 이동 목표는 만들지 않음 → 대기·종료 판단")])]),
    ]),
], entry="IViewpointPlanner.generateRegions", 시작="임무 시작 또는 지도·임무 경계 변경 (MissionRunner)",
   사후="미관측·오래된 관측·우선구역·후보 주변 영역이 생성된다 (시설·통로 분류 포함). 관측만 바뀌면 재분할 대신 상태를 갱신한다.")
S("SD-0502", "05", [RN, VP, ("rd", "SearchRegionDAO", "dao"), ("ps", "PriorityScorer", "component"), ("gv", "GlobalVisitPlanner", "component"),
                    ("cp", "RegionCostProvider", "interface"), ("pd", "SearchPlanDAO", "dao")], [
    call("rn", "vp", "prioritize(missionId)", "GlobalPlanDTO", "탐색 우선순위 산정을 요청한다.", [
        call("vp", "rd", "findOpen(missionId)", "List<SearchRegion>", "미완료 영역을 읽는다."),
        call("vp", "ps", "score(regions, state, policy)", "ScoredRegions", "기본 모드의 동일 가중치·미서비스 우선 조건을 확인한다."),
        call("vp", "gv", "buildOrder(regions, state)", "GlobalVisitPlan", "가까운 미완료 영역부터 순서를 잇는다."),
        call("vp", "gv", "improveOrder(order, costs)", "VisitOrder", "방향성 전체 비용으로 2-opt 개선을 검사한다.", [
            loop("2-opt 후보마다 · 마감 안", [call("gv", "cp", "transitionCost(exit, entry)", "CostResult", "방향성 이동 비용을 얻는다.")]),
        ]),
        call("vp", "gv", "fitBudget(order, budget)", "BudgetedVisitPlan", "prefix 끝점 복귀·예비·처리 시간을 포함한다."),
        call("vp", "pd", "insert(SearchPlan GLOBAL)", "planId", "채택·보류 목록과 비용 출처를 저장한다."),
    ]),
], entry="IViewpointPlanner.prioritize", 시작="수색 영역 생성 또는 재계획 이벤트 (MissionRunner)")
S("SD-0503", "05", [RN, VP, ("tc", "RegionTaskCoordinator", "component"), ("lp", "LocalIPPPlanner", "component"),
                    ("rb", "ReturnBudgetChecker", "component"), ("pd", "SearchPlanDAO", "dao")], [
    call("rn", "vp", "selectNext(missionId, VehicleStateDTO)", "ViewpointPlanDTO", "다음 관측지점을 요청한다.", [
        call("vp", "tc", "requestNext(globalPlan)", "RegionTask", "진행 작업 종료 후 첫 미완료 영역의 셀·예산을 받는다."),
        call("vp", "lp", "plan(RegionTask, PlanningSnapshot)", "ObservationPlan", "기본·가림·정밀 관측 후보를 구역 예산 안에서 비교한다."),
        call("vp", "rb", "check(route, endpoint, budget)", "Decision", "구간 끝점에서 복귀 시간·Wh 와 예비량을 검증한다."),
        alt([("안전·예산 통과", [call("vp", "pd", "insert(SearchPlan LOCAL)", "planId", "선택 이유·입력 버전·만료·첫 실행 구간을 확정한다.")]),
             ("수동·복귀·착륙·안전 제한 (2a)", [note("vp", "pd", "새 정찰 명령 차단 · 현장 동작 유지")])]),
    ]),
], entry="IViewpointPlanner.selectNext", 시작="활성 임무의 다음 관측점 선택 이벤트 (MissionRunner)")
S("SD-0504", "05", [("ts", "TargetService", "service"), VP, ("rq", "ReobservationRequestDAO", "dao"), ("ti", "ITargetService", "interface"),
                    ("lp", "LocalIPPPlanner", "component"), ("rb", "ReturnBudgetChecker", "component"), ("pd", "SearchPlanDAO", "dao")], [
    call("ts", "vp", "requestPrecision(PrecisionRequestDTO)", "ViewpointPlanDTO?", "정밀 관측 경로를 요청한다.", [
        call("vp", "rq", "insert(ReobservationRequest)", "requestId", "요청의 가설·최대 시간·횟수를 대기열에 넣는다."),
        call("vp", "ti", "getCandidate(candidateId)", "CandidateDetailDTO", "원관측 시각·위치 분포·보류 사유를 확인한다."),
        call("vp", "lp", "plan(RegionTask, PlanningSnapshot)", "ObservationPlan", "다른 시선에서 추가로 보이는 표면을 비교한다."),
        call("vp", "rb", "check(route, endpoint, budget)", "Decision", "잔여 구역과 복귀 예산을 보존한다."),
        alt([("경로 생성", [call("vp", "pd", "insert(SearchPlan REOBSERVE)", "planId", "요청·구역 계획·실행 구간을 잇는다."),
                            call("vp", "rq", "updateStatus(requestId, PLANNED, planId)", "bool", "요청 상태를 바꾼다.")]),
             ("안전 제한·RC 수동·경로 미생성 (2a)", [call("vp", "rq", "updateStatus(requestId, DEFERRED)", "bool", "실행을 거부하고 사유를 남긴다.")])]),
    ]),
], entry="IViewpointPlanner.requestPrecision", 시작="UC-0706 재관측 요청 접수 (TargetService)")
S("SD-0505", "05", [("mm", "MissionMapService", "service"), VP, ("tc", "RegionTaskCoordinator", "component"), ("gv", "GlobalVisitPlanner", "component"),
                    ("pd", "SearchPlanDAO", "dao")], [
    call("mm", "vp", "replan(missionId, ChangeEventDTO)", "SearchPlanDTO?", "변경 사건으로 재계획을 요청한다.", [
        call("vp", "tc", "applyResult(RegionResultDTO)", "RegionStateUpdate", "현재 작업과 실제 관측 결과를 대조해 잔여 셀을 유지한다."),
        opt("전역 재순서 사건", [call("vp", "gv", "buildOrder(regions, state)", "GlobalVisitPlan", "아직 기회를 못 받은 영역을 먼저 배정한다.")]),
        alt([("계획 생성", [call("vp", "pd", "supersede(oldPlanId, SearchPlan)", "planId", "이전 대기 목표를 무효화하고 새 버전을 잇는다.")]),
             ("경로 미생성·안전 제한 (3a)", [note("vp", "pd", "계획 보류 · 사유 기록")])]),
    ]),
], entry="IViewpointPlanner.replan", 시작="유효 관측·대상 요청·운용 상태 변화 이벤트 (MissionMapService · MissionRunner)")
S("SD-0506", "05", [("op", "관제 운영자", "actor"), ("ctl", "PlanController", "controller"), VP, ("pd", "SearchPlanDAO", "dao"),
                    ("cs", "ICommandService", "interface")], [
    call("op", "ctl", "getPlans(missionId, filter)", "List<PlanHistoryDTO>", "탐색 계획 이력을 조회한다.", [
        call("ctl", "vp", "queryHistory(missionId, filter)", "List<PlanHistoryDTO>", "결정 근거와 실행 결과를 묶는다.", [
            call("vp", "pd", "findByMission(missionId, filter)", "List<SearchPlan>", "후보·평가 근거·버전을 읽는다."),
            call("vp", "cs", "getStatusByPlan(planId)", "List<CommandStatusDTO>", "계획의 실제 실행 결과와 대조한다."),
            opt("기록 미등록 (2a)", [note("vp", "cs", "조회 가능 범위와 누락 상태를 표시")]),
        ]),
    ]),
], entry="PlanController.getPlans")
S("SD-X02", "05", [RN, VP, ("gv", "GlobalVisitPlanner", "component"), ("cp", "RegionCostProvider", "interface"), ("tc", "RegionTaskCoordinator", "component"),
                   ("lp", "LocalIPPPlanner", "component")], [
    call("rn", "vp", "prioritize(missionId)", "GlobalPlanDTO", "전역 방문 순서를 만든다.", [
        call("vp", "gv", "buildOrder(regions, state)", "GlobalVisitPlan", "대표점으로 방문 순서를 만든다.", [
            loop("영역 쌍마다", [call("gv", "cp", "transitionCost(exit, entry)", "CostResult", "대표점 사이 방향성 비용을 얻는다.")]),
        ]),
    ]),
    call("rn", "vp", "planRegion(RegionTaskDTO)", "ObservationPlanDTO", "영역 작업을 구역 계획에 위임한다.", [
        call("vp", "tc", "requestNext(globalPlan)", "RegionTask", "한 번에 한 영역 요청을 발행한다."),
        call("vp", "lp", "plan(RegionTask, PlanningSnapshot)", "ObservationPlan", "첫 실행 구간만 반환한다."),
    ]),
    call("rn", "vp", "applyRegionResult(RegionResultDTO)", "TaskUpdateDTO", "영역 처리 결과를 반영한다.", [
        call("vp", "tc", "applyResult(RegionResultDTO)", "RegionStateUpdate", "실제 가시 합집합만 누적한다."),
        alt([("부분 완료", [note("vp", "lp", "다음 순회에 배정")]),
             ("종료·실패·임무 변경", [call("vp", "gv", "buildOrder(regions, state)", "GlobalVisitPlan", "순서를 갱신한다 (진행 작업은 취소 확인 후 교체).")])]),
    ]),
], title="전역 순서·영역 작업·구역 계획 연결", uc="UC-0501~0506, UC-0803, UC-0806",
   개요="대표점은 방문 비용 계산에 사용한다. 전역 계획기는 영역 요청과 방문 순서 갱신 시점을 관리한다.",
   시작="MissionRunner (임무 실행 반복)", 선행="수색 영역과 전역 계획이 있다.", 사후="영역 상태·전역 순서·구역 계획이 버전과 함께 갱신된다.",
   예외="계획 미해결과 접근 불가는 별도 상태로 기록한다.", 경계="내부 이벤트 처리 | 결과 화면: UI-04 | 연결 시험: AT-F07")
