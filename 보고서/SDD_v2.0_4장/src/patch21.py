import zipfile, re, sys
from lxml import etree
SRC = sys.argv[2] if len(sys.argv) > 2 else "v2.0_관리단위재구성.hwpx"   # 팀 드라이브 원본 (학번 포함 · git 제외)
DST = sys.argv[1]
HP = "http://www.hancom.co.kr/hwpml/2011/paragraph"
ACT = {
 "관제 운영자 → 웹/PWA": "관제 운영자 (웹/PWA)",
 "명령 요청 (CommandService · MissionRunner 내부 호출)": "CommandService · MissionRunner (명령 요청)",
 "지도자료 수신 또는 표시·계획 변환 요청 (MissionService 등 내부 호출)": "MissionService 등 내부 서비스 (지도자료 수신 · 표시·계획 변환 요청)",
 "임무 시작 또는 지도·임무 경계 변경 (MissionRunner)": "MissionRunner (임무 시작 · 지도·임무 경계 변경)",
 "수색 영역 생성 또는 재계획 이벤트 (MissionRunner)": "MissionRunner (수색 영역 생성 · 재계획)",
 "활성 임무의 다음 관측점 선택 이벤트 (MissionRunner)": "MissionRunner (다음 관측점 선택)",
 "UC-0706 재관측 요청 접수 (TargetService)": "TargetService (UC-0706 재관측 요청)",
 "유효 관측·대상 요청·운용 상태 변화 이벤트 (MissionMapService · MissionRunner)": "MissionMapService · MissionRunner (유효 관측·대상 요청·운용 상태 변화)",
 "현장 영상 수신 (카메라 → 게이트웨이 VideoRelay)": "게이트웨이 VideoRelay (카메라 영상 수신)",
 "중앙 서버의 새 프레임 수신 (VideoIngestService)": "VideoIngestService (새 프레임 수신)",
 "비전 작업자 처리 가능 이벤트 (FrameIngestor)": "FrameIngestor (비전 작업자 처리 가능)",
 "녹화 구간 확정 (게이트웨이 SegmentManifestStore → 서버)": "게이트웨이 SegmentManifestStore (녹화 구간 확정)",
 "첫 탐지 또는 명시적 스냅샷 생성 (TargetService)": "TargetService (첫 탐지 · 스냅샷 생성)",
 "프로파일 협상 (VideoProfileManager)": "VideoProfileManager (프로파일 협상)",
 "최신 유효 분석 프레임 입력 (FrameIngestor)": "FrameIngestor (최신 유효 분석 프레임)",
 "새 탐지 관측 또는 추적 ID 유실·재연결 (CandidateRegistry)": "CandidateRegistry (새 탐지 관측 · 추적 ID 유실·재연결)",
 "사람 탐지와 영상·비행정보 연결 완료 (PersonDetectionService)": "PersonDetectionService (탐지·영상·비행정보 연결 완료)",
 "인상착의 조건이 있는 새 탐지 또는 비교 요청 (CandidateRegistry)": "CandidateRegistry (인상착의 조건 비교)",
 "개별 추적 시작/종료 요청 또는 새 대상 관측": "관제 운영자 (개별 추적 시작/종료 요청)",
 "새 분석 프레임 (FrameIngestor)": "FrameIngestor (새 분석 프레임)",
 "학습 서버 작업 (DatasetBuilder)": "DatasetBuilder (학습 서버 작업)",
 "영상·비행정보가 연결된 프레임 입력 (MissionMapService)": "MissionMapService (영상·비행정보 연결 프레임)",
 "프레임 관측영역·품질 평가 완료 (FrameIngestor)": "FrameIngestor (관측영역·품질 평가 완료)",
 "유효 관측 누적 또는 완료도 조회 (MissionService · HistoryService)": "MissionService · HistoryService (관측 누적 · 완료도 조회)",
 "상황지도·관측정보 갱신 이벤트 (MissionMapService)": "MissionMapService (상황지도·관측정보 갱신)",
 "분석 완료 프레임 (FrameIngestor)": "FrameIngestor (분석 완료 프레임)",
 "임무·정밀 관측·개별 추적·안전 동작 요청 (MissionRunner · TargetTrackingService)": "MissionRunner · TargetTrackingService (임무·정밀 관측·추적·안전 동작 요청)",
 "중앙 명령 발행 전 및 게이트웨이 수신 직후 (CommandService)": "CommandService (명령 발행 전 · 게이트웨이 수신 직후)",
 "유효 명령 구성 완료 (CommandService)": "CommandService (유효 명령 구성 완료)",
 "명령 응답 또는 실제 기체 상태 갱신 (ServerReporter)": "ServerReporter (명령 응답 · 기체 상태 갱신)",
 "현장 RC 인수 (LocalControlMonitor)": "LocalControlMonitor (현장 RC 인수)",
 "임무 파일 업로드 요청 (MissionRunner)": "MissionRunner (임무 파일 업로드)",
 "주기 상태 감시 또는 상태 변화 (MissionRunner · CommandService)": "MissionRunner · CommandService (주기 상태 감시 · 상태 변화)",
 "현장 조종자 → RC → 비행제어기 (서버와 독립)": "현장 조종자 (RC → 비행제어기 · 서버와 독립)",
 "현장·중앙 독립 감시 타이머": "감시 타이머 (현장·중앙 독립)",
 "장애 감지 또는 비행제어기 보호 상태 변화 (Pi 감시)": "LocalControlMonitor (장애 · 비행제어기 보호 상태 변화)",
 "게이트웨이·서버·기체 링크 복구 이벤트 (GatewayController)": "GatewayController (링크 복구)",
 "운영자·현장 조종자의 준비 점검": "관제 운영자 · 현장 조종자 (준비 점검)",
 "탐지·상태·명령 처리 이벤트 (TargetService · SafetySupervisor · CommandService)": "TargetService · SafetySupervisor · CommandService (알림 원인 사건)",
 "임무 실행 중 각 모듈의 상태 변화": "각 서비스 — 명령·안전·비전·임무 (상태 변화)",
 "현장 수신 영상·비행정보 (VideoRelay)": "VideoRelay (현장 영상·비행정보 수신)",
 "Pi 조각 확정 후 회수 (SegmentManifestStore)": "SegmentManifestStore (Pi 조각 확정 후 회수)",
}
SUBS = [
 ("시작 주체/사건", "액터"),
 ("첫 생명선은 실제 시작 주체(", "첫 생명선은 액터("),
 ("DB-01~54 테이블, 키, 인덱스와 기존 데이터 이관", "ERD 테이블, 키, 인덱스와 기존 데이터 이관"),
 ("— DB-01~18 과 1:1", "— ERD 18개 테이블과 1:1"),
 (", 테이블은 DB-01~54", ""),
 (", 테이블은 DB-01~18", ""),
]
cnt = {"act": 0, "sub": 0, "db": 0}
z = zipfile.ZipFile(SRC)
out = zipfile.ZipFile(DST, "w")
for info in z.infolist():
    data = z.read(info.filename)
    if re.match(r"Contents/section\d+\.xml", info.filename):
        root = etree.fromstring(data)
        for t in root.iter("{%s}t" % HP):
            if not t.text: continue
            s = t.text
            if s.strip() in ACT: s = s.replace(s.strip(), ACT[s.strip()]); cnt["act"] += 1
            for a, b in SUBS:
                if a in s: s = s.replace(a, b); cnt["sub"] += 1
            s2 = re.sub(r"DB-\d{2} (?=[a-z_]+)", "", s)
            if s2 != s: cnt["db"] += 1; s = s2
            t.text = s
        data = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)
    elif info.filename == "BinData/image15.png":
        data = open("out2/pd01.png", "rb").read()
    ct = zipfile.ZIP_STORED if info.filename == "mimetype" else zipfile.ZIP_DEFLATED
    out.writestr(info, data, compress_type=ct)
out.close(); print(cnt)
