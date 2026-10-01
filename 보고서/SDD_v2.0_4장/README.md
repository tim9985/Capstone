# SDD v2.0 — 4장 객체지향설계 전면 재작성 (10-01)

> 9.30 설계 발표 피드백 2번 ("설계는 객체지향 · DTO 패키지 · DAO · Service 인터페이스 · 패키지 간 호출은 인터페이스 · HTTP 를 관계로 그리지 않음") 반영
> HWPX: `3. 설계명세서_…_v2.1.hwpx` (최신 · 관리단위 묶음) · `…_v2.0.hwpx` (v1.4 (1) 에서 4장만 교체) — 둘 다 git 밖

## 목차 — v2.1 (관리단위 묶음 · 10-01)

개발자가 한 관리단위를 한곳에서 보도록 **관리단위 → 다이어그램** 순으로 묶는다

| 절 | 내용 |
|---|---|
| 4.1 Deployment Diagram | DD-01 (무채색) |
| 4.2 설계 모델 | 공통 기준 (클래스 · 시퀀스 · 클래스 명세) + 패키지 다이어그램 PD-01 + 패키지 표 |
| 4.2.n 관리단위 (n = 1~12) | 4.2.n.1 클래스 다이어그램 · 4.2.n.2 클래스 명세 · 4.2.n.3 시퀀스 다이어그램 |

v2.0 → v2.1 변경
- 목차를 관리단위 기준으로 재배치 (팀 드라이브 `v2.0_관리단위재구성.hwpx` 기준)
- 시퀀스 표 `시작 주체/사건` → **`액터`** — 값은 `액터 (계기)` 형태 (예: `MissionRunner (수색 영역 생성 · 재계획)`)
- **DB-\* 식별자 제거** — ERD 상세(DB-01~) 가 삭제됐으므로 테이블은 이름(`user_account` 등)으로만 부른다 · PD-01 그림도 다시 그림
- 적용: `src/patch21.py <출력.hwpx> <원본.hwpx>` (원본 HWPX 를 직접 고친다 · 생성 코드 `build20.py` 는 v2.0 배치 그대로)

v2.1 — GRACE 적용 구간 변경 (팀원 요청 · 10-01)

GRACE 는 **서버 → 관제 운용자 단말의 대상 크롭 제공**에만 쓴다 · 드론 → 게이트웨이 → 서버 원본(SRT) 경로와 원본 분석(탐지·좌표·관측 완료)은 그대로

| 위치 | 변경 |
|---|---|
| 2.1 파일 구조 | `research/grace/` → `services/media/crop-grace/` (서버 크롭 제공 모듈) · `apps/web/` 에 전용 GRACE 디코더 · 시험 경로에 손실·참조 복구·호환·성능 |
| 4.1 DD-01 | 무채색으로 다시 그림 (`dd.py`) — 서버 GRACE 크롭 인코더 · 단말 GRACE 디코더 · 크롭 패킷(서버→단말) · 사용 패킷 피드백(단말→서버) · 게이트웨이 GRACE 인코딩 없음 |
| 4.2 공통 기준 · PD-01 · 패키지 표 | 경계 클래스 `CropStreamEndpoint` · 패키지 `services.media.crop_grace` · `research.grace` 삭제 |
| CD-06 (3/3) | 서버 인코딩·패킷화 ↔ 단말 디코딩·표시 분리 · DTO 4개로 `sourceFrameId · frameId · targetId · cropBox · captureTime · epoch · modelVersion · usedPacketBitmap` 연결 |
| 클래스 | C-0606 · 0608 · 0609 · 0610 재정의 (ID 유지) · 신규 C-0630~0638 · C-0704 TargetService 에 `ICropDeliveryService` 의존 · C-0619 에 크롭 표시 통계 |
| SD-X06 | `GRACE 대상 크롭 제공 — 서버 → 관제 운용자 단말` · 3장으로 나눠 그림 (`SD-X06_1~3.png`) · 26단계 · UC-0702·0705 중심, UC-0604·0607 연계 |

- 손실: `FULL` · `PARTIAL` · `NONE` — NONE 은 복원하지 않음 (마지막 영상 + 관측 시각·오래된 영상 표지)
- 대체: 디코더 미가용·호환 불일치·참조 복구 실패 → `CROP_BASELINE` (기존 크롭 제공)
- 600 ms · 1200 B · 32프레임 · 256 MiB 는 **실험 초기값** · 전송 기술 · 디코더 실행 환경은 **미검증**
- `그림/SD-X06.png` 는 이전(연구 경로) 그림 — 문서에서는 쓰지 않음
- 적용: `src/patch22.py <v2.1 입력.hwpx> <출력.hwpx>` (src 에서 `python render.py` · `python pd.py` · `python dd.py` 뒤 실행)

## 계층 규칙

`api`(Controller) → `services.*`(«interface» I… ← 구현 → 구성요소) → `storage.dao`(DAO · ERD 18개 테이블 1:1) → `storage.entity` · 계층 사이 `contracts`(DTO) · 서버·Pi 공유 규칙 `policies` · Pi 는 `gateway.*`
- 기존 클래스 88개는 ID·이름 유지 (Repository 3개만 DAO 로 이름 변경: MissionDAO · SearchPlanDAO · AlertDAO)
- 네트워크(HTTP·WSS·SRT·크롭 스트림)는 Controller · EventPublisher · CropStreamEndpoint · IGatewayLink(GatewayClient) · ServerReporter 뒤에 숨긴다

## 다시 만들기

`src/` — `python check.py` (정합성: 시퀀스 호출 메서드가 클래스에 있는지 · DTO/DAO 참조) → `python render.py` (그림) → `python pd.py` → `python build20.py` (HWPX)
