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

## 계층 규칙

`api`(Controller) → `services.*`(«interface» I… ← 구현 → 구성요소) → `storage.dao`(DAO · ERD 18개 테이블 1:1) → `storage.entity` · 계층 사이 `contracts`(DTO) · 서버·Pi 공유 규칙 `policies` · Pi 는 `gateway.*`
- 기존 클래스 88개는 ID·이름 유지 (Repository 3개만 DAO 로 이름 변경: MissionDAO · SearchPlanDAO · AlertDAO)
- 네트워크(HTTP·WSS·SRT)는 Controller · EventPublisher · IGatewayLink(GatewayClient) · ServerReporter 뒤에 숨긴다

## 다시 만들기

`src/` — `python check.py` (정합성: 시퀀스 호출 메서드가 클래스에 있는지 · DTO/DAO 참조) → `python render.py` (그림) → `python pd.py` → `python build20.py` (HWPX)
