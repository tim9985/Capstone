# SDD v2.0 — 4장 객체지향설계 전면 재작성 (10-01)

> 9.30 설계 발표 피드백 2번 ("설계는 객체지향 · DTO 패키지 · DAO · Service 인터페이스 · 패키지 간 호출은 인터페이스 · HTTP 를 관계로 그리지 않음") 반영
> HWPX: `3. 설계명세서_…_v2.0.hwpx` (v1.4 (1) 에서 **4장만 교체** · 1·2·3장 그대로 · git 밖)

## 목차 (학교 양식 순서)

| 절 | 내용 |
|---|---|
| 4.1 Deployment Diagram | 기존 DD-01 유지 (그림만 무채색) |
| 4.2 Class Diagram | 4.2.1 패키지 다이어그램 PD-01 + 패키지 표 · 4.2.2~4.2.13 관리단위 12개 CD-01~12 (20장) |
| 4.3 Sequence Diagram | 관리단위별 SD 82개 (UC 73 + 통합 9) — 표 1개(유스케이스·시작·선행·사후·예외·경계) + 그림 + 처리표 |
| 4.4 Design Classes | 클래스 263개 — Controller·인터페이스·구현·구성요소·DAO 는 클래스마다 표, DTO·Entity 는 단위별 한 표 |

## 계층 규칙

`api`(Controller) → `services.*`(«interface» I… ← 구현 → 구성요소) → `storage.dao`(DAO · ERD 18개 테이블 1:1) → `storage.entity` · 계층 사이 `contracts`(DTO) · 서버·Pi 공유 규칙 `policies` · Pi 는 `gateway.*`
- 기존 클래스 88개는 ID·이름 유지 (Repository 3개만 DAO 로 이름 변경: MissionDAO · SearchPlanDAO · AlertDAO)
- 네트워크(HTTP·WSS·SRT)는 Controller · EventPublisher · IGatewayLink(GatewayClient) · ServerReporter 뒤에 숨긴다

## 다시 만들기

`src/` — `python check.py` (정합성: 시퀀스 호출 메서드가 클래스에 있는지 · DTO/DAO 참조) → `python render.py` (그림) → `python pd.py` → `python build20.py` (HWPX)
