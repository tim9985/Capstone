# 비전 담당자 전달 — 백엔드 연동 규격 v1.0

- 작성일: 2026-10-06
- 대상 원문: `비전 출력 규격 - 후보 JSON (초안 v0.1).md`
- 범위: 학교 서버 REAL 데이터 수신 API와 비전 워커 연결. 모델 학습·탐지·좌표 산정 알고리즘은 비전 담당자 범위.
- 계약 파일: [openapi-real.json](openapi-real.json). 이 문서의 구현 설명은 검증 기록과 함께 확인한다.

## 1. 담당자에게 전달할 핵심 내용

제공한 초안은 비전 워커가 백엔드에 후보·프레임·재관측 요청을 전달하기 위한 문서가 맞습니다. 다만 DB 칸과 이름을 맞췄다는 이유만으로 현재 API에 바로 입력할 수는 없습니다. 임무·프레임·모델 ID, 데이터 저장 순서, 인증, 재전송 처리가 함께 맞아야 합니다.

연결은 **FastAPI + 워커 로컬 JSONL**로 정했습니다. 백엔드가 워커 파일을 직접 읽지 않습니다. 워커는 학교 서버의 별도 비전 컨테이너에서 입력을 API로 조회하고 결과를 API로 전송합니다. JSONL은 재전송을 위한 로컬 outbox이며 백엔드 DB·영상 원본 저장소를 공유하지 않습니다.

미확정 후보도 모두 저장·조회합니다. `auto_confirmed`는 모델의 자동 판정이고, 운영자 판단은 `judgement`로 별도 관리합니다. 워커는 운영자 판단과 DB `row_version`을 덮어쓰지 않습니다. 좌표가 보류된 후보도 목록에서 사라지지 않습니다.

## 2. 초안 검토 결과

| 초안 | 현재 계약·보완 사항 |
|---|---|
| 공유 JSONL 꼬리 읽기 | FastAPI 전송. JSONL은 워커 전용 `/scratch/outbox.jsonl` |
| 예시 `mission_id="m001"` | 백엔드가 등록한 UUID 사용. 임무마다 임의 문자열 생성 금지 |
| 워커 `rev` → DB `row_version` | `worker_revision`과 DB `row_version` 분리. 현재 프레임당 후보 관측은 후보별 증가 판을 요구 |
| 파일별 `seq` int | API에서는 정밀도 보존을 위해 64비트 카운터를 문자열로 전송: `"1"` |
| `candidate` 전체 상태 사건 | 현재 API는 프레임 관측 묶음 중심. `CREATED/LOST/GEO_CHANGED` 독립 사건은 아직 별도 계약이 없음 |
| 자동 `CONFIRMED` | `auto_confirmed`로 보존. 운영자 `CONFIRMED/REJECTED/UNDECIDED` 판단과 다름 |
| `SIDECAR` → 촬영 시각 검증됨 | 출처만으로 검증 판정하지 않음. 시계·자세 증거를 따로 검증해야 함 |
| `VALID` 좌표 | 워커 주장은 원본으로 보존하되 현재 공식 좌표는 `PENDING/PHYSICAL_VALIDATION_REQUIRED` |
| PENDING에 이전 VALID 위경도 포함 | 현재 위치와 마지막 유효 위치를 분리해야 함. 현재 계약의 PENDING/INVALID는 위경도 null |
| `ellipse.along_m/cross_m/bearing_deg` | `error_ellipse.major_m/minor_m/azimuth_deg/k`로 정규화. 큰 축 ≥ 작은 축, 방위 기준 명시 |
| 예시 큰 축 2.1m, `sigma_m=1.4` | 초안의 “큰 축 1σ” 정의와 불일치. 수정 또는 다른 통계량이면 이름 변경 |
| 위경도 소수 7자리 ≈ 1cm | 표현 정밀도와 측정 정확도는 다름. 실제 오차는 타원·시각·자세·보정 품질로 표시 |
| bbox 배열 | API에서는 이름 있는 객체. 원본 해상도 기준, 0≤x1<x2≤width / 0≤y1<y2≤height |
| 이미지 상대 경로 | 먼저 미디어 API 업로드 후 `snapshot_asset_id`·`clip_asset_id` 사용 |
| 모델 이름 | 등록된 `model_config_id` UUID와 SHA-256·pipeline version 연결 |
| `state` CRITICAL 등 | `tracking.reported_state`에 모델 주장으로 보존. 후보 삭제·비행 결정에 직접 사용하지 않음 |
| SYSTEM 재관측 | 전용 비전 API로 저장. `PENDING` 제안이며 실제 이동 명령이 아님 |

`track_id`는 영상 추적기 ID이고 `candidate_id`는 임무 내 후보 ID이다. 추적기 재시작 때 같은 숫자 track ID가 재사용될 수 있으므로 stream/session/epoch와 함께 해석해야 한다. 모델 confidence를 실제 사람 존재 확률이나 상태의 확정 진단으로 표현하지 않는다.

## 3. 인증·실행 위치

| 항목 | 값 |
|---|---|
| 학교 서버 sandbox | `/home/se/sandboxes/drone-real-20261006` |
| 워커 서비스 | `vision`, UID 2202, `network_mode: none` |
| 백엔드 논리 주소 | `https://backend.internal` |
| 전송 소켓 | `/run/backend-socket/service.sock` |
| 인증서·CA·토큰 | `/run/identity/client.crt`, `client.key`, `ca.crt`, `token` |
| 워커 쓰기 영역 | `/scratch` — outbox·체크포인트·임시 입력 |
| 모델 위치 | `/models` — 읽기 전용. 실제 모델 미배치 |
| DB 접근 | 없음. 백엔드 API로만 교환 |

TLS가 검증한 클라이언트 인증서와 서비스 Bearer 토큰이 함께 일치해야 한다. HTTP 헤더에 인증서 지문을 써서 인증을 대신하지 않는다. 비밀 파일은 문서·Git·JSONL에 넣지 않는다. 현재 인증서는 개발용으로 발급됐으므로 만료 전 갱신이 필요하다.

기존 `app.common.client()` 사용 예:

```python
from app.common import client, text

api = client('/run/backend-socket/service.sock', 'backend.internal')
headers = {'Authorization': 'Bearer ' + text('token')}
frames = api.get('/internal/v1/frames', headers=headers)
frames.raise_for_status()
```

DB 자격증명·원본 저장소·CA 개인키는 워커에 제공하지 않는다. 별도 컨테이너는 파일·서비스 경계를 만들지만 학교 서버의 CPU/GPU/RAM 자원을 독점시키지는 않는다. 현재 워커 연결기에는 GPU를 연결하지 않았고 모델 추론은 시작하지 않았다.

## 4. API 호출 순서

1. 관제 운영자가 REAL 임무를 등록한다. 기체 게이트웨이가 인증된 세션을 만들고 상태 데이터를 전송한다.
2. 게이트웨이가 영상·sidecar·프레임을 미디어 API로 업로드한다. 백엔드는 크기·SHA-256·저장 완료를 확인한다.
3. 게이트웨이가 `POST /device/v1/frames`로 프레임 ID·임무·입력 asset ID·시각 출처·해상도를 등록한다.
4. 비전 워커가 `GET /internal/v1/frames`로 아직 분석되지 않은 프레임을 조회한다. 현재 단일 워커 방식이며 다중 워커 작업 임대는 미구현이다.
5. `GET /internal/v1/frames/{frame_id}/input`으로 입력을 내려받아 응답의 `X-Content-SHA256`을 확인한다.
6. 필요하면 crop/clip을 전용 비전 미디어 API로 업로드한다.
7. `POST /internal/v1/vision/results`로 프레임 관측을 묶어서 전송한다. 백엔드는 수신 영수증·원본·후보·관측·분석 이력을 한 트랜잭션으로 저장한다.
8. 재관측은 `POST /internal/v1/vision/reobservations`에 제안한다. 운영자 조회·WebSocket 이벤트로 반영되며 경로 생성·기체 이동은 활성화되지 않는다.

| 용도 | API |
|---|---|
| 대기 프레임 | `GET /internal/v1/frames` |
| 입력 다운로드 | `GET /internal/v1/frames/{fid}/input` |
| 결과 전달 | `POST /internal/v1/vision/results` |
| 재관측 제안 | `POST /internal/v1/vision/reobservations` |
| crop/clip 예약 | `POST /internal/v1/vision/media-bundles` |
| 파일 전송 | `PUT /internal/v1/vision/media-assets/{aid}/content` |
| 저장 완료 | `POST /internal/v1/vision/media-bundles/{bid}/finalize` |
| 예약 상태/취소 | `GET /internal/v1/vision/media-bundles/{bid}` / `POST .../{bid}/cancel` |

미디어는 먼저 파일별 `asset_id`, `role`, `byte_size`, `sha256`을 예약하고 바이너리 전송 후 finalize한다. crop은 `FRAME`, clip은 `VIDEO`, 결과 파일은 `RESULT`로 전달한다. 비전 미디어 묶음에는 원본 VIDEO·SIDECAR 조합을 강제하지 않는다. 다른 생산자가 업로드한 파일은 워커가 변경할 수 없다. 경로 문자열을 이용한 호스트 파일 읽기는 허용하지 않는다.

## 5. 결과 JSON 예시

아래 UUID는 설명용이다. 실제 등록된 임무·프레임·모델·미디어 ID로 바꿔야 한다.

```json
{
  "schema_version": "vision-ingest/1.0",
  "message_id": "f34fbcd2-5758-47bb-a01c-8a02b921f58b",
  "producer_session_id": "8f913fce-f21b-4cff-a0e1-7a5b18fb652e",
  "sequence": "1",
  "mission_id": "34821ebc-68d1-4cc9-93e8-535b21bdcf2f",
  "frame_id": "d20fd6a9-5f14-4a84-906f-d31dcfc3b6ca",
  "model_config_id": "3fd65688-4cb6-4a56-bb9d-53af576162f7",
  "emitted_at": "2026-10-06T05:12:33.120Z",
  "analysis_state": "DONE",
  "observations": [{
    "observation_id": "32db1392-7acb-4535-a6bc-724768494012",
    "candidate_id": "6f1c2a1e-8a5b-4d6e-9a43-2f0c7d3b9e10",
    "detection_index": 0,
    "bbox": {"x1": 1012.4, "y1": 633.0, "x2": 1040.8, "y2": 701.2},
    "confidence": 0.48,
    "kind": "DETECT",
    "track_id": null,
    "worker_revision": "3",
    "auto_confirmed": true,
    "geo": {"status": "PENDING", "lat": null, "lon": null, "reasons": ["NO_POSE"], "method": "FOOT_RAY_DEM"},
    "appearance": {"upper": {"status": "OK", "top": [{"color": "red", "p": 0.62}]}},
    "state": null,
    "snapshot_asset_id": null,
    "clip_asset_id": null
  }]
}
```

`DONE` + `observations: []`는 정상적인 무탐지 결과다. `SKIPPED`/`DROPPED`는 `reason`이 필수이며 관측 배열은 비어 있어야 한다. 현재 프레임은 1회 결과만 반영한다. 재분석 판·다중 워커 임대는 후속 계약으로 추가하고, 기존 완료 결과를 덮어쓰지 않는다.

워커가 `geo.status=VALID`를 보내려면 위경도와 `error_ellipse`를 함께 보내야 한다. 수신 계약 검증은 형식 검증이며 실제 정확도 검증은 아니다. 현재 공식 `person_candidate.latest_position`은 비워 두고 워커 좌표는 `tracking.geo_claim`에 보존한다. 측정 시각·카메라 보정·기체 자세·좌표계의 검증 절차가 마련되기 전 관제 지도에서 정확한 위치로 표시해서는 안 된다.

## 6. 저장·재전송 규칙

결과마다 UUID `message_id`를 만든다. 같은 메시지를 재전송할 때는 ID와 본문을 그대로 사용한다. 같은 ID·같은 본문은 성공 응답의 `duplicate=true`, 같은 ID·다른 본문은 409다. 생산자·세션·종류·sequence도 중복 충돌을 검사한다. HTTP 성공 응답은 DB 커밋 후 반환하며 타임아웃 시 새 ID로 바꾸지 않고 기존 ID를 재전송한다.

JSONL 한 줄 형식:

```json
{"type":"result","body":{"schema_version":"vision-ingest/1.0","message_id":"실제 UUID","...":"위 결과 본문"}}
```

실제로는 `body`에 완전한 결과 본문을 넣는다. 재관측 메시지는 `type="reobservation"`과 해당 요청 본문을 사용한다. UTF-8, 한 줄 최대 64 KiB, 마지막 줄은 `\n`으로 마무리한다. 연결기는 성공 응답을 받은 줄까지만 `/scratch/ack.json`에 기록한다. 미완성 줄과 전송 실패 줄은 다음 실행에 다시 처리한다.

outbox는 한 생산자가 append한다. 체크포인트가 남아 있는 파일을 줄이거나 교체하지 않는다. 현재 연결기는 오류가 난 첫 메시지에서 대기하므로 잘못된 계약의 409/422는 담당자가 확인·격리해야 한다. 자동 로그 회전·병렬 전송은 미구현이며 필요 시 별도 계약으로 추가한다. 로컬 디스크 쓰기가 보장되기 전 결과를 폐기하지 않는다.

## 7. DB 저장 위치와 소유권

| 데이터 | 저장 위치 | 소유권 |
|---|---|---|
| 원본 프레임·시각·자세 증거 | `video_frame` | 게이트웨이 입력, 백엔드 수신 판정 |
| 결과 원문·해시·생산자 | `ingest_receipt` | 변경 불가 수신 기록 |
| 프레임별 탐지 | `detection_observation` | 변경 불가 관측 |
| 후보 요약 | `person_candidate` | 백엔드가 관리 |
| 자동 확정·모델 상태·좌표 주장 | `person_candidate.tracking` | 워커 주장으로 표시 |
| 운영자 판단 | `person_candidate.judgement` | 운영자만 변경 |
| 이미지·clip | `media_asset` + 백엔드 파일 저장소 | API로 업로드·조회 |
| 재관측 제안 | `reobservation_request` | SYSTEM 제안, 실행 결정 별도 |
| 분석 시작·완료/실패 | `mission_event` | 관제 이벤트·감사 이력 |

관제는 `GET /api/v1/candidates/{cid}/observations`로 탐지 confidence·bbox·원본 프레임·모델·좌표 주장을 조회한다. 재관측 상태와 텔레메트리 이력도 임무별 API에서 읽는다. 현재 조회 이력은 최대 1000건이며 장기 대용량 이력의 페이지 계약은 후속 보완 대상이다.

## 8. 담당자와 후속 합의할 사항

자동 확정 기준 2장/3초 또는 유효 관측 3회, 관측 묶음의 시간 기준, LOST와 재등장, PENDING의 마지막 유효 좌표 표시, 상태 모델 규격, 색상 판정 규격, sidecar·시계 동기화 증거, 좌표 검증 책임자, 모델 로딩/건강 확인, 작업 재분석, 운영자 판단 피드백 규격을 합의해야 한다. 이 문서는 초안의 임계값을 임의로 승인하지 않는다.

기존 v0 워커가 이 계약을 이미 구현한 것으로 간주하지 않는다. 담당자가 변환기를 연결하고 실제 모델·입력 영상으로 왕복 검증해야 한다. 현재 서버 연결기는 모델을 로딩하지 않으며 실제 추론·실기체 촬영 검증은 `NOT_RUN`이다.
