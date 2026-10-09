"""C2 — 재관측 제안 규칙 단위 시험 (10-09 · 판정 기준은 _학습 큐 「10-09 C」) · 실행: python tests_reobserve.py"""
import sys, uuid
sys.path.insert(0, ".")
from app.vision_core.registry import Candidate, CandidateRegistry
from app.vision_core.reobserve import ReobservePlanner, MAX_PER_CANDIDATE, MAX_PER_MIN

ok = []
def check(name, cond):
    ok.append(cond); print(("PASS " if cond else "FAIL ") + name)

def cand(hits, conf=0.8, confirmed=True, first=0.0):
    c = Candidate(cid="m-C1", first_t=first, last_t=hits[-1]); c.hits = list(hits); c.best_conf = conf; c.confirmed = confirmed; c.last_box = (10, 10, 30, 60)
    return c

ST_UNC = {"tracked": True, "observed_s": 5, "grade": "CHECK", "motion": {"moving": False, "still_s": 4}, "score_terms": {"lying": 0.5}}
ST_OK = {"tracked": True, "observed_s": 5, "grade": "NORMAL", "motion": {"moving": True, "still_s": 0}, "score_terms": {"lying": 0.1}}
P = ReobservePlanner(); cid = str(uuid.uuid4())
c = cand([0, 1, 2, 3, 4])
r = P.on_observation("m", 4.0, c, cid, {"status": "PENDING", "reasons": ["FOOT_AT_EDGE"]}, ST_UNC,
                     {"match": {"verdict": "UNDETERMINED", "query": ["red"]}})
check("GEO_PENDING · STATE_UNCERTAIN · COLOR_UNDETERMINED 한 번에 3개", sorted(x["reason"] for x in r) == ["COLOR_UNDETERMINED", "GEO_PENDING", "STATE_UNCERTAIN"])
r2 = P.on_observation("m", 5.0, c, cid, {"status": "PENDING", "reasons": ["FOOT_AT_EDGE"]}, ST_UNC, {"match": {"verdict": "UNDETERMINED", "query": ["red"]}})
check("같은 이유 반복 안 함", r2 == [])
check(f"후보당 최대 {MAX_PER_CANDIDATE}", len(P.sent[cid]) <= MAX_PER_CANDIDATE)
P2 = ReobservePlanner()
r = P2.on_observation("m", 4.0, cand([0, 1, 2, 3, 4]), str(uuid.uuid4()), {"status": "PENDING", "reasons": ["NO_POSE"]}, ST_OK, {})
check("NO_POSE 만이면 GEO_PENDING 안 냄", r == [])
r = P2.on_observation("m", 1.0, cand([0, 1]), str(uuid.uuid4()), {"status": "PENDING", "reasons": ["FOOT_AT_EDGE"]}, ST_OK, {})
check("첫 관측 3초 전엔 GEO_PENDING 안 냄", r == [])
r = P2.on_observation("m", 4.0, cand([4], confirmed=False, first=4.0), str(uuid.uuid4()), {"status": "PENDING", "reasons": ["FOOT_AT_EDGE"]}, ST_OK, {})
check("확정 안 된 후보엔 GEO_PENDING 안 냄", r == [])
r = P2.on_observation("m", 4.0, cand([0, 1, 2, 3, 4]), str(uuid.uuid4()), {"status": "VALID", "reasons": []}, ST_OK, {"match": {"verdict": "MATCH"}})
check("좌표 VALID · 상태 확실 · 색 일치면 제안 0", r == [])
# LOW_EVIDENCE
P3 = ReobservePlanner(); reg = CandidateRegistry("m"); low = reg._new(0.0); low.hits = [0.0]; low.best_conf = 0.2; low.last_t = 0.0; low.last_box = (1, 1, 5, 9)
hi = reg._new(0.0); hi.hits = [0.0]; hi.best_conf = 0.9; hi.last_t = 0.0
uid = {("m", low.cid): str(uuid.uuid4()), ("m", hi.cid): str(uuid.uuid4())}
check("LOW_EVIDENCE — 2초 안이면 아직 안 냄", P3.sweep("m", 1.5, reg, uid) == [])
r = P3.sweep("m", 2.5, reg, uid)
check("LOW_EVIDENCE — 확신도 낮은 1회 후보만 1번", len(r) == 1 and r[0]["reason"] == "LOW_EVIDENCE" and r[0]["candidate_id"] == uid[("m", low.cid)])
check("LOW_EVIDENCE — 다시 안 냄", P3.sweep("m", 3.0, reg, uid) == [])
# 분당 상한
P4 = ReobservePlanner(); n = 0
for i in range(20):
    n += len(P4.on_observation("m", 10.0 + i, cand([0, 1, 2, 3, 4]), str(uuid.uuid4()), {"status": "PENDING", "reasons": ["RANGE_GT_200"]}, ST_OK, {}))
check(f"임무당 분당 ≤ {MAX_PER_MIN} (20초 동안 후보 20개)", n == MAX_PER_MIN)
n2 = len(P4.on_observation("m", 80.0, cand([0, 1, 2, 3, 4]), str(uuid.uuid4()), {"status": "PENDING", "reasons": ["RANGE_GT_200"]}, ST_OK, {}))
check("1분 지나면 다시 냄", n2 == 1)
# STATE_UNCERTAIN — 방위를 90° 바꿔 다시 보기 (10-09)
P5 = ReobservePlanner()
r = P5.on_observation("m", 4.0, cand([0, 1, 2, 3, 4]), str(uuid.uuid4()), {"status": "VALID", "reasons": []}, ST_UNC, {}, view_bearing=30.0)
v = r[0]["desired_view"] if r else {}
check("STATE_UNCERTAIN → REVIEW_BEARING · 지금 30° → 120° · 300°", len(r) == 1 and v.get("action") == "REVIEW_BEARING"
      and v.get("current_bearing_deg") == 30.0 and v.get("desired_bearings_deg") == [120.0, 300.0] and v.get("dwell_s") == 10)
r = ReobservePlanner().on_observation("m", 4.0, cand([0, 1, 2, 3, 4]), str(uuid.uuid4()), {"status": "VALID", "reasons": []}, ST_UNC, {})
check("방위 모르면 (자세 없음) 바꿀 방위 None · 요청은 냄", len(r) == 1 and r[0]["desired_view"]["desired_bearings_deg"] is None)
r = ReobservePlanner().on_observation("m", 4.0, cand([0, 1, 2, 3, 4]), str(uuid.uuid4()), {"status": "VALID", "reasons": []}, ST_UNC, {}, view_bearing=350.0)
check("방위 360° 넘김 (350° → 80° · 260°)", r[0]["desired_view"]["desired_bearings_deg"] == [80.0, 260.0])
# 방위 계산 (geo.bearing) — 기수 동쪽 (90°) · 마운트 45°
import math
from app.vision_core.geo import GeoResolver, Telemetry
G = GeoResolver(); tel = Telemetry(36.0, 128.0, 20.0, yaw=math.radians(90), gimbal_pitch_deg=-45.0, stabilized=True)
b_mid = G.bearing((950, 520, 970, 560), tel); b_right = G.bearing((1900, 520, 1920, 560), tel)
f = G.cam.f; lat = (1910 - 960) / f; dn = (560 - 540) / f                 # 카메라 광선 (앞 1 · 오른쪽 lat · 아래 dn) 을 45° 숙이면
want = 90 + math.degrees(math.atan2(lat, math.cos(math.radians(45)) - dn * math.sin(math.radians(45))))   # 수평 성분의 방위 (해석값)
check(f"방위: 화면 가운데 = 기수 ({b_mid:.1f}°) · 오른쪽 끝 {b_right:.1f}° = 해석값 {want:.1f}° (45° 숙이면 가로 각이 벌어진다)",
      abs(b_mid - 90) < 1 and abs(b_right - want) < 0.5)
# 계약 형식 (Reobserve 스키마)
import json, jsonschema
spec = json.load(open("schema/openapi-vision.json"))
body = P.on_observation("m", 9.0, cand([0, 1, 2, 3, 4]), str(uuid.uuid4()), {"status": "PENDING", "reasons": ["TILT_GT_6"]}, ST_OK, {})[0]
jsonschema.validate(body, {"$ref": "#/components/schemas/Reobserve", "components": spec["components"]})
body_su = P5.on_observation("m", 9.0, cand([0, 1, 2, 3, 4]), str(uuid.uuid4()), {"status": "VALID", "reasons": []}, ST_UNC, {}, view_bearing=10.0)[0]
jsonschema.validate(body_su, {"$ref": "#/components/schemas/Reobserve", "components": spec["components"]})
check("Reobserve 스키마 통과 (GEO_PENDING · STATE_UNCERTAIN)", True)
print(f"\n{sum(ok)}/{len(ok)} 통과")
sys.exit(0 if all(ok) else 1)
