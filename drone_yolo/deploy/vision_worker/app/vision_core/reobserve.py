"""
reobserve.py — 재관측 제안 (설계명세서 SD-0706 · REAL `POST /internal/v1/vision/reobservations` · 2026-10-09)

  증거가 모자란 후보를 다시 보자고 **제안**만 한다 — 서버는 PENDING 으로 저장하고 비행 명령으로 바꾸지 않는다 (dispatch_enabled=false)
  이유 (계약의 4가지)
    GEO_PENDING        확정 후보 (2회 이상) 인데 첫 관측 3초 뒤에도 좌표가 보류 — 다가가거나 화면 가운데에 두면 풀리는 사유만
                       (FOOT_AT_EDGE · RANGE_GT_200 · RAY_ABOVE_HORIZON · TILT_GT_6 · RATE_GT_1_5 · NO_TERRAIN_HIT) · NO_POSE 는 다시 봐도 안 풀려 제외
    LOW_EVIDENCE       한 번만 잡혔고 (확신도 < 0.4) 2초 넘게 다시 안 보임 — 놓쳤을 수 있는 사람
    STATE_UNCERTAIN    추적 3초 이상 · 멈춰 있음 · 누움 확률 0.3~0.7 (등급 CHECK) — 정지 비행으로 자세 · 움직임을 더 보자
    COLOR_UNDETERMINED 외형 조건이 있는데 3회 이상 보고도 상의 색 판정 불가
  제한  후보마다 같은 이유 1번 · 후보마다 최대 MAX_PER_CANDIDATE · 임무마다 분당 MAX_PER_MIN (과다 제안 방지)
  desired_view = 권고 (자세한 비행 판단은 경로 계획 · 운영자) — 마운트 45° · 권장 고도 16~20 m · 정지 시간
"""
import uuid

FIXABLE = {"FOOT_AT_EDGE", "RANGE_GT_200", "RAY_ABOVE_HORIZON", "TILT_GT_6", "RATE_GT_1_5", "NO_TERRAIN_HIT"}
MAX_PER_CANDIDATE, MAX_PER_MIN = 3, 6
VIEW = {
    "GEO_PENDING": {"action": "APPROACH_CENTER", "alt_agl_m": [16, 20], "gimbal_pitch_deg": -45, "dwell_s": 3,
                    "note": "keep target in lower-middle of frame (foot ray inside image, range < 60 m)"},
    "LOW_EVIDENCE": {"action": "REVISIT", "alt_agl_m": [16, 20], "gimbal_pitch_deg": -45, "dwell_s": 3,
                     "note": "pass over last seen spot again (3 s observation rule)"},
    "STATE_UNCERTAIN": {"action": "HOVER", "alt_agl_m": [16, 20], "gimbal_pitch_deg": -45, "dwell_s": 10,
                        "note": "hold to measure motion and posture (URGENT needs >=10 s still)"},
    "COLOR_UNDETERMINED": {"action": "CLOSER", "alt_agl_m": [12, 16], "gimbal_pitch_deg": -45, "dwell_s": 3,
                           "note": "closer view of upper body for clothing color"},
}


class ReobservePlanner:
    def __init__(self):
        self.sent = {}                 # 후보 uuid → {이유}
        self.window = {}               # 임무 → [보낸 시각]

    def _ok(self, mid, cid, reason, t):
        s = self.sent.setdefault(cid, set())
        if reason in s or len(s) >= MAX_PER_CANDIDATE:
            return False
        w = [x for x in self.window.get(mid, []) if t - x < 60]
        if len(w) >= MAX_PER_MIN:
            self.window[mid] = w
            return False
        s.add(reason); self.window[mid] = w + [t]
        return True

    def _body(self, cid, reason, cand, extra):
        hyp = {"n_hits": len(cand.hits), "best_conf": round(cand.best_conf, 3), "first_seen_s": round(cand.first_t, 3),
               "last_seen_s": round(cand.last_t, 3), "last_box": [round(v, 1) for v in cand.last_box] if cand.last_box else None} | extra
        view = dict(VIEW[reason])
        if cand.lat is not None:
            view["target"] = {"lat": round(cand.lat, 7), "lon": round(cand.lon, 7), "sigma_m": round(cand.sigma_m or 0, 2)}
        return {"request_id": str(uuid.uuid4()), "candidate_id": cid, "reason": reason, "hypothesis": hyp, "desired_view": view}

    def on_observation(self, mid, t, cand, cid, geo, state, appearance):
        """이번 프레임에서 관측된 후보 하나 → 제안 목록"""
        out = []
        codes = set((geo or {}).get("reasons") or [])
        if (geo or {}).get("status") != "VALID" and cand.confirmed and t - cand.first_t >= 3 and codes & FIXABLE \
                and self._ok(mid, cid, "GEO_PENDING", t):
            out.append(self._body(cid, "GEO_PENDING", cand, {"geo_reasons": sorted(codes)}))
        if state and state.get("tracked") and state.get("observed_s", 0) >= 3 and not state["motion"].get("moving") \
                and 0.3 <= state["score_terms"]["lying"] < 0.7 and self._ok(mid, cid, "STATE_UNCERTAIN", t):
            out.append(self._body(cid, "STATE_UNCERTAIN", cand, {"lying_p": state["score_terms"]["lying"], "grade": state["grade"],
                                                                 "still_s": state["motion"].get("still_s")}))
        m = (appearance or {}).get("match")
        if m and m.get("verdict") == "UNDETERMINED" and len(cand.hits) >= 3 and self._ok(mid, cid, "COLOR_UNDETERMINED", t):
            out.append(self._body(cid, "COLOR_UNDETERMINED", cand, {"query": m.get("query")}))
        return out

    def sweep(self, mid, t, registry, uid):
        """프레임마다 — 한 번만 잡히고 사라진 후보 (LOW_EVIDENCE)"""
        out = []
        for c in registry.items:
            if len(c.hits) == 1 and c.best_conf < 0.4 and t - c.last_t > 2.0:
                cid = uid.get((mid, c.cid))
                if cid and self._ok(mid, cid, "LOW_EVIDENCE", t):
                    out.append(self._body(cid, "LOW_EVIDENCE", c, {}))
        return out
