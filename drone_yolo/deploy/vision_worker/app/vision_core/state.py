"""
state.py — 상태 인지 (V4 · 2026-10-07) — 추적 → 자세 · 움직임 · 무동작 → 등급 · 점수

  연구 코드 state_pipeline.py + Q1 채택 설정 (obsidian 「추적 고치기 Q1」) 을 worker 로 옮긴 것
    · 추적  BoT-SORT (ultralytics 8.4.102) · 추적 문턱 0.15 (운용 탐지 임계와 같게) · GMC sparseOptFlow · ReID 끔 · track_buffer 30
            탐지는 worker 의 타일 탐지 결과를 그대로 넣는다 (프레임마다)
    · 1초마다 (촬영 시각 기준) — 배경 호모그래피 (사람 박스 가림 · ORB 3000 · 1280 폭으로 줄여 계산) 로 기체 움직임을 빼고
            발끝점 이동 ÷ 몸 높이 = 속도 (몸높이/초) · 3 초과 = ID 바뀜 → 그 추적 이력 비움 (--reset-jump)
            최근 3초 속도 중앙값 < 0.25 가 이어진 초 = 무동작 · ≥ 0.4 = 이동 중
    · 자세  점수 · 등급 = B0 (SARD+NOMAD) **지금 한 장** 확률 (Q1 V2·single) · 표시 = V3 (Archangel 실제 + UE) — posture.json
    · 등급  CHECK 관측 < 3초 · URGENT 누움 ≥ 0.7 & 무동작 ≥ 10초 · HIGH 누움+앉음 ≥ 0.7 & 무동작 ≥ 10초
            · CHECK 누움 0.3~0.7 & 멈춤 · NORMAL 나머지
            점수 = 0.5 × 누움 + 0.2 × 앉음 + 0.3 × min(무동작, 20)/20 (이동 중이면 무동작 항 0)
  원칙  후보를 거르지 않고 순위만 바꾼다 · 등급은 운영자 우선순위 보조 (UC-0705) · 확정 진단이 아님
  확정 안 된 탐지 (추적기는 새 사람을 두 번째 프레임에 확정) · 1초 갱신 전 새 추적 → 한 장 자세 상태 (tracked: false · 등급 CHECK)
  ⚠ 10-07 발견: 연구용 state_pipeline 의 model.track(persist=k>0) 은 ultralytics 8.4.102 에서 매 프레임 추적기를 새로 만든다
     (첫 호출의 persist=False 가 콜백에 고정) → 연구의 "추적 이력" 은 확신도 순번이었다 · 이 모듈은 추적기를 직접 이어 쓴다
  한계  시선 방향으로 누운 사람은 박스가 선 사람과 같아 보인다 (Archangel 57 % 서기로) → 누움 확률이 낮게 나올 수 있음
"""
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

SCHEMA = "state/0.1"
JUMP, STILL, MOVING = 3.0, 0.25, 0.4
TRACK_ARGS = dict(tracker_type="botsort", track_high_thresh=0.15, track_low_thresh=0.1, new_track_thresh=0.15, track_buffer=30,
                  match_thresh=0.8, fuse_score=True, gmc_method="sparseOptFlow", proximity_thresh=0.5, appearance_thresh=0.8,
                  with_reid=False, model="auto")
GRADES = {"URGENT": "🔴", "HIGH": "🟠", "CHECK": "❔", "NORMAL": "⚪"}


class PostureLR:
    """박스 2특징 로지스틱 (posture.json 의 한 모델) — sklearn 없이 numpy"""
    def __init__(self, m):
        self.mean = None if m["mean"] is None else np.array(m["mean"])
        self.scale = None if m["scale"] is None else np.array(m["scale"])
        self.W, self.b, self.name = np.array(m["coef"]), np.array(m["intercept"]), m.get("name")

    def proba(self, wh1080):
        wh = np.maximum(np.asarray(wh1080, float).reshape(-1, 2), 1.0)
        X = np.c_[np.log(wh[:, 1] / wh[:, 0]), np.log(wh.max(1))]
        if self.mean is not None:
            X = (X - self.mean) / self.scale
        s = X @ self.W.T + self.b
        s = np.exp(s - s.max(1, keepdims=True))
        return s / s.sum(1, keepdims=True)


def load_posture(path):
    js = json.loads(Path(path).read_text())
    assert js.get("classes") == ["lying", "sitting", "standing"], "posture.json 클래스 순서"
    return PostureLR(js["score"]), PostureLR(js["display"])


class _Dets:
    """ultralytics 추적기가 받는 Results 비슷한 것 (conf · xywh · cls · 불 색인)"""
    def __init__(self, xyxy, conf):
        self.xyxy, self.conf = np.asarray(xyxy, np.float32).reshape(-1, 4), np.asarray(conf, np.float32).reshape(-1)
        self.cls = np.zeros(len(self.conf), np.float32)
        w, h = self.xyxy[:, 2] - self.xyxy[:, 0], self.xyxy[:, 3] - self.xyxy[:, 1]
        self.xywh = np.c_[self.xyxy[:, 0] + w / 2, self.xyxy[:, 1] + h / 2, w, h]

    def __len__(self):
        return len(self.conf)

    def __getitem__(self, m):
        return _Dets(self.xyxy[m], self.conf[m])


def homography(ga, gb, boxes_a, boxes_b, orb, bf):
    """사람 박스를 가린 배경으로 a → b 호모그래피 (okutama_motion.homography 와 같음) · 실패하면 None"""
    masks = []
    for g, bx in ((ga, boxes_a), (gb, boxes_b)):
        m = np.full(g.shape, 255, np.uint8)
        for (x1, y1, x2, y2) in bx:
            cv2.rectangle(m, (int(x1) - 8, int(y1) - 8), (int(x2) + 8, int(y2) + 8), 0, -1)
        masks.append(m)
    ka, da = orb.detectAndCompute(ga, masks[0])
    kb, db = orb.detectAndCompute(gb, masks[1])
    if da is None or db is None:
        return None
    good = [x for x, y in (p for p in bf.knnMatch(da, db, k=2) if len(p) == 2) if x.distance < 0.75 * y.distance]
    if len(good) < 40:
        return None
    pa = np.float32([ka[x.queryIdx].pt for x in good]); pb = np.float32([kb[x.trainIdx].pt for x in good])
    H, inl = cv2.findHomography(pa, pb, cv2.RANSAC, 3.0)
    if H is None or inl.sum() < 30:
        return None
    return H


def grade(ly, si, still, obs, moving):
    score = 0.5 * ly + 0.2 * si + (0 if moving else 0.3 * min(still, 20) / 20)
    if obs < 3:
        g = "CHECK"
    elif ly >= 0.7 and still >= 10:
        g = "URGENT"
    elif ly + si >= 0.7 and still >= 10:
        g = "HIGH"
    elif 0.3 <= ly < 0.7 and not moving:
        g = "CHECK"
    else:
        g = "NORMAL"
    return g, round(float(score), 3)


class _Hist:
    __slots__ = ("seen", "still", "sp", "state", "epoch", "h")

    def __init__(self):
        self.seen, self.still, self.sp, self.state, self.epoch, self.h = 0, 0, [], None, 0, []   # h = 1초 표본 (t, 누움, 앉음, w, h) 최근 10개


class StateTracker:
    """임무 (영상 흐름) 하나의 추적 · 상태 — update 를 프레임마다 시각 순서로 부른다"""
    def __init__(self, posture_path, work_w=1280):
        from ultralytics.trackers.bot_sort import BOTSORT
        self.score_m, self.display_m = load_posture(posture_path)
        self.tracker = BOTSORT(SimpleNamespace(**TRACK_ARGS))
        self.orb, self.bf = cv2.ORB_create(3000), cv2.BFMatcher(cv2.NORM_HAMMING)
        self.work_w, self.hist = work_w, {}
        self.prev = None                                     # (t, 작은 회색 영상, {추적: 박스 (작은 영상 px)})
        self.next_tick = None
        self.stats = {"frames": 0, "ticks": 0, "homography_fail": 0, "id_jumps": 0}
        self.track_boxes = {}                                # 마지막 프레임: 탐지 번호 → 추적기 박스 (검증용)

    def update(self, t, img, boxes, confs):
        """한 프레임 → 탐지마다 (추적 id 또는 None, 상태 dict 또는 None)"""
        n = len(confs)
        out = [(None, None)] * n
        self.stats["frames"] += 1
        tr = self.tracker.update(_Dets(boxes, confs), img) if n else self.tracker.update(_Dets(np.zeros((0, 4)), []), img)
        cur = {}                                             # 추적 id → (탐지 번호, 박스)
        for row in tr:
            tid, idx = int(row[4]), int(row[7])
            if 0 <= idx < n:
                cur[tid] = (idx, tuple(float(v) for v in row[:4]))   # 추적기 박스 (state_pipeline 과 같음)
        if self.next_tick is None or t >= self.next_tick - 1e-3:   # ── 1초마다 상태 갱신 (state_pipeline 의 fr % FPS == 0) ──
            self._tick(t, img, cur)
            self.next_tick = (t if self.next_tick is None else self.next_tick) + 1.0
            if self.next_tick <= t:                               # 프레임이 끊겼으면 지금 기준으로 다시 맞춤
                self.next_tick = t + 1.0
        self.track_boxes = {idx: b for idx, b in cur.values()}
        for tid, (idx, _) in cur.items():
            h = self.hist.get(tid)
            out[idx] = (f"{tid}.{h.epoch}" if h else str(tid), h.state if h else None)
        loose = [j for j in range(n) if out[j][1] is None]      # 아직 확정 안 된 추적 · 갱신 전 새 추적 → 한 장 자세만 (거르지 않는다)
        if loose:
            H0 = img.shape[0]
            wh = np.array([[max(boxes[j][2] - boxes[j][0], 1) * 1080 / H0, max(boxes[j][3] - boxes[j][1], 1) * 1080 / H0] for j in loose])
            P, D = self.score_m.proba(wh), self.display_m.proba(wh)
            for k, j in enumerate(loose):
                g, score = grade(float(P[k, 0]), float(P[k, 1]), 0, 0, False)
                out[j] = (out[j][0], self._state(t, g, score, P[k], D[k], None, False, 0, 0, tracked=False))
        return out

    @staticmethod
    def _state(t, g, score, p, dp, med, moving, still, seen, tracked=True):
        return {"schema": SCHEMA, "grade": g, "score": score, "tracked": tracked,
                "posture": {"label": ("lying", "sitting", "standing")[int(np.argmax(dp))],
                            "p": {"lying": round(float(dp[0]), 3), "sitting": round(float(dp[1]), 3), "standing": round(float(dp[2]), 3)},
                            "model": "box-geometry V3"},
                "score_terms": {"lying": round(float(p[0]), 3), "sitting": round(float(p[1]), 3), "model": "box-geometry B0 (single frame)"},
                "motion": {"speed_bh_s": None if med is None else round(med, 3), "moving": moving, "still_s": still, "measured": med is not None},
                "observed_s": seen, "updated_at_s": round(float(t), 3),
                "note": "priority aid only — operator decides (UC-0705)"}

    def _tick(self, t, img, cur):
        self.stats["ticks"] += 1
        H0, W0 = img.shape[:2]
        k = self.work_w / W0
        small = cv2.cvtColor(cv2.resize(img, (self.work_w, int(round(H0 * k)))), cv2.COLOR_BGR2GRAY)
        sb = {tid: tuple(v * k for v in b) for tid, (_, b) in cur.items()}
        Hm = None
        if self.prev is not None and t - self.prev[0] <= 1.6:   # 바로 앞 초가 있을 때만 (끊겼으면 속도 없음)
            Hm = homography(self.prev[1], small, list(self.prev[2].values()), list(sb.values()), self.orb, self.bf)
            if Hm is None:
                self.stats["homography_fail"] += 1
        tids = list(cur)
        if tids:
            wh = np.array([[max(cur[i][1][2] - cur[i][1][0], 1) * 1080 / H0, max(cur[i][1][3] - cur[i][1][1], 1) * 1080 / H0] for i in tids])
            P, D = self.score_m.proba(wh), self.display_m.proba(wh)
        for j, tid in enumerate(tids):
            h = self.hist.setdefault(tid, _Hist())
            h.seen += 1
            b = sb[tid]
            if Hm is not None and tid in self.prev[2]:
                a = self.prev[2][tid]
                wa = cv2.perspectiveTransform(np.float32([[(a[0] + a[2]) / 2, a[3]]]).reshape(1, 1, 2), Hm).ravel()
                sp = float(np.linalg.norm(np.array([(b[0] + b[2]) / 2, b[3]]) - wa) / max(((a[3] - a[1]) + (b[3] - b[1])) / 2, 1))
                if sp <= JUMP:
                    h.sp.append(sp)
                else:                                        # ID 바뀜 — 앞사람 이력을 버리고 지금부터 새로
                    self.stats["id_jumps"] += 1
                    h.sp, h.seen, h.still, h.epoch, h.h = [], 1, 0, h.epoch + 1, []
            med = float(np.median(h.sp[-3:])) if h.sp else None
            moving = med is not None and med >= MOVING
            h.still = h.still + 1 if (med is not None and med < STILL) else 0
            ly, si = float(P[j, 0]), float(P[j, 1])
            h.h = (h.h + [(round(float(t), 3), round(ly, 4), round(si, 4), round(float(wh[j, 0]), 1), round(float(wh[j, 1]), 1))])[-10:]
            g, score = grade(ly, si, h.still, h.seen, moving)
            h.state = self._state(t, g, score, P[j], D[j], med, moving, h.still, h.seen)
        self.prev = (t, small, sb)
