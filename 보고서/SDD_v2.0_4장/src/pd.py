"""pd.py — PD-01 패키지 다이어그램 (무채색)"""
from mono import Canvas, INK, LINE, MID, head, PKG_TAB
from matplotlib.patches import Rectangle, Polygon

W = 2000; fs = 27


def pkg(c, x, y, w, h, name, desc, bold=True, fill="#ffffff", dashed=False):
    tw_ = c.tw(name, fs * 0.82, bold=True) + fs * 1.0
    c.ax.add_patch(Rectangle((x, y), min(tw_, w * 0.95), fs * 1.2, fc=PKG_TAB, ec=LINE, lw=1.6, zorder=3))
    c.ax.add_patch(Rectangle((x, y + fs * 1.2), w, h - fs * 1.2, fc=fill, ec=LINE, lw=1.8, zorder=3, ls=(0, (6, 4)) if dashed else "-"))
    c.text(x + fs * 0.4, y + fs * 0.62, name, fs * 0.82, bold=True, zorder=4)
    yy = y + fs * 1.2 + (h - fs * 1.2) / 2
    lines = desc.split("\n")
    for i, l in enumerate(lines):
        c.text(x + w / 2, yy + (i - (len(lines) - 1) / 2) * fs * 1.0, l, fs * 0.72, ha="center", color=MID, zorder=4)
    return (x, y, w, h)


def frame(c, x, y, w, h, name):
    c.ax.add_patch(Rectangle((x, y), w, h, fc="#f7f7f7", ec=MID, lw=2.0, zorder=1))
    c.text(x + fs * 0.5, y + fs * 0.75, name, fs * 0.95, bold=True, zorder=2)


def arrow(c, pts, lab="", dashed=True, lab_at=0.5, dx=0, dy=-14):
    xs, ys = zip(*pts)
    c.ax.plot(xs, ys, color=LINE, lw=2.0, ls=(0, (7, 5)) if dashed else "-", zorder=5)
    head(c.ax, pts[-2], pts[-1], "dep", fs, LINE)
    if lab:
        (x0, y0), (x1, y1) = max(zip(pts[:-1], pts[1:]), key=lambda s: abs(s[1][0] - s[0][0]) + abs(s[1][1] - s[0][1]))
        c.text(x0 + (x1 - x0) * lab_at + dx, y0 + (y1 - y0) * lab_at + dy, lab, fs * 0.72, ha="center", color=MID, zorder=6,
               bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))


def render(path):
    H = 2010
    c = Canvas(W, H)
    # ── 중앙 서버 ──
    frame(c, 10, 10, 1530, 1700, "중앙 서버 (FastAPI · PostgreSQL/PostGIS · RTX 4080 SUPER)")
    api = pkg(c, 40, 70, 1470, 150, "api", "«controller» Auth · Control · Drone · Map · Mission · Plan · Video · Candidate · MissionMap · Command\n· Safety · Alert · History   «boundary» EventPublisher (WebSocket) · GatewayController (Pi 보고 WSS) · CropStreamEndpoint (크롭 패킷·피드백)")
    c.ax.add_patch(Rectangle((40, 270), 1470, 760, fc="#fbfbfb", ec=LINE, lw=1.8, zorder=2))
    c.text(60, 295, "services   — 패키지마다 «interface» I… + 구현 + 구성요소", fs * 0.85, bold=True, zorder=3)
    names = [("services.auth", "사용자·제어권"), ("services.vehicle", "기체·게이트웨이 상태"), ("services.spatial", "지도·좌표 기준"),
             ("services.mission", "임무 실행"), ("services.mission.global", "전역 방문 계획"), ("services.mission.local", "구역 IPP · 계획 진입점"),
             ("services.media", "영상 수신·저장"), ("services.vision", "사람 탐지·좌표·후보"), ("services.mission_map", "상황지도·관측"),
             ("services.command", "비행 명령 · IGatewayLink"), ("services.safety", "안전·복구"), ("services.alert", "알림"), ("services.history", "이력·결과"),
             ("services.media.crop_grace", "관제 단말 크롭 GRACE 제공")]
    bw, bh, gx, gy = 340, 150, 26, 26
    sv = {}
    for i, (n, d) in enumerate(names):
        r, col = divmod(i, 4)
        x = 66 + col * (bw + gx); y = 330 + r * (bh + gy)
        sv[n] = pkg(c, x, y, bw * 2 + gx if n == "services.media.crop_grace" else bw, bh, n, d)
    ct = pkg(c, 40, 1080, 460, 170, "contracts", "«DTO» 계층·패키지 사이 전달 객체\n(shared/contracts · 웹과 스키마 공유)")
    po = pkg(c, 540, 1080, 450, 170, "policies", "CommandValidator · SafetyPolicy\nDetectionLikelihoodModel (서버·Pi 공유)")
    dao = pkg(c, 1030, 1080, 480, 170, "storage.dao", "«DAO» 18개 — 테이블 1:1\nSQL 은 이 패키지에만")
    ent = pkg(c, 1030, 1330, 480, 170, "storage.entity", "«entity» 18개 — 테이블 1:1\n(2.2 전체 ERD 기준)")
    db = pkg(c, 540, 1330, 450, 170, "PostgreSQL / PostGIS", "storage/migrations · 중앙 DB", fill="#f0f0f0")
    # ── 게이트웨이 ──
    frame(c, 1560, 10, 430, 1700, "현장 게이트웨이 (Pi)")
    gw = {}
    for i, (n, d) in enumerate([("gateway.api", "ServerChannelController\nServerReporter"), ("gateway.command", "GatewayCommandGuard\nExecutionLeaseGuard\nCommandRecoveryReconciler"),
                                ("gateway.flight", "FlightAdapter (MAVLink)\nLocalControlMonitor"), ("gateway.media", "VideoRelay · FieldRecorder"),
                                ("gateway.storage", "«DAO» CommandJournalDAO\nSegmentManifestStore (로컬)")]):
        gw[n] = pkg(c, 1585, 70 + i * 250, 380, 200, n, d)
    # ── 웹 · 학습 ──
    frame(c, 10, 1740, 640, 260, "관제 운용자 단말 (React · PWA)")
    web = pkg(c, 40, 1800, 580, 170, "web", "«view model» MapViewModel · CropViewModel 등\nGraceCropDecoder — 전용 GRACE 디코더\napi 만 호출 (HTTP · WebSocket · 크롭 스트림)")
    frame(c, 670, 1740, 650, 260, "학습 서버 (RTX 3090)")
    vt = pkg(c, 700, 1800, 600, 170, "vision_train", "«tool» DatasetBuilder · Trainer · Evaluator\nPairedComparator · WeightSouper · EngineBuilder")
    note_x, note_y, note_w, note_h = 1340, 1760, 650, 230
    c.ax.add_patch(Polygon([(note_x, note_y), (note_x + note_w - 26, note_y), (note_x + note_w, note_y + 26), (note_x + note_w, note_y + note_h),
                            (note_x, note_y + note_h)], fc="#f4f4f4", ec=MID, lw=1.5, zorder=3))
    for i, l in enumerate(["GRACE 크롭 제공 경로 (서버 → 관제 운용자 단말)", "crop_grace 인코딩·패킷화 → CropStreamEndpoint", "→ web GraceCropDecoder 디코딩·표시",
                           "사용 패킷 피드백은 반대 방향 · 원본 분석과 분리"]):
        c.text(note_x + 20, note_y + 40 + i * 48, l, fs * (0.78 if i == 0 else 0.72), bold=(i == 0), color=INK if i == 0 else MID, zorder=4)
    # ── 의존 ──
    arrow(c, [(775, 220), (775, 270)], "«import» I… 만", dy=0, dx=120)
    arrow(c, [(1210, 1030), (1210, 1080)], "«import»", dy=0, dx=70)
    arrow(c, [(260, 1030), (260, 1080)], "«import»", dy=0, dx=70)
    arrow(c, [(760, 1030), (760, 1080)], "«import»", dy=0, dx=70)
    arrow(c, [(1270, 1250), (1270, 1330)], "CRUD", dy=0, dx=50)
    arrow(c, [(1030, 1415), (990, 1415)], "", dashed=True)
    arrow(c, [(40, 145), (25, 145), (25, 1165), (40, 1165)], "")
    c.text(30, 650, "«import»", fs * 0.7, color=MID, rotation=90, zorder=6, bbox=dict(fc="white", ec="none"))
    arrow(c, [(1775, 520), (1775, 570)], "", dashed=True)
    arrow(c, [(1775, 270), (1775, 320)], "", dashed=True)
    arrow(c, [(1585, 670), (1548, 670), (1548, 1292), (765, 1292), (765, 1250)], "공유 규칙 (policies)", dashed=True, lab_at=0.75, dy=-14)
    arrow(c, [(1010, 1800), (1010, 1030)], "IModelRegistry (services.vision)", lab_at=0.18, dy=0, dx=170)
    arrow(c, [(330, 1800), (330, 1720), (150, 1720), (150, 1250)], "DTO 스키마", lab_at=0.35)
    # 네트워크 경계 설명
    c.text(230, 1600, "HTTP · WSS · SRT · 크롭 스트림은 관계로 그리지 않는다 — api(Controller · CropStreamEndpoint) · IGatewayLink · ServerReporter 가 경계를 맡는다",
           fs * 0.72, color=MID, zorder=6)
    c.save(path)
    return H


if __name__ == "__main__":
    print(render("out2/pd01.png"))
