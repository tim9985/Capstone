"""dd.py — DD-01 실기체 3티어 운용도 (무채색) · GRACE 는 서버 인코더 → 관제 운용자 단말 디코더 구간에만 둔다"""
from mono import Canvas, INK, LINE, MID, head
from matplotlib.patches import Rectangle, Polygon

W = 2000; fs = 29; D = 16          # D: 노드 입체 깊이


def node(c, x, y, w, h, name, sub="", dashed=False):
    ls = (0, (6, 4)) if dashed else "-"
    c.ax.add_patch(Polygon([(x, y), (x + D, y - D), (x + w + D, y - D), (x + w, y)], fc="#e9e9e9", ec=LINE, lw=1.6, zorder=2, ls=ls))
    c.ax.add_patch(Polygon([(x + w, y), (x + w + D, y - D), (x + w + D, y + h - D), (x + w, y + h)], fc="#dcdcdc", ec=LINE, lw=1.6, zorder=2, ls=ls))
    c.ax.add_patch(Rectangle((x, y), w, h, fc="#f7f7f7", ec=LINE, lw=2.0, zorder=2, ls=ls))
    c.text(x + fs * 0.5, y + fs * 0.85, name, fs * 0.95, bold=True, zorder=3)
    if sub:
        c.text(x + w - fs * 0.5, y + fs * 0.85, sub, fs * 0.7, ha="right", color=MID, zorder=3)


def comp(c, x, y, w, h, name, lines=(), bold_name=True, fill="#ffffff", dashed=False):
    c.ax.add_patch(Rectangle((x, y), w, h, fc=fill, ec=LINE, lw=1.7, zorder=3, ls=(0, (6, 4)) if dashed else "-"))
    # 구성요소 표시 (작은 이중 사각형)
    ix, iy = x + w - fs * 1.1, y + fs * 0.35
    c.ax.add_patch(Rectangle((ix, iy), fs * 0.65, fs * 0.8, fc="white", ec=MID, lw=1.1, zorder=4))
    for k in (0.15, 0.45):
        c.ax.add_patch(Rectangle((ix - fs * 0.15, iy + fs * k), fs * 0.3, fs * 0.18, fc="white", ec=MID, lw=1.0, zorder=5))
    c.text(x + fs * 0.4, y + fs * 0.8, name, fs * 0.8, bold=bold_name, zorder=4)
    for i, l in enumerate(lines):
        c.text(x + fs * 0.4, y + fs * (1.8 + 0.92 * i), l, fs * 0.72, color=MID, zorder=4)


def link(c, pts, lab="", dashed=False, both=False, lab_at=0.5, dx=0, dy=-16, ha="center", lw=2.0):
    xs, ys = zip(*pts)
    c.ax.plot(xs, ys, color=INK, lw=lw, ls=(0, (7, 5)) if dashed else "-", zorder=5)
    head(c.ax, pts[-2], pts[-1], "dep", fs, INK)
    if both:
        head(c.ax, pts[1], pts[0], "dep", fs, INK)
    if lab:
        (x0, y0), (x1, y1) = max(zip(pts[:-1], pts[1:]), key=lambda s: abs(s[1][0] - s[0][0]) + abs(s[1][1] - s[0][1]))
        for i, l in enumerate(lab.split("\n")):
            c.text(x0 + (x1 - x0) * lab_at + dx, y0 + (y1 - y0) * lab_at + dy + i * fs * 0.85, l, fs * 0.7, ha=ha, color=INK, zorder=6,
                   bbox=dict(boxstyle="square,pad=0.12", fc="white", ec="none"))


def render(path):
    H = 1300
    c = Canvas(W, H)
    # ── 외부 지도 서비스 ──
    node(c, 1150, 40, 360, 120, "VWorld API", "외부 지도 서비스")
    c.text(1170, 120, "건물 윤곽·지형 높이 · 지도 타일", fs * 0.7, color=MID, zorder=3)
    # ── 기체 ──
    node(c, 20, 330, 260, 300, "기체 · F450")
    comp(c, 38, 390, 224, 100, "비행제어기 (FC)", ["ArduPilot · MAVLink"])
    comp(c, 38, 505, 224, 100, "카메라", ["H.264 원본 영상"])
    # ── 게이트웨이 ──
    node(c, 390, 330, 310, 300, "게이트웨이 · RPi 4B")
    comp(c, 408, 390, 274, 100, "명령 검증 · 현장 기록", ["CommandGuard · Lease"])
    comp(c, 408, 505, 274, 100, "원본 영상 중계", ["VideoRelay — SRT 송신", "GRACE 인코딩 없음"])
    # ── 중앙 서버 ──
    node(c, 850, 260, 640, 620, "중앙 서버 · RTX 4080 SUPER (Linux)")
    comp(c, 870, 320, 290, 140, "비전 (vision)", ["YOLO11m → TensorRT", "원본 영상으로 탐지·좌표"])
    comp(c, 1180, 320, 290, 140, "백엔드 (mission-core)", ["Python · FastAPI", "REST · WebSocket"])
    comp(c, 870, 480, 600, 150, "GRACE 크롭 인코더  (services/media/crop-grace)", ["원본 크롭 생성·보존 → GRACE 인코딩·패킷화",
                                                                         "제공 프로파일 협상 (CROP_GRACE / CROP_BASELINE)",
                                                                         "참조 동기화 · CropStreamEndpoint (송신·피드백)"], fill="#efefef")
    comp(c, 870, 650, 290, 110, "DB", ["PostgreSQL · PostGIS"])
    comp(c, 1180, 650, 290, 110, "미디어 저장", ["녹화본 · 스냅샷 · 원본 크롭"])
    c.text(870, 800, "탐지·좌표 계산·관측 완료 판정은 원본 분석 경로만 쓴다", fs * 0.7, color=MID, zorder=4)
    c.text(870, 835, "(GRACE 복원 크롭은 운용자 확인용)", fs * 0.7, color=MID, zorder=4)
    # ── 관제 운용자 단말 ──
    node(c, 1650, 330, 325, 430, "관제 운용자 단말")
    comp(c, 1668, 390, 289, 120, "관제 웹 (React · PWA)", ["UI-01~07 · 후보 화면 UI-05"])
    comp(c, 1668, 530, 289, 200, "GRACE 디코더", ["전용 실행 환경 (배포 조건)", "부분 패킷 디코딩", "FULL·PARTIAL·NONE 판정", "'복원 영상' 표지로 표시"], fill="#efefef")
    for i, l in enumerate(["크롭 전송 기술: 구현·호환 검증 후 확정", "PWA·WebCodecs 의 GRACE 기본", "지원은 가정하지 않는다"]):
        c.text(1650, 800 + i * 34, l, fs * 0.68, color=MID, zorder=6)
    # ── 개발·학습 ──
    node(c, 20, 1000, 260, 180, "시뮬레이션", "개발·시험", dashed=True)
    c.text(38, 1090, "UE 5.5.4 · AirSim · SITL", fs * 0.7, color=MID, zorder=3)
    c.text(38, 1130, "영상·MAVLink 모의 입력", fs * 0.7, color=MID, zorder=3)
    node(c, 390, 1000, 310, 180, "개발 도구", "", dashed=True)
    c.text(408, 1090, "Git · GitHub", fs * 0.7, color=MID, zorder=3)
    node(c, 850, 1000, 640, 180, "학습 서버 (오프라인) · RTX 3090", "", dashed=True)
    c.text(870, 1090, "DatasetBuilder · Trainer · Evaluator · EngineBuilder", fs * 0.7, color=MID, zorder=3)
    c.text(870, 1130, "운용 경로 밖 — 검증된 .engine 만 서버에 배포", fs * 0.7, color=MID, zorder=3)
    # ── 연결 ──
    link(c, [(280, 450), (390, 450)], "MAVLink", both=True)
    link(c, [(280, 560), (390, 560)], "5GHz 영상")
    link(c, [(700, 450), (850, 450)], "WireGuard VPN\n명령 · 상태", both=True, dy=-52)
    link(c, [(700, 560), (850, 560)], "SRT 원본 영상")
    link(c, [(1490, 410), (1650, 410)], "REST · WSS", both=True)
    link(c, [(1490, 565), (1650, 565)], "GRACE\n크롭 패킷", lw=2.6, dy=-42)
    link(c, [(1650, 690), (1490, 690)], "사용 패킷\n피드백", dashed=True, dy=-42)
    link(c, [(1330, 160), (1330, 260)], "건물·지형 높이", dx=12, dy=0, ha="left")
    link(c, [(1510, 100), (1810, 100), (1810, 330)], "지도 타일", lab_at=0.5)
    link(c, [(1015, 1000), (1015, 880)], ".engine 배포", dashed=True, dx=12, dy=0, ha="left")
    link(c, [(150, 1000), (150, 630)], "모의 입력", dashed=True, dx=12, dy=0, ha="left")
    c.text(20, 1240, "영상: 기체 → 게이트웨이 → 서버는 원본(SRT) 전송을 유지한다.  GRACE: 서버 크롭 인코딩·패킷화 → 관제 운용자 단말 디코딩·표시 → 사용 패킷 피드백(단말 → 서버).",
           fs * 0.7, color=INK, zorder=6)
    c.save(path)
    return H


if __name__ == "__main__":
    print(render("out2/dd01.png"))
