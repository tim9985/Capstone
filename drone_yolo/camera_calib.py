"""
camera_calib.py — 실기체 카메라 화각 · 왜곡 실측 (체커보드 · 10-30 · 계획 「20 최종발표 · 실기체 비전 계획」 4절)

  입력: 체커보드 사진 폴더 또는 영상 (1920×1080 · 운용 해상도 그대로 · 판 전체가 보이게 15~40장 · 가운데 · 네 귀퉁이 · 기울여서)
  출력: 초점거리 fx · fy · 주점 · 왜곡 (k1 k2 p1 p2 k3) · 재투영 오차 · **수평 화각 (worker VISION_HFOV_DEG)**
        + 지금 가정 (수평 80.2° · f 1,141 px · 왜곡 없음) 과 비교 — 화면 가장자리 · 아래쪽 (발끝이 잡히는 곳) 이 왜곡 때문에 몇 px 어긋나는지
        → configs/camera_<이름>.json (git · 수치만)
  자체 시험: `--selftest` — 정답을 아는 가상 카메라로 체커보드를 그려 되찾는지 (f · 왜곡 · 화각)
실행  python camera_calib.py <사진 폴더 | 영상> --board 9x6 --square 25 --name imx415_ln012 [--every 15]
      python camera_calib.py --selftest
"""
import argparse, json, math, sys
from pathlib import Path

import cv2
import numpy as np

BASE = Path(__file__).resolve().parent
ASSUMED = {"hfov_deg": 80.2, "fx": 1141.0, "w": 1920, "h": 1080}


def frames(src, every):
    p = Path(src)
    if p.is_dir():
        for f in sorted(p.iterdir()):
            if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
                yield f.name, cv2.imread(str(f))
    else:
        cap = cv2.VideoCapture(str(p)); i = 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            if i % every == 0:
                yield f"{p.stem}_{i:05d}", fr
            i += 1


def find(img, board):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ok, c = cv2.findChessboardCornersSB(g, board, flags=cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE)
    return c if ok else None


def calibrate(views, board, square, size):
    obj = np.zeros((board[0] * board[1], 3), np.float32)
    obj[:, :2] = np.mgrid[0:board[0], 0:board[1]].T.reshape(-1, 2) * square
    err, K, D, rv, tv = cv2.calibrateCamera([obj] * len(views), views, size, None, None)
    per = []
    for c, r, t in zip(views, rv, tv):
        pr, _ = cv2.projectPoints(obj, r, t, K, D); per.append(float(np.sqrt(np.mean(np.sum((pr.reshape(-1, 2) - np.asarray(c).reshape(-1, 2)) ** 2, -1)))))
    return err, K, D, per


def report(K, D, size, err, per, n_used, n_all):
    W, H = size
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    hfov = math.degrees(2 * math.atan(W / (2 * fx))); vfov = math.degrees(2 * math.atan(H / (2 * fy)))
    dfov = math.degrees(2 * math.atan(math.hypot(W, H) / (2 * math.sqrt(fx * fy))))
    # 왜곡 무시 시 어긋남 — 화면 점 (왜곡된 실제 위치) 을 왜곡 없는 핀홀로 볼 때와 왜곡 보정한 위치 차 (px)
    pts = {"가운데": (W / 2, H / 2), "오른쪽 끝 가운데": (W - 1, H / 2), "아래 가운데": (W / 2, H - 1), "오른쪽 아래 귀퉁이": (W - 1, H - 1),
           "아래 1/4 · 오른쪽 1/4 (발끝 흔한 곳)": (W * 0.75, H * 0.75)}
    shift = {}
    for k, (x, y) in pts.items():
        u = cv2.undistortPoints(np.float32([[[x, y]]]), K, D, P=K)[0, 0]
        shift[k] = round(float(math.hypot(u[0] - x, u[1] - y)), 1)
    return {"해상도": [W, H], "fx": round(fx, 1), "fy": round(fy, 1), "cx": round(cx, 1), "cy": round(cy, 1),
            "왜곡 k1 k2 p1 p2 k3": [round(float(v), 5) for v in D.ravel()[:5]],
            "수평 화각 °": round(hfov, 2), "수직 화각 °": round(vfov, 2), "대각 화각 °": round(dfov, 2),
            "재투영 오차 px (전체 RMS)": round(float(err), 3), "장별 오차 px 최대": round(max(per), 3) if per else None,
            "쓴 장 / 전체": f"{n_used}/{n_all}",
            "왜곡 무시 시 어긋남 px": shift,
            "지금 가정 대비": {"수평 화각 °": f"{ASSUMED['hfov_deg']} → {round(hfov, 2)}", "fx px": f"{ASSUMED['fx']} → {round(fx, 1)}",
                         "fx 차 %": round((fx - ASSUMED["fx"]) / ASSUMED["fx"] * 100, 2)},
            "worker 설정": {"VISION_HFOV_DEG": round(hfov, 2)}}


def selftest():
    """가상 카메라 (f 1,141 · 주점 약간 어긋남 · 술통 왜곡) 로 체커보드 25장 그려 되찾기"""
    W, H = 1920, 1080
    Kt = np.array([[1141.0, 0, 968.0], [0, 1139.0, 532.0], [0, 0, 1]]); Dt = np.array([-0.21, 0.06, 0.0005, -0.0003, -0.008])
    board, sq = (9, 6), 60.0
    obj = np.zeros((54, 3), np.float32); obj[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2) * sq
    tile = 60
    pat = np.full(((6 + 1) * tile + 2 * tile, (9 + 1) * tile + 2 * tile), 255, np.uint8)       # 흰 여백 + 10×7 칸 (안쪽 꼭짓점 9×6)
    for i in range(10):
        for j in range(7):
            if (i + j) % 2 == 0:
                pat[tile + j * tile:tile + (j + 1) * tile, tile + i * tile:tile + (i + 1) * tile] = 0
    rng = np.random.default_rng(0); views = []; n_all = 0
    for k in range(25):
        n_all += 1
        rv = rng.uniform(-0.5, 0.5, 3); tv = np.array([rng.uniform(-500, 100), rng.uniform(-300, 50), rng.uniform(900, 1600)])
        # 판 평면 → 화면: 판 좌표 (px 단위 그림) 를 3D 로 두고 왜곡까지 투영해 그림 (역방향 사상)
        ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
        und = cv2.undistortPoints(np.stack([xs.ravel(), ys.ravel()], 1)[:, None, :], Kt, Dt)[:, 0, :]     # 화면 → 정규 좌표 (왜곡 뺌)
        R, _ = cv2.Rodrigues(rv)
        n = R[:, 2]; d = np.c_[und, np.ones(len(und))]                                                     # 광선 · 판 평면 (z=0 · 판 좌표) 교차
        s = (n @ tv) / (d @ n)
        P = d * s[:, None] - tv; pb = P @ R                                                                 # 판 좌표
        u = (pb[:, 0] / sq + 1) * tile + tile * 0; v = (pb[:, 1] / sq + 1) * tile
        img = cv2.remap(pat, u.reshape(H, W).astype(np.float32), v.reshape(H, W).astype(np.float32), cv2.INTER_LINEAR, borderValue=128)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        c = find(img, board)
        if c is not None:
            views.append(c)
    err, K, D, per = calibrate(views, board, sq, (W, H))
    out = report(K, D, (W, H), err, per, len(views), n_all)
    out["정답"] = {"fx": 1141.0, "fy": 1139.0, "cx": 968.0, "cy": 532.0, "왜곡": Dt.tolist(), "수평 화각 °": round(math.degrees(2 * math.atan(W / 2 / 1141.0)), 2)}
    print(json.dumps(out, ensure_ascii=False, indent=1))
    ok = abs(K[0, 0] - 1141) < 5 and abs(D[0, 0] - Dt[0]) < 0.02 and err < 0.5
    print("자체 시험", "PASS" if ok else "FAIL")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", nargs="?"); ap.add_argument("--board", default="9x6"); ap.add_argument("--square", type=float, default=25.0)
    ap.add_argument("--name", default="camera"); ap.add_argument("--every", type=int, default=15); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    board = tuple(int(v) for v in a.board.lower().split("x"))
    views, names, size, n_all = [], [], None, 0
    for name, img in frames(a.src, a.every):
        n_all += 1
        size = (img.shape[1], img.shape[0])
        c = find(img, board)
        if c is not None:
            views.append(c); names.append(name)
    if len(views) < 8:
        sys.exit(f"체커보드 찾은 장 {len(views)} < 8 — 판 전체가 보이게 · 초점 · 각도 다양하게 더 찍기")
    err, K, D, per = calibrate(views, board, a.square, size)
    out = report(K, D, size, err, per, len(views), n_all)
    bad = [n for n, e in zip(names, per) if e > 1.0]
    out["오차 1 px 넘는 장 (빼고 다시 권장)"] = bad
    print(json.dumps(out, ensure_ascii=False, indent=1))
    (BASE / "configs" / f"camera_{a.name}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
