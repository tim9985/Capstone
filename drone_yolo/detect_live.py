"""
detect_live.py — 항공 영상에 사람 탐지 모델을 걸고 **실시간으로 보면서** 설정을 바꾼다

왜 필요한가
  · 서버에서 학습한 가중치가 실제 항공 영상에서 어떻게 보이는지 눈으로 확인한다
  · 신뢰도·IoU·입력 크기를 **재생 중에** 바꿔 가며 무엇이 사라지고 남는지 본다
  · 가중치를 여러 개 올려 두고 같은 장면에서 바로 갈아 끼운다
  정답이 없는 영상에 쓰는 도구다. 정량 평가는 eval_domain.py · eval_perflight.py.

실행 (노트북)
  python detect_live.py --source "data/raw/okutama/Drone2/Noon/2.2.10.mp4" \
      --weights weights/m1_11m_1280.pt weights/yolov8s_stage1_all.pt --imgsz 1280

조작 (창을 클릭해 활성화한 뒤)
  space        재생 / 일시정지          a d ← →   한 프레임 뒤/앞 (일시정지 중)
  + -          신뢰도 ±0.05            [ ]       입력 크기 한 단계 아래/위
  m            다음 가중치로 교체        t         추적기(ByteTrack) 켜기/끄기
  b            박스 크기(px) 표시        l         라벨 표시
  c            crop 모드 (4K 에서 1280x720 을 원본 크기로 잘라 넣기)   i j k n  crop 창 이동
  s            현재 화면 저장            r         주석 영상 녹화 시작/정지
  h            도움말                   q ESC     종료

  창 위 슬라이더로도 신뢰도·IoU·입력 크기·최대 탐지 수·최소 박스 크기·프레임 건너뛰기를 바꾼다.

주의
  · 슬라이더 값은 다음 프레임부터 적용된다 (모델을 다시 부르지 않는다)
  · 입력 크기를 키우면 느려진다. RTX 3050 기준 1280 약 20 ms · 1920 약 38 ms
  · 한글 경로 저장은 cv2.imwrite 가 실패해서 imencode 로 쓴다
"""
import argparse
import sys
import time
from collections import deque
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import cv2
import numpy as np

WIN = "drone detect - live"
IMGSZ_STEPS = [640, 960, 1280, 1600, 1920]
HELP_LINES = [
    "space play/pause   a d step   + - conf   [ ] imgsz",
    "m model   t tracker   b box px   l labels   s save   r record   h help   q quit",
]


def imwrite_u(path, img):
    """한글 경로에서도 저장되게 한다."""
    path = Path(path)
    ok, buf = cv2.imencode(path.suffix or ".jpg", img)
    if ok:
        path.parent.mkdir(parents=True, exist_ok=True)
        buf.tofile(str(path))
    return ok


def color_for(conf):
    """신뢰도가 낮을수록 빨강, 높을수록 초록 (BGR)."""
    c = float(np.clip(conf, 0.0, 1.0))
    return (60, int(80 + 150 * c), int(230 - 150 * c))


def draw_text(img, text, org, scale=0.55, color=(255, 255, 255), thick=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


class Player:
    """영상 한 개를 앞뒤로 넘기며 읽는다."""

    def __init__(self, source, start=0):
        src = int(source) if str(source).isdigit() else str(source)
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise SystemExit(f"영상을 열 수 없다: {source}")
        self.total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.idx = -1
        if start:
            self.seek(start)

    def seek(self, idx):
        idx = max(0, min(idx, self.total - 1) if self.total else max(0, idx))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        self.idx = idx - 1

    def read(self):
        ok, frame = self.cap.read()
        if ok:
            self.idx += 1
        return ok, frame

    def release(self):
        self.cap.release()


def find_weights():
    """--weights 를 안 주면 여기서 찾는다. 서버에서 받아 온 것을 먼저 올린다."""
    here = Path(__file__).resolve().parent
    dirs = [here / "weights", here.parent.parent / "drone_dev" / "weights"]
    prefer = ("fov_11s", "fov_11m", "l1_", "m2_", "m1_", "e1_", "yolov8s_stage1_all")
    found, seen = [], set()
    for d in dirs:
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.pt")):
            if "pose" in p.name or "cls" in p.name or p.name in seen:
                continue  # 자세 판별 트랙은 09-13 중단
            seen.add(p.name)
            found.append(p)
    found.sort(key=lambda p: next((i for i, k in enumerate(prefer) if p.name.startswith(k)), len(prefer)))
    if found:
        print("자동으로 찾은 가중치:", ", ".join(p.name for p in found[:6]))
    return found[:6]


def load_models(paths, device):
    from ultralytics import YOLO

    models = []
    for p in paths:
        p = Path(p)
        if not p.exists():
            print(f"건너뜀 (파일 없음): {p}")
            continue
        m = YOLO(str(p))
        try:
            m.to(device)
        except Exception:
            pass
        models.append((p.name, m))
        print(f"불러옴: {p.name}  클래스 {list(m.names.values())[:4]}")
    if not models:
        raise SystemExit("쓸 수 있는 가중치가 없다. --weights 경로를 확인할 것")
    return models


def main():
    ap = argparse.ArgumentParser(description="항공 영상 실시간 사람 탐지 테스트기")
    ap.add_argument("--source", required=True, help="영상 파일 · 웹캠 번호(0) · RTSP/HTTP 주소")
    ap.add_argument("--weights", nargs="*", default=[],
                    help="가중치 .pt 여러 개 (m 키로 교체). 비우면 weights 폴더에서 자동으로 찾는다")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--max-det", type=int, default=100)
    ap.add_argument("--device", default="0", help="0 = 첫 GPU · cpu")
    ap.add_argument("--half", action="store_true", help="FP16 추론 (GPU 에서 더 빠름)")
    ap.add_argument("--start", type=int, default=0, help="시작 프레임")
    ap.add_argument("--stride", type=int, default=1, help="N 프레임마다 추론 (1 = 모두)")
    ap.add_argument("--window", type=int, default=1280, help="창 가로 크기(픽셀)")
    ap.add_argument("--crop", default="", help="예: 1280x720 — 4K 영상에서 이 크기 창을 원본 해상도로 잘라 넣는다 "
                                              "(줄이지 않으므로 사람 픽셀 크기가 실제 운용과 같아진다). c 키로 끄고 켠다")
    ap.add_argument("--save-dir", default="runs_live", help="화면 저장·녹화 폴더")
    args = ap.parse_args()

    models = load_models(args.weights or find_weights(), args.device)
    mi = 0
    player = Player(args.source, args.start)
    save_dir = Path(args.save_dir)
    print(f"영상 {player.w}x{player.h} · {player.fps:.1f} fps · {player.total or '?'} 프레임")

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, args.window, int(args.window * player.h / max(player.w, 1)) + 60)
    noop = lambda v: None
    cv2.createTrackbar("conf %", WIN, int(args.conf * 100), 100, noop)
    cv2.createTrackbar("iou %", WIN, int(args.iou * 100), 100, noop)
    cv2.createTrackbar("imgsz", WIN, IMGSZ_STEPS.index(args.imgsz) if args.imgsz in IMGSZ_STEPS else 2,
                       len(IMGSZ_STEPS) - 1, noop)
    cv2.createTrackbar("max det", WIN, args.max_det, 300, noop)
    cv2.createTrackbar("min box px", WIN, 0, 120, noop)
    cv2.createTrackbar("stride", WIN, max(1, args.stride), 10, noop)
    cv2.createTrackbar("speed %", WIN, 100, 400, noop)

    crop_wh = None
    if args.crop:
        cw, ch = (int(v) for v in args.crop.lower().split("x"))
        crop_wh = (cw, ch)
    crop_on = crop_wh is not None
    crop_pos = [0.5, 0.5]  # 원본 안에서 창의 중심 (0~1)
    playing, show_labels, show_px, use_tracker, show_help = True, True, False, False, True
    writer, last_result, last_ms = None, None, (0.0, 0.0, 0.0)
    fps_hist, frame = deque(maxlen=30), None

    def cut(img):
        """crop 모드면 원본 해상도 그대로 창 하나를 잘라 낸다."""
        if not (crop_on and crop_wh):
            return img
        h, w = img.shape[:2]
        cw, ch = min(crop_wh[0], w), min(crop_wh[1], h)
        x = int(np.clip(crop_pos[0] * w - cw / 2, 0, w - cw))
        y = int(np.clip(crop_pos[1] * h - ch / 2, 0, h - ch))
        return img[y:y + ch, x:x + cw]

    def infer(img, conf, iou, imgsz, max_det):
        name, model = models[mi]
        t0 = time.perf_counter()
        if use_tracker:
            res = model.track(img, imgsz=imgsz, conf=conf, iou=iou, max_det=max_det,
                              device=args.device, half=args.half, persist=True,
                              tracker="bytetrack.yaml", verbose=False)[0]
        else:
            res = model.predict(img, imgsz=imgsz, conf=conf, iou=iou, max_det=max_det,
                                device=args.device, half=args.half, verbose=False)[0]
        total_ms = (time.perf_counter() - t0) * 1000
        sp = getattr(res, "speed", {}) or {}
        return res, (sp.get("preprocess", 0.0), sp.get("inference", 0.0), sp.get("postprocess", 0.0), total_ms)

    while True:
        conf = max(cv2.getTrackbarPos("conf %", WIN), 1) / 100
        iou = max(cv2.getTrackbarPos("iou %", WIN), 1) / 100
        imgsz = IMGSZ_STEPS[cv2.getTrackbarPos("imgsz", WIN)]
        max_det = max(1, cv2.getTrackbarPos("max det", WIN))
        min_px = cv2.getTrackbarPos("min box px", WIN)
        stride = max(1, cv2.getTrackbarPos("stride", WIN))
        speed = max(cv2.getTrackbarPos("speed %", WIN), 10) / 100

        if playing:
            ok, new_frame = player.read()
            if not ok:
                playing = False
                print("영상 끝")
            else:
                frame = new_frame
                if player.idx % stride == 0:
                    last_result, last_ms = infer(cut(frame), conf, iou, imgsz, max_det)
        if frame is None:
            ok, frame = player.read()
            if not ok:
                break
            last_result, last_ms = infer(cut(frame), conf, iou, imgsz, max_det)

        view = cut(frame).copy()
        n, confs, longs = 0, [], []
        if last_result is not None and last_result.boxes is not None:
            ids = last_result.boxes.id
            for k, box in enumerate(last_result.boxes):
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                c = float(box.conf[0])
                long_side = max(x2 - x1, y2 - y1)
                if long_side < min_px:
                    continue
                n += 1
                confs.append(c)
                longs.append(long_side)
                col = color_for(c)
                cv2.rectangle(view, (x1, y1), (x2, y2), col, 2)
                if show_labels:
                    tag = f"{c:.2f}"
                    if show_px:
                        tag += f" {long_side}px"
                    if ids is not None and k < len(ids):
                        tag = f"#{int(ids[k])} " + tag
                    draw_text(view, tag, (x1, max(14, y1 - 6)), 0.5, col)

        dt = sum(fps_hist) / len(fps_hist) if fps_hist else 0.0
        name = models[mi][0]
        pre, inf_ms, post, total_ms = last_ms
        pos = f"{player.idx + 1}/{player.total}" if player.total else f"{player.idx + 1}"
        lines = [
            f"{name}  imgsz {imgsz}  conf {conf:.2f}  iou {iou:.2f}  {'TRACK' if use_tracker else 'DETECT'}"
            + (f"  CROP {crop_wh[0]}x{crop_wh[1]} @({crop_pos[0]:.2f},{crop_pos[1]:.2f})" if crop_on and crop_wh else "  FULL"),
            f"{total_ms:5.1f} ms/frame (inf {inf_ms:.1f})  {1000 / total_ms if total_ms else 0:4.1f} fps   frame {pos}"
            f"  stride {stride}  speed {speed:.1f}x",
            f"boxes {n}   conf avg {np.mean(confs) if confs else 0:.2f}   box long side median "
            f"{int(np.median(longs)) if longs else 0} px   min px {min_px}",
        ]
        if not playing:
            lines[0] += "   [PAUSED]"
        if writer is not None:
            lines[0] += "   [REC]"
        y = 22
        for text in lines:
            draw_text(view, text, (10, y), 0.6)
            y += 24
        if show_help:
            for text in HELP_LINES:
                draw_text(view, text, (10, view.shape[0] - 34 + (0 if text is HELP_LINES[0] else 18)), 0.45,
                          (200, 200, 200))

        if writer is not None:
            writer.write(view)
        cv2.imshow(WIN, view)

        wait = 1 if not playing else max(1, int(1000 / (player.fps * speed)))
        t_key = time.perf_counter()
        key = cv2.waitKey(wait) & 0xFF
        fps_hist.append(max(time.perf_counter() - t_key, 1e-6))

        if key in (ord("q"), 27):
            break
        elif key == ord(" "):
            playing = not playing
        elif key in (ord("d"), 83):
            playing = False
            ok, f2 = player.read()
            if ok:
                frame = f2
                last_result, last_ms = infer(frame, conf, iou, imgsz, max_det)
        elif key in (ord("a"), 81):
            playing = False
            player.seek(max(0, player.idx - 1))
            ok, f2 = player.read()
            if ok:
                frame = f2
                last_result, last_ms = infer(frame, conf, iou, imgsz, max_det)
        elif key in (ord("+"), ord("=")):
            cv2.setTrackbarPos("conf %", WIN, min(100, int(conf * 100) + 5))
        elif key in (ord("-"), ord("_")):
            cv2.setTrackbarPos("conf %", WIN, max(1, int(conf * 100) - 5))
        elif key == ord("]"):
            cv2.setTrackbarPos("imgsz", WIN, min(len(IMGSZ_STEPS) - 1, IMGSZ_STEPS.index(imgsz) + 1))
        elif key == ord("["):
            cv2.setTrackbarPos("imgsz", WIN, max(0, IMGSZ_STEPS.index(imgsz) - 1))
        elif key == ord("m"):
            mi = (mi + 1) % len(models)
            print(f"가중치 교체 → {models[mi][0]}")
            last_result, last_ms = infer(frame, conf, iou, imgsz, max_det)
        elif key == ord("t"):
            use_tracker = not use_tracker
            print(f"추적기 {'켬 (ByteTrack)' if use_tracker else '끔'}")
        elif key == ord("c"):
            if crop_wh is None:
                crop_wh = (1280, 720)
            crop_on = not crop_on
            print(f"crop 모드 {'켬' if crop_on else '끔'}")
        elif key in (ord("i"), ord("k"), ord("j"), ord("n")) and crop_on:
            dx = {"j": -0.08, "n": 0.08}.get(chr(key), 0.0)
            dy = {"i": -0.08, "k": 0.08}.get(chr(key), 0.0)
            crop_pos[0] = float(np.clip(crop_pos[0] + dx, 0.0, 1.0))
            crop_pos[1] = float(np.clip(crop_pos[1] + dy, 0.0, 1.0))
        elif key == ord("l"):
            show_labels = not show_labels
        elif key == ord("b"):
            show_px = not show_px
        elif key == ord("h"):
            show_help = not show_help
        elif key == ord("s"):
            out = save_dir / f"frame_{Path(str(args.source)).stem}_{player.idx + 1:06d}_{name}.jpg"
            imwrite_u(out, view)
            print(f"저장: {out}")
        elif key == ord("r"):
            if writer is None:
                save_dir.mkdir(parents=True, exist_ok=True)
                out = save_dir / f"rec_{Path(str(args.source)).stem}_{time.strftime('%H%M%S')}.mp4"
                writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"),
                                         player.fps, (view.shape[1], view.shape[0]))
                print(f"녹화 시작: {out}")
            else:
                writer.release()
                writer = None
                print("녹화 정지")

    if writer is not None:
        writer.release()
    player.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
