"""
sources.py — 영상 읽기 (RTSP 실시간 · 녹화 파일)

  RTSP: 읽기 스레드가 계속 받고 **가장 최근 장만** 남긴다 (밀리면 오래된 장은 버린다 · LatestFrameBuffer)
  파일: 원본 fps 에서 목표 fps 로 건너뛰며 읽는다 (재처리 · 시험용)
  반환 (frame, pts_ms, recv_unix) — pts_ms 는 촬영 시각 sidecar 와 맞출 때 쓴다
"""
import os
import threading
import time

import cv2

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")   # MediaMTX 는 TCP 만 연다


class LiveSource:
    def __init__(self, url, reconnect_s=2.0):
        self.url, self.reconnect_s = url, reconnect_s
        self._lock, self._frame, self._seq = threading.Lock(), None, 0
        self.read_frames = 0
        self.dropped = 0
        self._stop = False
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def _run(self):
        while not self._stop:
            cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                time.sleep(self.reconnect_s); continue
            while not self._stop:
                ok, fr = cap.read()
                if not ok:
                    break
                with self._lock:
                    if self._frame is not None:
                        self.dropped += 1
                    self._frame = (fr, cap.get(cv2.CAP_PROP_POS_MSEC), time.time())
                    self._seq += 1
                    self.read_frames += 1
            cap.release()
            time.sleep(self.reconnect_s)

    def latest(self, timeout=1.0):
        """새 장이 오면 (frame, pts_ms, recv_unix) · 없으면 None"""
        end = time.time() + timeout
        while time.time() < end:
            with self._lock:
                if self._frame is not None:
                    f, self._frame = self._frame, None
                    return f
            time.sleep(0.005)
        return None

    def close(self):
        self._stop = True


class FileSource:
    def __init__(self, path, fps=10.0):
        self.cap = cv2.VideoCapture(str(path))
        src = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.step = max(1, round(src / fps))
        self.i = 0
        self.read_frames = self.dropped = 0

    def latest(self, timeout=None):
        while True:
            ok = self.cap.grab()
            if not ok:
                return None
            self.i += 1
            if (self.i - 1) % self.step:
                self.dropped += 1
                continue
            ok, fr = self.cap.retrieve()
            if not ok:
                return None
            self.read_frames += 1
            return fr, self.cap.get(cv2.CAP_PROP_POS_MSEC), time.time()

    def close(self):
        self.cap.release()
