"""Stage 1: threaded ingestion with auto-reconnect and a 150-frame RAM ring buffer.

The reader thread owns the camera. The buffer holds (wall_clock_time, frame)
pairs and never touches the disk. Frames are shared by reference, so consumers
must treat them as READ-ONLY (draw on a copy).
"""
import os
import re
import time
import threading
from collections import deque

import cv2

import config


def mask_url(url):
    """rtsp://user:pass@host/x  ->  rtsp://***@host/x  (never show or store camera credentials)."""
    return re.sub(r"://[^/@]*@", "://***@", str(url))


class ThreadedCapture:
    def __init__(self, src=None, width=None, height=None, buffer_frames=None):
        self.src = config.CAMERA_SOURCE if src is None else src
        self.width = width or config.CAPTURE_WIDTH
        self.height = height or config.CAPTURE_HEIGHT
        self.buffer = deque(maxlen=buffer_frames or config.BUFFER_FRAMES)
        self._cond = threading.Condition()
        self._seq = 0
        self._stop = threading.Event()
        self.connected = False
        self.offline_since = None
        self.capture_fps = 0.0
        self.reconnect_count = 0
        self.cap = None
        # API shared with VideoFileCapture (recorded-footage mode)
        self.is_file, self.ended, self.paused = False, False, False
        self.generation, self.idx = 0, 0
        self.source_kind = "webcam" if isinstance(self.src, int) else "cctv"
        self.source_name = "camera" if isinstance(self.src, int) else mask_url(self.src)
        self.position_sec = self.duration_sec = None
        self._open()
        self._thread = threading.Thread(target=self._run, name="capture", daemon=True)
        self._thread.start()

    # ---- camera handling ----
    def _open(self):
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass
        try:
            if os.name == "nt" and isinstance(self.src, int):
                self.cap = cv2.VideoCapture(self.src, cv2.CAP_DSHOW)   # DirectShow backend
            else:
                self.cap = self._open_url_or_index()
            if isinstance(self.src, int):
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        except Exception as e:
            print(f"[CAPTURE] open failed: {e}")
            self.cap = None

    def _open_url_or_index(self):
        if isinstance(self.src, str) and "://" in self.src:
            os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")   # TCP is far more reliable than UDP
            try:                                                   # time out instead of hanging on a dead camera
                return cv2.VideoCapture(self.src, cv2.CAP_FFMPEG,
                                        [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 6000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 6000])
            except Exception:
                pass
        return cv2.VideoCapture(self.src)

    def _mark_offline(self):
        self.connected = False
        if self.offline_since is None:
            self.offline_since = time.time()

    def _run(self):
        backoff, fails, fps_ema, prev = 0.5, 0, 0.0, None
        while not self._stop.is_set():
            if self.cap is None or not self.cap.isOpened():
                self._mark_offline()
                time.sleep(backoff)
                backoff = min(backoff * 2, 5.0)
                self.reconnect_count += 1
                self._open()
                continue
            try:
                ok, frame = self.cap.read()
            except Exception:
                ok, frame = False, None
            if not ok or frame is None:
                fails += 1
                if fails >= 5:
                    self._mark_offline()
                    try:
                        self.cap.release()
                    except Exception:
                        pass
                    self.cap = None
                else:
                    time.sleep(0.02)
                continue
            fails, backoff = 0, 0.5
            now = time.time()
            self.connected, self.offline_since = True, None
            if prev is not None and now > prev:
                inst = 1.0 / (now - prev)
                fps_ema = inst if fps_ema == 0 else 0.9 * fps_ema + 0.1 * inst
                self.capture_fps = fps_ema
            prev = now
            with self._cond:
                self.buffer.append((now, frame))     # deque(maxlen) silently drops the oldest
                self._seq += 1
                self._cond.notify_all()

    # ---- consumer API ----
    def get_latest(self, last_seq=0, timeout=1.0):
        """Return (frame, timestamp, seq) for a frame newer than last_seq, else None."""
        with self._cond:
            if self._seq == last_seq:
                self._cond.wait(timeout)
            if self._seq == last_seq or not self.buffer:
                return None
            t, f = self.buffer[-1]
            return f, t, self._seq

    def last_frame(self):
        with self._cond:
            return self.buffer[-1] if self.buffer else None

    def get_window(self, t0, t1):
        with self._cond:
            items = list(self.buffer)
        return [(t, f) for t, f in items if t0 <= t <= t1]

    def now(self):
        return time.time()

    def describe_position(self):
        return ""

    def toggle_pause(self):
        pass

    def restart(self):
        pass

    @property
    def buffer_len(self):
        return len(self.buffer)

    def release(self):
        self._stop.set()
        self._thread.join(timeout=2.0)
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass