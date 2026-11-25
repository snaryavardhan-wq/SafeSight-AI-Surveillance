"""Recorded-CCTV input (demo mode). Same API as ThreadedCapture.

Two pacing modes:
  real-time (default)  frames are released at the footage's native speed. If the
                       analyser is slower it simply skips frames, exactly like a
                       live camera. Timestamps are wall-clock.
  every-frame          the reader never runs ahead of the analyser, so every
                       frame is analysed (slower than real time, most accurate).
                       Timestamps are a virtual footage clock.
"""
import os
import time
import threading
from collections import deque

import cv2

import config


def fmt_time(sec):
    sec = int(max(sec or 0, 0))
    return f"{sec // 60:02d}:{sec % 60:02d}"


class VideoFileCapture:
    is_file = True

    def __init__(self, path, every_frame=False, speed=1.0, loop=False, start_paused=False, buffer_frames=None):
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        self.path = path
        self.source_name = os.path.basename(path)
        self.source_kind = "file"
        self.every_frame = every_frame
        self.speed = max(float(speed), 0.05)
        self.loop = loop
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise ValueError(f"OpenCV could not open this video: {path}")
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if fps and fps == fps and 1 <= fps <= 240 else 25.0
        total = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.total_frames = int(total) if total and total > 0 else 0
        self.duration_sec = self.total_frames / self.fps if self.total_frames else None

        self.buffer = deque(maxlen=buffer_frames or config.BUFFER_FRAMES)
        self._cond = threading.Condition()
        self._io = threading.Lock()
        self._seq = 0
        self._taken = 0
        self._stop = threading.Event()
        self.connected, self.offline_since = True, None
        self.capture_fps, self.reconnect_count = self.fps, 0
        self.ended, self.generation = False, 0
        self.paused = start_paused
        self.idx = 0
        self._vbase = time.time()
        self._last_ts = self._vbase
        self._wall0 = None
        self._pause_start = None
        self._thread = threading.Thread(target=self._run, name="footage", daemon=True)
        self._thread.start()

    # ---------------- reader thread ----------------
    def _fit(self, frame):
        h, w = frame.shape[:2]
        if w > config.CAPTURE_WIDTH:                       # keep the RAM ring buffer at ~415 MB
            nh = int(h * config.CAPTURE_WIDTH / w)
            frame = cv2.resize(frame, (config.CAPTURE_WIDTH, nh), interpolation=cv2.INTER_AREA)
        return frame

    def _rewind(self):
        with self._io:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        with self._cond:
            self.buffer.clear()
            self.idx = 0
            self.generation += 1
            self._vbase = self._last_ts + 1.0              # footage clock keeps moving forward
            self._wall0 = None
            self._taken = self._seq
            self._cond.notify_all()

    def _run(self):
        while not self._stop.is_set():
            if self.paused and self._seq > 0:              # one frame is still read so the first frame can be previewed
                time.sleep(0.03)
                continue
            if self.every_frame:                           # back-pressure: never outrun the analyser
                with self._cond:
                    while self._seq != self._taken and not self._stop.is_set():
                        self._cond.wait(0.1)
                if self._stop.is_set():
                    break
                if self.paused and self._seq > 0:
                    continue
            with self._io:
                ok, frame = self.cap.read()
            if not ok or frame is None:
                if self.loop and self.idx > 0:
                    self._rewind()
                    continue
                with self._cond:
                    self.ended = True
                    self._cond.notify_all()
                while self.ended and not self._stop.is_set():   # idle until restart()
                    time.sleep(0.05)
                continue
            frame = self._fit(frame)
            if self.every_frame:
                ts = self._vbase + self.idx / self.fps
            else:
                if self._wall0 is None:
                    self._wall0 = time.time()
                delay = self._wall0 + self.idx / (self.fps * self.speed) - time.time()
                if delay > 0:
                    time.sleep(min(delay, 1.0))
                ts = time.time()
            with self._cond:
                self.idx += 1
                self.buffer.append((ts, frame))
                self._last_ts = ts
                self._seq += 1
                self._cond.notify_all()

    # ---------------- consumer API ----------------
    def get_latest(self, last_seq=0, timeout=1.0):
        with self._cond:
            if self._seq == last_seq:
                self._cond.wait(timeout)
            if self._seq == last_seq or not self.buffer:
                return None
            t, f = self.buffer[-1]
            self._taken = self._seq
            self._cond.notify_all()
            return f, t, self._seq

    def last_frame(self):
        with self._cond:
            return self.buffer[-1] if self.buffer else None

    def get_window(self, t0, t1):
        with self._cond:
            items = list(self.buffer)
        return [(t, f) for t, f in items if t0 <= t <= t1]

    def now(self):
        return self._last_ts if self.every_frame else time.time()

    @property
    def buffer_len(self):
        return len(self.buffer)

    @property
    def position_sec(self):
        return self.idx / self.fps

    def describe_position(self):
        return f" | footage {fmt_time(self.position_sec)}"

    # ---------------- transport controls ----------------
    def toggle_pause(self):
        if self.paused:
            if self._wall0 is not None and self._pause_start is not None:
                self._wall0 += time.time() - self._pause_start   # don't "catch up" after a pause
            self.paused = False
        else:
            self._pause_start = time.time()
            self.paused = True

    def restart(self):
        self._rewind()
        self.ended = False

    def release(self):
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        self._thread.join(timeout=2.0)
        try:
            self.cap.release()
        except Exception:
            pass