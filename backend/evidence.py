"""Stage 5: asynchronous evidence flushing.

On a verified threat: snapshot now, collect the 2.5 s PRE-roll from the RAM
buffer immediately, wait for the 2.5 s POST-roll, encode a ~5 s MP4, hash it,
then append a sealed record to the hash-chained ledger and notify Telegram.
Nothing here ever blocks the video loop.
"""
import os
import re
import time
import shutil
import threading
import traceback
import subprocess
from datetime import datetime

import cv2

import config
import database
from arbiter import level_name


def _ffmpeg_exe():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


class EvidenceManager:
    def __init__(self, capture, on_sealed=None):
        self.capture = capture
        self.on_sealed = on_sealed
        self.pending = 0
        self.enabled = True           # benchmark.py / evaluate.py turn this off so no files are written
        self._last = {}
        self._lock = threading.Lock()
        self._slots = threading.BoundedSemaphore(config.MAX_CONCURRENT_CLIPS)

    def trigger(self, event_type, track_id, score, details, frame, key=None):
        """Non-blocking. Returns False if suppressed by the per-key cooldown."""
        if not self.enabled:
            return False
        now = self.capture.now()          # wall clock for a camera, footage clock for recorded video
        key = key or f"{event_type}:{track_id}"
        if now - self._last.get(key, -1e18) < config.EVENT_COOLDOWN_SEC:
            return False
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ok:
            return False
        self._last[key] = now
        want_clip = self._slots.acquire(blocking=False)      # bounded RAM
        pre = self.capture.get_window(now - config.CLIP_PRE_SEC, now) if want_clip else []
        ts_str = database.now_str()
        with self._lock:
            self.pending += 1
        threading.Thread(
            target=self._worker, name="evidence", daemon=True,
            args=(event_type, track_id, score, details, buf.tobytes(), now, ts_str, pre, want_clip, self.capture)).start()
        return True

    def wait_idle(self, timeout=8.0):
        end = time.time() + timeout
        while self.pending > 0 and time.time() < end:
            time.sleep(0.1)

    # ---------------------------------------------------------------
    def _worker(self, event_type, track_id, score, details, snap_bytes, t_event, ts_str, pre, want_clip, cap):
        snap_path = video_path = video_sha = None
        try:
            slug = re.sub(r"[^A-Za-z0-9]+", "_", event_type).strip("_").lower()
            base = f"{slug}_{track_id if track_id is not None else 'x'}_{int(t_event)}"
            os.makedirs(config.SNAPSHOT_DIR, exist_ok=True)
            os.makedirs(config.VIDEO_DIR, exist_ok=True)
            try:
                snap_path = os.path.join(config.SNAPSHOT_DIR, base + ".jpg")
                with open(snap_path, "wb") as f:
                    f.write(snap_bytes)
            except OSError as e:
                print(f"[EVIDENCE] snapshot write failed: {e}")
                snap_path = None

            if want_clip:
                self._wait_until(cap, t_event + config.CLIP_POST_SEC)
                post = cap.get_window(t_event, t_event + config.CLIP_POST_SEC)
                last_pre = pre[-1][0] if pre else 0.0
                frames = pre + [p for p in post if p[0] > last_pre]
                pre = None
                if len(frames) >= 2 and self._disk_ok():
                    path = os.path.join(config.VIDEO_DIR, base + ".mp4")
                    if self._write_clip(frames, path):
                        video_path, video_sha = path, database.sha256_file(path)
                frames = None

            rec = None
            for attempt in range(3):
                try:
                    rec = database.commit_event(event_type, track_id, score, level_name(score), details,
                                                snap_path, snap_bytes, video_path, video_sha, ts_str)
                    break
                except Exception as e:
                    print(f"[EVIDENCE] ledger commit failed ({attempt + 1}/3): {e}")
                    time.sleep(1.0)
            if rec and self.on_sealed:
                self.on_sealed(rec)
        except Exception:
            traceback.print_exc()
        finally:
            if want_clip:
                self._slots.release()
            with self._lock:
                self.pending -= 1

    def _wait_until(self, cap, t_target, timeout=120.0):
        end = time.time() + timeout
        while (cap.now() < t_target and not getattr(cap, "ended", False)
               and time.time() < end):
            time.sleep(0.05)
        time.sleep(0.1)

    @staticmethod
    def _disk_ok():
        try:
            return shutil.disk_usage(".").free > config.MIN_FREE_DISK_MB * 1024 * 1024
        except OSError:
            return True

    @staticmethod
    def _write_clip(frames, path):
        try:
            h, w = frames[0][1].shape[:2]
            span = max(frames[-1][0] - frames[0][0], 1e-3)
            fps = min(max((len(frames) - 1) / span, 5.0), 60.0)   # real fps, so playback speed is true
            raw = path[:-4] + "_raw.mp4"
            writer, used = None, None
            for cc in ("avc1", "H264", "mp4v"):
                wr = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*cc), fps, (w, h))
                if wr.isOpened():
                    writer, used = wr, cc
                    break
                wr.release()
            if writer is None:
                return False
            for _, f in frames:
                writer.write(f if f.shape[:2] == (h, w) else cv2.resize(f, (w, h)))
            writer.release()
            ff = _ffmpeg_exe() if used == "mp4v" else None   # mp4v does not play in browsers
            if ff:
                try:
                    subprocess.run([ff, "-y", "-loglevel", "error", "-i", raw, "-c:v", "libx264",
                                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", path],
                                   check=True, timeout=90)
                    os.remove(raw)
                    return True
                except Exception as e:
                    print(f"[EVIDENCE] ffmpeg transcode failed, keeping raw clip: {e}")
            os.replace(raw, path)
            return os.path.getsize(path) > 1000
        except Exception as e:
            print(f"[EVIDENCE] clip encode failed: {e}")
            return False