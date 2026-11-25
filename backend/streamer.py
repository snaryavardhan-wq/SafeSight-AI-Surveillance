"""Live view for the dashboard: a tiny in-memory MJPEG server.

The engine hands over each annotated frame by reference; JPEG encoding only
happens while a browser is actually watching, and nothing is written to disk.
Every request needs the random per-run token (published in telemetry.json,
which the logged-in dashboard reads), and the server listens on 127.0.0.1
unless you set SAFESIGHT_STREAM_HOST.
"""
import json
import time
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import cv2

import config


class LiveStreamer:
    def __init__(self, host=None, port=None):
        self.host = host or config.LIVE_STREAM_HOST
        self.port = config.LIVE_STREAM_PORT if port is None else port
        self.token = secrets.token_urlsafe(12)
        self.running = False
        self.clients = 0
        self._cond = threading.Condition()
        self._lock = threading.Lock()
        self._frame = None
        self._seq = 0
        self._jpeg = (0, b"")
        self._server = None
        self.on_command = None          # callable(dict) -> bool, set by the engine

    # ---- producer (engine thread) ----
    def update(self, frame):
        with self._cond:
            self._frame = frame
            self._seq += 1
            self._cond.notify_all()

    # ---- consumers (HTTP threads) ----
    def wait_frame(self, last_seq, timeout=1.0):
        with self._cond:
            if self._seq == last_seq:
                self._cond.wait(timeout)
            return self._frame, self._seq

    def jpeg(self, frame, seq):
        with self._lock:                                   # encode once, share between viewers
            if self._jpeg[0] == seq:
                return self._jpeg[1]
            h, w = frame.shape[:2]
            if w > config.LIVE_STREAM_WIDTH:
                frame = cv2.resize(frame, (config.LIVE_STREAM_WIDTH, int(h * config.LIVE_STREAM_WIDTH / w)),
                                   interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), config.LIVE_STREAM_QUALITY])
            data = buf.tobytes() if ok else b""
            self._jpeg = (seq, data)
            return data

    # ---- server ----
    def start(self):
        streamer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _deny(self, code, msg):
                self.send_response(code)
                self.send_header("Content-Type", "text/plain")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(msg.encode())

            def do_OPTIONS(self):
                self.send_response(204)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")
                self.end_headers()

            def do_POST(self):
                u = urlparse(self.path)
                tok = (parse_qs(u.query).get("t") or [""])[0]
                if not secrets.compare_digest(tok, streamer.token):
                    return self._deny(403, "forbidden")
                if u.path != "/command":
                    return self._deny(404, "not found")
                try:
                    n = int(self.headers.get("Content-Length", "0"))
                    if not 0 < n <= 65536:
                        raise ValueError
                    cmd = json.loads(self.rfile.read(n).decode("utf-8"))
                    if not isinstance(cmd, dict):
                        raise ValueError
                except (ValueError, UnicodeDecodeError):
                    return self._deny(400, "bad request")
                cb = streamer.on_command
                if cb is None:
                    return self._deny(503, "no handler")
                if not cb(cmd):
                    return self._deny(400, "rejected")
                self.send_response(200)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Content-Length", "2")
                self.end_headers()
                self.wfile.write(b"ok")

            def do_GET(self):
                u = urlparse(self.path)
                tok = (parse_qs(u.query).get("t") or [""])[0]
                if not secrets.compare_digest(tok, streamer.token):
                    return self._deny(403, "forbidden")
                if u.path == "/snapshot.jpg":
                    frame, seq = streamer.wait_frame(-1, 0.0)
                    if frame is None:
                        return self._deny(503, "no frame yet")
                    data = streamer.jpeg(frame, seq)
                    self.send_response(200)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(data)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(data)
                elif u.path == "/stream.mjpg":
                    self._stream()
                else:
                    self._deny(404, "not found")

            def _stream(self):
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                with streamer._lock:
                    streamer.clients += 1
                last_seq, last_sent, interval = -1, 0.0, 1.0 / config.LIVE_STREAM_FPS
                try:
                    while streamer.running:
                        frame, seq = streamer.wait_frame(last_seq, 1.0)
                        if frame is None:
                            continue
                        now = time.time()
                        if seq == last_seq and now - last_sent < 1.0:      # keep-alive resend only
                            continue
                        if now - last_sent < interval:
                            time.sleep(interval - (now - last_sent))
                        data = streamer.jpeg(frame, seq)
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                         + str(len(data)).encode() + b"\r\n\r\n" + data + b"\r\n")
                        last_seq, last_sent = seq, time.time()
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                    pass
                finally:
                    with streamer._lock:
                        streamer.clients -= 1

        try:
            self._server = ThreadingHTTPServer((self.host, self.port), Handler)
        except OSError as e:
            print(f"[STREAM] disabled, cannot listen on {self.host}:{self.port} ({e})")
            return False
        self._server.daemon_threads = True
        self.port = self._server.server_address[1]
        self.running = True
        threading.Thread(target=self._server.serve_forever, name="live-stream", daemon=True).start()
        print(f"[STREAM] live view on http://{self.host}:{self.port}/  (shown inside the dashboard)")
        return True

    def stop(self):
        self.running = False
        with self._cond:
            self._cond.notify_all()
        if self._server:
            self._server.shutdown()
            self._server.server_close()