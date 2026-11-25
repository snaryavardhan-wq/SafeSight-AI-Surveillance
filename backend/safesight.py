"""SafeSight autonomous AI sentry: main engine.

Pipeline: ThreadedCapture (RAM ring buffer) -> [TamperDetector || YOLOv8n+ByteTrack]
-> spatial analytics -> ThreatArbiter -> EvidenceManager (clip + SHA-256 ledger)
-> voice / Telegram / dashboard telemetry.

Keys: q quit | v enrol VIP | b enrol blacklist | x purge faces | c clear zone
      f toggle face scan | t thermal | n night-vision | o normal
Zone: left-click adds a polygon vertex, right-click closes it.
"""
import os
import re
import json
import queue
import argparse
import threading
from urllib.parse import urlparse
import math
import time
import shutil
import traceback

import cv2
import numpy as np

import lapshim
lapshim.install_lap_shim()      # must run before ultralytics is imported (Windows may block lap's DLL)
from ultralytics import YOLO

import config
import database
import telegram_notifier as tg
from capture import ThreadedCapture, mask_url
from tamper import TamperDetector
from spatial import SprintDetector, BagTracker, PolygonZone
from arbiter import ThreatArbiter, level_name
from audio import SoundAlertManager
from evidence import EvidenceManager
from filecapture import VideoFileCapture, fmt_time
from streamer import LiveStreamer

LEVEL_COLORS = {"NORMAL": (0, 200, 0), "CAUTION": (0, 215, 255),
                "HIGH": (0, 140, 255), "CRITICAL": (0, 0, 255)}


def ensure_dirs():
    for d in (config.SNAPSHOT_DIR, config.VIDEO_DIR, config.FACES_DIR, config.REPORTS_DIR):
        os.makedirs(d, exist_ok=True)


# ======================================================================
# Face engine (auxiliary; can be toggled at runtime to save CPU)
# ======================================================================
class FaceSecurityEngine:
    def __init__(self):
        self.cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.profiles = {}
        self.load()

    @staticmethod
    def _prep(gray_face):
        return cv2.equalizeHist(cv2.resize(gray_face, (100, 100)))

    def load(self):
        self.profiles.clear()
        if not os.path.exists(config.FACES_DIR):
            return
        for folder in os.listdir(config.FACES_DIR):
            path = os.path.join(config.FACES_DIR, folder)
            if not os.path.isdir(path):
                continue
            parts = folder.split("_", 1)
            role = parts[0].upper() if len(parts) > 1 else "UNKNOWN"
            name = parts[1] if len(parts) > 1 else parts[0]
            tmpl = []
            for fn in os.listdir(path):
                img = cv2.imread(os.path.join(path, fn), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    tmpl.append(self._prep(img))
            if tmpl:
                self.profiles[name] = {"role": role, "templates": tmpl}

    def purge_all(self):
        shutil.rmtree(config.FACES_DIR, ignore_errors=True)
        os.makedirs(config.FACES_DIR, exist_ok=True)
        self.profiles.clear()

    def detect(self, gray):
        small = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5)
        boxes = self.cascade.detectMultiScale(small, scaleFactor=1.15, minNeighbors=4, minSize=(30, 30))
        return [[x * 2, y * 2, w * 2, h * 2] for (x, y, w, h) in boxes]

    def match(self, crop):
        if not self.profiles:
            return "Visitor", "VISITOR", 0.0
        norm = self._prep(crop)
        best_n, best_r, best_s = "Visitor", "VISITOR", -1.0
        for name, d in self.profiles.items():
            for t in d["templates"]:
                s = float(cv2.matchTemplate(norm, t, cv2.TM_CCOEFF_NORMED)[0][0])
                if s > best_s:
                    best_n, best_r, best_s = name, d["role"], s
        return (best_n, best_r, best_s) if best_s >= 0.40 else ("Visitor", "VISITOR", best_s)

    def analyze(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        out = []
        for x, y, w, h in self.detect(gray):
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
            if (x2 - x1) < 25 or (y2 - y1) < 25:
                continue
            name, role, s = self.match(gray[y1:y2, x1:x2])
            out.append({"box": (x1, y1, x2 - x1, y2 - y1), "name": name, "role": role, "confidence": s})
        return out


# ======================================================================
# Engine
# ======================================================================
class SentryEngine:
    def __init__(self, capture, streamer=None):
        ensure_dirs()
        database.init_db()
        self.cap = capture
        self.streamer = streamer
        self.is_file = bool(getattr(capture, "is_file", False))
        self.source_name = getattr(capture, "source_name", "camera")
        self.source_kind = getattr(capture, "source_kind", "webcam")
        self.zone_key = "zone_polygon" if self.source_kind == "webcam" else f"zone_polygon:{self.source_name}"
        self._cmd_q = queue.Queue(maxsize=64)
        self._frame_size = (config.CAPTURE_WIDTH, config.CAPTURE_HEIGHT)
        self.source_status = {"state": "ok", "message": ""}
        self._last_poll, self._switching, self._pending_cap = 0.0, False, None
        self.run_events, self.complete = [], False
        self.frames_done, self._t_first = 0, None
        self.summary_text, self.summary_path = "", ""
        self._gen = getattr(capture, "generation", 0)
        self._last_status = (0, 0, 0, "NORMAL", [], False)
        self.timing, self._tsum, self._tcnt = {}, {}, {}
        self.detect_calls = self.detect_skipped = 0
        self.last_motion = 0.0
        self.detections = []                    # every alert the engine raised: {event, track, score, t, key, details}
        self.model = self._load_model()
        self.sound = SoundAlertManager()
        self.evidence = EvidenceManager(capture, on_sealed=self._on_sealed)
        self.face = FaceSecurityEngine()
        self.zone = PolygonZone()
        self._load_zone()
        self.vision_mode = "NORMAL"
        self.clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        self.enrolling = False
        self.enroll_role, self.enroll_name, self.enroll_dir = "VIP", "Ashish", ""
        self.enroll_count, self.enroll_last, self.enroll_target = 0, 0.0, 10
        self.fps = 0.0
        self._tele_t = 0.0
        if self.streamer:
            self.streamer.on_command = self.remote_command
        database.take_source_request()          # ignore any stale request left over from a previous run
        self.reset_state()

    @staticmethod
    def _load_model():
        """Load the configured model; an exported (OpenVINO/ONNX) model that fails to load falls back to the .pt file."""
        try:
            return YOLO(config.YOLO_MODEL)
        except Exception as e:
            if config.YOLO_MODEL == "yolov8n.pt":
                raise
            print(f"[MODEL] could not load {config.YOLO_MODEL} ({e}); falling back to yolov8n.pt")
            return YOLO("yolov8n.pt")

    def mute(self):
        """No voice or siren (benchmarks and evaluation runs)."""
        class _Silent:
            def speak(self, *a, **k): return False
            def siren(self, *a, **k): return False
        self.sound = _Silent()

    # ---------------- profiling ----------------
    def _tick(self, name, t0):
        ms = (time.perf_counter() - t0) * 1000.0
        prev = self.timing.get(name)
        self.timing[name] = ms if prev is None else 0.9 * prev + 0.1 * ms
        self._tsum[name] = self._tsum.get(name, 0.0) + ms
        self._tcnt[name] = self._tcnt.get(name, 0) + 1

    def reset_timing(self):
        self._tsum, self._tcnt = {}, {}
        self.detect_calls = self.detect_skipped = 0

    def timing_avg(self):
        return {k: self._tsum[k] / self._tcnt[k] for k in self._tsum if self._tcnt.get(k)}

    # ---------------- state ----------------
    def reset_state(self):
        self._last_res, self._ref_small, self._cur_small, self._last_detect_ts = None, None, None, -1e18
        self.tamper = TamperDetector()
        self.arbiter = ThreatArbiter()
        self.sprint = SprintDetector()
        self.bags = BagTracker()
        self.fired = {}
        self.zone_entry = {}
        self.loiter_said = set()
        self.crowd_start = None
        self.frame_counter = 0
        self.cached_faces = []

    def recover(self):
        """Called after repeated per-frame errors: rebuild trackers and the model."""
        print("[ENGINE] Too many consecutive errors, rebuilding pipeline state...")
        self.reset_state()
        try:
            self.model = self._load_model()
        except Exception:
            traceback.print_exc()

    def _load_zone(self):
        raw = database.get_config_val(self.zone_key, "")
        try:
            pts = json.loads(raw) if raw else []
            if len(pts) >= 3:
                self.zone.set_points(pts)
        except (ValueError, TypeError):
            pass

    def zone_mouse(self, event, x, y, flags, param):
        was_closed = self.zone.closed
        self.zone.on_mouse(event, x, y)
        if self.zone.closed and not was_closed:
            database.set_config_val(self.zone_key, json.dumps(self.zone.points))

    def _on_sealed(self, rec):
        print(f"[LEDGER] Event #{rec['id']} sealed: {rec['event_type']} seal={rec['chain_hash'][:16]}")
        self.run_events.append(rec)
        if rec["score"] >= config.TELEGRAM_MIN_SCORE:
            tg.send_event_alert(rec)

    @staticmethod
    def _sig(score, label, event=None, track=None, details="", key=None, voice=None):
        return {"score": score, "label": label, "event": event, "track": track,
                "details": details, "key": key or f"{event}:{track}", "voice": voice}

    # ---------------- arbitration + evidence ----------------
    def _dispatch(self, signals, frame, now):
        score, level, labels = self.arbiter.update([(s["score"], s["label"]) for s in signals], now)
        for s in signals:
            if not s["event"] or s["score"] < config.EVIDENCE_MIN_SCORE:
                continue
            last = self.fired.get(s["key"])
            self.fired[s["key"]] = now
            if last is None or now - last > config.REARM_SEC:      # edge-trigger, not level-trigger
                self.detections.append({"event": s["event"], "track": s["track"], "score": s["score"], "key": s["key"],
                                        "t": round(self.cap.position_sec, 3) if self.is_file else now,
                                        "details": s["details"]})
                if frame is not None:
                    self.evidence.trigger(s["event"], s["track"], s["score"], s["details"] + self.cap.describe_position(),
                                          frame, key=s["key"])
                self.sound.speak(s["voice"])
        for k in [k for k, v in self.fired.items() if now - v > config.REARM_SEC * 4]:
            del self.fired[k]
        return score, level, labels

    # ---------------- per frame ----------------
    def step(self, frame, ts):
        gen = getattr(self.cap, "generation", 0)
        if gen != self._gen:                       # footage restarted or looped: start every tracker fresh
            self._gen = gen
            self.reset_state()
        if self._t_first is None:
            self._t_first = time.time()
        self.frames_done += 1
        self._frame_size = (frame.shape[1], frame.shape[0])
        t_start = time.perf_counter()
        render = self._step_inner(frame, ts)
        self._tick("total", t_start)
        if self.streamer:
            t0 = time.perf_counter()
            self.streamer.update(render)
            self._tick("stream", t0)
        return render

    def _step_inner(self, frame, ts):
        self.frame_counter += 1
        if self.enrolling:
            return self._enroll_step(frame, ts)

        signals = []
        t0 = time.perf_counter()
        render = self._render_base(frame)
        self._tick("prep", t0)

        # A. anti-tamper
        t0 = time.perf_counter()
        t_state, t_reason, t_dur = self.tamper.analyze(frame, ts)
        m = self.tamper.metrics
        is_tampered = t_state == "TAMPERED"
        if is_tampered:
            signals.append(self._sig(
                config.SCORE_TAMPER, f"CAMERA TAMPER: {t_reason} ({t_dur:.1f}s)", "Hardware Camera Tampering",
                None, f"{t_reason} | Lum {m['mu']:.0f} Var {m['lap']:.1f} Sat {m['sat']:.0f}", "tamper",
                "Alert! Camera tampering detected."))
            self.sound.siren()                                        # cooldown-limited

        self._tick("tamper", t0)

        # B. faces (auxiliary)
        t0 = time.perf_counter()
        face_on = database.get_config_val("face_detection_enabled", "1") == "1"
        if face_on:
            if self.frame_counter % 3 == 0:
                self.cached_faces = self.face.analyze(frame)
            for it in self.cached_faces:
                fx, fy, fw, fh = it["box"]
                color = {"BLACKLIST": (0, 0, 255), "VIP": (0, 255, 0)}.get(it["role"], (255, 200, 0))
                if it["role"] == "BLACKLIST":
                    signals.append(self._sig(config.SCORE_HIGH, f"BLACKLIST: {it['name']}", "Blacklisted Face Match",
                                             None, f"{it['name']} score {it['confidence']:.2f}",
                                             f"face:{it['name']}", f"Alert! Blacklisted target {it['name']} spotted."))
                cv2.rectangle(render, (fx, fy), (fx + fw, fy + fh), color, 2)
                cv2.putText(render, f"{it['name']} ({it['role']})", (fx, max(12, fy - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        else:
            self.cached_faces = []

        self._tick("faces", t0)

        # C. detection + ByteTrack
        t0 = time.perf_counter()
        res = self._detect(frame, ts)
        self._tick("detect", t0)
        t0 = time.perf_counter()
        person_boxes, bag_dets = {}, []
        zone_breached = False
        if res.boxes is not None and len(res.boxes) > 0:
            xyxy = res.boxes.xyxy.cpu().numpy()
            cls = res.boxes.cls.cpu().numpy()
            ids = res.boxes.id.int().cpu().tolist() if res.boxes.id is not None else [None] * len(xyxy)
            for i, (box, c, tid) in enumerate(zip(xyxy, cls, ids)):
                name = self.model.names[int(c)]
                x1, y1, x2, y2 = map(int, box)
                if name in ("backpack", "handbag", "suitcase"):
                    bag_dets.append((x1, y1, x2, y2, name.upper()))
                    continue
                if name != "person":
                    continue
                tid = tid if tid is not None else -(i + 1)
                person_boxes[tid] = (x1, y1, x2, y2)
                cx, cy, h = (x1 + x2) // 2, (y1 + y2) // 2, max(y2 - y1, 1)

                v, sprinting = self.sprint.update(tid, ts, cx, cy, h)
                d = [x for x in (self.zone.signed_distance((cx, cy)), self.zone.signed_distance((cx, y2)))
                     if x is not None]
                dist = max(d) if d else None
                inside = dist is not None and dist >= 0
                near = dist is not None and -config.ZONE_PROXIMITY_PX <= dist < 0

                color = (0, 255, 0)
                if inside:
                    zone_breached, color = True, (0, 0, 255)
                    signals.append(self._sig(config.SCORE_HIGH, f"ZONE BREACH: ID {tid}", "Restricted Zone Intrusion",
                                             tid, f"Centroid ({cx},{cy}), {dist:.0f}px inside perimeter",
                                             f"zone:{tid}", "Warning! Restricted perimeter breached."))
                    entered = self.zone_entry.setdefault(tid, ts)
                    if ts - entered >= config.LOITER_SEC:
                        signals.append(self._sig(config.SCORE_HIGH, f"LOITERING ID {tid} ({ts - entered:.0f}s)"))
                        if tid not in self.loiter_said:
                            self.loiter_said.add(tid)
                            self.sound.speak("Security notice. Loitering detected.")
                else:
                    self.zone_entry.pop(tid, None)
                    self.loiter_said.discard(tid)
                    if near:
                        color = (0, 215, 255)
                        signals.append(self._sig(config.SCORE_CAUTION, f"Zone proximity: ID {tid}"))
                if sprinting:
                    color = (0, 140, 255)
                    signals.append(self._sig(config.SCORE_HIGH, f"SPRINT ID {tid} ({v:.2f} bh/s)", "Sprint Anomaly",
                                             tid, f"v_norm = {v:.2f} body-heights/s", f"sprint:{tid}"))
                cv2.rectangle(render, (x1, y1), (x2, y2), color, 2)
                cv2.putText(render, f"ID:{tid} v:{v:.2f}bh/s", (x1, max(12, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        self.sprint.prune(set(person_boxes))
        for tid in [t for t in self.zone_entry if t not in person_boxes]:
            self.zone_entry.pop(tid, None)
            self.loiter_said.discard(tid)
        self.zone.draw(render, zone_breached)

        self._tick("spatial", t0)

        # D. unattended baggage (AABB edge distance)
        t0 = time.perf_counter()
        for b in self.bags.update(bag_dets, ts, person_boxes):
            x1, y1, x2, y2 = b["box"]
            if b["state"] == "UNATTENDED":
                color = (0, 0, 255)
                signals.append(self._sig(
                    config.SCORE_UNATTENDED, f"UNATTENDED {b['label']} #{b['id']} ({b['seconds']:.0f}s)",
                    "Unattended Baggage Anomaly", b["id"],
                    f"{b['label']} unattended {b['seconds']:.1f}s, D_edge > {config.BAG_ABANDON_DIST_PX}px",
                    f"bag:{b['id']}", f"Security alert. Unattended {b['label'].lower()} detected."))
            elif b["state"] == "SEPARATING":
                color = (0, 165, 255)
                signals.append(self._sig(config.SCORE_CAUTION, f"Bag #{b['id']} separating ({b['seconds']:.1f}s)"))
            else:
                color = (0, 255, 0)
            cv2.rectangle(render, (x1, y1), (x2, y2), color, 2)
            cv2.putText(render, f"{b['label']} #{b['id']} [{b['state']}]", (x1, max(12, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        # E. crowd (auxiliary, CAUTION)
        if len(person_boxes) >= config.CROWD_THRESHOLD:
            self.crowd_start = self.crowd_start or ts
            if ts - self.crowd_start >= config.CROWD_SEC:
                signals.append(self._sig(config.SCORE_CAUTION, f"Crowd: {len(person_boxes)} people"))
        else:
            self.crowd_start = None

        self._tick("analytics", t0)

        # F. arbitrate, record, display
        t0 = time.perf_counter()
        score, level, labels = self._dispatch(signals, frame, ts)
        self._hud(render, score, level, labels, face_on, is_tampered)
        self._telemetry(len(person_boxes), len(bag_dets), score, level, labels, face_on)
        self._tick("hud", t0)
        return render

    def _detect(self, frame, ts):
        """YOLOv8n + ByteTrack, optionally skipped while the scene is static (motion gate)."""
        if config.MOTION_GATE and self._last_res is not None and not self._motion_or_due(frame, ts):
            self.detect_skipped += 1
            return self._last_res                  # same boxes as the last detection: a static scene hasn't changed
        res = self.model.track(frame, persist=True, tracker=config.TRACKER_CFG, verbose=False,
                               classes=config.TARGET_CLASSES, conf=config.YOLO_CONF, imgsz=config.YOLO_IMGSZ)[0]
        self._last_res, self._last_detect_ts = res, ts
        self._ref_small = self._cur_small       # motion is measured against the scene at the last detection
        self.detect_calls += 1
        return res

    def _motion_or_due(self, frame, ts):
        small = cv2.GaussianBlur(cv2.cvtColor(cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA),
                                              cv2.COLOR_BGR2GRAY), (5, 5), 0)
        self._cur_small = small
        ref = self._ref_small
        if ref is None:
            return True
        self.last_motion = float((cv2.absdiff(small, ref) > config.MOTION_PIXEL_DELTA).mean())
        return self.last_motion >= config.MOTION_MIN_FRACTION or ts - self._last_detect_ts >= config.MOTION_REFRESH_SEC

    # ---------------- signal lost (camera unplugged / sabotaged) ----------------
    def handle_idle(self):
        off = self.cap.offline_since
        if off is None or time.time() - off < config.SIGNAL_LOST_SEC:
            return None
        dur = time.time() - off
        last = self.cap.last_frame()
        frame = last[1] if last else None
        canvas = frame.copy() if frame is not None else np.zeros(
            (config.CAPTURE_HEIGHT, config.CAPTURE_WIDTH, 3), np.uint8)
        sig = [self._sig(config.SCORE_TAMPER, f"CAMERA SIGNAL LOST ({dur:.0f}s)", "Camera Signal Lost", None,
                         f"No frames for {dur:.1f}s, reconnect attempts: {self.cap.reconnect_count}",
                         "lost", "Alert! Camera signal lost.")]
        score, level, labels = self._dispatch(sig, frame, time.time())
        self.sound.siren()
        self._hud(canvas, score, level, labels, False, True)
        self._telemetry(0, 0, score, level, labels, False)
        if self.streamer:
            self.streamer.update(canvas)
        return canvas

    # ---------------- rendering ----------------
    def _render_base(self, frame):
        if self.vision_mode == "THERMAL":
            return cv2.applyColorMap(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLORMAP_INFERNO)
        if self.vision_mode == "NIGHT_VISION":
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            out = cv2.cvtColor(cv2.merge((self.clahe.apply(l), a, b)), cv2.COLOR_LAB2BGR).astype(np.float32)
            out[:, :, 0] *= 0.7
            out[:, :, 2] *= 0.7
            out[:, :, 1] += 30
            return np.clip(out, 0, 255).astype(np.uint8)
        return frame.copy()

    def _hud(self, img, score, level, labels, face_on, tampered):
        color = LEVEL_COLORS[level]
        overlay = img.copy()
        cv2.rectangle(overlay, (12, 12), (520, 84), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)
        cv2.putText(img, f"SAFESIGHT [{self.vision_mode}] FPS:{self.fps:.0f} Face:{'ON' if face_on else 'OFF'} "
                         f"Buf:{self.cap.buffer_len}/{config.BUFFER_FRAMES}",
                    (22, 32), cv2.FONT_HERSHEY_DUPLEX, 0.42, (255, 255, 255), 1)
        cv2.putText(img, f"STATUS: {level} ({score}/100)", (22, 56), cv2.FONT_HERSHEY_DUPLEX, 0.55, color, 2)
        if labels:
            cv2.putText(img, labels[0][:60], (22, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 180, 255), 1)
        if not self.zone.closed:
            cv2.putText(img, "Zone: L-click add point, R-click close", (22, img.shape[0] - (50 if self.is_file else 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        if tampered:
            cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), (0, 0, 255), 6)
        if self.is_file:
            self._footage_overlay(img)

    def tick_telemetry(self):
        self._telemetry(*self._last_status)

    def _telemetry(self, persons, bags, score, level, labels, face_on):
        self._last_status = (persons, bags, score, level, labels, face_on)
        now = time.time()
        if now - self._tele_t < config.TELEMETRY_INTERVAL:
            return
        self._tele_t = now
        data = {"ts": now, "fps": round(self.fps, 1), "capture_fps": round(self.cap.capture_fps, 1),
                "persons": persons, "bags": bags, "score": score, "level": level, "reasons": labels[:3],
                "buffer": self.cap.buffer_len, "buffer_cap": config.BUFFER_FRAMES,
                "camera_connected": self.cap.connected, "mode": self.vision_mode,
                "face_enabled": face_on, "pending_evidence": self.evidence.pending,
                "source_type": "file" if self.is_file else "camera", "source": self.source_name,
                "source_kind": self.source_kind, "source_status": self.source_status,
                "timing_ms": {k: round(v, 1) for k, v in self.timing.items()},
                "detect_calls": self.detect_calls, "detect_skipped": self.detect_skipped,
                "position": round(self.cap.position_sec, 1) if self.cap.position_sec is not None else None,
                "duration": self.cap.duration_sec, "paused": bool(self.cap.paused),
                "complete": self.complete, "run_events": len(self.run_events),
                "analysis_mode": "every-frame" if getattr(self.cap, "every_frame", False) else "real-time",
                "stream": ({"port": self.streamer.port, "token": self.streamer.token}
                           if self.streamer and self.streamer.running else None)}
        try:                                                     # atomic replace: dashboard never reads a half file
            tmp = config.TELEMETRY_PATH + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f)
            os.replace(tmp, config.TELEMETRY_PATH)
        except OSError:
            pass

    # ---------------- remote control (dashboard) ----------------
    def remote_command(self, cmd):
        """Called from the stream server's HTTP thread: validate and queue; the main loop executes it."""
        name = cmd.get("cmd")
        if name == "zone":
            pts = cmd.get("points")
            try:
                ok = isinstance(pts, list) and 3 <= len(pts) <= 64 and all(
                    len(p) == 2 and 0.0 <= float(p[0]) <= 1.0 and 0.0 <= float(p[1]) <= 1.0 for p in pts)
            except (TypeError, ValueError):
                ok = False
            if not ok:
                return False
        elif name not in ("zone_clear", "toggle_pause", "restart"):
            return False
        try:
            self._cmd_q.put_nowait(cmd)
            return True
        except queue.Full:
            return False

    def process_commands(self):
        while True:
            try:
                cmd = self._cmd_q.get_nowait()
            except queue.Empty:
                return
            name = cmd["cmd"]
            if name == "zone":
                w, h = self._frame_size
                self.zone.set_points([(int(float(x) * w), int(float(y) * h)) for x, y in cmd["points"]])
                database.set_config_val(self.zone_key, json.dumps(self.zone.points))
            elif name == "zone_clear":
                self.zone.clear()
                database.set_config_val(self.zone_key, "")
            elif name == "toggle_pause" and self.is_file:
                self.cap.toggle_pause()
            elif name == "restart" and self.is_file:
                self.restart_run()

    # ---------------- source switching (dashboard "Camera Source" panel) ----------------
    def _validate_request(self, req):
        kind = req.get("type")
        spec = {"kind": kind, "every_frame": bool(req.get("every_frame")), "loop": bool(req.get("loop"))}
        if kind == "webcam":
            spec["label"] = "webcam"
        elif kind == "cctv":
            url = str(req.get("value", "")).strip()
            parsed = urlparse(url)
            if parsed.scheme.lower() not in config.ALLOWED_URL_SCHEMES or not parsed.hostname:
                raise ValueError("Enter a camera address that starts with rtsp://, rtsps://, http:// or https://")
            spec.update(value=url, label=mask_url(url))
        elif kind == "file":
            path = os.path.realpath(str(req.get("value", "")))
            root = os.path.realpath(config.FOOTAGE_DIR)
            try:
                inside = os.path.commonpath([path, root]) == root
            except ValueError:
                inside = False
            if not inside:
                raise ValueError(f"Videos must be inside the {config.FOOTAGE_DIR} folder (use the uploader).")
            if os.path.splitext(path)[1].lower() not in config.VIDEO_EXTENSIONS:
                raise ValueError("Unsupported video type. Use .mp4, .avi, .mov, .mkv or .wmv")
            if not os.path.isfile(path):
                raise ValueError("That video file no longer exists.")
            spec.update(value=path, label=os.path.basename(path))
        else:
            raise ValueError("Unknown source type.")
        return spec

    def poll_requests(self):
        """Main loop, about once a second. Returns True when the source was swapped (reset frame counters)."""
        now = time.time()
        if now - self._last_poll >= 1.0 and not self._switching:
            self._last_poll = now
            req = database.take_source_request()
            if req:
                self._begin_switch(req)
        if self._pending_cap is not None:
            self._complete_switch()
            return True
        return False

    def _begin_switch(self, req):
        try:
            spec = self._validate_request(req)
        except ValueError as e:
            self.source_status = {"state": "error", "message": str(e)}
            return
        if spec["kind"] == "webcam" and self.source_kind == "webcam":
            self.source_status = {"state": "ok", "message": "Already using the webcam."}
            return
        self._switching = True
        self.source_status = {"state": "connecting", "message": f"Connecting to {spec['label']}..."}
        threading.Thread(target=self._build_capture, args=(spec,), name="source-switch", daemon=True).start()

    def _build_capture(self, spec):
        """Worker thread: opening an RTSP camera can take seconds, so the sentry keeps running meanwhile."""
        try:
            if spec["kind"] == "file":
                paused = not database.get_config_val(f"zone_polygon:{spec['label']}", "", max_age=0.0)
                cap = VideoFileCapture(spec["value"], every_frame=spec["every_frame"], loop=spec["loop"],
                                       start_paused=paused)
            else:
                cap = ThreadedCapture(src=config.CAMERA_SOURCE if spec["kind"] == "webcam" else spec["value"])
                end = time.time() + config.SOURCE_CONNECT_TIMEOUT
                while time.time() < end and not (cap.connected and cap.buffer_len > 0):
                    time.sleep(0.1)
                if not (cap.connected and cap.buffer_len > 0):
                    cap.release()
                    raise RuntimeError("No video received from the source. Check the address, login and network.")
            self._pending_cap = cap
        except Exception as e:
            self.source_status = {"state": "error", "message": str(e) or e.__class__.__name__}
            self._switching = False

    def _complete_switch(self):
        cap, self._pending_cap = self._pending_cap, None
        old = self.cap
        self.cap = cap
        self.evidence.capture = cap
        self.is_file = bool(getattr(cap, "is_file", False))
        self.source_name = cap.source_name
        self.source_kind = getattr(cap, "source_kind", "webcam")
        self.zone_key = "zone_polygon" if self.source_kind == "webcam" else f"zone_polygon:{self.source_name}"
        self.zone.clear()
        self._load_zone()
        self.run_events, self.complete = [], False
        self.detections = []
        self.frames_done, self._t_first = 0, None
        self._gen = getattr(cap, "generation", 0)
        self.reset_state()
        self.source_status = {"state": "ok", "message": f"Now analysing {self.source_name}"}
        self._switching = False
        print(f"[SOURCE] switched to {self.source_kind}: {self.source_name}")
        threading.Thread(target=lambda: (self.evidence.wait_idle(20), old.release()), daemon=True).start()

    # ---------------- recorded-footage helpers ----------------
    def _footage_overlay(self, img):
        h, w = img.shape[:2]
        pos, dur = self.cap.position_sec or 0, self.cap.duration_sec
        txt = f"FOOTAGE {self.source_name}  {fmt_time(pos)}" + (f" / {fmt_time(dur)}" if dur else "")
        cv2.rectangle(img, (0, h - 34), (w, h), (15, 15, 15), -1)
        cv2.putText(img, txt, (12, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        if dur:
            cv2.rectangle(img, (0, h - 5), (int(w * min(pos / dur, 1.0)), h), (0, 200, 255), -1)

    def _banner(self, img, text, color=(0, 200, 255)):
        h, w = img.shape[:2]
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, 44), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.8, img, 0.2, 0, img)
        cv2.putText(img, text, (14, 29), cv2.FONT_HERSHEY_DUPLEX, 0.62, color, 1)

    def preview(self, frame):
        """Paused / zone-setup view: the frame, the zone being drawn, no analysis."""
        self._frame_size = (frame.shape[1], frame.shape[0])
        render = frame.copy()
        self.zone.draw(render, False)
        self._footage_overlay(render)
        if self.cap.idx <= 1:
            self._banner(render, "SETUP: draw the restricted zone (optional), then press Play / SPACE")
        else:
            self._banner(render, "PAUSED: SPACE to resume, R to restart")
        if self.streamer:
            self.streamer.update(render)
        return render

    def complete_view(self):
        last = self.cap.last_frame()
        render = last[1].copy() if last else np.zeros((config.CAPTURE_HEIGHT, config.CAPTURE_WIDTH, 3), np.uint8)
        crit = sum(1 for r in self.run_events if r["level"] == "CRITICAL")
        high = sum(1 for r in self.run_events if r["level"] == "HIGH")
        self._footage_overlay(render)
        self._banner(render, f"ANALYSIS COMPLETE: {len(self.run_events)} incident(s) "
                             f"({crit} CRITICAL, {high} HIGH)   R restart | Q quit", (0, 255, 120))
        if self.streamer:
            self.streamer.update(render)
        return render

    def restart_run(self):
        if not self.is_file:
            return
        self.cap.restart()
        self.complete = False
        self.run_events.clear()
        self.detections.clear()
        self.frames_done, self._t_first = 0, None
        self.reset_state()

    def finalize(self, write_pdf=False):
        """End of footage: wait for evidence to seal, then write and print the run summary."""
        self.evidence.wait_idle(90)
        recs = sorted(self.run_events, key=lambda r: r["id"])
        elapsed = max(time.time() - (self._t_first or time.time()), 1e-3)
        dur = self.cap.duration_sec
        by_level = {}
        for r in recs:
            by_level[r["level"]] = by_level.get(r["level"], 0) + 1
        lines = ["=" * 72, f"  SAFESIGHT FOOTAGE ANALYSIS: {self.source_name}", "=" * 72,
                 f"  Footage length : {fmt_time(dur) if dur else 'unknown'}",
                 f"  Mode           : {'every frame' if getattr(self.cap, 'every_frame', False) else 'real-time'}",
                 f"  Frames analysed: {self.frames_done} (average {self.frames_done / elapsed:.1f} FPS)",
                 f"  Incidents      : {len(recs)}  " + ", ".join(f"{n} {k}" for k, n in sorted(by_level.items())),
                 "-" * 72]
        for r in recs:
            lines.append(f"  #{r['id']:<4} {r['level']:<9} {r['event_type']:<28} {r['details']}")
        if not recs:
            lines.append("  No incidents detected.")
        lines.append("=" * 72)
        self.summary_text = "\n".join(lines)
        try:
            os.makedirs(config.REPORTS_DIR, exist_ok=True)
            safe = re.sub(r"[^A-Za-z0-9]+", "_", os.path.splitext(self.source_name)[0]).strip("_") or "footage"
            self.summary_path = os.path.join(config.REPORTS_DIR,
                                             f"footage_summary_{safe}_{time.strftime('%Y%m%d_%H%M%S')}.txt")
            with open(self.summary_path, "w", encoding="utf-8") as f:
                f.write(self.summary_text + "\n")
        except OSError as e:
            print(f"[SUMMARY] could not save: {e}")
        print(self.summary_text)
        if write_pdf:
            try:
                from pdf_report_generator import generate_pdf_dossier
                print("[PDF] dossier:", generate_pdf_dossier())
            except Exception as e:
                print(f"[PDF] could not generate dossier: {e}")
        self.complete = True
        return self.summary_text

    # ---------------- enrolment ----------------
    def _start_enroll(self, role, name):
        self.enroll_role, self.enroll_name = role, name
        self.enroll_dir = os.path.join(config.FACES_DIR, f"{role}_{name}")
        os.makedirs(self.enroll_dir, exist_ok=True)
        self.enroll_count, self.enrolling = 0, True

    def _enroll_step(self, frame, ts):
        render = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.putText(render, f"ENROLLING {self.enroll_role}: {self.enroll_name.upper()} "
                            f"({self.enroll_count}/{self.enroll_target})",
                    (20, 36), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 255), 2)
        faces = self.face.detect(gray)
        if faces:
            x, y, w, h = faces[0]
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
            cv2.rectangle(render, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if ts - self.enroll_last > 0.18 and (y2 > y1 and x2 > x1):
                cv2.imwrite(os.path.join(self.enroll_dir, f"sample_{self.enroll_count + 1}.jpg"), gray[y1:y2, x1:x2])
                self.enroll_count += 1
                self.enroll_last = ts
            if self.enroll_count >= self.enroll_target:
                self.enrolling = False
                self.face.load()
                self.sound.speak(f"{self.enroll_role} enrolled successfully.")
        return render

    # ---------------- keyboard ----------------
    def handle_key(self, k):
        """Return False to quit."""
        if k in (ord("q"), ord("Q")):
            return False
        if self.is_file and k == 32:                       # SPACE
            self.cap.toggle_pause()
            return True
        if self.is_file and k in (ord("r"), ord("R")):
            self.restart_run()
            return True
        if self.enrolling:
            return True
        if k in (ord("v"), ord("V")):
            self._start_enroll("VIP", "Ashish")
        elif k in (ord("b"), ord("B")):
            self._start_enroll("BLACKLIST", "Suspect")
        elif k in (ord("x"), ord("X")):
            self.face.purge_all()
        elif k in (ord("c"), ord("C")):
            self.zone.clear()
            database.set_config_val(self.zone_key, "")
        elif k in (ord("f"), ord("F")):
            cur = database.get_config_val("face_detection_enabled", "1", max_age=0)
            database.set_config_val("face_detection_enabled", "0" if cur == "1" else "1")
        elif k in (ord("t"), ord("T")):
            self.vision_mode = "THERMAL"
        elif k in (ord("n"), ord("N")):
            self.vision_mode = "NIGHT_VISION"
        elif k in (ord("o"), ord("O")):
            self.vision_mode = "NORMAL"
        return True


# ======================================================================
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="SafeSight sentry engine: live camera or recorded CCTV footage")
    ap.add_argument("--video", help="analyse a recorded CCTV file instead of the live camera")
    ap.add_argument("--loop", action="store_true", help="replay the footage forever")
    ap.add_argument("--every-frame", action="store_true",
                    help="analyse every frame (slower than real time, most accurate)")
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed in real-time mode (default 1.0)")
    ap.add_argument("--zone", help='restricted zone as "x,y;x,y;x,y" in video pixels')
    ap.add_argument("--no-window", action="store_true", help="no OpenCV window (use the dashboard live view)")
    ap.add_argument("--no-stream", action="store_true", help="disable the dashboard live-view stream")
    ap.add_argument("--no-setup", action="store_true", help="start at once, skip the zone-drawing pause")
    ap.add_argument("--exit-when-done", action="store_true", help="exit after the footage ends (batch mode)")
    ap.add_argument("--pdf", action="store_true", help="also generate the PDF dossier when the footage ends")
    ap.add_argument("--imgsz", type=int, help="detector input size, e.g. 480 or 320 (smaller = faster, default 640)")
    ap.add_argument("--model", help="YOLO weights or an exported folder (see export_model.py)")
    ap.add_argument("--no-face", action="store_true", help="start with face scanning off (saves CPU)")
    ap.add_argument("--motion-gate", action="store_true", help="skip the detector while the scene is static")
    return ap.parse_args(argv)


def main(argv=None, stop_event=None):
    args = parse_args(argv)
    ensure_dirs()
    database.init_db()
    if args.imgsz:
        config.YOLO_IMGSZ = args.imgsz
    if args.model:
        config.YOLO_MODEL = args.model
    if args.motion_gate:
        config.MOTION_GATE = True
    if args.no_face:
        database.set_config_val("face_detection_enabled", "0")
    is_file = bool(args.video)
    show = not args.no_window

    print("=" * 65)
    print("  SAFESIGHT SURVEILLANCE MATRIX STARTING" + (f"  [FOOTAGE: {os.path.basename(args.video)}]" if is_file else ""))
    print("=" * 65)
    if not tg.is_configured():
        print("[INFO] Telegram not configured: set TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID in .env")

    zone_pts = None
    if args.zone:
        try:
            zone_pts = [tuple(int(v) for v in p.split(",")) for p in args.zone.split(";")]
            assert len(zone_pts) >= 3 and all(len(p) == 2 for p in zone_pts)
        except (ValueError, AssertionError):
            raise SystemExit('--zone must look like "x,y;x,y;x,y" (at least 3 points)')

    if is_file:
        saved = database.get_config_val(f"zone_polygon:{os.path.basename(args.video)}", "")
        setup = show and not args.no_setup and not saved and not zone_pts
        try:
            cap = VideoFileCapture(args.video, every_frame=args.every_frame, speed=args.speed,
                                   loop=args.loop, start_paused=setup)
        except FileNotFoundError:
            raise SystemExit(f"\nVideo file not found: {args.video}\n"
                             f"  Current folder : {os.getcwd()}\n"
                             f"  Record one     : python backend/record_demo.py --seconds 45\n"
                             f"  Or use any .mp4: python backend/safesight.py --video \"C:\\full\\path\\clip.mp4\"")
        except ValueError as e:
            raise SystemExit(f"\n{e}\nTry another file; .mp4 (H.264) or .avi usually works.")
    else:
        cap = ThreadedCapture()

    streamer = None
    if config.LIVE_STREAM_ENABLED and not args.no_stream:
        streamer = LiveStreamer()
        if not streamer.start():
            streamer = None

    engine = SentryEngine(cap, streamer)
    if zone_pts:
        engine.zone.set_points(zone_pts)
        database.set_config_val(engine.zone_key, json.dumps(engine.zone.points))

    win = "SafeSight Autonomous AI Sentry Matrix"
    if show:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 1280, 720)
        cv2.setMouseCallback(win, engine.zone_mouse)

    last_seq, errors, prev = 0, 0, time.time()
    try:
        while not (stop_event and stop_event.is_set()):
            engine.process_commands()
            if engine.poll_requests():                 # dashboard switched the source
                last_seq = 0
            cap = engine.cap
            render = None
            if engine.is_file and cap.paused:
                fr = cap.last_frame()
                render = engine.preview(fr[1]) if fr else None
                time.sleep(0.03)
            elif engine.complete:
                render = engine.complete_view()
                time.sleep(0.03)
            else:
                item = cap.get_latest(last_seq, timeout=0.5)
                if item is None:
                    if engine.is_file and cap.ended:
                        engine.finalize(write_pdf=args.pdf)
                        if args.exit_when_done:
                            break
                    else:
                        render = engine.handle_idle()
                else:
                    frame, ts, last_seq = item
                    now = time.time()
                    inst = 1 / max(now - prev, 1e-3)
                    engine.fps = 0.9 * engine.fps + 0.1 * inst if engine.fps else inst
                    prev = now
                    try:
                        render = engine.step(frame, ts)
                        errors = 0
                    except Exception:                  # one bad frame must never kill the sentry
                        traceback.print_exc()
                        errors += 1
                        render = frame
                        if errors >= config.MAX_CONSECUTIVE_ERRORS:
                            engine.recover()
                            errors = 0
                            time.sleep(1.0)
            engine.tick_telemetry()
            if show:
                if render is not None:
                    cv2.imshow(win, render)
                if not engine.handle_key(cv2.waitKey(1) & 0xFF):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        engine.evidence.wait_idle(8.0)                 # let in-flight clips finish and seal
        engine.cap.release()
        if streamer:
            streamer.stop()
        if show:
            cv2.destroyAllWindows()
    return engine


if __name__ == "__main__":
    main()