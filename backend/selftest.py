"""Camera-free self-test:  python backend/selftest.py
Runs in a temp folder; needs only opencv-python + numpy (YOLO is stubbed)."""
import os
import sys
import time
import types
import shutil
import sqlite3
import tempfile
import threading
from collections import deque

import logging
import cv2
import numpy as np

logging.getLogger("streamlit").setLevel(logging.ERROR)   # keep the output readable

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
TMP = tempfile.mkdtemp(prefix="safesight_test_")
os.chdir(TMP)

import config
config.DB_PATH = os.path.join(TMP, "test.db")
config.TELEMETRY_PATH = os.path.join(TMP, "telemetry.json")
import database
from arbiter import ThreatArbiter
from tamper import TamperDetector
from spatial import box_edge_distance, SprintDetector, PolygonZone, BagTracker
from evidence import EvidenceManager

PASSED, FAILED = 0, 0


FAILED_NAMES = []


def check(name, cond):
    global PASSED, FAILED
    PASSED, FAILED = (PASSED + 1, FAILED) if cond else (PASSED, FAILED + 1)
    if not cond:
        FAILED_NAMES.append(name)
    print(("  PASS  " if cond else "  FAIL  ") + name)


class FakeCapture:
    """Same API as ThreadedCapture, fed with synthetic frames at 30 fps."""
    def __init__(self):
        self.buffer = deque(maxlen=config.BUFFER_FRAMES)
        self.lock = threading.Lock()
        self.connected, self.offline_since, self.capture_fps, self.reconnect_count = True, None, 30.0, 0
        self.is_file, self.ended, self.paused, self.generation, self.idx = False, False, False, 0, 0
        self.source_name, self.position_sec, self.duration_sec = 'camera', None, None
        self._stop = False
        threading.Thread(target=self._feed, daemon=True).start()

    def _feed(self):
        i = 0
        while not self._stop:
            f = np.full((360, 640, 3), 90, np.uint8)
            cv2.putText(f, str(i), (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
            f[::7, ::5] = np.random.randint(0, 255, 3)       # texture so tamper stays CLEAR
            with self.lock:
                self.buffer.append((time.time(), f))
            i += 1
            time.sleep(1 / 30)

    def get_window(self, t0, t1):
        with self.lock:
            return [(t, f) for t, f in self.buffer if t0 <= t <= t1]

    def last_frame(self):
        with self.lock:
            return self.buffer[-1] if self.buffer else None

    def now(self):
        return time.time()

    def describe_position(self):
        return ''

    @property
    def buffer_len(self):
        return len(self.buffer)

    def release(self):
        self._stop = True


# ------------------------------------------------------------------
print("\n[1] Ledger: WAL + hash chain")
database.init_db()
conn = sqlite3.connect(config.DB_PATH)
check("WAL mode enabled", conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal")
conn.close()
os.makedirs("snaps", exist_ok=True)
recs = []
for i in range(3):
    data = os.urandom(2000)
    p = f"snaps/e{i}.jpg"
    open(p, "wb").write(data)
    recs.append(database.commit_event(f"Test Event {i}", i, 90, "CRITICAL", "d", p, data))
check("clean ledger verifies", database.verify_ledger()["ok"])
with open("snaps/e1.jpg", "r+b") as f:
    f.write(b"X")
rep = database.verify_ledger()
check("modified snapshot detected", any(p["id"] == recs[1]["id"] for p in rep["problems"]))
open("snaps/e1.jpg", "wb").write(open("snaps/e1.jpg", "rb").read())
c = sqlite3.connect(config.DB_PATH)
c.execute("UPDATE security_events SET threat_score=10 WHERE id=?", (recs[2]["id"],))
c.commit()
check("edited score detected", any(p["id"] == recs[2]["id"] for p in database.verify_ledger()["problems"]))
c.execute("DELETE FROM security_events WHERE id=?", (recs[0]["id"],))
c.commit()
c.close()
check("deleted record breaks chain", any("CHAIN_BROKEN" in p["issue"] for p in database.verify_ledger()["problems"]))

# ------------------------------------------------------------------
print("\n[2] Arbiter (monotonic, max-based)")
a = ThreatArbiter()
s1 = a.update([(50, "x")], 0.0)[0]
s2 = a.update([(90, "x"), (50, "y")], 0.1)[0]
s3 = a.update([(50, "y")], 0.2)[0]
check("escalates 50 -> 90", (s1, s2) == (50, 90))
check("never decreases inside window", s3 == 90)
a.update([], 0.3)
check("still held 0.5s after clear", a.update([], 0.8)[0] == 90)
check("resets to NORMAL after quiet", a.update([], 1.5)[0] == 0)

# ------------------------------------------------------------------
print("\n[3] Anti-tamper")
rng = np.random.default_rng(1)
textured = rng.integers(0, 255, (720, 1280, 3), dtype=np.uint8)
cases = {"BLACKOUT": np.zeros((720, 1280, 3), np.uint8),
         "BLINDING/LASER": np.full((720, 1280, 3), 255, np.uint8),
         "OCCLUSION/DEFOCUS": np.full((720, 1280, 3), 128, np.uint8)}
for want, img in cases.items():
    d = TamperDetector()
    st0 = d.analyze(img, 0.0)[0]
    st1, reason, _ = d.analyze(img, 1.5)
    check(f"{want}: verifying then TAMPERED after 1.45s", st0 == "VERIFYING" and st1 == "TAMPERED" and reason == want)
d = TamperDetector()
check("textured frame is CLEAR", d.analyze(textured, 0.0)[0] == "CLEAR")

# ------------------------------------------------------------------
print("\n[4] Spatial analytics")
check("D_edge = 0 when overlapping", box_edge_distance((0, 0, 50, 50), (40, 40, 90, 90)) == 0)
check("D_edge diagonal gap", abs(box_edge_distance((0, 0, 10, 10), (13, 14, 20, 20)) - 5.0) < 1e-6)


def run_speed(h, px_per_s):
    sd, hit, v = SprintDetector(), False, 0
    for i in range(12):
        v, s = sd.update(1, i * 0.1, 100 + px_per_s * i * 0.1, 300, h)
        hit = hit or s
    return v, hit


vw, hw = run_speed(500, 120)      # close, slow walk
vr, hr = run_speed(100, 90)       # far, fast (0.9 bh/s)
check("near walker not flagged (raw pixels would be huge)", not hw and vw < 0.65)
check("far runner flagged (raw pixels tiny)", hr and vr >= 0.65)

z = PolygonZone()
z.set_points([(100, 100), (300, 100), (300, 300), (100, 300)])
check("inside polygon >= 0", z.signed_distance((200, 200)) >= 0)
check("near outside is -(0..60)", -60 <= z.signed_distance((330, 200)) < 0)
concave = PolygonZone()
concave.set_points([(0, 0), (200, 0), (200, 200), (100, 100), (0, 200)])
check("concave notch is outside", concave.signed_distance((100, 180)) < 0)

bt = BagTracker()
bag = [(500, 100, 560, 160, "BACKPACK")]
person = {1: (10, 10, 60, 200)}
st = None
for i in range(45):
    st = bt.update(bag, i * 0.1, person)[0]
check("bag far from owner -> UNATTENDED after 3.5s", st["state"] == "UNATTENDED")
check("bag next to owner -> CARRIED", bt.update(bag, 5.0, {1: (470, 90, 520, 250)})[0]["state"] == "CARRIED")

# ------------------------------------------------------------------
print("\n[5] Evidence: pre+post clip, hash, ledger")
database.init_db()
fresh = sqlite3.connect(config.DB_PATH)
fresh.execute("DELETE FROM security_events")
fresh.commit()
fresh.close()
cap = FakeCapture()
time.sleep(3.0)
ev = EvidenceManager(cap)
frame = cap.last_frame()[1]
check("trigger accepted", ev.trigger("Unit Test Event", 7, 90, "synthetic", frame, key="k1"))
check("cooldown suppresses duplicate", not ev.trigger("Unit Test Event", 7, 90, "synthetic", frame, key="k1"))
ev.wait_idle(15)
row = sqlite3.connect(config.DB_PATH).execute(
    "SELECT video_path, video_sha256, chain_hash FROM security_events").fetchone()
check("event sealed in ledger", row is not None and row[2])
vc = cv2.VideoCapture(row[0]) if row and row[0] else None
dur = (vc.get(cv2.CAP_PROP_FRAME_COUNT) / max(vc.get(cv2.CAP_PROP_FPS), 1)) if vc else 0
check(f"clip is ~5 s long (got {dur:.1f}s)", 4.0 <= dur <= 6.0)
check("full ledger verifies incl. video hash", database.verify_ledger()["ok"])

# ------------------------------------------------------------------
print("\n[6] Full engine step with stubbed YOLO")


class _T:
    def __init__(self, a): self.a = np.array(a)
    def cpu(self): return self
    def numpy(self): return self.a
    def int(self): return self
    def tolist(self): return self.a.tolist()


class _Boxes:
    def __init__(self, rows):
        self.xyxy = _T([r[1:5] for r in rows]); self.cls = _T([r[0] for r in rows])
        self.id = _T([r[5] for r in rows]); self._n = len(rows)
    def __len__(self): return self._n


class FakeYOLO:
    names = {0: "person", 24: "backpack", 26: "handbag", 28: "suitcase"}
    def __init__(self, *_): self.t = 0
    def track(self, *a, **k):
        self.t += 1
        rows = [(0, 150, 120, 250, 320, 1),            # person inside the zone
                (24, 900, 400, 960, 460, 5)]           # bag far from everyone
        return [types.SimpleNamespace(boxes=_Boxes(rows))]


sys.modules["ultralytics"] = types.SimpleNamespace(YOLO=FakeYOLO)
import safesight
safesight.tg.send_event_alert = lambda rec: None
fresh = sqlite3.connect(config.DB_PATH)
fresh.execute("DELETE FROM security_events")
fresh.commit()
fresh.close()
engine = safesight.SentryEngine(cap)
engine.zone.set_points([(100, 100), (300, 100), (300, 340), (100, 340)])
engine.fps = 10
errors = 0
last_level = None
for i in range(60):
    f = cap.last_frame()[1]
    try:
        r = engine.step(f, i * 0.1)
        last_level = engine.arbiter.window_peak
    except Exception as e:
        errors += 1
        print("   step error:", repr(e))
check("60 frames processed without exceptions", errors == 0)
check("arbiter escalated to CRITICAL (bag 90)", last_level == 90)
engine.evidence.wait_idle(15)
types_logged = {r[0] for r in sqlite3.connect(config.DB_PATH).execute("SELECT event_type FROM security_events")}
check("zone intrusion event recorded", "Restricted Zone Intrusion" in types_logged)
check("unattended baggage event recorded", "Unattended Baggage Anomaly" in types_logged)
check("edge-triggered (no spam): 2 events from 60 frames", len(types_logged) == 2 and
      sqlite3.connect(config.DB_PATH).execute("SELECT COUNT(*) FROM security_events").fetchone()[0] == 2)
check("telemetry file written", os.path.exists(config.TELEMETRY_PATH))
check("ledger still verifies", database.verify_ledger()["ok"])
for k in (ord("t"), ord("n"), ord("o"), ord("c")):
    engine.handle_key(k)
check("hotkeys do not crash", True)


# ------------------------------------------------------------------
print("\n[7] Live-view streamer (in-memory MJPEG)")
import http.client
from streamer import LiveStreamer
streamer = LiveStreamer(host="127.0.0.1", port=0)
check("streamer starts on a free port", streamer.start())
streamer.update(np.full((540, 960, 3), 120, np.uint8))
c = http.client.HTTPConnection("127.0.0.1", streamer.port, timeout=5)
c.request("GET", f"/snapshot.jpg?t={streamer.token}")
r = c.getresponse(); body = r.read()
check("snapshot with token is a JPEG", r.status == 200 and body[:2] == b"\xff\xd8")
c = http.client.HTTPConnection("127.0.0.1", streamer.port, timeout=5)
c.request("GET", "/snapshot.jpg?t=wrong")
check("wrong token is refused (403)", c.getresponse().status == 403)
c = http.client.HTTPConnection("127.0.0.1", streamer.port, timeout=5)
c.request("GET", f"/stream.mjpg?t={streamer.token}")
r = c.getresponse(); chunk = r.read(3000)
check("MJPEG stream delivers multipart JPEG frames", r.status == 200 and b"--frame" in chunk and b"image/jpeg" in chunk)
check("viewer is counted while connected", streamer.clients == 1)
r.close(); c.close()                                    # like a browser tab closing
for _ in range(6):                                     # the server notices a closed browser on its next write
    streamer.update(np.full((540, 960, 3), 60, np.uint8)); time.sleep(0.4)
check("viewer count returns to 0 after disconnect", streamer.clients == 0)
streamer.stop()

# ------------------------------------------------------------------
print("\n[8] Recorded footage input: both pacing modes, pause, restart")
from filecapture import VideoFileCapture, fmt_time
VID = os.path.join(TMP, "cctv_demo.mp4")
vw = cv2.VideoWriter(VID, cv2.VideoWriter_fourcc(*"mp4v"), 15, (640, 360))
for i in range(225):                                    # 15 s @ 15 fps
    f = np.full((360, 640, 3), 90, np.uint8)
    cv2.putText(f, str(i), (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    f[::7, ::5] = np.random.randint(0, 255, 3)
    vw.write(f)
vw.release()
check("time formatter", fmt_time(125) == "02:05")

short = os.path.join(TMP, "short.mp4")
vw = cv2.VideoWriter(short, cv2.VideoWriter_fourcc(*"mp4v"), 20, (320, 180))
for i in range(60):                                     # 3 s @ 20 fps
    f = np.full((180, 320, 3), 100, np.uint8); f[::5, ::3] = i * 4 % 255
    vw.write(f)
vw.release()

t0 = time.time(); seen = 0; last = 0
vc = VideoFileCapture(short)
while not vc.ended and time.time() - t0 < 10:
    it = vc.get_latest(last, 0.2)
    if it:
        seen += 1; last = it[2]
el = time.time() - t0
check(f"real-time mode paces at native speed (3 s footage took {el:.1f}s)", 2.5 <= el <= 4.5)
check("real-time mode reports end of footage", vc.ended)
vc.release()

vc = VideoFileCapture(short, every_frame=True)
stamps = []; last = 0; t0 = time.time()
while time.time() - t0 < 10:
    it = vc.get_latest(last, 0.3)
    if it:
        stamps.append(it[1]); last = it[2]
    elif vc.ended:
        break
check(f"every-frame mode delivers all 60 frames (got {len(stamps)})", len(stamps) == 60)
gaps = [b - a for a, b in zip(stamps, stamps[1:])]
check("footage clock advances 1/fps per frame", all(abs(g - 0.05) < 1e-6 for g in gaps))
check("position and duration are reported", abs(vc.duration_sec - 3.0) < 0.2 and abs(vc.position_sec - 3.0) < 0.2)
g0 = vc.generation
vc.restart(); time.sleep(0.4)
check("restart bumps the generation and plays again", vc.generation == g0 + 1 and vc.get_latest(last, 1.0) is not None)
vc.release()

vr = VideoFileCapture(short)                           # real-time mode: pause must not cause a catch-up burst
time.sleep(0.8)
vr.toggle_pause(); time.sleep(0.2); idx_p = vr.idx; time.sleep(0.8)
check("pause holds the frame counter", vr.idx == idx_p)
vr.toggle_pause(); time.sleep(0.4)
check("resume continues", vr.idx > idx_p)
check("no catch-up burst after a pause", vr.idx - idx_p <= 0.4 * 20 + 4)
vr.release()

# ------------------------------------------------------------------
print("\n[9] Whole program on recorded footage: safesight.main() with stubbed YOLO")
import json, glob
database.set_config_val("face_detection_enabled", "0")
fresh = sqlite3.connect(config.DB_PATH)
fresh.execute("DELETE FROM security_events"); fresh.commit(); fresh.close()
config.LIVE_STREAM_PORT = 0
result = {}


def run_main():
    result["engine"] = safesight.main(["--video", VID, "--every-frame", "--no-window", "--exit-when-done",
                                       "--zone", "100,100;300,100;300,340;100,340"])


th = threading.Thread(target=run_main, daemon=True); th.start()
tele, snap_ok = None, False
for _ in range(80):                                     # poll telemetry for the stream details
    time.sleep(0.25)
    try:
        tele = json.load(open(config.TELEMETRY_PATH))
    except Exception:
        continue
    if tele.get("stream") and tele.get("source_type") == "file":
        try:
            c = http.client.HTTPConnection("127.0.0.1", tele["stream"]["port"], timeout=3)
            c.request("GET", f"/snapshot.jpg?t={tele['stream']['token']}")
            rr = c.getresponse()
            snap_ok = rr.status == 200 and rr.read()[:2] == b"\xff\xd8"
        except Exception:
            snap_ok = False
        if snap_ok:
            break
check("telemetry advertises the live stream and a viewer can fetch a frame", snap_ok)
check("telemetry marks the source as recorded footage", bool(tele) and tele.get("source") == "cctv_demo.mp4")
th.join(timeout=120)
check("program ran start-to-end and exited by itself", not th.is_alive())
eng = result.get("engine")
rows = sqlite3.connect(config.DB_PATH).execute(
    "SELECT event_type, details, video_path FROM security_events ORDER BY id").fetchall()
kinds = {r[0] for r in rows}
check("all 225 frames analysed", eng is not None and eng.frames_done == 225)
check("zone intrusion found in footage", "Restricted Zone Intrusion" in kinds)
check("unattended baggage found in footage", "Unattended Baggage Anomaly" in kinds)
check("events carry their footage timestamp", all("footage " in r[1] for r in rows))
check("exactly one event per incident (no spam)", len(rows) == 2)
bag = [r for r in rows if r[0].startswith("Unattended")]
vcx = cv2.VideoCapture(bag[0][2]) if bag and bag[0][2] else None
dur = vcx.get(cv2.CAP_PROP_FRAME_COUNT) / max(vcx.get(cv2.CAP_PROP_FPS), 1) if vcx else 0
check(f"evidence clip uses footage time (~5 s, got {dur:.1f}s)", 4.0 <= dur <= 6.0)
summ = glob.glob(os.path.join("reports", "footage_summary_*.txt"))
txt = open(summ[0]).read() if summ else ""
check("end-of-footage summary file written", bool(summ) and "Frames analysed: 225" in txt and "Unattended Baggage" in txt)
check("ledger verifies after the footage run", database.verify_ledger()["ok"])

# restart resets the trackers (generation change)
cap2 = VideoFileCapture(short, every_frame=True)
eng2 = safesight.SentryEngine(cap2)
it = None
while it is None:
    it = cap2.get_latest(0, 0.5)
eng2.step(it[0], it[1]); eng2.bags.bags[99] = {"stale": True}
cap2.restart(); time.sleep(0.3)
it = None
while it is None:
    it = cap2.get_latest(it[2] if it else 0, 0.5) if False else cap2.get_latest(0, 0.5)
eng2.step(it[0], it[1])
check("engine resets trackers when footage restarts/loops", 99 not in eng2.bags.bags)
cap2.release()

# ------------------------------------------------------------------
print("\n[10] Dashboard control: switch source at runtime, draw zone, play/pause")
config.SOURCE_CONNECT_TIMEOUT = 3.0
os.makedirs(config.FOOTAGE_DIR, exist_ok=True)
clip_a = os.path.join(TMP, config.FOOTAGE_DIR, "clip_a.mp4")
clip_b = os.path.join(TMP, config.FOOTAGE_DIR, "clip_b.mp4")
shutil.copy(short, clip_a); shutil.copy(VID, clip_b)
stop = threading.Event()
th = threading.Thread(target=lambda: safesight.main(["--video", clip_a, "--no-window", "--loop", "--no-setup"],
                                                    stop_event=stop), daemon=True)
th.start()


def tele():
    try:
        return json.load(open(config.TELEMETRY_PATH))
    except Exception:
        return {}


def wait_for(pred, timeout=20):
    end = time.time() + timeout
    while time.time() < end:
        t = tele()
        if pred(t):
            return t
        time.sleep(0.25)
    return None


def post(path_q, body, token=None):
    t = tele()["stream"]
    c = http.client.HTTPConnection("127.0.0.1", t["port"], timeout=5)
    data = body if isinstance(body, (bytes, str)) else json.dumps(body)
    c.request("POST", f"{path_q}?t={token or t['token']}", body=data)
    return c.getresponse().status


t = wait_for(lambda t: t.get("source") == "clip_a.mp4" and t.get("stream"))
check("engine starts on clip_a and advertises its stream", t is not None)
check("zone drawn in the browser is accepted", post("/command", {"cmd": "zone", "points": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]]}) == 200)
time.sleep(1.5)
saved = json.loads(database.get_config_val("zone_polygon:clip_a.mp4", "[]", max_age=0))
check(f"zone is converted to pixels and saved for this video {saved}", saved == [[32, 18], [288, 18], [288, 162], [32, 162]])
check("wrong token is refused", post("/command", {"cmd": "zone_clear"}, token="nope") == 403)
check("unknown command is rejected", post("/command", {"cmd": "rm -rf"}) == 400)
check("zone outside 0..1 is rejected", post("/command", {"cmd": "zone", "points": [[2, 2], [3, 3], [4, 4]]}) == 400)
check("garbage body is rejected", post("/command", "not json") == 400)
check("zone can be cleared", post("/command", {"cmd": "zone_clear"}) == 200)
time.sleep(1.5)
check("cleared zone is removed from the DB", database.get_config_val("zone_polygon:clip_a.mp4", "x", max_age=0) == "")

check("pause command accepted", post("/command", {"cmd": "toggle_pause"}) == 200)
check("engine reports paused", wait_for(lambda t: t.get("paused") is True, 8) is not None)
post("/command", {"cmd": "toggle_pause"})
check("play command resumes", wait_for(lambda t: t.get("paused") is False, 8) is not None)

database.post_source_request({"type": "file", "value": os.path.join(TMP, "short.mp4")})
t = wait_for(lambda t: (t.get("source_status") or {}).get("state") == "error", 10)
check("video outside the footage folder is refused", t is not None and "inside" in t["source_status"]["message"] and t["source"] == "clip_a.mp4")
database.post_source_request({"type": "cctv", "value": "file:///etc/passwd"})
t = wait_for(lambda t: "rtsp://" in (t.get("source_status") or {}).get("message", ""), 10)
check("non-camera address scheme is refused", t is not None)
database.post_source_request({"type": "cctv", "value": "http://user:secret@127.0.0.1:9/video"})
t = wait_for(lambda t: (t.get("source_status") or {}).get("state") == "connecting", 10)
check("unreachable CCTV shows 'connecting' first", t is not None)
check("camera password is never shown", t is None or "secret" not in json.dumps(t))
check("…nor stored in the DB request", "secret" not in database.get_config_val("source_request", "", max_age=0))
t = wait_for(lambda t: (t.get("source_status") or {}).get("state") == "error", 20)
check("unreachable CCTV fails cleanly and keeps the old source running", t is not None and t["source"] == "clip_a.mp4")
config.CAMERA_SOURCE = 97            # no such device, so this test behaves the same on PCs with or without a webcam
database.post_source_request({"type": "webcam"})
t = wait_for(lambda t: (t.get("source_status") or {}).get("state") == "connecting", 10)
t = wait_for(lambda t: (t.get("source_status") or {}).get("state") == "error", 20)
check("missing webcam fails cleanly too", t is not None and t["source"] == "clip_a.mp4")

database.post_source_request({"type": "file", "value": clip_b, "every_frame": True})
t = wait_for(lambda t: t.get("source") == "clip_b.mp4", 20)
check("dashboard switches to the other recorded video", t is not None)
check("it starts paused on the first frame (no zone saved yet)", t is not None and t.get("paused") is True)
check("every-frame mode requested from the dashboard is honoured", t is not None and t.get("analysis_mode") == "every-frame")
post("/command", {"cmd": "toggle_pause"})
t1 = wait_for(lambda t: t.get("paused") is False and (t.get("position") or 0) > 0.5, 15)
check("Play starts the new video", t1 is not None)
stop.set(); th.join(timeout=30)
check("engine shuts down cleanly", not th.is_alive())

# ------------------------------------------------------------------
def dashboard_tests():
    print("\n[11] Real dashboard script under Streamlit's test harness")
    import hashlib
    from streamlit.testing.v1 import AppTest
    con = sqlite3.connect(config.DB_PATH)
    for u, pw, role in (("admin", "Admin@123", "admin"), ("guard", "Guard@123", "operator")):
        con.execute("INSERT OR REPLACE INTO users (username, password_hash, role, full_name) VALUES (?,?,?,?)",
                    (u, hashlib.sha256(pw.encode()).hexdigest(), role, u.title()))
    con.commit(); con.close()
    json.dump({"ts": time.time(), "fps": 5, "capture_fps": 30, "persons": 1, "bags": 0, "score": 0, "level": "NORMAL",
               "reasons": [], "buffer": 150, "buffer_cap": 150, "camera_connected": True, "mode": "NORMAL",
               "face_enabled": False, "pending_evidence": 0, "source_type": "camera", "source": "camera",
               "source_kind": "webcam", "source_status": {"state": "error", "message": "demo error text"},
               "position": None, "duration": None, "paused": False, "complete": False, "run_events": 0,
               "analysis_mode": "real-time", "stream": {"port": 8765, "token": "abc123"}},
              open(config.TELEMETRY_PATH, "w"))
    DASH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard.py")


    def login(user, pw):
        at = AppTest.from_file(DASH, default_timeout=60)
        at.run()
        if len(at.text_input) < 2:
            raise RuntimeError("login page did not render: " + "; ".join(str(e.value)[:200] for e in at.exception))
        at.text_input[0].set_value(user); at.text_input[1].set_value(pw)
        [b for b in at.button if "Authenticate" in b.label][0].click()
        return at.run()


    try:
        at = login("admin", "Admin@123")
    except RuntimeError as e:
        check("dashboard page renders (" + str(e)[:160] + ")", False)
        return
    check("admin logs in and the page renders without exceptions", not at.exception and at.session_state["authenticated"])
    labels = [b.label for b in at.button]
    check("Camera Source panel shows Webcam / Connect CCTV / Use Recorded Video",
          any("Webcam" in l for l in labels) and any("Connect CCTV" in l for l in labels) and any("Use Recorded Video" in l for l in labels))
    check("failed-switch message from the engine is shown", any("demo error text" in e.value for e in at.error))
    database.set_config_val("source_request", "")
    [b for b in at.button if "Webcam" in b.label][0].click()
    at.run()
    check("Webcam button posts a request to the engine", '"webcam"' in database.get_config_val("source_request", "", max_age=0))
    at.text_input(key="cctv_url").set_value("rtsp://u:p@10.0.0.5:554/s")
    [b for b in at.button if "Connect CCTV" in b.label][0].click()
    at.run()
    req = json.loads(database.get_config_val("source_request", "{}", max_age=0))
    check("Connect CCTV posts the address", req.get("type") == "cctv" and req.get("value", "").startswith("rtsp://u:p@10.0.0.5"))
    database.set_config_val("source_request", "")
    at.text_input(key="cctv_url").set_value("")
    [b for b in at.button if "Connect CCTV" in b.label][0].click()
    at.run()
    check("empty address is refused with a hint", any("address" in w.value.lower() for w in at.warning) and
          database.get_config_val("source_request", "", max_age=0) == "")
    json.dump({"ts": time.time() - 60}, open(config.TELEMETRY_PATH, "w"))
    database.set_config_val("source_request", "")
    [b for b in at.button if "Webcam" in b.label][0].click()
    at.run()
    check("with the engine offline the dashboard refuses and explains",
          any("not running" in e.value for e in at.error) and database.get_config_val("source_request", "", max_age=0) == "")
    at2 = login("guard", "Guard@123")
    check("operator sees the status but no source controls",
          not at2.exception and not any("Connect CCTV" in b.label for b in at2.button)
          and any("Only administrators" in c.value for c in at2.caption))


try:
    import pandas  # noqa: F401  (the dashboard needs it)
    _dash_ok = True
except Exception as _e:                                  # e.g. Windows Application Control blocking pandas' DLL
    _dash_ok = False
    print("\n[11] Real dashboard script under Streamlit's test harness")
    print("  SKIP  dashboard tests: pandas cannot be loaded on this PC:", _e.__class__.__name__, str(_e)[:160])
    print("        The real dashboard will not start either until this is fixed (see the fix steps).")
if _dash_ok:
    dashboard_tests()

# ------------------------------------------------------------------
print("\n[12] Speed work: stage profiler and the optional motion gate")


def make_scene_video(path, moving):
    rng = np.random.default_rng(7)
    base = np.zeros((360, 640, 3), np.uint8)
    for _ in range(40):                                   # smooth blocks: static texture that survives compression
        x, y = int(rng.integers(0, 560)), int(rng.integers(0, 300))
        base[y:y + 60, x:x + 80] = rng.integers(30, 230, 3)
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 20, (640, 360))
    for i in range(200):                                  # 10 s @ 20 fps
        f = base.copy()
        if moving and 60 <= i < 140:                      # a white square crosses the scene
            x = 20 + (i - 60) * 7
            f[150:230, x:x + 60] = 255
        w.write(f)
    w.release()


static_vid, moving_vid = os.path.join(TMP, "static_scene.mp4"), os.path.join(TMP, "moving_scene.mp4")
make_scene_video(static_vid, False); make_scene_video(moving_vid, True)


def run_engine(path, gate):
    config.MOTION_GATE = gate
    c2 = VideoFileCapture(path, every_frame=True)
    e = safesight.SentryEngine(c2)
    e.evidence.enabled = False; e.mute()
    last = 0
    while True:
        it = c2.get_latest(last, 2.0)
        if it is None:
            if c2.ended:
                break
            continue
        e.step(it[0], it[1]); last = it[2]
    c2.release(); config.MOTION_GATE = False
    return e


e_off = run_engine(static_vid, False)
e_on = run_engine(static_vid, True)
e_mov = run_engine(moving_vid, True)
avg = e_off.timing_avg()
check("profiler reports every stage", all(k in avg for k in ("prep", "tamper", "faces", "detect", "spatial", "analytics", "hud", "total")))
check("total time covers the detector time", avg["total"] >= avg["detect"])
check("gate off: detector runs on all 200 frames", e_off.detect_calls == 200 and e_off.detect_skipped == 0)
check(f"gate on, static scene: detector mostly skipped ({e_on.detect_calls}/200 runs)", 10 < e_on.detect_calls <= 40)
check("skipped + run frames add up", e_on.detect_calls + e_on.detect_skipped == 200)
check(f"gate on, moving scene: detector wakes for the motion ({e_mov.detect_calls} runs)", e_mov.detect_calls > e_on.detect_calls + 30)
check("tamper check still runs on every frame with the gate on", e_on._tcnt["tamper"] == 200)
bag_alerts = [d for d in e_on.detections if d["event"] == "Unattended Baggage Anomaly"]
check("unattended bag still raised on time with the gate on", len(bag_alerts) == 1 and 3.4 <= bag_alerts[0]["t"] <= 4.2)
check("alerts record their time in the footage", all(isinstance(d["t"], float) for d in e_on.detections))

a = safesight.parse_args(["--imgsz", "480", "--model", "x_openvino_model", "--no-face", "--motion-gate"])
check("new command-line flags parse", a.imgsz == 480 and a.model == "x_openvino_model" and a.no_face and a.motion_gate)

orig_yolo = safesight.YOLO
def picky(name, *_):
    if name != "yolov8n.pt":
        raise RuntimeError("cannot load exported model")
    return FakeYOLO()
safesight.YOLO = picky
config.YOLO_MODEL = "broken_openvino_model"
e3 = safesight.SentryEngine(cap)
check("an exported model that fails to load falls back to yolov8n.pt", isinstance(e3.model, FakeYOLO))
config.YOLO_MODEL = "yolov8n.pt"; safesight.YOLO = orig_yolo

# ------------------------------------------------------------------
print("\n[13] Evaluation: scoring maths")
from evaluate import score_events, merge_predictions, summarize, normalize_type
GT = [{"type": "zone", "start": 0, "end": 10}, {"type": "bag", "start": 3, "end": 15}, {"type": "sprint", "start": 20, "end": 25}]
PR = [{"event": "Restricted Zone Intrusion", "t": 0.5}, {"event": "Restricted Zone Intrusion", "t": 1.0},
      {"event": "Unattended Baggage Anomaly", "t": 3.5}, {"event": "Hardware Camera Tampering", "t": 40}]
r = score_events(GT, PR, tolerance=2, merge=5)
z, b, sp, tm = r["Restricted Zone Intrusion"], r["Unattended Baggage Anomaly"], r["Sprint Anomaly"], r["Hardware Camera Tampering"]
check("repeated alerts inside the merge window count once", z["tp"] == 1 and z["fp"] == 0)
check("matched events: TP with delay measured from the labelled start", b["tp"] == 1 and abs(b["delays"][0] - 0.5) < 1e-9 and abs(z["delays"][0] - 0.5) < 1e-9)
check("an event nobody alerted on is a miss (FN)", sp["fn"] == 1 and sp["tp"] == 0)
check("an alert with no labelled event is a false positive", tm["fp"] == 1 and tm["false_alarms"] == [40])
check("tolerance is inclusive at the end of the window",
      score_events(GT[2:], [{"event": "Sprint Anomaly", "t": 27.0}], 2, 5)["Sprint Anomaly"]["tp"] == 1)
check("…and exclusive just beyond it",
      score_events(GT[2:], [{"event": "Sprint Anomaly", "t": 27.5}], 2, 5)["Sprint Anomaly"]["tp"] == 0)
two = score_events([{"type": "zone", "start": 0, "end": 2}], [{"event": "Restricted Zone Intrusion", "t": 1},
                   {"event": "Restricted Zone Intrusion", "t": 30}], 2, 5)["Restricted Zone Intrusion"]
check("a later separate alert is a false positive", two["tp"] == 1 and two["fp"] == 1)
check("aliases map to the engine's event names", normalize_type("running") == "Sprint Anomaly" and normalize_type("BAG") == "Unattended Baggage Anomaly")
try:
    normalize_type("shoplifting"); bad = False
except ValueError:
    bad = True
check("unknown event types are rejected with a message", bad)
rows = summarize([r], hours=0.5)
allr = rows["ALL EVENTS"]
check("overall precision/recall/F1 are right (TP2 FP1 FN1)",
      (allr["tp"], allr["fp"], allr["fn"]) == (2, 1, 1) and abs(allr["precision"] - 2 / 3) < 1e-9
      and abs(allr["recall"] - 2 / 3) < 1e-9 and abs(allr["f1"] - 2 / 3) < 1e-9)
check("false alarms per hour use the footage length", abs(allr["fp_per_hour"] - 2.0) < 1e-9)
check("zero denominators give n/a instead of crashing", summarize([score_events([], [], 2, 5)], 1)["ALL EVENTS"]["precision"] is None)

# ------------------------------------------------------------------
print("\n[14] Evaluation: full run on a clip, and the speed benchmark")
import evaluate, benchmark, io, contextlib
db_before = sqlite3.connect(config.DB_PATH).execute("SELECT COUNT(*) FROM security_events").fetchone()[0]
edir = os.path.join(TMP, "eval_clips"); os.makedirs(edir, exist_ok=True)
shutil.copy(VID, os.path.join(edir, "clip1.mp4"))
json.dump({"video": "clip1.mp4", "zone": "100,100;300,100;300,340;100,340",
           "events": [{"type": "zone", "start": 0, "end": 15}, {"type": "bag", "start": 3, "end": 15},
                      {"type": "sprint", "start": 5, "end": 8}]}, open(os.path.join(edir, "clip1.labels.json"), "w"))
out_base = os.path.join(TMP, "eval_out")
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    rc = evaluate.main(["--dir", edir, "--out", out_base])
rep_json = json.load(open(out_base + ".json"))
allr = rep_json["summary"]["ALL EVENTS"]
check("evaluate runs a labelled clip start to end", rc == 0 and rep_json["clips"][0]["frames"] == 225)
check("it finds the zone and bag events and reports the missing sprint", (allr["tp"], allr["fp"], allr["fn"]) == (2, 0, 1))
check("markdown report for your write-up is produced", "| Restricted Zone Intrusion |" in open(out_base + ".md").read())
db_after = sqlite3.connect(config.DB_PATH).execute("SELECT COUNT(*) FROM security_events").fetchone()[0]
check("evaluation leaves your real database untouched", db_after == db_before and config.DB_PATH.endswith("test.db"))
with contextlib.redirect_stdout(io.StringIO()) as tbuf:
    evaluate.main(["--template", "clip.mp4"])
check("--template prints valid JSON to fill in", "events" in json.loads(tbuf.getvalue()))
try:
    with contextlib.redirect_stderr(io.StringIO()):
        evaluate.main(["--video", "x.mp4"]); usage = False
except SystemExit:
    usage = True
check("missing --labels is refused with usage help", usage)

# speed / accuracy sweep: several detector sizes in one run
sweep_base = os.path.join(TMP, "sweep_out")
with contextlib.redirect_stdout(io.StringIO()):
    evaluate.main(["--dir", edir, "--imgsz", "640", "320", "--motion-gate", "on", "--out", sweep_base])
sw = json.load(open(sweep_base + ".json"))
check("sweep evaluates every requested setting", [x["imgsz"] for x in sw["settings"]] == [640, 320] and all(x["gate"] for x in sw["settings"]))
check("sweep reports FPS and accuracy per setting", all(x["fps"] > 0 and x["summary"]["ALL EVENTS"]["tp"] == 2 for x in sw["settings"]))
check("sweep markdown has the comparison table", "| Setting | FPS |" in open(sweep_base + ".md").read())
check("sweep restores your settings afterwards", config.YOLO_IMGSZ == 640 and config.MOTION_GATE is False)

# a zone drawn earlier for this video (stored in the real DB) is reused when the labels file has none
shutil.copy(VID, os.path.join(TMP, "clip2.mp4"))
database.set_config_val("zone_polygon:clip2.mp4", json.dumps([[100, 100], [300, 100], [300, 340], [100, 340]]))
json.dump({"events": [{"type": "zone", "start": 0, "end": 15}, {"type": "bag", "start": 3, "end": 15}]},
          open(os.path.join(TMP, "clip2.labels.json"), "w"))
with contextlib.redirect_stdout(io.StringIO()):
    evaluate.main(["--video", os.path.join(TMP, "clip2.mp4"), "--labels", os.path.join(TMP, "clip2.labels.json"),
                   "--out", os.path.join(TMP, "eval_out2")])
r2 = json.load(open(os.path.join(TMP, "eval_out2.json")))["summary"]["ALL EVENTS"]
check("evaluate reuses the zone you drew in the dashboard when the labels have none", (r2["tp"], r2["fp"], r2["fn"]) == (2, 0, 0))

bench_out = os.path.join(TMP, "bench_out")
with contextlib.redirect_stdout(io.StringIO()):
    brows = benchmark.main(["--video", static_vid, "--frames", "100", "--warmup", "5", "--imgsz", "640", "320",
                            "--motion-gate", "both", "--out", bench_out])
check("benchmark measures every combination (2 sizes x gate off/on)", len(brows) == 4 and all(r["fps"] > 0 for r in brows))
off, on = [r for r in brows if not r["gate"]], [r for r in brows if r["gate"]]
check("benchmark shows the gate skipping detector work", all(o["detect_skipped"] == 0 for o in off) and all(g["detect_skipped"] > 0 for g in on))
check("benchmark reports per-stage milliseconds", all("detect" in r["ms"] and "total" in r["ms"] for r in brows))
check("benchmark writes markdown + json", os.path.exists(bench_out + ".md") and os.path.exists(bench_out + ".json"))
check("benchmark restores your settings afterwards", config.YOLO_IMGSZ == 640 and config.MOTION_GATE is False and config.DB_PATH.endswith("test.db"))

# ------------------------------------------------------------------
print("\n[15] Tracker keeps working when Windows blocks lap's DLL")
import lapshim, importlib, io, contextlib
x_, y_ = lapshim._lapjv(np.array([[1.0, 5.0], [6.0, 2.0]]), extend_cost=True, cost_limit=0.9 + 10)[1:]
check("replacement lapjv finds the cheapest one-to-one matching", list(x_) == [0, 1] and list(y_) == [0, 1])
tot, x_, y_ = lapshim._lapjv(np.array([[0.2, 0.95], [0.9, 0.97]]), extend_cost=True, cost_limit=0.9)
check("pairs costing more than the limit stay unmatched (like Ultralytics' IoU threshold)", list(x_) == [0, -1] and list(y_) == [0, -1] and abs(tot - 0.2) < 1e-9)
x_, y_ = lapshim._lapjv(np.random.rand(3, 2), extend_cost=True, cost_limit=2.0)[1:]
check("more tracks than detections: one track is left unmatched", len(x_) == 3 and (x_ < 0).sum() == 1 and (y_ >= 0).all())
check("empty input does not crash", lapshim._lapjv(np.zeros((0, 3)), cost_limit=0.5)[1].shape == (0,))
matches = [[ix, mx] for ix, mx in enumerate(lapshim._lapjv(np.array([[0.1, 0.8], [0.7, 0.2]]), extend_cost=True, cost_limit=0.5)[1]) if mx >= 0]
check("output feeds Ultralytics' own matching code unchanged", matches == [[0, 0], [1, 1]])

import tempfile as _tf
blk = _tf.mkdtemp()
open(os.path.join(blk, "lap.py"), "w").write('raise ImportError("DLL load failed while importing _lapjv: An Application Control policy has blocked this file.")\n')
sys.path.insert(0, blk); sys.modules.pop("lap", None)
with contextlib.redirect_stdout(io.StringIO()):
    installed = lapshim.install_lap_shim()
import lap as _lap
check("blocked lap is replaced by the SciPy version", installed and _lap.__version__.endswith("shim") and callable(_lap.lapjv))
sys.path.remove(blk); sys.modules.pop("lap", None)

good = _tf.mkdtemp()
open(os.path.join(good, "lap.py"), "w").write('__version__ = "9.9"\ndef lapjv(*a, **k):\n    return 0, [], []\n')
sys.path.insert(0, good); sys.modules.pop("lap", None)
check("a working lap is left alone", lapshim.install_lap_shim() is False and sys.modules["lap"].__version__ == "9.9")
sys.path.remove(good); sys.modules.pop("lap", None)

lazy = _tf.mkdtemp()                                   # exactly what happened on your PC: import works, the first call is blocked
open(os.path.join(lazy, "lap.py"), "w").write('__version__ = "0.10.0"\ndef lapjv(*a, **k):\n    raise ImportError("DLL load failed while importing _lapjv: An Application Control policy has blocked this file.")\n')
sys.path.insert(0, lazy); sys.modules.pop("lap", None)
with contextlib.redirect_stdout(io.StringIO()):
    swapped = lapshim.install_lap_shim()
check("importable-but-blocked lap (lazy DLL) is detected and replaced", swapped and sys.modules["lap"].__version__.endswith("shim"))
sys.path.remove(lazy); sys.modules.pop("lap", None)

open(os.path.join(blk, "torch.py"), "w").write('raise ImportError("DLL load failed: An Application Control policy has blocked this file.")\n')
sys.path.insert(0, blk); sys.modules.pop("torch", None); sys.modules.pop("lap", None)
import check_env
out = io.StringIO()
with contextlib.redirect_stdout(out):
    code = check_env.main()
sys.path.remove(blk); sys.modules.pop("torch", None)
text = out.getvalue()
check("check_env lists what loads and what fails", "OK       numpy" in text and "FAILED" in text and code == 1)
check("check_env tests the real tracker matching step", "ByteTrack assignment step" in text)
check("check_env explains the Windows block and the fix", "Windows blocked this file" in text)

cap.release()
shutil.rmtree(TMP, ignore_errors=True)
print(f"\n{PASSED} passed, {FAILED} failed")
for _n in FAILED_NAMES:
    print("  FAILED:", _n)
sys.exit(1 if FAILED else 0)