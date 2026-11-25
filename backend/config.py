"""SafeSight central configuration.

Every threshold lives here so the code, the documentation and the demo all
agree. Secrets (Telegram token) come from environment variables or a local
.env file (already git-ignored), never from source code.
"""
import os


def _load_dotenv(path=".env"):
    if not os.path.exists(path):
        return
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    except OSError:
        pass


_load_dotenv()

# ---------------- paths ----------------
DB_PATH = os.environ.get("SAFESIGHT_DB", "safesight.db")
SNAPSHOT_DIR = "alerts_snapshots"
VIDEO_DIR = "alerts_videos"
FACES_DIR = "known_faces"
REPORTS_DIR = "reports"
TELEMETRY_PATH = "telemetry.json"

# camera index (0) or an RTSP/file URL via SAFESIGHT_CAMERA
_cam = os.environ.get("SAFESIGHT_CAMERA", "0")
CAMERA_SOURCE = int(_cam) if _cam.isdigit() else _cam

# ---------------- Stage 1: ingestion + RAM ring buffer ----------------
CAPTURE_WIDTH = 1280
CAPTURE_HEIGHT = 720          # 1280x720x3 x150 frames ~ 415 MB; 1080p ~ 0.93 GB
BUFFER_FRAMES = 150
CLIP_PRE_SEC = 2.5
CLIP_POST_SEC = 2.5
SIGNAL_LOST_SEC = 3.0         # no frames this long => treated as sabotage
MAX_CONCURRENT_CLIPS = 2      # caps RAM held by in-flight evidence clips
MIN_FREE_DISK_MB = 500        # below this, events are logged without a clip

# ---------------- Stage 2A: anti-tamper ----------------
TAMPER_SIZE = (320, 180)
TAMPER_CONFIRM_SEC = 1.45
BLACKOUT_LUM = 20.0
BLIND_LUM = 240.0
OCCLUSION_LAP = 50.0
OCCLUSION_STD = 20.0
PAPER_LUM = 120.0
PAPER_SAT = 35.0
PAPER_LAP = 200.0
PAPER_STD = 45.0

# ---------------- Stage 2B: perception ----------------
YOLO_MODEL = os.environ.get("SAFESIGHT_MODEL", "yolov8n.pt")   # or an exported folder, e.g. yolov8n_openvino_model
TRACKER_CFG = "bytetrack.yaml"
TARGET_CLASSES = [0, 24, 26, 28]   # person, backpack, handbag, suitcase
YOLO_CONF = 0.20
YOLO_IMGSZ = int(os.environ.get("SAFESIGHT_IMGSZ", "640"))      # smaller = faster, but small objects (bags) get harder
# Motion gate (opt-in): skip the detector while the scene is static, refreshing it every MOTION_REFRESH_SEC.
# Measure its effect on accuracy with evaluate.py before turning it on for a real deployment.
MOTION_GATE = os.environ.get("SAFESIGHT_MOTION_GATE", "0") == "1"
MOTION_PIXEL_DELTA = 18          # grey-level change that counts as 'this pixel moved' (on a 160x90 thumbnail)
MOTION_MIN_FRACTION = 0.003      # share of pixels that must move to wake the detector
MOTION_REFRESH_SEC = 0.5         # detector always runs at least this often

# ---------------- Stage 3: spatial analytics ----------------
SPRINT_WINDOW_FRAMES = 5
SPRINT_VNORM = 0.65           # body-heights / second
SPRINT_CONFIRM_EVALS = 2
BAG_ABANDON_DIST_PX = 180     # D_edge threshold
BAG_ABANDON_TIME_SEC = 3.5
BAG_MATCH_RADIUS_PX = 70
BAG_FORGET_SEC = 8.0
ZONE_PROXIMITY_PX = 60        # outside-but-close => CAUTION
LOITER_SEC = 6.0
CROWD_THRESHOLD = 3
CROWD_SEC = 5.0

# ---------------- Stage 4: arbiter ----------------
SCORE_CAUTION = 50
SCORE_HIGH = 75
SCORE_UNATTENDED = 90
SCORE_TAMPER = 100
ARBITER_CLEAR_SEC = 1.0       # quiet time before the monotonic window resets

# ---------------- Stage 5: evidence / dispatch ----------------
# The project report says >= 90. Zone intrusions are HIGH (75) and were always
# recorded by the original code, so the default is 75. Set to 90 to match the
# report literally.
EVIDENCE_MIN_SCORE = 75
EVENT_COOLDOWN_SEC = 10.0
REARM_SEC = 5.0               # a condition must be clear this long to re-fire
MAX_CONSECUTIVE_ERRORS = 25
TELEMETRY_INTERVAL = 1.0

# ---------------- live view stream + demo footage ----------------
LIVE_STREAM_ENABLED = os.environ.get("SAFESIGHT_STREAM", "1") != "0"
LIVE_STREAM_HOST = os.environ.get("SAFESIGHT_STREAM_HOST", "127.0.0.1")   # 0.0.0.0 to view from other PCs
LIVE_STREAM_PORT = int(os.environ.get("SAFESIGHT_STREAM_PORT", "8765"))
LIVE_STREAM_WIDTH = 960
LIVE_STREAM_FPS = 12
LIVE_STREAM_QUALITY = 70
FOOTAGE_DIR = "demo_footage"
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v")
ALLOWED_URL_SCHEMES = ("rtsp", "rtsps", "http", "https")   # CCTV addresses typed in the dashboard
SOURCE_CONNECT_TIMEOUT = 10.0                               # seconds to wait for the first frame of a new source

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
TELEGRAM_MIN_SCORE = 75