import os
import sys
import sqlite3
import hashlib
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
from datetime import datetime, date
import re
import json
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

import config
import database

try:
    from pdf_report_generator import generate_pdf_dossier
except Exception:
    generate_pdf_dossier = None


@st.cache_resource
def _boot_db():
    database.init_db()      # creates every table (incl. users) and enables WAL: no more login crash
    return True


_boot_db()

st.set_page_config(
    page_title="SafeSight AI | Tactical SOC",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"  # Forces the sidebar to stay open
)

DB_PATH = config.DB_PATH
SNAPSHOT_DIR = "alerts_snapshots"
VIDEO_DIR = "alerts_videos"
REPORTS_DIR = "reports"
os.makedirs(SNAPSHOT_DIR, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# ------------------------------------------------------------------
# Advanced High-Contrast Cyber Defense Theme
# ------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600;700&display=swap');
    
    /* 1. HIDE ONLY DEPLOY & MENU BUTTONS (KEEP SIDEBAR TOGGLE VISIBLE) */
    .stDeployButton {
        display: none !important;
        visibility: hidden !important;
    }
    #MainMenu {
        visibility: hidden !important;
        display: none !important;
    }
    footer {
        visibility: hidden !important;
        display: none !important;
    }
    header[data-testid="stHeader"] {
        background: transparent !important;
        color: #38BDF8 !important;
    }

    /* 2. ROOT THEME & FULL SCREEN UTILIZATION */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    .stApp {
        background: #06080F;
        color: #E2E8F0;
    }
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 2rem !important;
        max-width: 98% !important;
    }
    
    /* 3. SIDEBAR STYLING */
    section[data-testid="stSidebar"] {
        background-color: #0A0F1D !important;
        border-right: 1px solid rgba(56, 189, 248, 0.15) !important;
    }
    
    /* 4. SLEEK TOP COMMAND BAR */
    .soc-nav {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.85) 0%, rgba(30, 41, 59, 0.45) 100%);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 12px;
        padding: 12px 20px;
        margin-bottom: 16px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        backdrop-filter: blur(12px);
    }
    .soc-brand {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .soc-title-text {
        font-size: 1.35rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        background: linear-gradient(90deg, #38BDF8, #818CF8, #C084FC);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .soc-subtitle-text {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* 5. LIVE PULSE BADGE */
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.3);
        padding: 5px 12px;
        border-radius: 20px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        font-weight: 700;
        color: #34D399;
    }
    .pulse-dot {
        width: 7px;
        height: 7px;
        background-color: #10B981;
        border-radius: 50%;
        box-shadow: 0 0 8px #10B981;
        animation: pulse 1.8s infinite;
    }
    @keyframes pulse {
        0% { transform: scale(0.9); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
        70% { transform: scale(1.1); box-shadow: 0 0 0 6px rgba(16, 185, 129, 0); }
        100% { transform: scale(0.9); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }
    
    /* 6. KPI METRICS CARDS */
    .kpi-box {
        background: #0B1120;
        border: 1px solid rgba(148, 163, 184, 0.12);
        border-radius: 10px;
        padding: 12px 16px;
        position: relative;
        overflow: hidden;
    }
    .kpi-box::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 3px;
    }
    .kpi-cyan::after { background: #38BDF8; }
    .kpi-red::after { background: #EF4444; }
    .kpi-amber::after { background: #F59E0B; }
    .kpi-emerald::after { background: #10B981; }
    .kpi-purple::after { background: #A855F7; }
    
    .kpi-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem;
        text-transform: uppercase;
        color: #64748B;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .kpi-num {
        font-size: 1.6rem;
        font-weight: 800;
        color: #F8FAFC;
        margin-top: 2px;
    }
    
    /* 7. SEVERITY STATUS BADGES */
    .badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 4px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem;
        font-weight: 700;
    }
    .badge-critical { background: rgba(239, 68, 68, 0.2); color: #F87171; border: 1px solid rgba(239, 68, 68, 0.4); }
    .badge-high { background: rgba(245, 158, 11, 0.2); color: #FBBF24; border: 1px solid rgba(245, 158, 11, 0.4); }
    .badge-caution { background: rgba(6, 182, 212, 0.2); color: #38BDF8; border: 1px solid rgba(6, 182, 212, 0.4); }
    .badge-normal { background: rgba(16, 185, 129, 0.2); color: #34D399; border: 1px solid rgba(16, 185, 129, 0.4); }
</style>
""", unsafe_allow_html=True)

def fetch_security_logs():
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    try:
        conn = sqlite3.connect(DB_PATH, timeout=5)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS security_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                event_type TEXT NOT NULL,
                track_id INTEGER,
                threat_score INTEGER NOT NULL,
                threat_level TEXT NOT NULL,
                details TEXT,
                snapshot_path TEXT,
                video_path TEXT
            )
        """)
        conn.commit()
        cursor.execute("PRAGMA table_info(security_events)")
        cols = [col[1] for col in cursor.fetchall()]
        if "video_path" not in cols:
            cursor.execute("ALTER TABLE security_events ADD COLUMN video_path TEXT")
            conn.commit()
        df = pd.read_sql_query("SELECT * FROM security_events ORDER BY id DESC", conn)
        conn.close()
        if "video_path" not in df.columns:
            df["video_path"] = None
        return df
    except Exception:
        return pd.DataFrame()

def verify_user(username, password):
    if not os.path.exists(DB_PATH):
        return None
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    pwd_hash = hashlib.sha256(password.encode("utf-8")).hexdigest()
    cursor.execute("SELECT username, role, full_name FROM users WHERE username = ? AND password_hash = ?", (username, pwd_hash))
    user = cursor.fetchone()
    conn.close()
    if user:
        return {"username": user[0], "role": user[1], "full_name": user[2]}
    return None

def add_new_user(username, password, role, full_name):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    pwd_hash = hashlib.sha256(password.encode("utf-8")).hexdigest()
    try:
        cursor.execute("INSERT INTO users (username, password_hash, role, full_name) VALUES (?, ?, ?, ?)",
                       (username, pwd_hash, role, full_name))
        conn.commit()
        return True, f"Account '{username}' created successfully!"
    except sqlite3.IntegrityError:
        return False, f"Username '{username}' already exists."
    finally:
        conn.close()

def delete_user(username):
    if username == "admin":
        return False, "Cannot delete master admin."
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM users WHERE username = ?", (username,))
    conn.commit()
    conn.close()
    return True, f"User '{username}' removed."

def get_all_users():
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT username, role, full_name, created_at FROM users", conn)
    conn.close()
    return df

# Authentication
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
    st.session_state["user"] = None

if not st.session_state["authenticated"]:
    st.markdown("<div style='margin-top:60px;'></div>", unsafe_allow_html=True)
    _, login_col, _ = st.columns([1, 1.2, 1])
    with login_col:
        st.markdown("""
            <div style="text-align:center; margin-bottom:20px;">
                <span style="font-size:3rem;">🛡️</span>
                <h2 style="color:#F8FAFC; margin:0; font-weight:800;">SafeSight SOC Portal</h2>
                <p style="color:#64748B; font-size:0.85rem;">Autonomous Edge AI Surveillance Operations</p>
            </div>
        """, unsafe_allow_html=True)
        if not database.has_users():
            st.warning("No accounts exist yet. Run `python backend/setup_users.py` once, then refresh.")
        with st.form("login_form"):
            user_in = st.text_input("Security Identifier")
            pass_in = st.text_input("Access Key", type="password")
            if st.form_submit_button("Authenticate Access", width="stretch"):
                u_info = verify_user(user_in.strip(), pass_in.strip())
                if u_info:
                    st.session_state["authenticated"] = True
                    st.session_state["user"] = u_info
                    st.rerun()
                else:
                    st.error("Authentication failed: Invalid credentials.")
    st.stop()

current_user = st.session_state["user"]
is_admin = current_user["role"] == "admin"

# Sidebar
with st.sidebar:
    st.markdown(f"""
        <div style="display:flex; align-items:center; gap:10px; margin-bottom:12px;">
            <span style="font-size:1.8rem;">🛡️</span>
            <div>
                <h3 style="margin:0; font-size:1.05rem; color:#F8FAFC;">{current_user['full_name']}</h3>
                <span style="background:rgba(56,189,248,0.15); color:#38BDF8; border:1px solid rgba(56,189,248,0.3); padding:2px 8px; border-radius:4px; font-size:0.72rem; font-family:'JetBrains Mono';">{current_user['role'].upper()} CLEARANCE</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    if st.button("🚪 Sign Out", width="stretch"):
        st.session_state["authenticated"] = False
        st.session_state["user"] = None
        st.rerun()

    st.divider()
    st.markdown("##### 📅 Timeline & Calendar Filter")
    filter_mode = st.radio("Timeline Scope:", ["Show All Records", "Filter by Specific Date"])
    
    selected_date = None
    if filter_mode == "Filter by Specific Date":
        selected_date = st.date_input("Select Incident Day:", value=datetime.now().date())

    st.markdown("##### 🎛️ Telemetry Filtering")
    severity_filter = st.multiselect(
        "Incident Severity",
        ["CRITICAL", "HIGH", "CAUTION", "NORMAL"],
        default=["CRITICAL", "HIGH", "CAUTION", "NORMAL"]
    )

    if is_admin:
        st.divider()
        st.markdown("##### 🔒 Master Maintenance")
        with st.expander("⚡ Purge Incident Logs"):
            confirm = st.text_input("Type 'CONFIRM'", type="password")
            if st.button("🚨 Purge Database", width="stretch") and confirm == "CONFIRM":
                conn = sqlite3.connect(DB_PATH)
                conn.cursor().execute("DELETE FROM security_events")
                conn.commit()
                conn.close()
                st.toast("Security logs purged by Administrator.", icon="🗑️")
                st.rerun()

def _load_telemetry():
    try:
        with open(config.TELEMETRY_PATH, "r", encoding="utf-8") as f:
            t = json.load(f)
        t["age"] = time.time() - t.get("ts", 0)
        return t
    except Exception:
        return None


@st.fragment(run_every="2s")
def live_telemetry_panel():
    """Live engine telemetry (FPS, tracks, threat gauge). Only this small fragment refreshes."""
    t = _load_telemetry()
    online = t is not None and t["age"] < 5
    colors = {"NORMAL": "#10B981", "CAUTION": "#38BDF8", "HIGH": "#F59E0B", "CRITICAL": "#EF4444"}
    if not online:
        st.markdown('<div class="kpi-box kpi-red"><div class="kpi-label">SENTRY ENGINE</div>'
                    '<div class="kpi-num" style="color:#F87171;">OFFLINE</div></div>', unsafe_allow_html=True)
        return
    lvl, score = t["level"], t["score"]
    c = st.columns(6)
    items = [("ENGINE", "ONLINE" if t["camera_connected"] else "NO CAMERA", "kpi-emerald" if t["camera_connected"] else "kpi-red"),
             ("LIVE FPS", f'{t["fps"]:.0f}', "kpi-cyan"),
             ("PEOPLE / BAGS", f'{t["persons"]} / {t["bags"]}', "kpi-purple"),
             ("THREAT", f"{lvl} {score}", "kpi-red" if lvl in ("HIGH", "CRITICAL") else "kpi-amber"),
             ("RAM BUFFER", f'{t["buffer"]}/{t["buffer_cap"]}', "kpi-cyan"),
             ("EVIDENCE QUEUE", str(t["pending_evidence"]), "kpi-amber")]
    for col, (lab, val, cls) in zip(c, items):
        col.markdown(f'<div class="kpi-box {cls}"><div class="kpi-label">{lab}</div><div class="kpi-num">{val}</div></div>',
                     unsafe_allow_html=True)
    st.markdown(f"""
        <div style="margin:10px 0 4px 0;background:#0B1120;border:1px solid rgba(148,163,184,.12);border-radius:8px;height:14px;">
            <div style="width:{max(score, 2)}%;height:100%;border-radius:8px;background:{colors.get(lvl, '#10B981')};"></div>
        </div>
        <div style="font-family:'JetBrains Mono';font-size:.72rem;color:#64748B;">
            Threat gauge {score}/100 | {' | '.join(t['reasons']) if t['reasons'] else 'no active anomalies'}
        </div>""", unsafe_allow_html=True)
    if t.get("source_type") == "file":
        pos, dur = t.get("position") or 0, t.get("duration") or 0
        pct = min(100, int(100 * pos / dur)) if dur else 0
        state = "ANALYSIS COMPLETE" if t.get("complete") else ("PAUSED" if t.get("paused") else "ANALYZING")
        mm = lambda x: f"{int(x) // 60:02d}:{int(x) % 60:02d}"
        st.markdown(f"""
            <div style="margin-top:10px;font-family:'JetBrains Mono';font-size:.75rem;color:#38BDF8;">
                🎞️ {state}: {t.get('source')} | {mm(pos)} / {mm(dur)} | {t.get('run_events', 0)} incident(s) this run | {t.get('analysis_mode')} mode
            </div>
            <div style="margin-top:4px;background:#0B1120;border:1px solid rgba(148,163,184,.12);border-radius:8px;height:8px;">
                <div style="width:{max(pct, 1)}%;height:100%;border-radius:8px;background:#38BDF8;"></div>
            </div>""", unsafe_allow_html=True)


@st.fragment(run_every="3s")
def new_event_watcher():
    """Reruns the whole page only when a NEW incident is recorded, so video playback isn't interrupted."""
    latest = database.latest_event_id()
    seen = st.session_state.get("seen_event_id")
    if seen is None:
        st.session_state["seen_event_id"] = latest
    elif latest != seen:
        st.session_state["seen_event_id"] = latest
        st.rerun(scope="app")


_STREAM_HTML = """
<style>
  body{margin:0;background:transparent;font-family:'Segoe UI',Arial,sans-serif}
  .wrap{background:#06080F;border:1px solid rgba(56,189,248,.25);border-radius:10px;padding:8px}
  .bar{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:8px;align-items:center}
  button{background:#0B1120;color:#E2E8F0;border:1px solid rgba(148,163,184,.3);border-radius:6px;
         padding:5px 10px;font-size:12px;cursor:pointer}
  button:hover{border-color:#38BDF8}
  .hint{color:#94A3B8;font-size:12px}
  .stage{position:relative}
  #v{width:100%;display:block;border-radius:6px;background:#000;max-height:500px;object-fit:contain}
  #c{position:absolute;left:0;top:0;pointer-events:none}
</style>
<div class="wrap">
  <div class="bar">
    <button id="draw">✏️ Draw zone</button>
    <button id="undo" style="display:none">↶ Undo point</button>
    <button id="finish" style="display:none">✔ Finish zone</button>
    <button id="clear">🗑 Clear zone</button>
    <span id="fbar" style="display:none;gap:8px">
      <button id="play">⏯ Play / Pause</button> <button id="restart">⟲ Restart</button>
    </span>
    <span class="hint" id="hint"></span>
  </div>
  <div class="stage"><img id="v" alt="live view"/><canvas id="c"></canvas></div>
</div>
<script>
(function() {
  var host = 'localhost';
  try { host = window.parent.location.hostname || host; } catch (e) {}
  var BASE = 'http://' + host + ':__PORT__', T = '__TOKEN__';
  var img = document.getElementById('v'), cv = document.getElementById('c'), ctx = cv.getContext('2d');
  var hint = document.getElementById('hint'), drawing = false, pts = [];
  if (__IS_FILE__) document.getElementById('fbar').style.display = 'inline-flex';
  img.src = BASE + '/stream.mjpg?t=' + T;

  function vrect() {                       // where the video really is inside the <img> (object-fit: contain)
    var r = img.getBoundingClientRect(), nw = img.naturalWidth || 16, nh = img.naturalHeight || 9;
    var sc = Math.min(r.width / nw, r.height / nh), w = nw * sc, h = nh * sc;
    return {x: (r.width - w) / 2, y: (r.height - h) / 2, w: w, h: h, W: r.width, H: r.height};
  }
  function redraw() {
    var v = vrect(); cv.width = v.W; cv.height = v.H; ctx.clearRect(0, 0, v.W, v.H);
    if (!pts.length) return;
    ctx.strokeStyle = '#38BDF8'; ctx.fillStyle = '#38BDF8'; ctx.lineWidth = 2; ctx.beginPath();
    pts.forEach(function(p, i) { var x = v.x + p[0] * v.w, y = v.y + p[1] * v.h; i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
    ctx.stroke();
    pts.forEach(function(p) { ctx.beginPath(); ctx.arc(v.x + p[0] * v.w, v.y + p[1] * v.h, 4, 0, 6.3); ctx.fill(); });
  }
  function ui() {
    ['undo', 'finish'].forEach(function(id) { document.getElementById(id).style.display = drawing ? 'inline-block' : 'none'; });
    hint.textContent = drawing ? 'Click on the video to place corners (' + pts.length + ' so far, need 3+), then Finish zone.' : '';
    redraw();
  }
  function send(cmd) {
    return fetch(BASE + '/command?t=' + T, {method: 'POST', mode: 'no-cors',
                 headers: {'Content-Type': 'text/plain'}, body: JSON.stringify(cmd)});
  }
  img.addEventListener('click', function(e) {
    if (!drawing) return;
    var r = img.getBoundingClientRect(), v = vrect();
    var x = (e.clientX - r.left - v.x) / v.w, y = (e.clientY - r.top - v.y) / v.h;
    if (x < 0 || y < 0 || x > 1 || y > 1) return;
    pts.push([x, y]); ui();
  });
  document.getElementById('draw').onclick = function() { drawing = true; pts = []; ui(); };
  document.getElementById('undo').onclick = function() { pts.pop(); ui(); };
  document.getElementById('finish').onclick = function() {
    if (pts.length < 3) { hint.textContent = 'A zone needs at least 3 corners.'; return; }
    send({cmd: 'zone', points: pts}); drawing = false; pts = []; ui(); hint.textContent = 'Zone applied.';
  };
  document.getElementById('clear').onclick = function() { send({cmd: 'zone_clear'}); drawing = false; pts = []; ui(); hint.textContent = 'Zone cleared.'; };
  document.getElementById('play').onclick = function() { send({cmd: 'toggle_pause'}); };
  document.getElementById('restart').onclick = function() { send({cmd: 'restart'}); };
  setInterval(redraw, 500); window.addEventListener('resize', redraw);
})();
</script>
"""


def _stream_html(port, token, is_file):
    return (_STREAM_HTML.replace("__PORT__", str(int(port))).replace("__TOKEN__", re.sub(r"[^A-Za-z0-9_-]", "", str(token)))
            .replace("__IS_FILE__", "true" if is_file else "false"))


@st.fragment(run_every="3s")
def live_camera_panel():
    """Live annotated video from the engine (in-memory MJPEG stream, nothing is written to disk)."""
    t = _load_telemetry()
    online = t is not None and t["age"] < 5
    stream = t.get("stream") if online else None
    with st.expander("📺 Live Camera View", expanded=True):
        if not stream:
            st.info("The live view appears here while the sentry engine is running (streaming on, "
                    "dashboard and engine on the same PC unless SAFESIGHT_STREAM_HOST is set).")
            return
        src = f"recorded footage {t.get('source')}" if t.get("source_type") == "file" else "live camera"
        st.caption(f"Source: {src}. Use Draw zone to mark the restricted area directly on the video.")
        html = _stream_html(stream["port"], stream["token"], t.get("source_type") == "file").strip()
        if hasattr(st, "iframe"):
            st.iframe(html, height=600)            # replaces the deprecated components.html
        else:
            components.html(html, height=600)


ALLOWED_VIDEO_EXT = tuple(config.VIDEO_EXTENSIONS)


def _save_upload(uploaded):
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", uploaded.name)
    base, ext = os.path.splitext(name)
    if ext.lower() not in ALLOWED_VIDEO_EXT:
        raise ValueError("Unsupported video type. Use .mp4, .avi, .mov, .mkv or .wmv")
    os.makedirs(config.FOOTAGE_DIR, exist_ok=True)
    path = os.path.abspath(os.path.join(config.FOOTAGE_DIR, f"{time.strftime('%Y%m%d_%H%M%S')}_{base}{ext.lower()}"))
    with open(path, "wb") as f:
        f.write(uploaded.getbuffer())
    return path


@st.fragment(run_every="2s")
def source_status_line():
    t = _load_telemetry()
    if t is None or t["age"] >= 5:
        st.markdown('<span class="status-pill" style="color:#F87171;border-color:rgba(239,68,68,.4);">'
                    'ENGINE OFFLINE: start it with run_safesight.bat</span>', unsafe_allow_html=True)
        return
    kind, label = t.get("source_kind", "webcam"), t.get("source", "")
    text = {"webcam": "WEBCAM", "cctv": f"CCTV {label}", "file": f"RECORDED VIDEO {label}"}.get(kind, label)
    st.markdown(f'<span class="status-pill">CURRENT: {text}</span>', unsafe_allow_html=True)
    stt = t.get("source_status") or {}
    if stt.get("state") == "connecting":
        st.info(stt.get("message", "Connecting..."))
    elif stt.get("state") == "error":
        st.error(stt.get("message", "Could not switch source."))


def _engine_online():
    t = _load_telemetry()
    return t is not None and t["age"] < 5


def render_camera_source_panel():
    st.markdown("""
        <div class="soc-nav" style="margin-bottom:8px;">
            <div>
                <div style="font-size:1.05rem;font-weight:700;color:#F8FAFC;">Camera Source</div>
                <div class="soc-subtitle-text">Choose what SafeSight should analyze</div>
            </div>
        </div>""", unsafe_allow_html=True)
    source_status_line()
    if not is_admin:
        st.caption("Only administrators can change the camera source.")
        return
    with st.container(border=True):
        c1, c2, c3 = st.columns([1.1, 3.4, 1.3])
        use_webcam = c1.button("💻 Webcam", width="stretch")
        cctv_url = c2.text_input("CCTV / RTSP address", type="password", key="cctv_url",
                                 placeholder="rtsp://username:password@camera-ip:554/stream",
                                 label_visibility="collapsed")
        use_cctv = c3.button("📹 Connect CCTV", width="stretch")

        d1, d2 = st.columns([3.6, 1.2])
        uploaded = d1.file_uploader("Upload a recorded video", type=[e.lstrip(".") for e in ALLOWED_VIDEO_EXT],
                                    key="video_upload", label_visibility="collapsed")
        use_upload = d2.button("🎞️ Use Recorded Video", width="stretch")

        e1, e2 = st.columns([3.6, 1.2])
        existing = sorted((f for f in os.listdir(config.FOOTAGE_DIR) if f.lower().endswith(ALLOWED_VIDEO_EXT)),
                          reverse=True) if os.path.isdir(config.FOOTAGE_DIR) else []
        picked = e1.selectbox("Or pick a video already on this PC", ["(none)"] + existing, key="video_pick",
                              label_visibility="collapsed")
        use_picked = e2.button("▶ Use selected", width="stretch")
        o1, o2 = st.columns(2)
        mode = o1.radio("Playback", ["Real-time", "Every frame (accurate, slower)"], horizontal=True, key="play_mode")
        loop = o2.checkbox("Loop the video", key="play_loop")

    req = None
    try:
        if use_webcam:
            req = {"type": "webcam"}
        elif use_cctv:
            if not cctv_url.strip():
                st.warning("Type the camera address first.")
            else:
                req = {"type": "cctv", "value": cctv_url.strip()}
        elif use_upload:
            if uploaded is None:
                st.warning("Choose a video file first.")
            else:
                req = {"type": "file", "value": _save_upload(uploaded)}
        elif use_picked:
            if picked == "(none)":
                st.warning("Pick a video from the list.")
            else:
                req = {"type": "file", "value": os.path.abspath(os.path.join(config.FOOTAGE_DIR, picked))}
    except ValueError as e:
        st.error(str(e))
    if req:
        if not _engine_online():
            st.error("The sentry engine is not running. Start it first (run_safesight.bat), then try again.")
        else:
            if req["type"] == "file":
                req.update(every_frame=mode.startswith("Every"), loop=bool(loop))
            database.post_source_request(req)
            st.toast("Switching source...", icon="🔄")


def render_soc_interface():
    df = fetch_security_logs()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    st.markdown(f"""
        <div class="soc-nav">
            <div class="soc-brand">
                <span style="font-size:1.6rem;">🛡️</span>
                <div>
                    <div class="soc-title-text">SAFESIGHT AI SURVEILLANCE MATRIX</div>
                    <div class="soc-subtitle-text">SOC NODE: ALPHA-1 | OPERATOR: {current_user['username'].upper()}</div>
                </div>
            </div>
            <div style="display:flex; align-items:center; gap:14px;">
                <div class="status-pill">
                    <div class="pulse-dot"></div>
                    SENTRY ACTIVE
                </div>
                <div style="font-family:'JetBrains Mono'; font-size:0.75rem; color:#64748B;">🕒 {now_str}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    live_telemetry_panel()
    render_camera_source_panel()
    live_camera_panel()
    new_event_watcher()

    if is_admin:
        tab_live, tab_export, tab_users = st.tabs([
            "🔴 Incident Forensic Studio",
            "📥 Audit Proofs & PDF Dossier",
            "⚙️ Admin Subsystems & Users"
        ])
    else:
        tab_live = st.container()

    # TAB 1: STUDIO
    with tab_live:
        if df.empty:
            st.markdown("""
                <div style="text-align:center; padding:60px 20px; background:#0B1120; border-radius:12px; border:1px dashed #1E293B; margin-top:10px;">
                    <span style="font-size:2.5rem;">📡</span>
                    <h4 style="margin-top:12px; color:#F8FAFC;">SURVEILLANCE SENSORS SCANNING</h4>
                    <p style="color:#64748B; font-size:0.85rem; font-family:'JetBrains Mono';">No security anomalies recorded yet.</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            df['date_only'] = pd.to_datetime(df['timestamp']).dt.date
            working_df = df[df['date_only'] == selected_date] if selected_date is not None else df
            filtered_df = working_df[working_df["threat_level"].isin(severity_filter)] if severity_filter else working_df

            k1, k2, k3, k4, k5 = st.columns(5)
            k1.markdown(f'<div class="kpi-box kpi-cyan"><div class="kpi-label">TOTAL LOGS</div><div class="kpi-num">{len(filtered_df)}</div></div>', unsafe_allow_html=True)
            k2.markdown(f'<div class="kpi-box kpi-red"><div class="kpi-label">CRITICAL THREATS</div><div class="kpi-num" style="color:#F87171;">{len(filtered_df[filtered_df["threat_level"]=="CRITICAL"])}</div></div>', unsafe_allow_html=True)
            k3.markdown(f'<div class="kpi-box kpi-amber"><div class="kpi-label">HIGH SEVERITY</div><div class="kpi-num" style="color:#FBBF24;">{len(filtered_df[filtered_df["threat_level"]=="HIGH"])}</div></div>', unsafe_allow_html=True)
            k4.markdown(f'<div class="kpi-box kpi-emerald"><div class="kpi-label">ZONE BREACHES</div><div class="kpi-num" style="color:#34D399;">{len(filtered_df[filtered_df["event_type"].str.contains("Intrusion", na=False)])}</div></div>', unsafe_allow_html=True)
            k5.markdown(f'<div class="kpi-box kpi-purple"><div class="kpi-label">TAMPER EVENTS</div><div class="kpi-num" style="color:#C084FC;">{len(filtered_df[filtered_df["event_type"].str.contains("Tampering", na=False)])}</div></div>', unsafe_allow_html=True)

            st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)

            if not filtered_df.empty:
                options = []
                opt_map = {}
                for idx, r in filtered_df.iterrows():
                    label = f"#{r['id']} | {r['timestamp']} | [{r['threat_level']}] {r['event_type']} (Score: {r['threat_score']})"
                    options.append(label)
                    opt_map[label] = r

                selected_opt = st.selectbox(
                    f"📌 Select Incident from History ({len(options)} total events):",
                    options,
                    index=0
                )
                active_record = opt_map[selected_opt]

                insp_l, insp_r = st.columns([1.3, 1])

                with insp_l:
                    st.markdown("##### 📹 5-Second Forensic Video Playback")
                    v_path = str(active_record["video_path"]) if pd.notna(active_record["video_path"]) else ""
                    if v_path and os.path.exists(v_path) and os.path.getsize(v_path) > 1000:
                        with open(v_path, 'rb') as vf:
                            video_bytes = vf.read()
                        st.video(video_bytes, format="video/mp4")
                        st.download_button(
                            label=f"⬇️ Download Event #{active_record['id']} MP4 Video Clip",
                            data=video_bytes,
                            file_name=os.path.basename(v_path),
                            mime="video/mp4",
                            width="stretch"
                        )
                    else:
                        st.info("⚠️ Video buffer clip was not recorded for this event.")

                with insp_r:
                    st.markdown("##### 📸 Photographic Evidence Snapshot")
                    s_path = str(active_record["snapshot_path"]) if pd.notna(active_record["snapshot_path"]) else ""
                    if s_path and os.path.exists(s_path):
                        img = Image.open(s_path)
                        st.image(img, width="stretch")
                        with open(s_path, "rb") as sf:
                            st.download_button(
                                label=f"⬇️ Download Event #{active_record['id']} JPEG Snapshot",
                                data=sf.read(),
                                file_name=os.path.basename(s_path),
                                mime="image/jpeg",
                                width="stretch"
                            )
                        st.markdown(f"""
                            <div style="margin-top:6px;">
                                <span class="badge badge-{active_record['threat_level'].lower()}">{active_record['threat_level']}</span>
                                <span style="font-family:'JetBrains Mono'; font-size:0.75rem; color:#94A3B8; margin-left:8px;">🕒 {active_record['timestamp']}</span>
                                <div style="font-size:0.8rem; color:#E2E8F0; margin-top:4px;"><b>Telemetry:</b> {active_record['details']}</div>
                            </div>
                        """, unsafe_allow_html=True)
                        seal = active_record.get("chain_hash")
                        if seal is not None and pd.notna(seal) and str(seal):
                            st.caption(f"🔐 Ledger seal (SHA-256 chain): {str(seal)[:32]}…")
                        else:
                            st.caption("⚠️ Legacy record: not sealed in the hash-chained ledger.")
                    else:
                        st.caption("No snapshot recorded for this entry.")
            else:
                st.info("No security incidents match the current timeline or severity filters.")

            st.divider()
            st.markdown(f"#### 📋 Complete Incident Audit Stream ({len(filtered_df)} Records)")
            cols_to_show = ["id", "timestamp", "threat_level", "threat_score", "event_type", "track_id", "details"]
            st.dataframe(
                filtered_df[cols_to_show].rename(
                    columns={
                        "id": "Log ID",
                        "timestamp": "Timestamp",
                        "threat_level": "Severity",
                        "threat_score": "Score",
                        "event_type": "Incident Type",
                        "track_id": "Target ID",
                        "details": "Telemetry Details"
                    }
                ),
                width="stretch",
                hide_index=True
            )

    # TAB 2: EXPORT
    if is_admin:
        with tab_export:
            st.markdown("#### 🔐 Ledger Integrity Verification")
            st.caption("Recomputes every SHA-256 digest from the stored snapshots and clips and checks the hash chain for deleted or altered records.")
            if st.button("🔍 Verify Ledger Integrity", width="stretch"):
                rep_ = database.verify_ledger()
                if rep_["ok"]:
                    st.success(f"Ledger intact: {rep_['sealed']} sealed record(s) verified"
                               + (f", {rep_['legacy']} legacy unsealed." if rep_["legacy"] else "."))
                else:
                    st.error(f"INTEGRITY FAILURE: {len(rep_['problems'])} problem(s) detected")
                    for p in rep_["problems"]:
                        st.write(f"• Event #{p['id']}: {p['issue']}")
                st.code(f"Chain head: {rep_['head']}")
            st.divider()
            st.markdown("#### 📥 Official Forensic Evidentiary Proofs")
            st.write("Generate and download legal proof dossiers containing cryptographic signatures, photo logs, and incident telemetry:")
            
            c_pdf, c_csv = st.columns(2)
            with c_pdf:
                st.markdown("##### 📄 Courtroom Forensic PDF Dossier")
                st.caption("Compiles recent security breach photos, timestamps, telemetry details, and a SHA-256 digital signature into a PDF.")
                if generate_pdf_dossier is not None:
                    if st.button("⚡ Generate Legal PDF Dossier", width="stretch"):
                        pdf_file = generate_pdf_dossier()
                        if pdf_file and os.path.exists(pdf_file):
                            with open(pdf_file, "rb") as f:
                                st.download_button(
                                    label="⬇️ Click Here to Download Generated PDF Dossier",
                                    data=f.read(),
                                    file_name=os.path.basename(pdf_file),
                                    mime="application/pdf",
                                    width="stretch"
                                )
                            st.success(f"Dossier ready: {os.path.basename(pdf_file)}")
                        else:
                            st.warning("No security incidents available to generate report.")
                else:
                    st.error("ReportLab library missing. Run: pip install reportlab")

            with c_csv:
                st.markdown("##### 📊 Raw Audit Evidence Ledger (CSV)")
                st.caption("Export all raw database records, object tracking IDs, normalized velocity records, and threat scores.")
                if not df.empty:
                    st.download_button(
                        label="⬇️ Download Complete Raw CSV Audit Logs",
                        data=df.to_csv(index=False).encode('utf-8'),
                        file_name=f"SafeSight_Full_Audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        width="stretch"
                    )
                else:
                    st.info("No security logs available for export yet.")

    # TAB 3: USERS
    if is_admin:
        with tab_users:
            st.markdown("#### ⚙️ Edge AI Subsystem Configuration")
            conn = sqlite3.connect(DB_PATH)
            cur = conn.cursor()
            cur.execute("CREATE TABLE IF NOT EXISTS system_config (key TEXT PRIMARY KEY, val TEXT NOT NULL)")
            cur.execute("SELECT val FROM system_config WHERE key = 'face_detection_enabled'")
            row = cur.fetchone()
            current_face_state = (row[0] == "1") if row else True
            conn.close()

            c_toggle1, c_toggle2 = st.columns([1.2, 2])
            with c_toggle1:
                toggle_val = st.toggle("Enable Facial Recognition Scanner", value=current_face_state)
                if toggle_val != current_face_state:
                    conn = sqlite3.connect(DB_PATH)
                    cur = conn.cursor()
                    cur.execute("""
                        INSERT INTO system_config (key, val) VALUES ('face_detection_enabled', ?)
                        ON CONFLICT(key) DO UPDATE SET val=excluded.val
                    """, ("1" if toggle_val else "0",))
                    conn.commit()
                    conn.close()
                    st.toast(f"Face Scanner {'ENABLED' if toggle_val else 'DISABLED'} by Admin", icon="⚙️")
                    st.rerun()

            with c_toggle2:
                if toggle_val:
                    st.success("🟢 Facial Recognition Active: Scans for VIP clearance and Blacklist targets.")
                else:
                    st.warning("🟡 Facial Recognition Disabled: Suppresses face boxes and optimizes FPS.")

            st.divider()
            st.markdown("#### 👥 Personnel Accounts")
            u1, u2 = st.columns([1.3, 1])
            with u1:
                st.write("##### Active Security Personnel")
                st.dataframe(get_all_users(), width="stretch", hide_index=True)
                st.write("##### 🗑️ Remove User")
                del_u = st.text_input("Username to remove")
                if st.button("Delete User", width="stretch") and del_u:
                    ok, msg = delete_user(del_u.strip())
                    if ok:
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)
            with u2:
                st.write("##### ➕ Provision New User")
                with st.form("new_user_form"):
                    nu = st.text_input("Username *")
                    nn = st.text_input("Full Name *")
                    np = st.text_input("Password *", type="password")
                    nr = st.selectbox("Role", ["operator", "admin"])
                    if st.form_submit_button("Create Account", width="stretch"):
                        if nu and np and nn:
                            ok, msg = add_new_user(nu.strip(), np.strip(), nr, nn.strip())
                            if ok:
                                st.success(msg)
                                st.rerun()
                            else:
                                st.error(msg)

render_soc_interface()