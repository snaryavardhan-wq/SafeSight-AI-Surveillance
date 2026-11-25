"""SQLite (WAL mode) persistence and the SHA-256 hash-chained forensic ledger.

content_hash = SHA256(S | T | s | e)           (snapshot bytes, timestamp, score, event)
chain_hash   = SHA256(prev_chain | content_hash | video_sha256)

Deleting, reordering or editing any sealed record (or its snapshot / clip) is
detected by verify_ledger(). Deleting the *newest* records cannot be detected
from the DB alone, which is why the chain hash is also sent to Telegram: the
security channel acts as an external witness.
"""
import os
import json
import uuid
import time
import sqlite3
import hashlib
import threading
from datetime import datetime

import config

GENESIS_HASH = "0" * 64
_write_lock = threading.Lock()
_cfg_cache = {}

_NEW_COLS = {
    "content_hash": "TEXT",
    "prev_hash": "TEXT",
    "chain_hash": "TEXT",
    "video_sha256": "TEXT",
}


def _connect():
    conn = sqlite3.connect(config.DB_PATH, timeout=10, isolation_level=None)
    conn.execute("PRAGMA busy_timeout=10000")
    return conn


def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")   # local time, consistent everywhere


def init_db():
    """Create/upgrade all tables and switch the DB to WAL mode. Safe to call repeatedly."""
    with _write_lock:
        conn = _connect()
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("""
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
                )""")
            cols = {r[1] for r in conn.execute("PRAGMA table_info(security_events)")}
            if "video_path" not in cols:
                conn.execute("ALTER TABLE security_events ADD COLUMN video_path TEXT")
            for c, t in _NEW_COLS.items():
                if c not in cols:
                    conn.execute(f"ALTER TABLE security_events ADD COLUMN {c} {t}")
            conn.execute("CREATE TABLE IF NOT EXISTS system_config (key TEXT PRIMARY KEY, val TEXT NOT NULL)")
            conn.execute("INSERT OR IGNORE INTO system_config (key, val) VALUES ('face_detection_enabled', '1')")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    username TEXT PRIMARY KEY,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL,
                    full_name TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )""")
        finally:
            conn.close()


# ---------------- small helpers ----------------
def has_users():
    try:
        conn = _connect()
        try:
            return conn.execute("SELECT COUNT(*) FROM users").fetchone()[0] > 0
        finally:
            conn.close()
    except sqlite3.Error:
        return False


def latest_event_id():
    try:
        conn = _connect()
        try:
            return conn.execute("SELECT COALESCE(MAX(id), 0) FROM security_events").fetchone()[0]
        finally:
            conn.close()
    except sqlite3.Error:
        return 0


def get_config_val(key, default="1", max_age=2.0):
    """Cached: the engine calls this every frame, so don't hit the DB each time."""
    now = time.time()
    hit = _cfg_cache.get(key)
    if hit and now - hit[1] < max_age:
        return hit[0]
    try:
        conn = _connect()
        try:
            row = conn.execute("SELECT val FROM system_config WHERE key = ?", (key,)).fetchone()
        finally:
            conn.close()
        val = row[0] if row else default
    except sqlite3.Error:
        val = hit[0] if hit else default
    _cfg_cache[key] = (val, now)
    return val


def set_config_val(key, val):
    try:
        with _write_lock:
            conn = _connect()
            try:
                conn.execute(
                    "INSERT INTO system_config (key, val) VALUES (?, ?) "
                    "ON CONFLICT(key) DO UPDATE SET val=excluded.val", (key, str(val)))
            finally:
                conn.close()
        _cfg_cache[key] = (str(val), time.time())
    except sqlite3.Error as e:
        print(f"[DB] config write failed: {e}")


# ---------------- dashboard -> engine control channel ----------------
def post_source_request(req):
    """The dashboard asks the running engine to switch its video source."""
    set_config_val("source_request", json.dumps({**req, "id": uuid.uuid4().hex}))


def take_source_request():
    """Engine side: fetch the pending request (if any) and clear it, so RTSP passwords don't linger in the DB."""
    raw = get_config_val("source_request", "", max_age=0.0)
    if not raw:
        return None
    set_config_val("source_request", "")
    try:
        req = json.loads(raw)
        return req if isinstance(req, dict) else None
    except ValueError:
        return None


# ---------------- hashing ----------------
def compute_content_hash(snapshot_bytes, timestamp, score, event):
    h = hashlib.sha256()
    h.update(snapshot_bytes)
    h.update(b"|" + str(timestamp).encode("utf-8"))
    h.update(b"|" + str(int(score)).encode("utf-8"))
    h.update(b"|" + str(event).encode("utf-8"))
    return h.hexdigest()


def compute_chain_hash(prev_hash, content_hash, video_sha):
    return hashlib.sha256(f"{prev_hash}|{content_hash}|{video_sha or ''}".encode("utf-8")).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------- ledger ----------------
def commit_event(event_type, track_id, score, level, details, snapshot_path,
                 snapshot_bytes, video_path=None, video_sha=None, timestamp=None):
    """Atomically append a sealed record. Returns a dict describing it."""
    timestamp = timestamp or now_str()
    content_hash = compute_content_hash(snapshot_bytes, timestamp, score, event_type)
    with _write_lock:
        conn = _connect()
        try:
            conn.execute("BEGIN IMMEDIATE")   # also serialises against other processes
            row = conn.execute(
                "SELECT chain_hash FROM security_events WHERE chain_hash IS NOT NULL "
                "ORDER BY id DESC LIMIT 1").fetchone()
            prev = row[0] if row else GENESIS_HASH
            chain = compute_chain_hash(prev, content_hash, video_sha)
            cur = conn.execute(
                "INSERT INTO security_events (timestamp, event_type, track_id, threat_score, threat_level, "
                "details, snapshot_path, video_path, content_hash, prev_hash, chain_hash, video_sha256) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (timestamp, event_type, track_id, int(score), level, details, snapshot_path,
                 video_path, content_hash, prev, chain, video_sha))
            rid = cur.lastrowid
            conn.execute("COMMIT")
        except Exception:
            try:
                conn.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            raise
        finally:
            conn.close()
    return {"id": rid, "timestamp": timestamp, "event_type": event_type, "score": int(score),
            "level": level, "details": details, "snapshot_path": snapshot_path,
            "video_path": video_path, "content_hash": content_hash, "prev_hash": prev,
            "chain_hash": chain, "video_sha256": video_sha}


def verify_ledger():
    """Recompute every digest from the files on disk and check the chain."""
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT id, timestamp, event_type, threat_score, snapshot_path, video_path, "
            "content_hash, prev_hash, chain_hash, video_sha256 FROM security_events ORDER BY id").fetchall()
    finally:
        conn.close()

    prev, head = GENESIS_HASH, GENESIS_HASH
    problems, sealed, legacy = [], 0, 0
    for rid, ts, ev, score, snap, vid, content, prev_h, chain, vsha in rows:
        if not chain:
            legacy += 1
            continue
        sealed += 1
        if prev_h != prev:
            problems.append({"id": rid, "issue": "CHAIN_BROKEN: a previous record was deleted or altered"})
        if compute_chain_hash(prev_h, content, vsha) != chain:
            problems.append({"id": rid, "issue": "RECORD_ALTERED: stored hashes do not match"})
        if snap and os.path.exists(snap):
            with open(snap, "rb") as f:
                data = f.read()
            if compute_content_hash(data, ts, score, ev) != content:
                problems.append({"id": rid, "issue": "EVIDENCE_OR_METADATA_MODIFIED: snapshot, time, score or event changed"})
        else:
            problems.append({"id": rid, "issue": "SNAPSHOT_MISSING"})
        if vsha:
            if vid and os.path.exists(vid):
                if sha256_file(vid) != vsha:
                    problems.append({"id": rid, "issue": "VIDEO_MODIFIED"})
            else:
                problems.append({"id": rid, "issue": "VIDEO_MISSING"})
        prev = head = chain
    return {"total": len(rows), "sealed": sealed, "legacy": legacy,
            "ok": not problems, "problems": problems, "head": head}