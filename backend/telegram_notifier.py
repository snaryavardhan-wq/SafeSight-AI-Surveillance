"""Telegram dispatch. Token/chat id come from env vars or .env (never hard-coded).

A single worker thread sends messages in order (Telegram allows ~1 msg/s per
chat) so bursts are queued, not dropped, and the video loop never blocks.
"""
import os
import html
import queue
import threading
import time

import requests

import config

_q = queue.Queue(maxsize=50)
_started = False
_start_lock = threading.Lock()


def is_configured():
    return bool(config.TELEGRAM_BOT_TOKEN and config.TELEGRAM_CHAT_ID)


def _post(message, photo_path):
    base = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}"
    try:
        if photo_path and os.path.exists(photo_path):
            with open(photo_path, "rb") as ph:
                res = requests.post(f"{base}/sendPhoto", timeout=10,
                                    data={"chat_id": config.TELEGRAM_CHAT_ID, "caption": message[:1000],
                                          "parse_mode": "HTML"},
                                    files={"photo": ph})
        else:
            res = requests.post(f"{base}/sendMessage", timeout=10,
                                json={"chat_id": config.TELEGRAM_CHAT_ID, "text": message[:4000],
                                      "parse_mode": "HTML"})
        if res.status_code == 200:
            return True
        print(f"[TELEGRAM] API error: {res.text[:200]}")
    except Exception as e:
        print(f"[TELEGRAM] dispatch failed: {e}")
    return False


def _worker():
    while True:
        msg, photo = _q.get()
        if not _post(msg, photo):
            time.sleep(2.0)
            _post(msg, photo)
        time.sleep(1.0)


def send_telegram_alert(message, photo_path=None):
    global _started
    if not is_configured():
        print("[TELEGRAM] Not configured (set TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID). Skipping.")
        return False
    with _start_lock:
        if not _started:
            threading.Thread(target=_worker, name="telegram", daemon=True).start()
            _started = True
    try:
        _q.put_nowait((message, photo_path))
        return True
    except queue.Full:
        return False


def send_event_alert(rec):
    """Format a sealed ledger record. The chain hash doubles as an external witness."""
    e = html.escape
    msg = (f"🚨 <b>SafeSight {e(rec['level'])}</b>: {e(rec['event_type'])}\n"
           f"Score: {rec['score']}/100 | Event #{rec['id']}\n"
           f"Time: {e(rec['timestamp'])}\n"
           f"Details: {e(str(rec['details']))}\n"
           f"Ledger seal: <code>{rec['chain_hash'][:32]}</code>")
    return send_telegram_alert(msg, rec.get("snapshot_path"))