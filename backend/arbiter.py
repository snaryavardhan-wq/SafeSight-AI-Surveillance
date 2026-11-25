"""Stage 4: four-tier monotonic threat arbiter.

NORMAL 0 | CAUTION 50 | HIGH 75 | CRITICAL 90-100.
The score is the MAX of active signals (not a sum), so a high-priority threat
overrides minor ones. Within an event window the score never decreases; it
resets to NORMAL once every condition has been clear for ARBITER_CLEAR_SEC.
"""
import config


def level_name(score):
    if score >= 90:
        return "CRITICAL"
    if score >= 75:
        return "HIGH"
    if score >= 50:
        return "CAUTION"
    return "NORMAL"


class ThreatArbiter:
    def __init__(self):
        self.window_peak = 0
        self._quiet_since = None

    def update(self, signals, now):
        """signals: list of (score, label). Returns (score, level, labels_desc)."""
        raw = max((s for s, _ in signals), default=0)
        if raw > 0:
            self._quiet_since = None
            self.window_peak = max(self.window_peak, raw)
        else:
            if self._quiet_since is None:
                self._quiet_since = now
            if now - self._quiet_since >= config.ARBITER_CLEAR_SEC:
                self.window_peak = 0
        labels = [l for _, l in sorted(signals, key=lambda x: -x[0])]
        return self.window_peak, level_name(self.window_peak), labels