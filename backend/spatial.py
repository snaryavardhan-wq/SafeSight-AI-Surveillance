"""Stage 3: spatial analytics (AABB edge distance, perspective-invariant velocity,
unattended-bag tracker, polygon restricted zone)."""
import math
from collections import deque

import cv2
import numpy as np

import config


def box_edge_distance(a, b):
    """D_edge: AABB edge-to-edge separation; 0 when the boxes overlap or touch."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    dx = max(0, max(ax1, bx1) - min(ax2, bx2))
    dy = max(0, max(ay1, by1) - min(ay2, by2))
    return math.hypot(dx, dy)


class SprintDetector:
    """v_norm = displacement over a 5-frame window / (body height * elapsed seconds)."""

    def __init__(self):
        self._hist = {}
        self._hits = {}

    def update(self, tid, t, cx, cy, h):
        dq = self._hist.get(tid)
        if dq is None:
            dq = self._hist[tid] = deque(maxlen=config.SPRINT_WINDOW_FRAMES)
        dq.append((t, cx, cy))
        if len(dq) < dq.maxlen:
            return 0.0, False
        (t0, x0, y0), (t1, x1, y1) = dq[0], dq[-1]
        dt = t1 - t0
        if dt <= 1e-3:
            return 0.0, False
        v = math.hypot(x1 - x0, y1 - y0) / (max(h, 1) * dt)
        hits = self._hits.get(tid, 0) + 1 if v >= config.SPRINT_VNORM else 0
        self._hits[tid] = hits
        return v, hits >= config.SPRINT_CONFIRM_EVALS

    def prune(self, active_ids):
        for tid in list(self._hist):
            if tid not in active_ids:
                self._hist.pop(tid, None)
                self._hits.pop(tid, None)


class BagTracker:
    """Persistent bag identities (one-to-one nearest matching) + separation timers."""

    def __init__(self):
        self.bags = {}
        self.next_id = 1

    def update(self, detections, now, person_boxes):
        """detections: [(x1,y1,x2,y2,label)]. Returns a list of status dicts."""
        unmatched = set(self.bags)
        for x1, y1, x2, y2, label in detections:
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            best, best_d = None, config.BAG_MATCH_RADIUS_PX
            for b_id in unmatched:
                px, py = self.bags[b_id]["center"]
                d = math.hypot(cx - px, cy - py)
                if d < best_d:
                    best, best_d = b_id, d
            if best is None:
                best = self.next_id
                self.next_id += 1
                self.bags[best] = {"separated_start": None}
            unmatched.discard(best)
            self.bags[best].update(box=(x1, y1, x2, y2), center=(cx, cy), label=label, last_seen=now)

        for b_id in [i for i, b in self.bags.items() if now - b["last_seen"] > config.BAG_FORGET_SEC]:
            del self.bags[b_id]

        out = []
        for b_id, b in self.bags.items():
            dist = min((box_edge_distance(b["box"], pb) for pb in person_boxes.values()),
                       default=float("inf"))
            if dist <= config.BAG_ABANDON_DIST_PX:
                b["separated_start"] = None
                out.append({"id": b_id, "box": b["box"], "label": b["label"],
                            "state": "CARRIED", "seconds": 0.0, "dist": dist})
                continue
            if b["separated_start"] is None:
                b["separated_start"] = now
            secs = now - b["separated_start"]
            state = "UNATTENDED" if secs >= config.BAG_ABANDON_TIME_SEC else "SEPARATING"
            out.append({"id": b_id, "box": b["box"], "label": b["label"],
                        "state": state, "seconds": secs, "dist": dist})
        return out


class PolygonZone:
    """Operator-drawn convex/concave polygon. L-click adds a vertex, R-click closes it."""

    def __init__(self):
        self.points = []
        self.closed = False
        self._contour = None

    def set_points(self, pts):
        self.points = [(int(x), int(y)) for x, y in pts]
        self.closed = len(self.points) >= 3
        self._rebuild()

    def clear(self):
        self.points, self.closed, self._contour = [], False, None

    def _rebuild(self):
        self._contour = np.array(self.points, np.int32).reshape(-1, 1, 2) if self.closed else None

    def on_mouse(self, event, x, y):
        if event == cv2.EVENT_LBUTTONDOWN and not self.closed:
            self.points.append((int(x), int(y)))
        elif event == cv2.EVENT_RBUTTONDOWN and not self.closed and len(self.points) >= 3:
            self.closed = True
            self._rebuild()

    def signed_distance(self, pt):
        """>= 0 inside (pixels from edge), < 0 outside; None if no zone is defined."""
        if not self.closed:
            return None
        return cv2.pointPolygonTest(self._contour, (float(pt[0]), float(pt[1])), True)

    def draw(self, img, breached=False):
        if not self.points:
            return
        color = (0, 0, 255) if breached else (255, 255, 255)
        pts = np.array(self.points, np.int32).reshape(-1, 1, 2)
        if self.closed:
            overlay = img.copy()
            cv2.fillPoly(overlay, [pts], color)
            cv2.addWeighted(overlay, 0.18, img, 0.82, 0, img)
        cv2.polylines(img, [pts], self.closed, color, 2)
        for p in self.points:
            cv2.circle(img, p, 4, color, -1)
        x, y = self.points[0]
        cv2.putText(img, "RESTRICTED PERIMETER" if self.closed else "DRAWING ZONE (R-click closes)",
                    (x + 6, max(18, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)