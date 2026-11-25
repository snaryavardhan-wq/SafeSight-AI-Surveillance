"""Stage 2A: multi-vector camera anti-tamper heuristics on a 320x180 thumbnail."""
import cv2

import config


class TamperDetector:
    def __init__(self):
        self.start = None
        self.metrics = {}

    def analyze(self, frame, now):
        """Return (state, reason, duration); state is CLEAR, VERIFYING or TAMPERED."""
        small = cv2.resize(frame, config.TAMPER_SIZE, interpolation=cv2.INTER_NEAREST)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        mu = float(gray.mean())
        sd = float(gray.std())
        lap = float(cv2.Laplacian(gray, cv2.CV_32F).var())
        sat = float(hsv[:, :, 1].mean())
        self.metrics = {"mu": mu, "sd": sd, "lap": lap, "sat": sat}

        reason = None
        if mu < config.BLACKOUT_LUM:
            reason = "BLACKOUT"
        elif mu > config.BLIND_LUM:
            reason = "BLINDING/LASER"
        elif lap < config.OCCLUSION_LAP and sd < config.OCCLUSION_STD:
            reason = "OCCLUSION/DEFOCUS"
        elif (mu > config.PAPER_LUM and sat < config.PAPER_SAT
              and lap < config.PAPER_LAP and sd < config.PAPER_STD):
            reason = "PAPER/PAINT COVER"

        if reason is None:
            self.start = None
            return "CLEAR", None, 0.0
        if self.start is None:
            self.start = now
        dur = now - self.start
        return ("TAMPERED" if dur >= config.TAMPER_CONFIRM_SEC else "VERIFYING"), reason, dur