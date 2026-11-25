"""Record a staged scenario from the webcam so it can be replayed as demo footage.

    python backend/record_demo.py                 # records until you press q
    python backend/record_demo.py --seconds 60    # stops by itself
Then:  python backend/safesight.py --video demo_footage/<file>.mp4

Tip: stage a story: walk into the zone, put a bag down and walk away, run across
the frame, cover the lens for 3 seconds. The preview shows a red REC dot; the
saved video itself stays clean (no overlay).
"""
import os
import sys
import time
import argparse

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


def main(argv=None):
    ap = argparse.ArgumentParser(description="Record webcam footage for the SafeSight demo")
    ap.add_argument("--out", help="output .mp4 (default demo_footage/demo_<time>.mp4)")
    ap.add_argument("--seconds", type=float, default=0, help="stop after N seconds (0 = until q)")
    ap.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    ap.add_argument("--fps", type=float, default=30.0, help="video frame rate (default 30)")
    args = ap.parse_args(argv)

    os.makedirs(config.FOOTAGE_DIR, exist_ok=True)
    out = args.out or os.path.join(config.FOOTAGE_DIR, time.strftime("demo_%Y%m%d_%H%M%S.mp4"))
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAPTURE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAPTURE_HEIGHT)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("Could not read from the camera.")
    h, w = frame.shape[:2]
    writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
    if not writer.isOpened():
        raise SystemExit("Could not open the output video file.")

    print(f"Recording to {out}  ({w}x{h}). Press q in the preview window to stop.")
    t0, written = time.time(), 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            elapsed = time.time() - t0
            # write as many frames as wall-clock time requires, so playback speed is true
            # even when the camera delivers fewer frames per second than args.fps
            due = int(elapsed * args.fps) + 1 - written
            for _ in range(max(due, 1)):
                writer.write(frame)
                written += 1
            preview = frame.copy()
            if int(elapsed * 2) % 2 == 0:
                cv2.circle(preview, (30, 30), 10, (0, 0, 255), -1)
            cv2.putText(preview, f"REC {int(elapsed // 60):02d}:{int(elapsed % 60):02d}   press q to stop",
                        (50, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("SafeSight - record demo footage", preview)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                break
            if args.seconds and elapsed >= args.seconds:
                break
    finally:
        writer.release()
        cap.release()
        cv2.destroyAllWindows()
    print(f"Saved {written} frames ({written / args.fps:.1f} s): {out}")
    print(f"Analyse it with:  python backend/safesight.py --video \"{out}\"")


if __name__ == "__main__":
    main()