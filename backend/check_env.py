"""Check that every package SafeSight needs can actually be loaded on this PC.

    python backend/check_env.py

Prints OK or the exact reason for each one, plus what to do. Useful when Windows (Application Control /
Smart App Control / antivirus) blocks a compiled file.
"""
import sys
import importlib

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))

CHECKS = [
    ("numpy", True), ("cv2", True), ("scipy", True), ("pandas", True), ("PIL", True), ("requests", True),
    ("torch", True), ("ultralytics", True), ("streamlit", True), ("reportlab", True),
    ("lap", False), ("pyttsx3", False), ("imageio_ffmpeg", False),
]
HINTS = {
    "Application Control": "Windows blocked this file. Reinstall a different version "
                           "(pip install --force-reinstall --no-cache-dir <package>==<older version>), "
                           "or ask the PC's administrator to allow the folder.",
    "CascadeClassifier": "OpenCV 5 removed face detection. Run: pip install \"opencv-python>=4.10,<5\"",
    "No module named": "Not installed. Run: pip install -r requirements.txt",
}


def hint(msg):
    for key, text in HINTS.items():
        if key in msg:
            return text
    return ""


def main():
    bad = 0
    for name, required in CHECKS:
        try:
            mod = importlib.import_module(name)
            ver = getattr(mod, "__version__", "")
            extra = ""
            if name == "cv2" and not hasattr(mod, "CascadeClassifier"):
                raise ImportError("cv2 has no attribute 'CascadeClassifier'")
            print(f"  OK       {name:<16}{ver}{extra}")
        except Exception as e:
            msg = f"{e.__class__.__name__}: {e}"
            tag = "FAILED" if required else "optional"
            print(f"  {tag:<8} {name:<16}{msg[:110]}")
            h = hint(msg)
            if h:
                print(f"           -> {h}")
            if name == "lap":
                print("           -> SafeSight has a built-in replacement for this one (lapshim.py), so this is not fatal.")
            elif required:
                bad += 1

    import numpy as np
    import lapshim
    lapshim.install_lap_shim()
    try:
        importlib.import_module("ultralytics.trackers.byte_tracker")
        print("  OK       ByteTrack tracker loads")
    except Exception as e:
        bad += 1
        print(f"  FAILED   ByteTrack tracker: {e.__class__.__name__}: {str(e)[:110]}")
    try:                                                  # the check that matters: really run the matching step
        lap = sys.modules["lap"]
        lap.lapjv(np.array([[0.1, 0.9], [0.8, 0.2]]), extend_cost=True, cost_limit=0.5)
        print(f"  OK       ByteTrack assignment step works ({'SafeSight replacement' if 'shim' in lap.__version__ else 'lap'})")
    except Exception as e:
        bad += 1
        print(f"  FAILED   ByteTrack assignment step: {e.__class__.__name__}: {str(e)[:110]}")
        print("           -> Run: pip install scipy   (SafeSight then uses it instead of the blocked file)")
    print("\nAll required packages load." if not bad else f"\n{bad} problem(s) above need fixing before SafeSight can run.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())