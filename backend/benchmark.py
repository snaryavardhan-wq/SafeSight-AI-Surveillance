"""Where does the time go, and which settings make SafeSight faster on THIS computer?

    python backend/benchmark.py --video demo_footage/clip.mp4
    python backend/benchmark.py --video clip.mp4 --imgsz 640 480 320 --face both --motion-gate both
    python backend/benchmark.py --video clip.mp4 --model yolov8n.pt --model yolov8n_openvino_model

Each combination processes the same frames (every frame, no skipping) and reports:
  FPS            frames the whole pipeline handles per second
  detect / ...   average milliseconds per frame spent in each stage
  detector runs  how often YOLO really ran (the motion gate skips it on static scenes)

Nothing is saved except the report in reports/. Speed is only half the story: check each fast setting with
evaluate.py so you know what accuracy it costs.
"""
import os
import sys
import json
import time
import shutil
import argparse
import tempfile
import itertools

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

STAGES = ["prep", "tamper", "faces", "detect", "spatial", "analytics", "hud", "total"]


def run_setting(video, frames, warmup):
    """Process `warmup + frames` frames with the current config and return timing results."""
    import safesight
    from filecapture import VideoFileCapture
    cap = VideoFileCapture(video, every_frame=True)
    engine = safesight.SentryEngine(cap)
    engine.evidence.enabled = False
    engine.mute()
    last, n, t0 = 0, 0, None
    try:
        while n < warmup + frames:
            item = cap.get_latest(last, timeout=2.0)
            if item is None:
                if cap.ended:
                    break
                continue
            frame, ts, last = item
            if n == warmup:                       # warm-up frames (model load, caches) are not counted
                engine.reset_timing()
                t0 = time.perf_counter()
            engine.step(frame, ts)
            n += 1
    finally:
        cap.release()
    measured = max(n - warmup, 0)
    wall = (time.perf_counter() - t0) if t0 else 0.0
    return {"frames": measured, "fps": measured / wall if wall > 0 else 0.0, "ms": engine.timing_avg(),
            "detect_calls": engine.detect_calls, "detect_skipped": engine.detect_skipped}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Benchmark SafeSight speed on this computer")
    ap.add_argument("--video", required=True)
    ap.add_argument("--frames", type=int, default=200, help="frames to measure per setting (default 200)")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--imgsz", type=int, nargs="+", default=[config.YOLO_IMGSZ], help="detector sizes to try")
    ap.add_argument("--model", action="append", help="model(s) to try; repeat the flag for several")
    ap.add_argument("--face", choices=["on", "off", "both"], default="off")
    ap.add_argument("--motion-gate", choices=["on", "off", "both"], default="off")
    ap.add_argument("--out", help="report base name (default reports/benchmark_<time>)")
    args = ap.parse_args(argv)

    if not os.path.exists(args.video):
        raise SystemExit(f"Video not found: {args.video}")
    models = args.model or [config.YOLO_MODEL]
    faces = {"on": [True], "off": [False], "both": [False, True]}[args.face]
    gates = {"on": [True], "off": [False], "both": [False, True]}[args.motion_gate]

    import database
    saved = (config.DB_PATH, config.YOLO_MODEL, config.YOLO_IMGSZ, config.MOTION_GATE)
    tmp = tempfile.mkdtemp(prefix="safesight_bench_")
    config.DB_PATH = os.path.join(tmp, "bench.db")           # scratch DB: your real one is never touched
    rows = []
    try:
        database.init_db()
        for model, imgsz, face, gate in itertools.product(models, args.imgsz, faces, gates):
            config.YOLO_MODEL, config.YOLO_IMGSZ, config.MOTION_GATE = model, imgsz, gate
            database.set_config_val("face_detection_enabled", "1" if face else "0")
            label = f"{os.path.basename(str(model))} | imgsz {imgsz} | face {'on' if face else 'off'} | gate {'on' if gate else 'off'}"
            print(f"\n>> {label}", flush=True)
            try:
                r = run_setting(args.video, args.frames, args.warmup)
            except Exception as e:                           # e.g. an exported model that fails to load
                print(f"   skipped: {e}")
                continue
            r.update(label=label, model=str(model), imgsz=imgsz, face=face, gate=gate)
            rows.append(r)
            print(f"   {r['fps']:.1f} FPS over {r['frames']} frames, detector ran {r['detect_calls']}x "
                  f"(skipped {r['detect_skipped']}x)")
    finally:
        config.DB_PATH, config.YOLO_MODEL, config.YOLO_IMGSZ, config.MOTION_GATE = saved
        shutil.rmtree(tmp, ignore_errors=True)

    if not rows:
        raise SystemExit("No setting could be measured.")
    table = format_table(rows)
    print("\n" + table)
    base = args.out or os.path.join(config.REPORTS_DIR, time.strftime("benchmark_%Y%m%d_%H%M%S"))
    os.makedirs(os.path.dirname(base) or ".", exist_ok=True)
    with open(base + ".md", "w", encoding="utf-8") as f:
        f.write(markdown(rows, args))
    with open(base + ".json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(f"\nSaved {base}.md and {base}.json")
    best = max(rows, key=lambda r: r["fps"])
    print(f"Fastest: {best['label']}  ({best['fps']:.1f} FPS)")
    return rows


def format_table(rows):
    head = f"{'Setting':<64}{'FPS':>6}" + "".join(f"{s:>10}" for s in STAGES[3:6]) + f"{'detector runs':>15}"
    lines = [head, "-" * len(head)]
    for r in rows:
        ms = r["ms"]
        runs = f"{r['detect_calls']}/{r['detect_calls'] + r['detect_skipped']}"
        lines.append(f"{r['label']:<64}{r['fps']:>6.1f}" + "".join(f"{ms.get(s, 0):>9.1f}ms"[-10:] for s in STAGES[3:6]) + f"{runs:>15}")
    return "\n".join(lines)


def markdown(rows, args):
    out = ["# SafeSight speed benchmark", "",
           f"Video: `{os.path.basename(args.video)}` | {args.frames} frames per setting (after {args.warmup} warm-up) | every frame analysed", "",
           "| Setting | FPS | " + " | ".join(f"{s} ms" for s in STAGES) + " | Detector runs |",
           "|---|---|" + "---|" * len(STAGES) + "---|"]
    for r in rows:
        ms = r["ms"]
        out.append(f"| {r['label']} | {r['fps']:.1f} | " + " | ".join(f"{ms.get(s, 0):.1f}" for s in STAGES)
                   + f" | {r['detect_calls']} of {r['detect_calls'] + r['detect_skipped']} |")
    out += ["", "_ms = average milliseconds per frame in that stage. `detect` is YOLOv8n + ByteTrack; `faces` is the optional face "
                "scanner; `total` includes drawing the display. Verify the accuracy cost of any faster setting with evaluate.py._"]
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    main()