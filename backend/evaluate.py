"""Measure how well SafeSight finds the incidents in your clips.

1.  Watch a clip and write down what happens and when (times are seconds in the video):

        python backend/evaluate.py --template demo_footage/clip.mp4  > demo_footage/clip.labels.json
        (then edit the events: type, start, end)

2.  Score it:

        python backend/evaluate.py --video demo_footage/clip.mp4 --labels demo_footage/clip.labels.json
        python backend/evaluate.py --dir demo_footage            # every clip that has a .labels.json beside it

3.  Compare speed settings (what does a faster setting cost in accuracy?):

        python backend/evaluate.py --dir demo_footage --imgsz 640 480 320 --motion-gate both

Every frame is analysed (footage clock, so the result does not depend on how fast your PC is). Nothing is written
to your real database or evidence folders. Output: a table, plus reports/evaluation_<time>.md and .json.

How events are scored
  * An alert is a TRUE POSITIVE if it has the right type and fires between (start - tolerance) and (end + tolerance).
  * Repeated alerts of one type within --merge seconds count once (an operator sees one incident).
  * Alerts that match nothing are FALSE POSITIVES; labelled events nobody alerted on are FALSE NEGATIVES (missed).
  * Delay = alert time minus the labelled start.
"""
import os
import re
import sys
import json
import time
import shutil
import argparse
import tempfile
import statistics
from contextlib import contextmanager

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

ALIASES = {
    "zone": "Restricted Zone Intrusion", "intrusion": "Restricted Zone Intrusion",
    "restricted zone intrusion": "Restricted Zone Intrusion",
    "bag": "Unattended Baggage Anomaly", "baggage": "Unattended Baggage Anomaly", "unattended": "Unattended Baggage Anomaly",
    "unattended baggage anomaly": "Unattended Baggage Anomaly",
    "sprint": "Sprint Anomaly", "running": "Sprint Anomaly", "sprint anomaly": "Sprint Anomaly",
    "tamper": "Hardware Camera Tampering", "tampering": "Hardware Camera Tampering",
    "hardware camera tampering": "Hardware Camera Tampering",
    "signal lost": "Camera Signal Lost", "camera signal lost": "Camera Signal Lost",
    "blacklist": "Blacklisted Face Match", "blacklisted face match": "Blacklisted Face Match",
}


def normalize_type(name):
    key = str(name).strip().lower()
    if key not in ALIASES:
        raise ValueError(f"Unknown event type {name!r}. Use one of: zone, bag, sprint, tamper, signal lost, blacklist")
    return ALIASES[key]


# ---------------------------------------------------------------- scoring (pure functions)
def merge_predictions(preds, window):
    """Collapse alerts of the same type that follow each other within `window` seconds into the first one."""
    out, last = [], {}
    for p in sorted(preds, key=lambda p: p["t"]):
        prev = last.get(p["event"])
        if prev is not None and p["t"] - prev["t"] <= window:
            prev["merged"] = prev.get("merged", 1) + 1
            prev["t_last"] = p["t"]                   # chain: a long run of alerts stays one incident
            continue
        q = dict(p, merged=1, t_last=p["t"])
        out.append(q)
        last[p["event"]] = q
    return out


def score_events(gt, preds, tolerance=2.0, merge=5.0):
    """gt: [{type,start,end}], preds: [{event,t}]. Returns {type: stats} plus per-event match details."""
    gt = [dict(type=normalize_type(g["type"]), start=float(g["start"]), end=float(g.get("end", g["start"]))) for g in gt]
    preds = merge_predictions(preds, merge)
    types = sorted({g["type"] for g in gt} | {p["event"] for p in preds})
    result = {}
    for ty in types:
        g_list = sorted((g for g in gt if g["type"] == ty), key=lambda g: g["start"])
        p_list = sorted((p for p in preds if p["event"] == ty), key=lambda p: p["t"])
        used, matches = set(), []
        for g in g_list:
            hit = None
            for i, p in enumerate(p_list):
                if i not in used and g["start"] - tolerance <= p["t"] <= g["end"] + tolerance:
                    hit = i
                    break
            if hit is not None:
                used.add(hit)
                matches.append({"gt": g, "t": p_list[hit]["t"], "delay": p_list[hit]["t"] - g["start"]})
        result[ty] = {"tp": len(matches), "fn": len(g_list) - len(matches), "fp": len(p_list) - len(used),
                      "matches": matches, "delays": [m["delay"] for m in matches],
                      "false_alarms": [p["t"] for i, p in enumerate(p_list) if i not in used],
                      "missed": [g for g in g_list if not any(m["gt"] is g for m in matches)]}
    return result


def _ratio(a, b):
    return None if b == 0 else a / b


def summarize(per_type_list, hours):
    """Combine several clips: micro-average per type and overall."""
    agg = {}
    for per_type in per_type_list:
        for ty, r in per_type.items():
            a = agg.setdefault(ty, {"tp": 0, "fp": 0, "fn": 0, "delays": []})
            a["tp"] += r["tp"]; a["fp"] += r["fp"]; a["fn"] += r["fn"]; a["delays"] += r["delays"]
    rows = {}
    for ty, a in agg.items():
        p, r = _ratio(a["tp"], a["tp"] + a["fp"]), _ratio(a["tp"], a["tp"] + a["fn"])
        f1 = None if (p is None or r is None or p + r == 0) else 2 * p * r / (p + r)
        rows[ty] = {**a, "precision": p, "recall": r, "f1": f1,
                    "mean_delay": statistics.fmean(a["delays"]) if a["delays"] else None,
                    "median_delay": statistics.median(a["delays"]) if a["delays"] else None,
                    "fp_per_hour": a["fp"] / hours if hours > 0 else None}
    tot = {k: sum(r[k] for r in rows.values()) for k in ("tp", "fp", "fn")}
    p, r = _ratio(tot["tp"], tot["tp"] + tot["fp"]), _ratio(tot["tp"], tot["tp"] + tot["fn"])
    f1 = None if (p is None or r is None or p + r == 0) else 2 * p * r / (p + r)
    all_delays = [d for a in agg.values() for d in a["delays"]]
    rows["ALL EVENTS"] = {**tot, "delays": all_delays, "precision": p, "recall": r, "f1": f1,
                          "mean_delay": statistics.fmean(all_delays) if all_delays else None,
                          "median_delay": statistics.median(all_delays) if all_delays else None,
                          "fp_per_hour": tot["fp"] / hours if hours > 0 else None}
    return rows


# ---------------------------------------------------------------- running the engine on a clip
@contextmanager
def isolated_state():
    """Scratch database and no evidence files, so evaluating never touches your real data."""
    old_db, old_gate, old_sz, old_model = config.DB_PATH, config.MOTION_GATE, config.YOLO_IMGSZ, config.YOLO_MODEL
    tmp = tempfile.mkdtemp(prefix="safesight_eval_")
    config.DB_PATH = os.path.join(tmp, "eval.db")
    try:
        yield
    finally:
        config.DB_PATH = old_db
        config.MOTION_GATE, config.YOLO_IMGSZ, config.YOLO_MODEL = old_gate, old_sz, old_model
        shutil.rmtree(tmp, ignore_errors=True)


def run_clip(video, zone=None):
    """Analyse every frame of `video`; return (alerts, duration_sec, frames, avg_fps)."""
    import safesight                                           # imported late: needs ultralytics
    from filecapture import VideoFileCapture
    from database import init_db, set_config_val
    init_db()
    set_config_val("face_detection_enabled", "0")               # evaluation measures the core detectors
    cap = VideoFileCapture(video, every_frame=True)
    engine = safesight.SentryEngine(cap)
    engine.evidence.enabled = False
    engine.mute()
    if zone:
        engine.zone.set_points(zone)
    last, t0, frames = 0, time.time(), 0
    try:
        while True:
            item = cap.get_latest(last, timeout=2.0)
            if item is None:
                if cap.ended:
                    break
                continue
            frame, ts, last = item
            engine.step(frame, ts)
            frames += 1
    finally:
        engine.evidence.wait_idle(5)
        cap.release()
    elapsed = max(time.time() - t0, 1e-3)
    return engine.detections, (cap.duration_sec or frames / cap.fps), frames, frames / elapsed


def parse_zone(text):
    if not text:
        return None
    pts = [tuple(int(v) for v in p.split(",")) for p in text.split(";")]
    if len(pts) < 3:
        raise ValueError("zone needs at least 3 points")
    return pts


def load_labels(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    events = data.get("events", [])
    for e in events:
        if "start" not in e or "type" not in e:
            raise ValueError(f"{path}: every event needs 'type' and 'start'")
    return data


# ---------------------------------------------------------------- reporting
def _pct(x):
    return "n/a" if x is None else f"{100 * x:.0f}%"


def _sec(x):
    return "n/a" if x is None else f"{x:+.1f}s"


def format_table(rows):
    head = f"{'Event type':<28}{'TP':>4}{'FP':>4}{'FN':>4}{'Precision':>11}{'Recall':>8}{'F1':>6}{'Mean delay':>12}{'FP/hour':>9}"
    lines = [head, "-" * len(head)]
    for ty in sorted(rows, key=lambda k: (k == "ALL EVENTS", k)):
        r = rows[ty]
        fph = "n/a" if r["fp_per_hour"] is None else f"{r['fp_per_hour']:.1f}"
        lines.append(f"{ty:<28}{r['tp']:>4}{r['fp']:>4}{r['fn']:>4}{_pct(r['precision']):>11}{_pct(r['recall']):>8}"
                     f"{_pct(r['f1']):>6}{_sec(r['mean_delay']):>12}{fph:>9}")
    return "\n".join(lines)


def format_markdown(rows, clips, tolerance, merge):
    out = ["# SafeSight detection evaluation", "",
           f"Clips: {len(clips)} | footage: {sum(c['duration'] for c in clips) / 60:.1f} min | "
           f"tolerance: {tolerance}s | merge window: {merge}s | every frame analysed", "",
           "| Event type | TP | FP | FN | Precision | Recall | F1 | Mean delay | False alarms / hour |",
           "|---|---|---|---|---|---|---|---|---|"]
    for ty in sorted(rows, key=lambda k: (k == "ALL EVENTS", k)):
        r = rows[ty]
        fph = "n/a" if r["fp_per_hour"] is None else f"{r['fp_per_hour']:.1f}"
        out.append(f"| {ty} | {r['tp']} | {r['fp']} | {r['fn']} | {_pct(r['precision'])} | {_pct(r['recall'])} | "
                   f"{_pct(r['f1'])} | {_sec(r['mean_delay'])} | {fph} |")
    out += ["", "## Per clip", "", "| Clip | Length | Frames | Avg FPS | Missed events | False alarms |", "|---|---|---|---|---|---|"]
    for c in clips:
        missed = "; ".join(f"{m['type']} @{m['start']:.0f}s" for r in c["per_type"].values() for m in r["missed"]) or "none"
        fa = "; ".join(f"{ty} @{t:.1f}s" for ty, r in c["per_type"].items() for t in r["false_alarms"]) or "none"
        out.append(f"| {c['name']} | {c['duration']:.0f}s | {c['frames']} | {c['fps']:.1f} | {missed} | {fa} |")
    out += ["", "_Precision = TP/(TP+FP), recall = TP/(TP+FN). Counts of 0 give n/a. "
                "With only a few events the percentages are indicative, not statistical proof._"]
    return "\n".join(out) + "\n"


def template_for(video):
    return json.dumps({"video": os.path.basename(video),
                       "zone": "x1,y1;x2,y2;x3,y3;x4,y4   <- optional: delete this line to reuse the zone you drew for this video",
                       "events": [{"type": "zone", "start": 0.0, "end": 0.0, "note": "person enters the restricted zone"},
                                  {"type": "bag", "start": 0.0, "end": 0.0, "note": "bag left behind"}]}, indent=2)


def format_sweep_markdown(results, tolerance, merge):
    out = ["# SafeSight speed / accuracy comparison", "",
           f"tolerance: {tolerance}s | merge window: {merge}s | every frame analysed | face scanning off", "",
           "| Setting | FPS | TP | FP | FN | Precision | Recall | F1 | Mean delay | False alarms / hour |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        a = r["rows"]["ALL EVENTS"]
        fph = "n/a" if a["fp_per_hour"] is None else f"{a['fp_per_hour']:.1f}"
        out.append(f"| {r['label']} | {r['fps']:.1f} | {a['tp']} | {a['fp']} | {a['fn']} | {_pct(a['precision'])} | "
                   f"{_pct(a['recall'])} | {_pct(a['f1'])} | {_sec(a['mean_delay'])} | {fph} |")
    out += ["", "_Pick the fastest row whose recall and false alarms you can live with. FPS here is the whole pipeline "
                "on this PC. With few events, differences of one or two events are noise._", ""]
    for r in results:
        out += [f"## {r['label']}", "", format_markdown(r["rows"], r["clips"], tolerance, merge).split("\n", 2)[2]]
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Score SafeSight against clips you labelled by hand")
    ap.add_argument("--video"); ap.add_argument("--labels")
    ap.add_argument("--dir", help="evaluate every video that has <name>.labels.json beside it")
    ap.add_argument("--template", metavar="VIDEO", help="print an empty labels file for this video")
    ap.add_argument("--tolerance", type=float, default=2.0, help="seconds of slack around each labelled event")
    ap.add_argument("--merge", type=float, default=5.0, help="alerts of one type this close together count once")
    ap.add_argument("--imgsz", type=int, nargs="+", help="detector size(s); several sizes = a speed/accuracy sweep")
    ap.add_argument("--motion-gate", choices=["on", "off", "both"], help="motion gate setting(s) to try")
    ap.add_argument("--model", action="append", help="model(s) to try; repeat the flag for several")
    ap.add_argument("--out", help="report base name (default reports/evaluation_<time>)")
    args = ap.parse_args(argv)

    if args.template:
        print(template_for(args.template))
        return 0
    pairs = []
    if args.dir:
        for fn in sorted(os.listdir(args.dir)):
            base, ext = os.path.splitext(fn)
            lab = os.path.join(args.dir, base + ".labels.json")
            if ext.lower() in config.VIDEO_EXTENSIONS and os.path.exists(lab):
                pairs.append((os.path.join(args.dir, fn), lab))
    elif args.video and args.labels:
        pairs.append((args.video, args.labels))
    if not pairs:
        ap.error("give --video and --labels, or --dir with <clip>.labels.json files next to the videos")

    sizes = args.imgsz or [config.YOLO_IMGSZ]
    gates = {"on": [True], "off": [False], "both": [False, True], None: [config.MOTION_GATE]}[args.motion_gate]
    models = args.model or [config.YOLO_MODEL]
    settings = [(m, z, g) for m in models for z in sizes for g in gates]

    import database                                            # zones you already drew for these videos (real DB, read-only)
    saved = {}
    for video, _ in pairs:
        raw = database.get_config_val(f"zone_polygon:{os.path.basename(video)}", "", max_age=0.0)
        try:
            saved[video] = [tuple(p) for p in json.loads(raw)] if raw else None
        except ValueError:
            saved[video] = None
    results = []
    with isolated_state():
        for model, imgsz, gate in settings:
            config.YOLO_MODEL, config.YOLO_IMGSZ, config.MOTION_GATE = model, imgsz, gate
            label = f"{os.path.basename(str(model))} | imgsz {imgsz} | gate {'on' if gate else 'off'}"
            if len(settings) > 1:
                print(f"\n=== {label} ===")
            clips, per_type_list = [], []
            for video, lab in pairs:
                data = load_labels(lab)
                print(f"\n>> {os.path.basename(video)} ...", flush=True)
                ztxt = data.get("zone")
                zone = parse_zone(ztxt) if ztxt and "x1" not in str(ztxt) else None     # the template's placeholder is ignored
                zone = zone or saved.get(video)
                print(f"   zone: {'from the labels file' if ztxt and zone and 'x1' not in str(ztxt) else 'drawn earlier in the dashboard' if zone else 'none (zone events cannot fire)'}")
                alerts, dur, frames, fps = run_clip(video, zone)
                per_type = score_events(data["events"], alerts, args.tolerance, args.merge)
                per_type_list.append(per_type)
                clips.append({"name": os.path.basename(video), "duration": dur, "frames": frames, "fps": fps,
                              "per_type": per_type, "alerts": alerts})
                print(f"   {frames} frames, {len(alerts)} alert(s), {fps:.1f} FPS")
            rows = summarize(per_type_list, sum(c["duration"] for c in clips) / 3600.0)
            results.append({"label": label, "model": str(model), "imgsz": imgsz, "gate": gate, "rows": rows, "clips": clips,
                            "fps": sum(c["frames"] for c in clips) / max(sum(c["frames"] / max(c["fps"], 1e-6) for c in clips), 1e-6)})

    base = args.out or os.path.join(config.REPORTS_DIR, time.strftime("evaluation_%Y%m%d_%H%M%S"))
    os.makedirs(os.path.dirname(base) or ".", exist_ok=True)
    if len(results) == 1:
        rows, clips = results[0]["rows"], results[0]["clips"]
        print("\n" + format_table(rows))
        with open(base + ".md", "w", encoding="utf-8") as f:
            f.write(format_markdown(rows, clips, args.tolerance, args.merge))
        slim = {k: {kk: vv for kk, vv in v.items() if kk != "delays"} for k, v in rows.items()}
        with open(base + ".json", "w", encoding="utf-8") as f:
            json.dump({"summary": slim, "clips": [{k: c[k] for k in ("name", "duration", "frames", "fps", "alerts")} for c in clips],
                       "tolerance": args.tolerance, "merge": args.merge}, f, indent=2)
    else:
        head = f"{'Setting':<44}{'FPS':>6}{'TP':>4}{'FP':>4}{'FN':>4}{'Precision':>11}{'Recall':>8}{'F1':>6}{'Delay':>8}"
        print("\n" + head + "\n" + "-" * len(head))
        for r in results:
            a = r["rows"]["ALL EVENTS"]
            print(f"{r['label']:<44}{r['fps']:>6.1f}{a['tp']:>4}{a['fp']:>4}{a['fn']:>4}{_pct(a['precision']):>11}"
                  f"{_pct(a['recall']):>8}{_pct(a['f1']):>6}{_sec(a['mean_delay']):>8}")
        with open(base + ".md", "w", encoding="utf-8") as f:
            f.write(format_sweep_markdown(results, args.tolerance, args.merge))
        with open(base + ".json", "w", encoding="utf-8") as f:
            json.dump({"settings": [{"label": r["label"], "model": r["model"], "imgsz": r["imgsz"], "gate": r["gate"],
                                     "fps": r["fps"], "summary": {k: {kk: vv for kk, vv in v.items() if kk != "delays"}
                                                                   for k, v in r["rows"].items()}} for r in results],
                       "tolerance": args.tolerance, "merge": args.merge}, f, indent=2)
    print(f"\nSaved {base}.md and {base}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())