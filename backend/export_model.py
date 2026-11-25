"""Export YOLOv8n to a CPU-friendly format (EXPERIMENTAL; benchmark it before trusting it).

    pip install openvino                      # Intel CPUs / integrated graphics
    python backend/export_model.py --format openvino
    python backend/benchmark.py --video demo_footage/clip.mp4 --model yolov8n.pt --model yolov8n_openvino_model
    python backend/evaluate.py  --video clip.mp4 --labels clip.labels.json   (with SAFESIGHT_MODEL set, to check accuracy)

To use the exported model:   $env:SAFESIGHT_MODEL = "yolov8n_openvino_model"    or    --model yolov8n_openvino_model
If the exported model fails to load, SafeSight prints a message and falls back to yolov8n.pt.
"""
import argparse


def main(argv=None):
    ap = argparse.ArgumentParser(description="Export YOLOv8n for faster CPU inference")
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--format", choices=["openvino", "onnx"], default="openvino")
    ap.add_argument("--imgsz", type=int, default=640, help="keep equal to the size you will run with")
    args = ap.parse_args(argv)
    from ultralytics import YOLO
    try:
        path = YOLO(args.model).export(format=args.format, imgsz=args.imgsz)
    except Exception as e:
        raise SystemExit(f"Export failed: {e}\nTry:  pip install openvino   (or: pip install onnx onnxruntime)")
    print(f"\nExported to: {path}")
    print(f'Use it with:  python backend/safesight.py --model "{path}"')
    print("Then run benchmark.py and evaluate.py to confirm it is both faster and as accurate.")


if __name__ == "__main__":
    main()