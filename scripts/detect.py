"""Run the trained YOLO11n detector on a webcam, a video file, or images.

Examples:
    python scripts/detect.py --source 0                          # live webcam (press q to quit)
    python scripts/detect.py --source demo/some_video.mp4        # video file
    python scripts/detect.py --source assets/samples/            # folder of images
    python scripts/detect.py --source photo.jpg --save           # save annotated output to runs/
"""
import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def run_stream(model, source, conf, save):
    """Frame-by-frame loop with an OpenCV window (webcam or video)."""
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open source: {source}")

    writer = None
    if save:
        out_dir = ROOT / "runs" / "detect" / "stream"
        out_dir.mkdir(parents=True, exist_ok=True)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_path = out_dir / "output.mp4"
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        results = model(frame, conf=conf, verbose=False)
        annotated = results[0].plot()
        if writer:
            writer.write(annotated)
        cv2.imshow("YOLO11n Real-Time Detection", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer:
        writer.release()
        print(f"Saved to {out_path}")
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", default=str(ROOT / "weights" / "best.pt"))
    parser.add_argument("--source", default="0", help="webcam index, video path, image path, or folder")
    parser.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--save", action="store_true", help="save annotated results")
    args = parser.parse_args()

    model = YOLO(args.weights)
    is_stream = args.source.isdigit() or Path(args.source).suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}
    if is_stream:
        run_stream(model, args.source, args.conf, args.save)
    else:
        model.predict(args.source, conf=args.conf, save=args.save, project=str(ROOT / "runs" / "detect"))


if __name__ == "__main__":
    main()
