"""Fine-tune YOLO11n on the 4-class campus dataset (car, fountain, person, tree).

Reproduces the original Colab run:
    yolo train model=yolo11n.pt data=data.yaml epochs=20 imgsz=640

Usage:
    python scripts/train.py                     # defaults of the original run
    python scripts/train.py --epochs 100 --device 0
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="yolo11n.pt", help="COCO-pretrained checkpoint to start from")
    parser.add_argument("--data", default=str(ROOT / "data" / "data.yaml"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None, help="'cpu', '0', '0,1', ... (auto if omitted)")
    args = parser.parse_args()

    model = YOLO(args.model)
    model.train(
        data=str(Path(args.data).resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(ROOT / "runs" / "detect"),
    )
    # Evaluate the best checkpoint on the held-out test split
    model.val(data=str(Path(args.data).resolve()), split="test", project=str(ROOT / "runs" / "detect"))


if __name__ == "__main__":
    main()
