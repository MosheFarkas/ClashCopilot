"""Validate a KataCR detector against the 260 real labeled frames.

Gives a repeatable number to compare weights before/after fine-tuning
(and to catch regressions). Run under .venv-katacr:

    .venv-katacr/bin/python scripts/val_detector.py --weights path.pt [--detector 1]
"""

import argparse
import sys
from pathlib import Path

KATACR = Path(__file__).parent.parent / "vendor" / "KataCR"
sys.path.insert(0, str(KATACR))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--detector", type=int, default=1)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--imgsz", type=int, default=896)
    args = parser.parse_args()

    from katacr.yolov8.train import YOLO_CR

    model = YOLO_CR(args.weights)
    metrics = model.val(
        data=str(KATACR / f"katacr/yolov8/detector{args.detector}/data.yaml"),
        imgsz=args.imgsz, device=args.device, batch=4, workers=0, plots=False, verbose=False,
    )
    box = metrics.box
    print(f"\nWEIGHTS: {args.weights}")
    print(f"mAP50={box.map50:.4f}  mAP50-95={box.map:.4f}  "
          f"precision={box.mp:.4f}  recall={box.mr:.4f}")


if __name__ == "__main__":
    main()
