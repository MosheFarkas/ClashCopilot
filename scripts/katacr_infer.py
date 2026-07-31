"""Run the KataCR dual-detector on arena crops (separate pinned venv).

KataCR's detector is the accuracy fix for our footage: trained on
568x896 arena-only crops from real phone recordings, ~150 classes that
include towers/bars/text (so decorations stop being misread as units),
and ally/enemy predicted natively as a `bel` attribute.

Must run under .venv-katacr (ultralytics==8.1.24, torch==2.2.2, numpy<2)
with vendor/KataCR on PYTHONPATH -- the checkpoints unpickle to
katacr.yolov8.custom_model.CRDetectionModel. Only the model + predictor
classes are imported (the trainer/validator path pulls in jax).

Reads frames, writes detections as JSON so the main venv can consume them:
    .venv-katacr/bin/python scripts/katacr_infer.py VIDEO --out dets.json
        [--start S] [--duration S] [--stride N] [--conf 0.5]
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

KATACR = Path(__file__).parent.parent / "vendor" / "KataCR"
sys.path.insert(0, str(KATACR))

from ultralytics.engine.model import Model  # noqa: E402

from katacr.constants.label_list import idx2unit, unit2idx  # noqa: E402
from katacr.yolov8.custom_model import CRDetectionModel  # noqa: E402
from katacr.yolov8.custom_predict import CRDetectionPredictor  # noqa: E402

# Arena crop (x, y, w, h as frame fractions). KataCR ships playback params
# only for 2.22-ratio video; ours is 2.16 (590x1280). Width/x follow their
# playback profile, height is derived from the 568x896 target aspect, and y
# was calibrated against their reference part2 crops.
ARENA_CROP = (0.024, 0.186, 0.954, 0.694)
PART2_SIZE = (568, 896)

# Non-unit classes (the "# object" block of katacr/constants/label_list.py):
# towers, HP/level bars, and UI chrome. Detecting them is what stops the
# model hallucinating troops on scenery, but they are not troops/spells.
NON_UNIT = {
    "king-tower", "queen-tower", "cannoneer-tower", "dagger-duchess-tower",
    "dagger-duchess-tower-bar", "tower-bar", "king-tower-bar", "bar",
    "bar-level", "clock", "emote", "text", "elixir", "selected",
    "skeleton-king-bar",
}


class YOLO_CR_Infer(Model):
    """YOLO_CR without the trainer/validator (those import jax)."""

    @property
    def task_map(self):
        return {"detect": {"model": CRDetectionModel, "predictor": CRDetectionPredictor}}


def arena_crop(frame: np.ndarray) -> tuple[np.ndarray, tuple[int, int, float, float]]:
    height, width = frame.shape[:2]
    x, y, w, h = ARENA_CROP
    x0, y0 = int(x * width), int(y * height)
    x1, y1 = x0 + int(w * width), y0 + int(h * height)
    crop = frame[y0:y1, x0:x1]
    resized = cv2.resize(crop, PART2_SIZE, interpolation=cv2.INTER_CUBIC)
    scale_x = (x1 - x0) / PART2_SIZE[0]
    scale_y = (y1 - y0) / PART2_SIZE[1]
    return resized, (x0, y0, scale_x, scale_y)


class ComboDetector:
    def __init__(self, weights: list[str], conf: float = 0.5, iou: float = 0.6,
                 units_only: bool = True, device: str = "cpu"):
        self.models = [YOLO_CR_Infer(str(w)) for w in weights]
        for model in self.models:
            model.to(device)
        self.conf, self.iou = conf, iou
        self.units_only = units_only

    def detect(self, crop: np.ndarray) -> list[dict]:
        """Detections on a 568x896 arena crop, in crop pixel coords."""
        import torch
        import torchvision

        rows = []
        for model in self.models:
            result = model.predict(crop, verbose=False, conf=self.conf)[0]
            boxes = result.orig_boxes.clone()  # xyxy, conf, cls, bel
            for i in range(len(boxes)):
                # map per-detector class index to the global unit index
                boxes[i, 5] = unit2idx[result.names[int(boxes[i, 5])]]
            rows.append(boxes)
        if not rows:
            return []
        merged = torch.cat(rows, 0)
        if len(merged) == 0:
            return []
        keep = torchvision.ops.nms(merged[:, :4], merged[:, 4], self.iou)
        merged = merged[keep]

        out = []
        for row in merged.tolist():
            x0, y0, x1, y1, conf, cls = row[:6]
            bel = int(row[6]) if len(row) > 6 else 0
            name = idx2unit[int(cls)]
            if self.units_only and name in NON_UNIT:
                continue
            out.append(
                {
                    "name": name,
                    "side": "enemy" if bel == 1 else "ally",
                    "conf": float(conf),
                    "bbox": [round(x0), round(y0), round(x1), round(y1)],
                }
            )
        return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--out", required=True)
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--all-classes", action="store_true",
                        help="also emit towers/bars/UI classes, not just troops and spells")
    parser.add_argument("--device", default="mps", help="cpu | mps | cuda")
    args = parser.parse_args()

    weights = [
        str(KATACR / "runs" / "detector1_v0.7.13.pt"),
        str(KATACR / "runs" / "detector2_v0.7.13.pt"),
    ]
    detector = ComboDetector(weights, conf=args.conf, units_only=not args.all_classes,
                             device=args.device)

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(args.start * fps))
    limit = int(args.duration * fps) if args.duration else None

    frames_out = {}
    index = 0
    while limit is None or index < limit:
        ok, frame = cap.read()
        if not ok:
            break
        if index % args.stride == 0:
            crop, (x0, y0, sx, sy) = arena_crop(frame)
            detections = detector.detect(crop)
            for d in detections:  # map back to full-frame pixels
                bx0, by0, bx1, by1 = d["bbox"]
                d["bbox"] = [
                    round(x0 + bx0 * sx), round(y0 + by0 * sy),
                    round(x0 + bx1 * sx), round(y0 + by1 * sy),
                ]
            frames_out[index] = detections
            if (index // args.stride) % 20 == 0:
                print(f"  frame {index}: {len(detections)} detections", flush=True)
        index += 1
    cap.release()

    Path(args.out).write_text(json.dumps({
        "video": args.video, "start": args.start, "fps": fps,
        "stride": args.stride, "frames": frames_out,
    }))
    total = sum(len(v) for v in frames_out.values())
    print(f"wrote {len(frames_out)} frames, {total} detections to {args.out}")


if __name__ == "__main__":
    main()
