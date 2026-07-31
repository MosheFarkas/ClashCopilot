"""Render an annotated match video: boxes around troops/spells with
name, ally/enemy side, and confidence.

Run:  .venv/bin/python scripts/annotate_video.py VIDEO [--out out.mp4]
          [--start SECONDS] [--duration SECONDS] [--min-conf 0.45]

Colors: enemy red, ally green. Processes every frame in the window
(CPU ONNX ~ several fps; pick a short window for quick previews).
"""

import argparse
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from clash_copilot.detection.arena import ArenaDetector  # noqa: E402

MODELS = Path("data/models")
COLORS = {"ally": (80, 220, 80), "enemy": (60, 60, 235)}  # BGR


def draw(frame, detections):
    for d in detections:
        x0, y0, x1, y1 = d.bbox
        color = COLORS[d.side]
        cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
        label = f"{d.name} {d.conf:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        ty = y0 - 4 if y0 - th - 8 >= 0 else y1 + th + 4
        cv2.rectangle(frame, (x0, ty - th - 4), (x0 + tw + 4, ty + 3), color, -1)
        cv2.putText(frame, label, (x0 + 2, ty), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--out", default=None)
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--min-conf", type=float, default=0.45)
    args = parser.parse_args()

    detector = ArenaDetector(
        MODELS / "units_M_480x352.onnx", MODELS / "side.onnx", min_conf=args.min_conf
    )
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(args.start * fps))
    n_frames = int(args.duration * fps) if args.duration else None

    out_path = args.out or str(Path(args.video).with_suffix("")) + "_annotated.mp4"
    writer = None
    count = 0
    started = time.time()
    while n_frames is None or count < n_frames:
        ok, frame = cap.read()
        if not ok:
            break
        detections = detector.detect(frame)
        frame = draw(frame, detections)
        if writer is None:
            height, width = frame.shape[:2]
            writer = cv2.VideoWriter(
                out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
            )
        writer.write(frame)
        count += 1
        if count % 100 == 0:
            print(f"  {count} frames ({count / (time.time() - started):.1f} fps)", flush=True)
    cap.release()
    if writer is not None:
        writer.release()
    print(f"wrote {count} frames to {out_path}"
          f" ({count / (time.time() - started):.1f} fps processing)")


if __name__ == "__main__":
    main()
