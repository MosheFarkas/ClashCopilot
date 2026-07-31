"""Render detections (JSON from katacr_infer.py) onto a video.

Runs in the main venv: applies ByteTrack association + per-track label
voting for stability, then draws red/green boxes labeled with unit name
and confidence.

Run:  .venv/bin/python scripts/render_dets.py dets.json --out annotated.mp4
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from clash_copilot.detection.tracking import TrackSmoother  # noqa: E402

COLORS = {"ally": (80, 220, 80), "enemy": (60, 60, 235)}


def draw(frame, items):
    for name, side, conf, bbox in items:
        x0, y0, x1, y1 = bbox
        color = COLORS[side]
        cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
        label = f"{name} {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        ty = y0 - 3 if y0 - th - 6 >= 0 else y1 + th + 3
        cv2.rectangle(frame, (x0, ty - th - 3), (x0 + tw + 4, ty + 2), color, -1)
        cv2.putText(frame, label, (x0 + 2, ty), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dets")
    parser.add_argument("--out", required=True)
    parser.add_argument("--no-track", action="store_true")
    args = parser.parse_args()

    data = json.loads(Path(args.dets).read_text())
    frames = {int(k): v for k, v in data["frames"].items()}
    fps, start, stride = data["fps"], data["start"], data.get("stride", 1)

    tracker = smoother = None
    if not args.no_track:
        import supervision as sv
        from supervision.tracker.byte_tracker.core import ByteTrack

        tracker = ByteTrack(frame_rate=int(round(fps / stride)))
        smoother = TrackSmoother(window=7)
        names = sorted({d["name"] for dets in frames.values() for d in dets})
        name_to_id = {n: i for i, n in enumerate(names)}

    cap = cv2.VideoCapture(data["video"])
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start * fps))
    writer = None
    last_items: list = []
    index = 0
    while index <= max(frames):
        ok, frame = cap.read()
        if not ok:
            break
        if index in frames:
            dets = frames[index]
            if tracker is None:
                last_items = [
                    (d["name"], d["side"], d["conf"], tuple(d["bbox"])) for d in dets
                ]
            else:
                import supervision as sv

                if dets:
                    detections = sv.Detections(
                        xyxy=np.array([d["bbox"] for d in dets], dtype=np.float32),
                        confidence=np.array([d["conf"] for d in dets], dtype=np.float32),
                        class_id=np.array([name_to_id[d["name"]] for d in dets]),
                        data={"side": np.array([d["side"] for d in dets])},
                    )
                    tracked = tracker.update_with_detections(detections)
                    items = []
                    for i in range(len(tracked)):
                        tid = int(tracked.tracker_id[i])
                        name, side, conf = smoother.update(
                            tid, names[int(tracked.class_id[i])],
                            str(tracked.data["side"][i]), float(tracked.confidence[i]),
                        )
                        x0, y0, x1, y1 = tracked.xyxy[i]
                        items.append((name, side, conf,
                                      (round(x0), round(y0), round(x1), round(y1))))
                    last_items = items
                else:
                    tracker.update_with_detections(sv.Detections.empty())
                    last_items = []
        frame = draw(frame, last_items)
        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        writer.write(frame)
        index += 1
    cap.release()
    if writer:
        writer.release()
    print(f"wrote {index} frames to {args.out}")


if __name__ == "__main__":
    main()
