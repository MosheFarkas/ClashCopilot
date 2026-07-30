"""Run the tracker over a recorded match video.

Usage:
    python -m clash_copilot VIDEO --layout layout.json --templates DIR [--cards cards.json]

The layout JSON (see geometry.Layout) is calibrated per recording setup;
templates are one PNG per card, named after the card. Try it without real
footage via scripts/make_synthetic_video.py, which generates all three.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from clash_copilot.capture.source import VideoFileSource
from clash_copilot.cards import load_card_costs
from clash_copilot.detection.template import TemplateCardDetector
from clash_copilot.geometry import Layout
from clash_copilot.pipeline import OpponentTracker
from clash_copilot.report import format_state, format_summary


def load_templates(directory: str | Path) -> dict[str, np.ndarray]:
    templates = {
        path.stem: image
        for path in sorted(Path(directory).glob("*.png"))
        if (image := cv2.imread(str(path))) is not None
    }
    if not templates:
        raise SystemExit(f"no PNG templates found in {directory}")
    return templates


def build_tracker(
    layout: Layout,
    templates: dict[str, np.ndarray],
    frame_shape: tuple[int, ...],
    card_costs: dict[str, int],
) -> OpponentTracker:
    w, h = layout.card_size_pixels(frame_shape)
    detector = TemplateCardDetector(
        {name: cv2.resize(image, (w, h)) for name, image in templates.items()},
        roi=layout.play_zone.to_pixels(frame_shape),
        threshold=layout.threshold,
    )
    return OpponentTracker(detector, card_costs)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="clash_copilot", description="Track the opponent in a recorded match."
    )
    parser.add_argument("video", help="path to a match recording")
    parser.add_argument("--layout", required=True, help="layout JSON for this recording setup")
    parser.add_argument("--templates", required=True, help="directory of per-card PNGs")
    parser.add_argument("--cards", default=None, help="cards.json (default: bundled sample)")
    args = parser.parse_args(argv)

    layout = Layout.from_json(args.layout)
    templates = load_templates(args.templates)
    costs = load_card_costs(args.cards)

    tracker = None
    last = None
    for frame in VideoFileSource(args.video).frames():
        if tracker is None:
            tracker = build_tracker(layout, templates, frame.image.shape, costs)
        state = tracker.process_frame(frame)
        if state is not None:
            print(format_state(state))
            last = state

    print()
    print(format_summary(last) if last else "No plays detected.")


if __name__ == "__main__":
    main()
