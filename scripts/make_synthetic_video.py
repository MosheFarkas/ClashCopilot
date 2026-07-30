"""Generate synthetic test footage for the CLI: video + templates + layout.

Run:  .venv/bin/python scripts/make_synthetic_video.py [out_dir]

Writes clip.avi, templates/<card>.png, and layout.json to out_dir
(default data/synthetic), then prints the CLI command to try.
"""

import json
import sys
from pathlib import Path

import cv2

from clash_copilot.synthetic import HOG_CYCLE, layout_dict, make_card_art, render_frames

FPS = 5


def main() -> None:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/synthetic")
    templates_dir = out_dir / "templates"
    templates_dir.mkdir(parents=True, exist_ok=True)

    art = make_card_art(HOG_CYCLE.deck)
    for card, tile in art.items():
        cv2.imwrite(str(templates_dir / f"{card}.png"), tile)

    frames = render_frames(HOG_CYCLE, art, fps=FPS)
    video_path = out_dir / "clip.avi"
    height, width = frames[0].image.shape[:2]
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (width, height)
    )
    for frame in frames:
        writer.write(frame.image)
    writer.release()

    layout_path = out_dir / "layout.json"
    layout_path.write_text(json.dumps(layout_dict(), indent=2))

    print(f"Wrote {video_path}, {len(art)} templates, {layout_path}")
    print("\nTry it:")
    print(
        f"  .venv/bin/python -m clash_copilot {video_path}"
        f" --layout {layout_path} --templates {templates_dir}"
    )


if __name__ == "__main__":
    main()
