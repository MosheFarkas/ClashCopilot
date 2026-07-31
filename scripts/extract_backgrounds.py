"""Extract clean 2026 arena backgrounds from match footage.

Samples frames across a window and takes the per-pixel median, so units
and spells (which move) disappear while the arena and towers (which do
not) remain. Output matches KataCR's background format: 568x896 arena
crops, ready for their synthetic generator.

Run:  .venv/bin/python scripts/extract_backgrounds.py VIDEO --out data/backgrounds
          [--windows 6] [--samples 25] [--span 40]
"""

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from clash_copilot.classify.background import median_background  # noqa: E402

ARENA_CROP = (0.024, 0.186, 0.954, 0.694)  # same profile as katacr_infer.py
PART2_SIZE = (568, 896)


def arena(frame):
    height, width = frame.shape[:2]
    x, y, w, h = ARENA_CROP
    x0, y0 = int(x * width), int(y * height)
    crop = frame[y0 : y0 + int(h * height), x0 : x0 + int(w * width)]
    return cv2.resize(crop, PART2_SIZE, interpolation=cv2.INTER_CUBIC)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--out", default="data/backgrounds")
    parser.add_argument("--windows", type=int, default=6, help="backgrounds to produce")
    parser.add_argument("--samples", type=int, default=25, help="frames per median")
    parser.add_argument("--span", type=float, default=40.0, help="seconds each window spans")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.video).stem

    written = 0
    for w in range(args.windows):
        start = int(total * (w + 0.5) / args.windows) - int(args.span * fps / 2)
        start = max(0, start)
        step = max(1, int(args.span * fps / args.samples))
        frames = []
        for s in range(args.samples):
            cap.set(cv2.CAP_PROP_POS_FRAMES, start + s * step)
            ok, frame = cap.read()
            if ok:
                frames.append(arena(frame))
        if len(frames) < 5:
            continue
        background = median_background(frames)
        path = out_dir / f"{stem}_bg{w:02d}.jpg"
        cv2.imwrite(str(path), background, [cv2.IMWRITE_JPEG_QUALITY, 95])
        written += 1
        print(f"wrote {path}")
    cap.release()
    print(f"{written} backgrounds -> {out_dir}")


if __name__ == "__main__":
    main()
