"""Harvest weak-labeled card crops from replay-view match recordings.

The replay UI shows BOTH players' hand bars (4 slots each). Videos contain
several matches, so decks are voted PER MATCH SEGMENT (segments found via
slot readability: menus/transitions read as garbage). A crop is saved only
when three independent signals agree -- the full-roster best match, the
deck-restricted best match, and a stable run of identical labels -- plus a
score floor and a variance floor that rejects transition-frame junk.
Precision over recall: the caps are generous and footage is cheap.

Slot geometry is normalized (calibrated on 590x1280 portrait replay
recordings; scales with resolution).

Run:  .venv/bin/python scripts/harvest_crops.py VIDEO [VIDEO ...]
          [--out data/train_crops] [--sample-fps 2] [--min-score 0.4]
"""

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from clash_copilot.cards import load_card_icon  # noqa: E402
from clash_copilot.classify.harvest import (  # noqa: E402
    color_agreement,
    readable_segments,
    smooth_labels,
    vote_deck,
)
from clash_copilot.classify.templates import load_templates  # noqa: E402
from clash_copilot.detection.template import best_template_match  # noqa: E402

ICONS = Path("data/icons")
ORIGIN = Path("data/footage/katacr-dataset/images/card_classification_origin")

# Normalized slot boxes for the replay UI (x, y, w, h); two bars of 4 slots.
SLOT_XS = [0.140, 0.270, 0.401, 0.531]
SLOT_W, SLOT_H = 0.118, 0.069
BARS = {"t": 0.109, "b": 0.865}  # top = opponent bar, bottom = recorder bar
CROP_SIZE = (64, 80)
MIN_RUN = 3
MAX_PER_CARD = 150
READABLE_SCORE = 0.3  # a slot "reads" when some card matches this well
MIN_STD = 25.0  # transition frames are near-uniform; real card crops are not
MIN_COLOR = 0.25  # colored crops must hue-agree with the card's API icon


def base(card: str | None) -> str | None:
    return card.split("#")[0] if card else None


def slot_keys() -> list[str]:
    return [f"{bar}{i}" for bar in BARS for i in range(len(SLOT_XS))]


def slot_boxes(shape) -> dict[str, tuple[int, int, int, int]]:
    height, width = shape[:2]
    return {
        f"{bar}{i}": (int(x * width), int(y0 * height), int(SLOT_W * width), int(SLOT_H * height))
        for bar, y0 in BARS.items()
        for i, x in enumerate(SLOT_XS)
    }


def sample_frames(video: Path, sample_fps: float):
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(1, round(fps / sample_fps))
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if index % step == 0:
            yield index, frame
        index += 1
    cap.release()


def harvest(video: Path, out: Path, sample_fps: float, min_score: float) -> Counter:
    templates = load_templates("origin", ICONS, ORIGIN)
    # color references from the CURRENT official icons: covers cards newer
    # than the exemplar set, which grayscale matching force-fits to old art
    color_refs = {
        p.stem: cv2.resize(icon, CROP_SIZE)
        for p in sorted(ICONS.glob("*.png"))
        if (icon := load_card_icon(p)) is not None
    }
    crops: dict[str, list] = defaultdict(list)  # slot key -> [(frame idx, gray, bgr)]
    for frame_index, frame in sample_frames(video, sample_fps):
        for key, (x, y, w, h) in slot_boxes(frame.shape).items():
            bgr = cv2.resize(frame[y : y + h, x : x + w], CROP_SIZE)
            crops[key].append((frame_index, cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), bgr))

    keys = slot_keys()
    full = {
        key: [best_template_match(gray, templates) for _, gray, _ in crops[key]]
        for key in keys
    }
    n = len(crops[keys[0]])
    readable = [
        sum(full[key][i][1] >= READABLE_SCORE for key in keys) >= 5 for i in range(n)
    ]
    segments = readable_segments(readable, min_len=int(30 * sample_fps))
    print(f"  {video.name}: {n} samples, {len(segments)} match segments")

    saved: Counter = Counter()
    for seg_index, (start, end) in enumerate(segments):
        for bar in BARS:
            bar_keys = [k for k in keys if k[0] == bar]
            reads = [
                (base(card), score)
                for key in bar_keys
                for card, score in full[key][start:end]
                if card and score >= min_score
            ]
            deck = vote_deck(reads)
            print(f"    segment {seg_index} {'opponent' if bar == 't' else 'recorder'}:"
                  f" {sorted(deck)}")
            restricted = {k: t for k, t in templates.items() if base(k) in deck}
            for key in bar_keys:
                labels = []
                for i in range(start, end):
                    _, gray, bgr = crops[key][i]
                    card_r, score_r = best_template_match(gray, restricted)
                    label = base(card_r)
                    agreed = (
                        label is not None
                        and score_r >= min_score
                        and base(full[key][i][0]) == label
                        and gray.std() >= MIN_STD
                    )
                    if agreed and label in color_refs:
                        hue = color_agreement(bgr, color_refs[label])
                        agreed = hue is None or hue >= MIN_COLOR
                    labels.append(label if agreed else None)
                smoothed = smooth_labels(labels, MIN_RUN)
                for offset, label in enumerate(smoothed):
                    if label is None or saved[label] >= MAX_PER_CARD:
                        continue
                    frame_index, _, bgr = crops[key][start + offset]
                    target = out / label
                    target.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(target / f"{video.stem}_{frame_index:06d}_{key}.jpg"), bgr)
                    saved[label] += 1
    return saved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("videos", nargs="+")
    parser.add_argument("--out", default="data/train_crops")
    parser.add_argument("--sample-fps", type=float, default=2.0)
    parser.add_argument("--min-score", type=float, default=0.4)
    args = parser.parse_args()

    total: Counter = Counter()
    for video in args.videos:
        print(f"harvesting {video} ...")
        total += harvest(Path(video), Path(args.out), args.sample_fps, args.min_score)
    print("\ncrops saved per card:")
    for card, count in total.most_common():
        print(f"  {card:24} {count}")
    print(f"total: {sum(total.values())}")


if __name__ == "__main__":
    main()
