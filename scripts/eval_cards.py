"""Evaluate card-identity template matching on real game footage crops.

Templates: official card portraits (data/icons, from scripts/fetch_cards.py).
Test set:  labeled hand-slot crops from the MIT-licensed KataCR dataset
           (data/footage/katacr-dataset/images/card_classification).

Each crop is classified against the FULL card roster (not just the deck's
8 cards); top-1 accuracy is reported per class.

Measured configurations (2026-07, 168 non-empty crops):
    color, equal-size:            41.7%
    color, sliding 0.8:           52.4%
    gray,  sliding 0.8:           53.0%   <- best, used below
    gray + equalizeHist, 0.8:     52.4%
    in-domain single-crop templates: worse (35-47%) -- intra-class variance
    (grey-out states, "next" slot countdown overlay, border styles) exceeds
    cross-domain art differences.
Conclusion: raw template matching tops out ~53% on real hand-slot crops.
Prior art (AmarSaini, KataCR) solved this with a small CNN classifier
trained on augmented card art -- that is the planned next step; this
script stays as the benchmark harness.

Run:  .venv/bin/python scripts/eval_cards.py
"""

import re
import sys
from collections import Counter
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from clash_copilot.detection.template import best_template_match  # noqa: E402

ICONS = Path("data/icons")
DATASET = Path("data/footage/katacr-dataset/images/card_classification")
SIZE = (64, 80)  # (w, h): crops compared at this scale
TEMPLATE_SCALE = 0.8  # templates smaller than crops -> alignment tolerance


def norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def to_gray(image) -> "cv2.Mat":
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def main() -> None:
    template_size = (int(SIZE[0] * TEMPLATE_SCALE), int(SIZE[1] * TEMPLATE_SCALE))
    templates = {}
    for path in sorted(ICONS.glob("*.png")):
        icon = cv2.imread(str(path))
        if icon is not None:
            templates[path.stem] = cv2.resize(to_gray(cv2.resize(icon, SIZE)), template_size)
    by_norm = {norm(name): name for name in templates}
    print(f"{len(templates)} roster templates (gray, slide {TEMPLATE_SCALE}) | test set: {DATASET}\n")

    empty_scores: list[float] = []
    correct_scores: list[float] = []
    overall = Counter()
    for class_dir in sorted(DATASET.iterdir()):
        if not class_dir.is_dir() or class_dir.name.startswith("_"):
            continue
        base = norm(class_dir.name.removesuffix("-evolution"))
        expected = by_norm.get(base)
        if expected is None and class_dir.name != "empty":
            print(f"{class_dir.name:24} SKIP (no roster match)")
            continue

        n = hits = 0
        scores: list[float] = []
        confusions: Counter = Counter()
        for jpg in sorted(class_dir.glob("*.jpg")):
            crop = cv2.imread(str(jpg))
            if crop is None:
                continue
            card, score = best_template_match(to_gray(cv2.resize(crop, SIZE)), templates)
            n += 1
            scores.append(score)
            if class_dir.name == "empty":
                empty_scores.append(score)
            elif card == expected:
                hits += 1
                correct_scores.append(score)
            else:
                confusions[card] += 1

        if class_dir.name == "empty":
            print(f"{'empty':24} n={n:4}  best-match scores "
                  f"max={max(scores):.2f} mean={sum(scores)/n:.2f}")
            continue
        overall["n"] += n
        overall["hits"] += hits
        top_confusion = confusions.most_common(1)
        note = f"  worst confusion: {top_confusion[0][0]} x{top_confusion[0][1]}" if top_confusion else ""
        print(f"{class_dir.name:24} n={n:4}  top1={hits/n:6.1%}  "
              f"mean score={sum(scores)/n:.2f}{note}")

    print(f"\nOverall top-1 (non-empty): {overall['hits']}/{overall['n']}"
          f" = {overall['hits']/overall['n']:.1%}")
    if correct_scores and empty_scores:
        print(f"Separation: correct-match scores min={min(correct_scores):.2f}"
              f" vs empty scores max={max(empty_scores):.2f}")


if __name__ == "__main__":
    main()
