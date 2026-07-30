"""Evaluate card-identity template matching on real game footage crops.

Templates: official card portraits (data/icons, from scripts/fetch_cards.py).
Test set:  labeled hand-slot crops from the MIT-licensed KataCR dataset
           (data/footage/katacr-dataset/images/card_classification).

Each crop is classified against the FULL card roster (not just the deck's
8 cards); top-1 accuracy is reported per class.

Measured configurations (2026-07, 168 non-empty crops):
    color, equal-size:            41.7%
    color, sliding 0.8:           52.4%
    gray,  sliding 0.8:           53.0%
    gray + equalizeHist, 0.8:     52.4%
    in-domain single-crop templates: worse (35-47%) -- intra-class variance
    (grey-out states, "next" slot countdown overlay, border styles) exceeds
    cross-domain art differences.
    + alpha-aware icon loading (load_card_icon; icons are RGBA with ~30%
      transparent margin that previously baked in a black silhouette):
      gray, sliding 0.8:          64.3%   <- current template baseline
CNN classifier trained on augmented icons (--model): 53.0% -- masters the
synthetic domain (loss ~0.1) but does not transfer past the template
baseline. Closing the remaining gap needs real labeled slot crops (see
README roadmap); this script stays as the benchmark harness for both.

Run:  .venv/bin/python scripts/eval_cards.py            # template baseline
      .venv/bin/python scripts/eval_cards.py --model data/models/card_cnn.pt
"""

import argparse
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from clash_copilot.cards import load_card_icon  # noqa: E402
from clash_copilot.detection.template import best_template_match  # noqa: E402

ICONS = Path("data/icons")
DATASET = Path("data/footage/katacr-dataset/images/card_classification")
SIZE = (64, 80)  # (w, h): crops compared at this scale
TEMPLATE_SCALE = 0.8  # templates smaller than crops -> alignment tolerance


def norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def to_gray(image) -> "cv2.Mat":
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def eval_classifier(model_path: str) -> None:
    import numpy as np

    from clash_copilot.classify.benchmark import load_labeled_crops
    from clash_copilot.classify.model import CardClassifier

    clf = CardClassifier.load(model_path)
    roster = [name for name in clf.class_names if name != "empty"]
    crops = load_labeled_crops(DATASET, roster=roster)
    predictions = clf.predict(np.stack([crop for crop, _ in crops]))

    by_class: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for (_, expected), (predicted, prob) in zip(crops, predictions):
        by_class[expected].append((predicted, prob))
    print(f"classifier {model_path} | {len(crops)} crops | full {len(roster)}-card roster\n")
    overall = Counter()
    for expected in sorted(by_class):
        results = by_class[expected]
        hits = sum(predicted == expected for predicted, _ in results)
        confusions = Counter(p for p, _ in results if p != expected)
        top_confusion = confusions.most_common(1)
        note = f"  worst confusion: {top_confusion[0][0]} x{top_confusion[0][1]}" if top_confusion else ""
        print(f"{expected:24} n={len(results):4}  top1={hits/len(results):6.1%}"
              f"  mean prob={sum(p for _, p in results)/len(results):.2f}{note}")
        if expected != "empty":
            overall["n"] += len(results)
            overall["hits"] += hits
    print(f"\nOverall top-1 (non-empty): {overall['hits']}/{overall['n']}"
          f" = {overall['hits']/overall['n']:.1%}")


def eval_templates() -> None:
    template_size = (int(SIZE[0] * TEMPLATE_SCALE), int(SIZE[1] * TEMPLATE_SCALE))
    templates = {}
    for path in sorted(ICONS.glob("*.png")):
        icon = load_card_icon(path)
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="CardClassifier weights; omit for template baseline")
    args = parser.parse_args()
    if args.model:
        eval_classifier(args.model)
    else:
        eval_templates()


if __name__ == "__main__":
    main()
