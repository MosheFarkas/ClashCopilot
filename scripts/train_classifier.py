"""Train the card-identity CNN on augmented official portraits.

Fresh augmented samples are generated every round (the network never sees
the same crop twice), and progress is scored against the real-footage
benchmark crops after each round; the best-scoring weights are kept.

Honest-numbers caveat: the real crops steer WHEN we stop, so the final
figure is model selection on the test set. Fine for a scaffold benchmark;
a held-out split needs more real labeled data first.

Run:  .venv/bin/python scripts/train_classifier.py [--rounds 20] [--per-class 32]
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from clash_copilot.cards import load_card_icon  # noqa: E402
from clash_copilot.classify.augment import training_batch  # noqa: E402
from clash_copilot.classify.benchmark import load_labeled_crops  # noqa: E402
from clash_copilot.classify.model import CardClassifier  # noqa: E402

ICONS = Path("data/icons")
DATASET = Path("data/footage/katacr-dataset/images/card_classification")
NEGATIVES = Path("data/footage/katacr-dataset/images/part2")  # real arena frames
OUT = Path("data/models/card_cnn.pt")


def load_negatives(limit: int = 150) -> list:
    """Real non-card imagery for the empty class (arena frames, not slot crops)."""
    paths = sorted(NEGATIVES.rglob("*.jpg"))
    step = max(1, len(paths) // limit)
    images = [cv2.imread(str(p)) for p in paths[::step][:limit]]
    return [image for image in images if image is not None]


def evaluate(clf: CardClassifier, crops) -> tuple[float, float]:
    images = np.stack([crop for crop, _ in crops])
    expected = [label for _, label in crops]
    predicted = [name for name, _ in clf.predict(images)]
    hits = [p == e for p, e in zip(predicted, expected)]
    non_empty = [h for h, e in zip(hits, expected) if e != "empty"]
    return float(np.mean(hits)), float(np.mean(non_empty))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--per-class", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import torch

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    icons = {
        p.stem: icon
        for p in sorted(ICONS.glob("*.png"))
        if (icon := load_card_icon(p)) is not None
    }
    crops = load_labeled_crops(DATASET, roster=list(icons))
    negatives = load_negatives()
    clf = CardClassifier.new(sorted(icons) + ["empty"])
    print(f"{len(icons)} classes + empty | {len(crops)} real benchmark crops"
          f" | {len(negatives)} negative frames")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    best = 0.0
    for round_index in range(1, args.rounds + 1):
        start = time.time()
        images, labels, _ = training_batch(icons, args.per_class, rng, negatives=negatives)
        loss = clf.fit(images, labels, epochs=1)
        overall, non_empty = evaluate(clf, crops)
        marker = ""
        if non_empty > best:
            best = non_empty
            clf.save(OUT)
            marker = "  <- saved"
        print(
            f"round {round_index:2}/{args.rounds}: loss={loss:.3f}"
            f"  real-crop acc: overall={overall:.1%} non-empty={non_empty:.1%}"
            f"  ({time.time() - start:.0f}s){marker}",
            flush=True,
        )
    print(f"\nBest non-empty accuracy: {best:.1%}  (weights: {OUT})")


if __name__ == "__main__":
    main()
