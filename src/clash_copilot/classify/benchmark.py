"""Load the labeled real-footage crops used as the card-identity benchmark.

Dataset layout (KataCR): one directory per class, kebab-case names;
"-evolution" suffixed classes map to their base card; "empty" is its own
class; underscore-prefixed dirs are auxiliary and skipped.
"""

import re
from pathlib import Path

import cv2
import numpy as np

from clash_copilot.classify.augment import SIZE


def norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def load_labeled_crops(
    dataset: str | Path, roster: list[str]
) -> list[tuple[np.ndarray, str]]:
    """(crop resized to SIZE, expected class) pairs.

    Expected class is a roster card name, or "empty". Class dirs that match
    nothing in the roster are skipped (mirrors the eval script's SKIP).
    """
    by_norm = {norm(name): name for name in roster}
    crops: list[tuple[np.ndarray, str]] = []
    for class_dir in sorted(Path(dataset).iterdir()):
        if not class_dir.is_dir() or class_dir.name.startswith("_"):
            continue
        if class_dir.name == "empty":
            expected = "empty"
        else:
            expected = by_norm.get(norm(class_dir.name.removesuffix("-evolution")))
            if expected is None:
                continue
        for jpg in sorted(class_dir.glob("*.jpg")):
            image = cv2.imread(str(jpg))
            if image is not None:
                crops.append((cv2.resize(image, SIZE), expected))
    return crops
