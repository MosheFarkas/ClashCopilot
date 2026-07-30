"""Grayscale card templates for correlation matching.

Two sources: official API icons (data/icons, alpha-aware) or in-game
exemplars (KataCR card_classification_origin; the shared card frame is
center-cropped away because it dominates normalized correlation).
Evolution exemplars are extra views keyed "<Card>#evo" -- strip the
suffix before scoring.
"""

import re
from pathlib import Path

import cv2
import numpy as np

from clash_copilot.cards import load_card_icon
from clash_copilot.classify.augment import SIZE

TEMPLATE_SCALE = 0.8  # templates smaller than crops -> alignment tolerance


def norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _prep(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(cv2.resize(image, SIZE), cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (int(SIZE[0] * TEMPLATE_SCALE), int(SIZE[1] * TEMPLATE_SCALE)))


def load_templates(source: str, icons_dir: str | Path, origin_dir: str | Path) -> dict[str, np.ndarray]:
    templates: dict[str, np.ndarray] = {}
    if source == "icons":
        for path in sorted(Path(icons_dir).glob("*.png")):
            icon = load_card_icon(path)
            if icon is not None:
                templates[path.stem] = _prep(icon)
        return templates
    roster = {norm(p.stem): p.stem for p in sorted(Path(icons_dir).glob("*.png"))}
    for path in sorted(Path(origin_dir).glob("*.jpg")):
        base = path.stem.removesuffix("-evolution")
        card = roster.get(norm(base))
        if card is None:
            continue
        image = cv2.imread(str(path))
        if image is None:
            continue
        h, w = image.shape[:2]
        my, mx = int(h * 0.15), int(w * 0.15)
        image = image[my : h - my, mx : w - mx]
        key = card if path.stem == base else f"{card}#evo"
        templates[key] = _prep(image)
    return templates
