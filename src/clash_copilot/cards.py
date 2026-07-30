"""Card metadata access.

Ships with a bundled sample (the cards used by the sample meta decks).
Regenerate the full roster from the official API with scripts/fetch_cards.py.
"""

import json
from importlib import resources
from pathlib import Path

import cv2
import numpy as np


def _bundled(name: str) -> str:
    return resources.files("clash_copilot.data").joinpath(name).read_text()


def load_card_icon(
    path: str | Path, background: tuple[int, int, int] = (60, 60, 60)
) -> np.ndarray | None:
    """Load a card portrait as BGR, handling the official icons' alpha channel.

    Official icons are RGBA with ~30% fully transparent margin; naive BGR
    loading bakes a black silhouette into every image. This crops to the
    opaque bounding box and composites residual transparency over
    `background`.
    """
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        return None
    if raw.ndim == 3 and raw.shape[2] == 4:
        alpha = raw[:, :, 3]
        ys, xs = np.where(alpha > 8)
        if len(ys):
            raw = raw[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
        a = raw[:, :, 3:4].astype(np.float32) / 255.0
        bg = np.full_like(raw[:, :, :3], background)
        blended = raw[:, :, :3].astype(np.float32) * a + bg.astype(np.float32) * (1 - a)
        return blended.astype(np.uint8)
    return raw


def load_card_costs(path: str | Path | None = None) -> dict[str, int]:
    """Map card name -> elixir cost. Reads the bundled sample unless a path is given."""
    raw = Path(path).read_text() if path else _bundled("cards_sample.json")
    return {name: int(cost) for name, cost in json.loads(raw)["cards"].items()}
