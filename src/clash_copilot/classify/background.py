"""Recover clean arena backgrounds from match footage.

KataCR's synthetic generator pastes unit sprites onto empty arena
backgrounds. Its background set is from 2024 arenas, which is a large part
of why its detector struggles on current seasonal skins. A per-pixel median
over frames sampled across a match removes everything that moves (units,
spells, projectiles) and keeps what does not (arena art, towers) -- giving
us 2026 backgrounds to composite onto, with no labeling.
"""

import numpy as np


def median_background(frames: list[np.ndarray]) -> np.ndarray:
    """Per-pixel median of frames; transient objects vanish."""
    if not frames:
        raise ValueError("need at least one frame")
    return np.median(np.stack(frames), axis=0).astype(np.uint8)
