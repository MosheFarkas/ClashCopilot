"""Screen-layout configuration.

All regions are normalized to frame size (fractions of width/height), so a
layout calibrated once works at any recording resolution -- hard-coded pixel
coordinates are what killed most prior-art projects.
"""

import json
from dataclasses import dataclass
from pathlib import Path

DEFAULT_THRESHOLD = 0.9


@dataclass(frozen=True)
class Region:
    """A rectangle in normalized [0, 1] frame coordinates."""

    x: float
    y: float
    w: float
    h: float

    def __post_init__(self) -> None:
        if not (0 <= self.x and 0 <= self.y and self.x + self.w <= 1 and self.y + self.h <= 1):
            raise ValueError(f"region out of [0, 1] bounds: {self}")

    def to_pixels(self, frame_shape: tuple[int, ...]) -> tuple[int, int, int, int]:
        """(x, y, w, h) in pixels for a numpy-shaped (height, width, ...) frame."""
        height, width = frame_shape[:2]
        return (
            round(self.x * width),
            round(self.y * height),
            round(self.w * width),
            round(self.h * height),
        )


@dataclass(frozen=True)
class Layout:
    play_zone: Region  # where played-card art appears
    card_size: tuple[float, float]  # card art (w, h), fractions of frame size
    threshold: float  # template-match confidence cutoff

    @classmethod
    def from_json(cls, path: str | Path) -> "Layout":
        raw = json.loads(Path(path).read_text())
        return cls(
            play_zone=Region(**raw["play_zone"]),
            card_size=(raw["card_size"]["w"], raw["card_size"]["h"]),
            threshold=raw.get("threshold", DEFAULT_THRESHOLD),
        )

    def card_size_pixels(self, frame_shape: tuple[int, ...]) -> tuple[int, int]:
        """Card art (w, h) in pixels for a numpy-shaped frame."""
        height, width = frame_shape[:2]
        return round(self.card_size[0] * width), round(self.card_size[1] * height)
