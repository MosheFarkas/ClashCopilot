"""Frame sources.

Everything downstream consumes an iterator of Frames, so swapping recorded
footage for live screen capture (e.g. mss) later only means adding another
source class here.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, NamedTuple, Protocol, Sequence

import cv2
import numpy as np


class Frame(NamedTuple):
    image: np.ndarray  # BGR, HxWx3
    t: float  # seconds since start of footage


class FrameSource(Protocol):
    def frames(self) -> Iterator[Frame]: ...


@dataclass
class ArraySource:
    """In-memory source for tests and synthetic demos."""

    items: Sequence[Frame]

    def frames(self) -> Iterator[Frame]:
        yield from self.items


class VideoFileSource:
    """Reads a recorded match (replay screen recording, exported clip, ...)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def frames(self) -> Iterator[Frame]:
        cap = cv2.VideoCapture(str(self.path))
        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            index = 0
            while True:
                ok, image = cap.read()
                if not ok:
                    break
                yield Frame(image=image, t=index / fps)
                index += 1
        finally:
            cap.release()
