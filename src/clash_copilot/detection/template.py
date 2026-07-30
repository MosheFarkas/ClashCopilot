"""v0 card-play detection: template matching with temporal debouncing.

Matches known card art against a region of interest and emits a PlayEvent
only after the same card is seen in `confirm_frames` consecutive frames
(prior-art lesson: single-frame matches misfire during reveal animations).
The event re-arms when the card disappears from the ROI or a different
card takes over.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from clash_copilot.capture.source import Frame


@dataclass(frozen=True)
class PlayEvent:
    card: str
    t: float  # seconds, time of the confirming frame
    score: float  # template-match confidence 0..1


def best_template_match(
    image: np.ndarray, templates: dict[str, np.ndarray]
) -> tuple[str | None, float]:
    """Best-scoring template anywhere in `image` (normalized correlation).

    Templates larger than the image are skipped; returns (None, 0.0) when
    nothing fits or the image is too uniform for normalized matching.
    """
    if image.std() < 1e-6:
        return None, 0.0
    best_card, best_score = None, 0.0
    for card, template in templates.items():
        if image.shape[0] < template.shape[0] or image.shape[1] < template.shape[1]:
            continue
        result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
        score = float(np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0).max())
        if score > best_score:
            best_card, best_score = card, score
    return best_card, best_score


class TemplateCardDetector:
    def __init__(
        self,
        templates: dict[str, np.ndarray],
        roi: tuple[int, int, int, int] | None = None,  # x, y, w, h
        threshold: float = 0.9,
        confirm_frames: int = 2,
    ):
        self.templates = templates
        self.roi = roi
        self.threshold = threshold
        self.confirm_frames = confirm_frames
        self._candidate: str | None = None
        self._streak = 0
        self._active: str | None = None

    def process(self, frame: Frame) -> PlayEvent | None:
        card, score = self._best_match(frame.image)
        if card is None:
            self._candidate = self._active = None
            self._streak = 0
            return None
        if card == self._active:
            return None
        if card == self._candidate:
            self._streak += 1
        else:
            self._candidate, self._streak = card, 1
        if self._streak >= self.confirm_frames:
            self._active, self._candidate, self._streak = card, None, 0
            return PlayEvent(card=card, t=frame.t, score=score)
        return None

    def _best_match(self, image: np.ndarray) -> tuple[str | None, float]:
        if self.roi is not None:
            x, y, w, h = self.roi
            image = image[y : y + h, x : x + w]
        best_card, best_score = best_template_match(image, self.templates)
        if best_score < self.threshold:
            return None, 0.0
        return best_card, best_score
