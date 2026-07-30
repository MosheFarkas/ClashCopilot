"""Weak-labeling helpers for harvesting real slot crops from match footage.

The recorder's deck is identified by voting over a whole video's slot
reads (a deck's 8 cards dominate any per-frame noise), and per-slot label
streams are smoothed so only stable runs survive -- transient misreads
during animations get dropped rather than mislabeled.
"""

from collections import Counter

import cv2
import numpy as np

MIN_SATURATED_FRACTION = 0.15  # below this, a crop carries no usable hue evidence


def color_agreement(crop, reference) -> float | None:
    """Hue-histogram correlation in [0, 1]-ish, or None when the crop is
    too desaturated to judge (greyed-out cards).

    Grayscale template matching and hue comparison fail independently,
    which makes this a real second opinion on a weak label.
    """
    def hue_hist(image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = (hsv[:, :, 1] > 60) & (hsv[:, :, 2] > 40)
        if mask.mean() < MIN_SATURATED_FRACTION:
            return None
        hist = cv2.calcHist([hsv], [0], mask.astype(np.uint8) * 255, [18], [0, 180])
        return cv2.normalize(hist, hist).flatten()

    crop_hist = hue_hist(crop)
    if crop_hist is None:
        return None
    ref_hist = hue_hist(reference)
    if ref_hist is None:
        return None
    return float(max(0.0, cv2.compareHist(crop_hist, ref_hist, cv2.HISTCMP_CORREL)))


def vote_deck(reads: list[tuple[str, float]], deck_size: int = 8) -> set[str]:
    """Top `deck_size` cards by score-weighted vote across all slot reads."""
    votes: Counter = Counter()
    for card, score in reads:
        votes[card] += score
    return {card for card, _ in votes.most_common(deck_size)}


def readable_segments(
    flags: list[bool], min_len: int, max_gap: int = 3
) -> list[tuple[int, int]]:
    """(start, end) index ranges of readable stretches (matches).

    Unreadable runs shorter than `max_gap` (animation blips) do not split
    a segment; segments shorter than `min_len` (menu flashes) are dropped.
    """
    segments: list[tuple[int, int]] = []
    start = None
    gap = 0
    for i, flag in enumerate(flags):
        if flag:
            if start is None:
                start = i
            gap = 0
        elif start is not None:
            gap += 1
            if gap >= max_gap:
                end = i - gap + 1
                if end - start >= min_len:
                    segments.append((start, end))
                start, gap = None, 0
    if start is not None and len(flags) - start >= min_len:
        segments.append((start, len(flags)))
    return segments


def smooth_labels(labels: list[str | None], min_run: int = 3) -> list[str | None]:
    """Keep labels only inside runs of >= min_run identical values."""
    out: list[str | None] = [None] * len(labels)
    i = 0
    while i < len(labels):
        j = i
        while j < len(labels) and labels[j] == labels[i]:
            j += 1
        if labels[i] is not None and j - i >= min_run:
            out[i:j] = labels[i:j]
        i = j
    return out
