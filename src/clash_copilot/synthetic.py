"""Synthetic match footage for demos and end-to-end testing.

Renders a scripted opponent's plays as frames: each play shows the card's
art tile in a fixed play zone for ~0.5s. Art tiles are low-frequency block
noise -- distinct per card, and coarse enough to survive video compression.
"""

from dataclasses import dataclass

import numpy as np

from clash_copilot.capture.source import Frame

FRAME_SHAPE = (200, 300, 3)
PLAY_ZONE_PX = (100, 60)  # x, y
TILE = 40
HOLD_SECONDS = 0.5
BACKGROUND = 40
MATCH_THRESHOLD = 0.8  # headroom for lossy video codecs


@dataclass(frozen=True)
class Scenario:
    deck: list[str]
    plays: list[tuple[float, str]]  # (seconds, card)
    duration: float


# A real archetype on a schedule the opponent could actually afford
# (checked against the 5-start / 2.8s-per-elixir regen model).
HOG_CYCLE = Scenario(
    deck=[
        "Ice Spirit", "Skeletons", "Hog Rider", "Cannon",
        "Musketeer", "Ice Golem", "Fireball", "The Log",
    ],
    plays=[
        (3.0, "Ice Spirit"),
        (6.0, "Skeletons"),
        (10.0, "Hog Rider"),
        (15.0, "Cannon"),
        (24.0, "Musketeer"),
        (30.0, "Ice Golem"),
        (42.0, "Fireball"),
        (48.0, "The Log"),
        (56.0, "Hog Rider"),  # cycle repeat: legal, 5 plays after its first use
    ],
    duration=60.0,
)


def make_card_art(cards: list[str]) -> dict[str, np.ndarray]:
    """Deterministic stand-in art per card (swap for real portraits later)."""
    art = {}
    block = TILE // 10
    for seed, card in enumerate(cards):
        coarse = np.random.default_rng(seed).integers(0, 255, (10, 10, 3), dtype=np.uint8)
        art[card] = np.kron(coarse, np.ones((block, block, 1), dtype=np.uint8))
    return art


def render_frames(
    scenario: Scenario, art: dict[str, np.ndarray], fps: int = 5
) -> list[Frame]:
    frames = []
    x, y = PLAY_ZONE_PX
    for i in range(int(scenario.duration * fps)):
        t = i / fps
        image = np.full(FRAME_SHAPE, BACKGROUND, dtype=np.uint8)
        for play_t, card in scenario.plays:
            if play_t <= t < play_t + HOLD_SECONDS:
                image[y : y + TILE, x : x + TILE] = art[card]
        frames.append(Frame(image=image, t=t))
    return frames


def layout_dict() -> dict:
    """Layout JSON (see geometry.Layout) matching the rendered geometry."""
    height, width = FRAME_SHAPE[:2]
    x, y = PLAY_ZONE_PX
    return {
        "play_zone": {"x": x / width, "y": y / height, "w": TILE / width, "h": TILE / height},
        "card_size": {"w": TILE / width, "h": TILE / height},
        "threshold": MATCH_THRESHOLD,
    }
