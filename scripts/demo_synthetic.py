"""End-to-end demo on synthetic footage (no game assets or network needed).

Renders a scripted opponent playing a Hog 2.6 cycle deck as video frames
(each play shows the card's art tile in a fixed "play zone" for ~0.5s),
then runs the real pipeline over those frames: template detection ->
cycle/elixir tracking -> printed running state.

Run:  .venv/bin/python scripts/demo_synthetic.py
"""

import numpy as np

from clash_copilot.capture.source import ArraySource, Frame
from clash_copilot.cards import load_card_costs
from clash_copilot.detection.template import TemplateCardDetector
from clash_copilot.pipeline import OpponentTracker

FPS = 5
TILE = 40
PLAY_ZONE = (100, 60)  # x, y where a played card's art appears
FRAME_SHAPE = (200, 300, 3)

# A real archetype, played on a schedule the opponent could actually afford
# (checked against the 5-start / 2.8s-per-elixir regen model).
OPPONENT_DECK = [
    "Ice Spirit", "Skeletons", "Hog Rider", "Cannon",
    "Musketeer", "Ice Golem", "Fireball", "The Log",
]
PLAY_SCRIPT = [
    (3.0, "Ice Spirit"),
    (6.0, "Skeletons"),
    (10.0, "Hog Rider"),
    (15.0, "Cannon"),
    (24.0, "Musketeer"),
    (30.0, "Ice Golem"),
    (42.0, "Fireball"),
    (48.0, "The Log"),
    (56.0, "Hog Rider"),  # cycle repeat: legal, 5 plays after its first use
]
DURATION = 60.0
HOLD_SECONDS = 0.5


def make_card_art(deck: list[str]) -> dict[str, np.ndarray]:
    """Deterministic stand-in art per card (swap for real portraits later)."""
    return {
        card: np.random.default_rng(seed).integers(0, 255, (TILE, TILE, 3), dtype=np.uint8)
        for seed, card in enumerate(deck)
    }


def render_frames(art: dict[str, np.ndarray]) -> list[Frame]:
    frames = []
    for i in range(int(DURATION * FPS)):
        t = i / FPS
        image = np.full(FRAME_SHAPE, 40, dtype=np.uint8)
        for play_t, card in PLAY_SCRIPT:
            if play_t <= t < play_t + HOLD_SECONDS:
                x, y = PLAY_ZONE
                image[y : y + TILE, x : x + TILE] = art[card]
        frames.append(Frame(image=image, t=t))
    return frames


def describe(state) -> str:
    line = (
        f"[t={state.t:5.1f}s] {state.event.card:<11}"
        f" ({state.event.score:.2f} conf) | opp elixir ≈ {state.elixir:4.1f}"
        f" | revealed {len(state.seen)}/8"
    )
    if state.deck_known:
        line += f" | hand: {', '.join(sorted(state.hand))} | next: {state.next_card}"
    else:
        line += f" | unknown slots: {state.unknown_count}"
    if state.anomaly_count:
        line += f" | detection anomalies: {state.anomaly_count}"
    return line


def main() -> None:
    art = make_card_art(OPPONENT_DECK)
    source = ArraySource(render_frames(art))
    tracker = OpponentTracker(
        TemplateCardDetector(art, confirm_frames=2), load_card_costs()
    )

    print(f"Synthetic match: opponent plays {len(PLAY_SCRIPT)} cards over {DURATION:.0f}s\n")
    last = None
    for frame in source.frames():
        state = tracker.process_frame(frame)
        if state is not None:
            print(describe(state))
            last = state

    assert last is not None, "no plays detected -- demo is broken"
    print("\nFinal read on the opponent:")
    print(f"  deck ({'complete' if last.deck_known else 'partial'}): "
          f"{', '.join(sorted(last.seen))}")
    if last.deck_known:
        print(f"  in hand right now: {', '.join(sorted(last.hand))}")
        print(f"  next card they draw: {last.next_card}")
    print(f"  elixir estimate: {last.elixir:.1f}")
    truth = set(OPPONENT_DECK)
    print(f"\nGround truth check: revealed set matches scripted deck: {last.seen == truth}")


if __name__ == "__main__":
    main()
