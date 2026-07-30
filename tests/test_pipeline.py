import numpy as np
import pytest

from clash_copilot.capture.source import Frame
from clash_copilot.pipeline import OpponentTracker
from clash_copilot.detection.template import TemplateCardDetector

TILE = 40
DECK = [
    "Hog Rider", "Musketeer", "Cannon", "Ice Golem",
    "Ice Spirit", "Skeletons", "Fireball", "The Log",
]
COSTS = {
    "Hog Rider": 4, "Musketeer": 4, "Cannon": 3, "Ice Golem": 2,
    "Ice Spirit": 1, "Skeletons": 1, "Fireball": 4, "The Log": 2,
}


def make_templates(names, seed=7):
    rng = np.random.default_rng(seed)
    return {n: rng.integers(0, 255, (TILE, TILE, 3), dtype=np.uint8) for n in names}


def frame_with(tile, t):
    img = np.full((200, 300, 3), 40, dtype=np.uint8)
    if tile is not None:
        img[60 : 60 + TILE, 100 : 100 + TILE] = tile
    return Frame(image=img, t=t)


def play_frames(templates, plays, hold_frames=2, gap_frames=1, dt=0.14):
    """Render a scripted sequence of plays into frames."""
    frames, t = [], 0.0
    for card in plays:
        for _ in range(hold_frames):
            frames.append(frame_with(templates[card], round(t, 3)))
            t += dt
        for _ in range(gap_frames):
            frames.append(frame_with(None, round(t, 3)))
            t += dt
    return frames


@pytest.fixture
def tracker():
    templates = make_templates(DECK)
    detector = TemplateCardDetector(templates, confirm_frames=2)
    return OpponentTracker(detector, COSTS), templates


def test_state_emitted_only_on_play_events(tracker):
    trk, templates = tracker
    frames = play_frames(templates, ["Hog Rider"])
    states = [s for s in (trk.process_frame(f) for f in frames) if s is not None]
    assert len(states) == 1
    assert states[0].seen == {"Hog Rider"}


def test_elixir_accounts_for_regen_and_cost(tracker):
    trk, templates = tracker
    # confirmed on 2nd frame at t=0.28: regen 0.28/2.8 = 0.1, spend 4
    frames = [
        frame_with(templates["Hog Rider"], 0.0),
        frame_with(templates["Hog Rider"], 0.28),
    ]
    states = [s for s in (trk.process_frame(f) for f in frames) if s is not None]
    assert states[-1].elixir == pytest.approx(5.1 - 4.0)


def test_deck_unknown_until_all_eight_seen(tracker):
    trk, templates = tracker
    frames = play_frames(templates, DECK[:7])
    last = [s for s in (trk.process_frame(f) for f in frames) if s is not None][-1]
    assert not last.deck_known
    assert last.unknown_count == 1
    assert last.hand is None


def test_full_deck_reveals_exact_hand_and_next_card(tracker):
    trk, templates = tracker
    frames = play_frames(templates, DECK)  # all 8 played in order
    last = [s for s in (trk.process_frame(f) for f in frames) if s is not None][-1]
    assert last.deck_known
    assert last.unknown_count == 0
    assert last.hand == set(DECK[:4])  # last 4 played are out of hand
    assert last.next_card == DECK[4]
