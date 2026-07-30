from clash_copilot.detection.template import PlayEvent
from clash_copilot.pipeline import GameState
from clash_copilot.report import format_state, format_summary


def make_state(**overrides):
    defaults = dict(
        t=10.0,
        event=PlayEvent(card="Hog Rider", t=10.0, score=0.97),
        elixir=2.6,
        seen={"Hog Rider", "Cannon", "The Log"},
        unknown_count=5,
        deck_known=False,
        hand=None,
        next_card=None,
        anomaly_count=0,
    )
    defaults.update(overrides)
    return GameState(**defaults)


def test_format_state_before_deck_known_shows_unknown_slots():
    line = format_state(make_state())
    assert "Hog Rider" in line
    assert "revealed 3/8" in line
    assert "unknown slots: 5" in line
    assert "hand:" not in line


def test_format_state_after_deck_known_shows_hand_and_next():
    line = format_state(
        make_state(
            seen={f"c{i}" for i in range(8)},
            unknown_count=0,
            deck_known=True,
            hand={"c0", "c1", "c2", "c3"},
            next_card="c4",
        )
    )
    assert "hand: c0, c1, c2, c3" in line
    assert "next: c4" in line


def test_format_state_reports_anomalies_when_present():
    assert "anomalies: 2" in format_state(make_state(anomaly_count=2))


def test_format_summary_partial_deck():
    text = format_summary(make_state())
    assert "deck (partial)" in text
    assert "elixir estimate: 2.6" in text
    assert "in hand" not in text


def test_format_summary_complete_deck():
    text = format_summary(
        make_state(
            seen={f"c{i}" for i in range(8)},
            unknown_count=0,
            deck_known=True,
            hand={"c0", "c1", "c2", "c3"},
            next_card="c4",
        )
    )
    assert "deck (complete)" in text
    assert "in hand right now: c0, c1, c2, c3" in text
    assert "next card they draw: c4" in text
