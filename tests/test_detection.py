import numpy as np

from clash_copilot.capture.source import Frame
from clash_copilot.detection.template import TemplateCardDetector

TILE = 40


def make_templates(names, seed=7):
    rng = np.random.default_rng(seed)
    return {
        name: rng.integers(0, 255, (TILE, TILE, 3), dtype=np.uint8) for name in names
    }


def frame_with(tile=None, t=0.0, pos=(60, 100)):
    img = np.full((200, 300, 3), 40, dtype=np.uint8)
    if tile is not None:
        y, x = pos
        img[y : y + TILE, x : x + TILE] = tile
    return Frame(image=img, t=t)


def feed(detector, sequence):
    """sequence: list of (tile or None, t); returns emitted events."""
    events = []
    for tile, t in sequence:
        event = detector.process(frame_with(tile, t))
        if event is not None:
            events.append(event)
    return events


def test_emits_single_event_after_confirm_frames():
    templates = make_templates(["Hog Rider"])
    det = TemplateCardDetector(templates, confirm_frames=2)
    tile = templates["Hog Rider"]
    events = feed(det, [(tile, 0.0), (tile, 0.2), (tile, 0.4)])
    assert len(events) == 1
    assert events[0].card == "Hog Rider"
    assert events[0].t == 0.2  # confirmed on the second frame


def test_blank_frames_produce_no_events():
    det = TemplateCardDetector(make_templates(["Hog Rider"]), confirm_frames=2)
    assert feed(det, [(None, 0.0), (None, 0.2)]) == []


def test_rearms_after_card_disappears():
    templates = make_templates(["Hog Rider"])
    det = TemplateCardDetector(templates, confirm_frames=2)
    tile = templates["Hog Rider"]
    events = feed(
        det,
        [(tile, 0.0), (tile, 0.2), (tile, 0.4), (None, 0.6), (tile, 0.8), (tile, 1.0)],
    )
    assert [e.t for e in events] == [0.2, 1.0]


def test_detects_card_switch_without_gap():
    templates = make_templates(["Hog Rider", "Fireball"])
    det = TemplateCardDetector(templates, confirm_frames=2)
    hog, fb = templates["Hog Rider"], templates["Fireball"]
    events = feed(det, [(hog, 0.0), (hog, 0.2), (fb, 0.4), (fb, 0.6)])
    assert [e.card for e in events] == ["Hog Rider", "Fireball"]


def test_ignores_cards_outside_roi():
    templates = make_templates(["Hog Rider"])
    det = TemplateCardDetector(templates, roi=(0, 0, 150, 200), confirm_frames=2)
    tile = templates["Hog Rider"]
    # pasted at x=200, outside the roi width of 150
    events = [
        e
        for e in (
            det.process(frame_with(tile, t, pos=(60, 200))) for t in (0.0, 0.2, 0.4)
        )
        if e is not None
    ]
    assert events == []
