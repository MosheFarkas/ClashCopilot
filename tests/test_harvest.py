import numpy as np

from clash_copilot.classify.harvest import (
    color_agreement,
    readable_segments,
    smooth_labels,
    vote_deck,
)


def solid(bgr):
    image = np.zeros((80, 64, 3), dtype=np.uint8)
    image[:] = bgr
    return image


def test_color_agreement_high_for_same_hue():
    red_a, red_b = solid((0, 0, 200)), solid((30, 30, 230))
    assert color_agreement(red_a, red_b) > 0.5


def test_color_agreement_low_for_different_hue():
    red, blue = solid((0, 0, 200)), solid((200, 30, 0))
    assert color_agreement(red, blue) < 0.2


def test_color_agreement_desaturated_crop_is_neutral():
    # greyed-out cards carry no hue evidence: return None (no verdict),
    # not a rejection
    grey, red = solid((128, 128, 128)), solid((0, 0, 200))
    assert color_agreement(grey, red) is None


def test_readable_segments_splits_on_unreadable_gaps():
    flags = [True] * 10 + [False] * 4 + [True] * 8
    assert readable_segments(flags, min_len=5) == [(0, 10), (14, 22)]


def test_readable_segments_drops_short_runs():
    flags = [True] * 3 + [False] * 3 + [True] * 6
    assert readable_segments(flags, min_len=5) == [(6, 12)]


def test_readable_segments_tolerates_brief_dropouts():
    # a 2-frame blip inside a long match must not split the segment
    flags = [True] * 10 + [False] * 2 + [True] * 10
    assert readable_segments(flags, min_len=5, max_gap=3) == [(0, 22)]


def test_vote_deck_returns_top_eight_by_weighted_votes():
    # slot label streams: 8 real cards dominate, noise appears rarely
    reads = []
    deck = [f"card{i}" for i in range(8)]
    for repeat in range(20):
        for card in deck:
            reads.append((card, 0.9))
    reads += [("Noise A", 0.95), ("Noise B", 0.6)]  # rare misreads
    assert vote_deck(reads) == set(deck)


def test_vote_deck_handles_fewer_than_eight_cards():
    reads = [("Knight", 0.9), ("Archers", 0.8)]
    assert vote_deck(reads) == {"Knight", "Archers"}


def test_smooth_labels_keeps_only_stable_runs():
    labels = ["A", "A", "A", "B", "A", "A", "A", None, "C", "C"]
    # min_run=3: the lone B and the C pair are unstable -> dropped
    assert smooth_labels(labels, min_run=3) == [
        "A", "A", "A", None, "A", "A", "A", None, None, None,
    ]


def test_smooth_labels_treats_none_as_run_break():
    labels = ["A", None, "A", "A"]
    assert smooth_labels(labels, min_run=2) == [None, None, "A", "A"]
