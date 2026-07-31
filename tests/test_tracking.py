import pytest

from clash_copilot.detection.tracking import TrackSmoother


def test_majority_name_survives_single_frame_misread():
    smoother = TrackSmoother(window=5)
    for _ in range(3):
        name, side, conf = smoother.update(1, "giant", "enemy", 0.8)
    name, side, conf = smoother.update(1, "royal_giant", "enemy", 0.9)  # misread
    assert name == "giant"


def test_side_votes_independently_of_name():
    smoother = TrackSmoother(window=5)
    smoother.update(2, "archer", "ally", 0.7)
    smoother.update(2, "archer", "enemy", 0.7)
    _, side, _ = smoother.update(2, "archer", "ally", 0.7)
    assert side == "ally"


def test_confidence_is_windowed_mean():
    smoother = TrackSmoother(window=3)
    smoother.update(3, "bat", "enemy", 0.4)
    smoother.update(3, "bat", "enemy", 0.6)
    _, _, conf = smoother.update(3, "bat", "enemy", 0.8)
    assert conf == pytest.approx((0.4 + 0.6 + 0.8) / 3)


def test_window_forgets_old_votes():
    smoother = TrackSmoother(window=2)
    smoother.update(4, "knight", "enemy", 0.9)
    smoother.update(4, "prince", "enemy", 0.9)
    name, _, _ = smoother.update(4, "prince", "enemy", 0.9)
    assert name == "prince"


def test_tracks_are_independent():
    smoother = TrackSmoother(window=3)
    smoother.update(1, "giant", "enemy", 0.9)
    name, _, _ = smoother.update(2, "bat", "ally", 0.5)
    assert name == "bat"
