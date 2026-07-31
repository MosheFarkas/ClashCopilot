import pytest

from clash_copilot.detection.tracking import TrackCoaster, TrackSmoother


def test_coaster_passes_through_live_tracks():
    coaster = TrackCoaster(max_age=3)
    out = coaster.update({1: ("giant", "enemy", 0.9, (0, 0, 10, 10))})
    assert out[1][0] == "giant"


def test_coaster_holds_a_briefly_lost_track():
    coaster = TrackCoaster(max_age=3)
    coaster.update({1: ("giant", "enemy", 0.9, (0, 0, 10, 10))})
    held = coaster.update({})  # detector missed it this frame
    assert 1 in held
    assert held[1][3] == (0, 0, 10, 10)


def test_coaster_drops_track_after_max_age():
    coaster = TrackCoaster(max_age=2)
    coaster.update({1: ("giant", "enemy", 0.9, (0, 0, 10, 10))})
    coaster.update({})
    coaster.update({})
    assert coaster.update({}) == {}


def test_coaster_revives_age_when_track_returns():
    coaster = TrackCoaster(max_age=2)
    coaster.update({1: ("giant", "enemy", 0.9, (0, 0, 10, 10))})
    coaster.update({})
    coaster.update({1: ("giant", "enemy", 0.8, (1, 1, 11, 11))})
    coaster.update({})
    assert 1 in coaster.update({})  # age restarted, so still held


def test_coaster_decays_confidence_while_coasting():
    coaster = TrackCoaster(max_age=3, decay=0.5)
    coaster.update({1: ("giant", "enemy", 0.8, (0, 0, 10, 10))})
    first = coaster.update({})[1][2]
    second = coaster.update({})[1][2]
    assert first == pytest.approx(0.4)
    assert second == pytest.approx(0.2)


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
