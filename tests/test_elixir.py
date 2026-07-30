import pytest

from clash_copilot.state.elixir import ElixirTracker


def test_starts_at_five():
    assert ElixirTracker().elixir == pytest.approx(5.0)


def test_regenerates_one_elixir_per_2_8_seconds():
    t = ElixirTracker()
    t.advance_to(2.8)
    assert t.elixir == pytest.approx(6.0)


def test_caps_at_ten_and_tracks_leak():
    t = ElixirTracker()
    t.advance_to(28.0)  # +10 regen on top of 5 -> capped
    assert t.elixir == pytest.approx(10.0)
    assert t.leaked == pytest.approx(5.0)


def test_spend_subtracts_cost():
    t = ElixirTracker()
    t.advance_to(2.8)
    t.spend(4)
    assert t.elixir == pytest.approx(2.0)


def test_spend_below_zero_clamps_and_counts_underflow():
    t = ElixirTracker()
    t.spend(9)
    assert t.elixir == pytest.approx(0.0)
    assert t.underflows == 1


def test_double_elixir_after_120_seconds():
    t = ElixirTracker()
    t.advance_to(120.0)
    t.spend(t.elixir)  # empty the bar at the boundary
    t.advance_to(122.8)
    assert t.elixir == pytest.approx(2.0)


def test_regen_integrates_across_rate_boundary():
    t = ElixirTracker()
    t.advance_to(117.2)
    t.spend(9)  # 10 (capped) - 9 = 1
    t.advance_to(121.4)  # 2.8s @ 1x = 1.0, then 1.4s @ 2x = 1.0
    assert t.elixir == pytest.approx(3.0)


def test_triple_elixir_after_240_seconds():
    t = ElixirTracker()
    t.advance_to(240.0)
    t.spend(t.elixir)
    t.advance_to(242.8)
    assert t.elixir == pytest.approx(3.0)


def test_advance_backwards_raises():
    t = ElixirTracker()
    t.advance_to(10.0)
    with pytest.raises(ValueError):
        t.advance_to(9.0)
