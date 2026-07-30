"""Opponent elixir simulation.

The opponent's elixir bar is never visible in a 1v1 match, so it must be
simulated: a regeneration clock minus the cost of observed plays. Rates:
1 elixir per 2.8s of match time, doubled after 120s, tripled after 240s
(the last overtime minute). Both players start with 5; the cap is 10.
"""

START = 5.0
CAP = 10.0
BASE_SECONDS_PER_ELIXIR = 2.8
_RATE_SEGMENTS = (  # (segment start, segment end, regen multiplier)
    (0.0, 120.0, 1.0),
    (120.0, 240.0, 2.0),
    (240.0, float("inf"), 3.0),
)


class ElixirTracker:
    """Tracks one player's estimated elixir over match time (seconds)."""

    def __init__(self) -> None:
        self.elixir = START
        self.leaked = 0.0  # regen lost to the 10-elixir cap
        self.underflows = 0  # spends that would have gone negative (tracking error signal)
        self._t = 0.0

    def advance_to(self, t: float) -> None:
        if t < self._t:
            raise ValueError(f"match clock went backwards: {t} < {self._t}")
        for start, end, mult in _RATE_SEGMENTS:
            lo, hi = max(self._t, start), min(t, end)
            if hi > lo:
                self._gain((hi - lo) * mult / BASE_SECONDS_PER_ELIXIR)
        self._t = t

    def spend(self, cost: float) -> None:
        self.elixir -= cost
        if self.elixir < 0:
            self.elixir = 0.0
            self.underflows += 1

    def _gain(self, amount: float) -> None:
        self.elixir += amount
        if self.elixir > CAP:
            self.leaked += self.elixir - CAP
            self.elixir = CAP
