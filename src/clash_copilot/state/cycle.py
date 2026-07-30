"""Card-cycle bookkeeping.

Deck = 8 cards, hand = 4. A played card goes to the back of the 8-card
queue, so it cannot reappear until 4 other cards have been played. The
last 4 plays are therefore exactly the out-of-hand queue: once all 8 deck
cards have been seen, the hand is (seen - last 4 plays) and the next card
to be drawn is the 4th-most-recent play.
"""


class CycleTracker:
    """Tracks the opponent's play sequence and derives hand/cycle facts."""

    def __init__(self) -> None:
        self.plays: list[str] = []
        # (play index, card) plays that violate the 4-card cycle rule --
        # impossible in-game, so they signal detection errors upstream.
        self.anomalies: list[tuple[int, str]] = []

    def record_play(self, card: str) -> None:
        if card in self.plays[-4:]:
            self.anomalies.append((len(self.plays), card))
        self.plays.append(card)

    @property
    def seen(self) -> set[str]:
        return set(self.plays)

    @property
    def deck_known(self) -> bool:
        return len(self.seen) == 8

    @property
    def known_in_hand(self) -> set[str]:
        """Cards that are certainly in hand: seen and not among the last 4 plays."""
        return self.seen - set(self.plays[-4:])

    @property
    def hand(self) -> set[str] | None:
        """The exact 4-card hand, or None until the full deck has been seen."""
        return self.known_in_hand if self.deck_known else None

    @property
    def next_card(self) -> str | None:
        """The next card the opponent will draw, if enough plays have been seen."""
        return self.plays[-4] if len(self.plays) >= 4 else None
