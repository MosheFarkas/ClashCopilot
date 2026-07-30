"""End-to-end wiring: frames -> play events -> game state.

Deterministic by design: card slots read "unknown" until the opponent has
revealed all 8 cards, after which the exact hand and next card fall out of
the cycle rule. No predictions -- only facts derived from observed plays.
"""

from dataclasses import dataclass

from clash_copilot.capture.source import Frame
from clash_copilot.detection.template import PlayEvent, TemplateCardDetector
from clash_copilot.state.cycle import CycleTracker
from clash_copilot.state.elixir import TIMER_START, ElixirTracker

DECK_SIZE = 8


@dataclass(frozen=True)
class GameState:
    t: float
    event: PlayEvent  # the play that produced this snapshot
    elixir: float  # opponent elixir estimate right after the play
    seen: set[str]
    unknown_count: int  # deck slots not yet revealed
    deck_known: bool
    hand: set[str] | None  # exact hand, only once deck_known
    next_card: str | None
    anomaly_count: int


class OpponentTracker:
    """Consumes frames; emits a GameState snapshot whenever a play is detected.

    Card costs must cover every detectable card (missing cost -> KeyError:
    fail loudly rather than silently corrupt the elixir estimate).
    """

    def __init__(
        self,
        detector: TemplateCardDetector,
        card_costs: dict[str, int],
        start_elixir: float = TIMER_START,
    ):
        self.detector = detector
        self.card_costs = card_costs
        self.elixir = ElixirTracker(start=start_elixir)
        self.cycle = CycleTracker()

    def process_frame(self, frame: Frame) -> GameState | None:
        event = self.detector.process(frame)
        if event is None:
            return None
        self.elixir.advance_to(event.t)
        self.elixir.spend(self.card_costs[event.card])
        self.cycle.record_play(event.card)
        return GameState(
            t=event.t,
            event=event,
            elixir=self.elixir.elixir,
            seen=self.cycle.seen,
            unknown_count=DECK_SIZE - len(self.cycle.seen),
            deck_known=self.cycle.deck_known,
            hand=self.cycle.hand,
            next_card=self.cycle.next_card,
            anomaly_count=len(self.cycle.anomalies),
        )
