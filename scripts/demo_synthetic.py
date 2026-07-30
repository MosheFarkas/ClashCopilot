"""End-to-end demo on synthetic footage (no game assets or network needed).

Renders a scripted opponent playing a Hog 2.6 cycle deck, then runs the
real pipeline over those frames: template detection -> cycle/elixir
tracking -> printed running state.

Run:  .venv/bin/python scripts/demo_synthetic.py
"""

from clash_copilot.capture.source import ArraySource
from clash_copilot.cards import load_card_costs
from clash_copilot.detection.template import TemplateCardDetector
from clash_copilot.pipeline import OpponentTracker
from clash_copilot.report import format_state, format_summary
from clash_copilot.synthetic import HOG_CYCLE, make_card_art, render_frames


def main() -> None:
    art = make_card_art(HOG_CYCLE.deck)
    source = ArraySource(render_frames(art=art, scenario=HOG_CYCLE))
    tracker = OpponentTracker(
        TemplateCardDetector(art, confirm_frames=2), load_card_costs()
    )

    print(
        f"Synthetic match: opponent plays {len(HOG_CYCLE.plays)} cards "
        f"over {HOG_CYCLE.duration:.0f}s\n"
    )
    last = None
    for frame in source.frames():
        state = tracker.process_frame(frame)
        if state is not None:
            print(format_state(state))
            last = state

    assert last is not None, "no plays detected -- demo is broken"
    print()
    print(format_summary(last))
    truth = set(HOG_CYCLE.deck)
    print(f"\nGround truth check: revealed set matches scripted deck: {last.seen == truth}")


if __name__ == "__main__":
    main()
