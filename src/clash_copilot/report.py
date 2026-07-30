"""Human-readable formatting of GameState snapshots."""

from clash_copilot.pipeline import GameState


def format_state(state: GameState) -> str:
    line = (
        f"[t={state.t:5.1f}s] {state.event.card:<11}"
        f" ({state.event.score:.2f} conf) | opp elixir ≈ {state.elixir:4.1f}"
        f" | revealed {len(state.seen)}/8"
    )
    if state.deck_known:
        line += f" | hand: {', '.join(sorted(state.hand))} | next: {state.next_card}"
    else:
        line += f" | unknown slots: {state.unknown_count}"
    if state.anomaly_count:
        line += f" | detection anomalies: {state.anomaly_count}"
    return line


def format_summary(state: GameState) -> str:
    lines = [
        "Final read on the opponent:",
        f"  deck ({'complete' if state.deck_known else 'partial'}): "
        f"{', '.join(sorted(state.seen))}",
    ]
    if state.deck_known:
        lines.append(f"  in hand right now: {', '.join(sorted(state.hand))}")
        lines.append(f"  next card they draw: {state.next_card}")
    lines.append(f"  elixir estimate: {state.elixir:.1f}")
    return "\n".join(lines)
