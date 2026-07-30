"""Validate the ElixirTracker's regen + accounting model on real matches.

Each KataCR episode carries the RECORDER's true elixir bar per step plus
their plays. Scope: this validates the simulation core -- given each
play's true cost, does start-value + regen schedule - spends track the
real bar over a full match? (In deployment, card recognition supplies the
costs; its accuracy is measured separately by eval_cards.py.)

Method notes, from inspecting the data:
- the bar reading at the play step itself is unreadable or mid-animation
  garbage, so each play's cost = (last stable reading before) - (first
  stable reading 2+ steps after) + regen over that gap, rounded;
- scoring excludes a small blackout window around each play for the same
  reason;
- the simulation is seeded from the first ground-truth reading (the
  pre-match countdown regen offset is not observable from the episode).

Run:  .venv/bin/python scripts/validate_elixir.py [episode.npy.xz ...]
"""

import sys

from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from clash_copilot.replaydata import load_episode, plays_from_episode  # noqa: E402
from clash_copilot.state.elixir import BASE_SECONDS_PER_ELIXIR, ElixirTracker  # noqa: E402

DEFAULT_EPISODES = ["data/footage/oyassu_2023_ep3.npy.xz", "data/footage/wty_2024_golem_ep4.npy.xz"]

# Known recorder decks (per the source dataset's folder docs). Decoded play
# costs snap to the nearest sum achievable with these costs -- ground
# knowledge that removes bar-OCR decode noise. None = deck unknown.
KNOWN_DECK_COSTS = {
    "oyassu_2023_ep3.npy.xz": [1, 1, 2, 2, 3, 4, 4, 4],  # Hog 2.6 cycle
    "wty_2024_golem_ep4.npy.xz": None,
}


def interpolated_times(states: list[dict]) -> list[float]:
    """Spread steps sharing the same OCR second evenly across that second.

    OCR misses (None) are forward-filled from the last good reading.
    """
    times, last, offset = [], 0.0, 0.0
    for state in states:
        if state["time"] is not None:
            raw = float(state["time"]) + offset
            if raw < last - 30:  # overtime timer reset: unwrap to keep time monotonic
                offset += last - raw + 1.0
                raw = float(state["time"]) + offset
            last = raw
        times.append(last)
    smooth, i = [], 0
    while i < len(times):
        j = i
        while j < len(times) and times[j] == times[i]:
            j += 1
        for k in range(i, j):
            smooth.append(times[i] + (k - i) / (j - i))
        i = j
    return smooth


def regen(t0: float, t1: float) -> float:
    total, segments = 0.0, ((0.0, 120.0, 1.0), (120.0, 240.0, 2.0), (240.0, 1e9, 3.0))
    for start, end, mult in segments:
        lo, hi = max(t0, start), min(t1, end)
        if hi > lo:
            total += (hi - lo) * mult / BASE_SECONDS_PER_ELIXIR
    return total


def snap_to_deck(cost: float, cluster_size: int, deck_costs: list[int] | None) -> float:
    """Nearest sum of `cluster_size` cards from the known deck's cost set."""
    if deck_costs is None:
        return float(round(cost))
    from itertools import combinations_with_replacement

    sums = {sum(combo) for combo in combinations_with_replacement(set(deck_costs), cluster_size)}
    return min(sums, key=lambda s: abs(s - cost))


def per_play_costs(states, actions, times, deck_costs=None) -> dict[int, float]:
    """Play state-index -> integer cost, from stable bar readings around it.

    Plays landing within a few steps of each other share one bar drop, so
    they are clustered and the combined cost is charged at the cluster's
    first play (equivalent accounting for the simulation).
    """
    indices = [index for index, _, _ in plays_from_episode(states, actions)]
    clusters: list[list[int]] = []
    for index in indices:
        if clusters and index - clusters[-1][-1] < 7:
            clusters[-1].append(index)
        else:
            clusters.append([index])

    def stable_reading(indices) -> tuple[int, float] | None:
        """(middle index, median) of the first 3 readable steps -- medians
        reject the transient garbage the bar shows mid-animation."""
        readings = [(j, states[j]["elixir"]) for j in indices if states[j]["elixir"] is not None][:3]
        if not readings:
            return None
        readings.sort(key=lambda pair: pair[1])
        return readings[len(readings) // 2]

    costs = {}
    for cluster in clusters:
        first, last = cluster[0], cluster[-1]
        before = stable_reading(range(first - 1, max(first - 8, -1), -1))
        after = stable_reading(range(last + 2, min(last + 12, len(states))))
        if before is None or after is None:
            continue
        (jb, eb), (ja, ea) = before, after
        raw = eb - ea + regen(times[jb], times[ja])
        cost = snap_to_deck(raw, len(cluster), deck_costs)
        if 1 <= cost <= 10 * len(cluster):
            for index in cluster:  # even split: totals match, timing stays local
                costs[index] = cost / len(cluster)
    return costs


def validate(path: str) -> None:
    episode = load_episode(path)
    states, actions = episode["state"], episode["action"]
    times = interpolated_times(states)
    plays = plays_from_episode(states, actions)
    deck_costs = KNOWN_DECK_COSTS.get(Path(path).name)
    costs = per_play_costs(states, actions, times, deck_costs)
    print(f"\n=== {path} ===")
    print(f"{len(states)} steps over {times[0]:.0f}-{times[-1]:.0f}s"
          f" | {len(plays)} plays, {len(costs)} with resolvable cost")
    print(f"per-play costs: {sorted(round(c, 1) for c in costs.values())}")

    blackout = {j for index, _, _ in plays for j in range(index - 3, index + 4)}
    first_reading = next(s["elixir"] for s in states if s["elixir"] is not None)
    tracker = ElixirTracker()
    tracker.advance_to(times[0])
    tracker.elixir = float(first_reading)  # seed from first good reading
    errors = []
    skipped = 0
    for i, state in enumerate(states):
        tracker.advance_to(times[i])
        if i in costs:
            tracker.spend(costs[i])
        if state["elixir"] is None or i in blackout:
            skipped += 1
            continue
        errors.append((times[i], tracker.elixir - float(state["elixir"])))
    print(f"steps scored: {len(errors)} (skipped {skipped}: unreadable or play blackout)")

    abs_errors = [abs(e) for _, e in errors]
    print(f"MAE={sum(abs_errors)/len(abs_errors):.2f} elixir | max={max(abs_errors):.2f}"
          f" | within ±1: {sum(e <= 1 for e in abs_errors)/len(abs_errors):.1%}")
    for label, lo, hi in (("single elixir (<120s)", 0, 120), ("double+ (>=120s)", 120, 1e9)):
        phase = [abs(e) for t, e in errors if lo <= t < hi]
        if phase:
            print(f"  {label:22} MAE={sum(phase)/len(phase):.2f}  n={len(phase)}")
    if tracker.underflows:
        print(f"  underflows (spend below 0): {tracker.underflows}")


def main() -> None:
    episodes = sys.argv[1:] or DEFAULT_EPISODES
    for path in episodes:
        if Path(path).exists():
            validate(path)
        else:
            print(f"skip missing {path}")


if __name__ == "__main__":
    main()
