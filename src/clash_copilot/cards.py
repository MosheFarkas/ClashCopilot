"""Card metadata access.

Ships with a bundled sample (the cards used by the sample meta decks).
Regenerate the full roster from the official API with scripts/fetch_cards.py.
"""

import json
from importlib import resources
from pathlib import Path


def _bundled(name: str) -> str:
    return resources.files("clash_copilot.data").joinpath(name).read_text()


def load_card_costs(path: str | Path | None = None) -> dict[str, int]:
    """Map card name -> elixir cost. Reads the bundled sample unless a path is given."""
    raw = Path(path).read_text() if path else _bundled("cards_sample.json")
    return {name: int(cost) for name, cost in json.loads(raw)["cards"].items()}
