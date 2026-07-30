"""Official Clash Royale API helpers (developer.clashroyale.com).

Offline use only: the API cannot identify a live opponent mid-match. Here
it supplies card metadata (elixir costs, icon URLs) via scripts/fetch_cards.py.
Auth: free token, sent as a Bearer header; keys are IP-allowlisted.
"""

import json
import urllib.request

BASE_URL = "https://api.clashroyale.com/v1"


def fetch_json(path: str, token: str) -> dict:
    request = urllib.request.Request(
        f"{BASE_URL}{path}", headers={"Authorization": f"Bearer {token}"}
    )
    with urllib.request.urlopen(request) as response:
        return json.load(response)


def cards_payload_to_costs(payload: dict) -> dict[str, int]:
    """Map card name -> elixir cost from a /cards response.

    Cards without a fixed cost (e.g. Mirror) are skipped.
    """
    return {
        card["name"]: int(card["elixirCost"])
        for card in payload["items"]
        if "elixirCost" in card
    }
