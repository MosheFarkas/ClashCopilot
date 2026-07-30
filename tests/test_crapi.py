from clash_copilot.crapi import cards_payload_to_costs

CARDS_PAYLOAD = {
    "items": [
        {"name": "Knight", "elixirCost": 3, "iconUrls": {"medium": "https://x/k.png"}},
        {"name": "Mirror", "iconUrls": {"medium": "https://x/m.png"}},  # no fixed cost
    ]
}


def test_cards_payload_to_costs_skips_cards_without_fixed_cost():
    assert cards_payload_to_costs(CARDS_PAYLOAD) == {"Knight": 3}
