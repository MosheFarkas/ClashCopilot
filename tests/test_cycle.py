from clash_copilot.state.cycle import CycleTracker

C = [f"card{i}" for i in range(1, 9)]


def play_all_eight(tracker):
    for card in C:
        tracker.record_play(card)


def test_seen_accumulates_distinct_cards():
    t = CycleTracker()
    t.record_play("Hog Rider")
    t.record_play("Ice Spirit")
    assert t.seen == {"Hog Rider", "Ice Spirit"}


def test_deck_not_known_until_eight_distinct_cards_seen():
    t = CycleTracker()
    for card in C[:7]:
        t.record_play(card)
    assert not t.deck_known
    t.record_play(C[7])
    assert t.deck_known


def test_hand_is_exact_once_deck_known():
    t = CycleTracker()
    play_all_eight(t)
    # queue = last 4 plays (card5..card8); hand = the other 4
    assert t.hand == {"card1", "card2", "card3", "card4"}


def test_next_card_is_fourth_most_recent_play():
    t = CycleTracker()
    play_all_eight(t)
    assert t.next_card == "card5"


def test_next_card_unknown_before_four_plays():
    t = CycleTracker()
    for card in C[:3]:
        t.record_play(card)
    assert t.next_card is None


def test_known_in_hand_is_partial_before_deck_known():
    t = CycleTracker()
    for card in C[:5]:
        t.record_play(card)
    # card1 was played 5 plays ago -> it is certainly back in hand
    assert t.known_in_hand == {"card1"}
    assert t.hand is None  # exact hand unknowable yet


def test_replay_within_four_plays_is_flagged_as_anomaly():
    t = CycleTracker()
    t.record_play("card1")
    t.record_play("card2")
    t.record_play("card1")  # impossible under the 4-card cycle rule
    assert len(t.anomalies) == 1
