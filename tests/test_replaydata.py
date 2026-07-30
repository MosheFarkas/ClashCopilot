import io
import lzma
import pickle

import numpy as np
import pytest

from clash_copilot.replaydata import load_episode, plays_from_episode


def write_episode(path, payload):
    buf = io.BytesIO()
    np.save(buf, np.array(payload, dtype=object), allow_pickle=True)
    path.write_bytes(lzma.compress(buf.getvalue()))


def test_load_episode_reads_numpy_object_payload(tmp_path):
    path = tmp_path / "ep.npy.xz"
    write_episode(path, {"state": [1, 2], "action": [3]})
    episode = load_episode(path)
    assert episode["state"] == [1, 2]


def test_load_episode_blocks_non_numpy_pickles(tmp_path):
    from pathlib import PurePosixPath  # any picklable non-numpy global

    buf = io.BytesIO()
    buf.write(b"\x93NUMPY\x01\x00")
    header = b"{'descr': '|O', 'fortran_order': False, 'shape': (), }\n"
    header += b" " * (64 - (len(header) + 10) % 64)
    buf.write(len(header).to_bytes(2, "little"))
    buf.write(header)
    buf.write(pickle.dumps(PurePosixPath("evil")))
    path = tmp_path / "evil.npy.xz"
    path.write_bytes(lzma.compress(buf.getvalue()))
    with pytest.raises(pickle.UnpicklingError):
        load_episode(path)


def test_plays_from_episode_pairs_actions_with_slot_contents():
    states = [
        {"time": 4, "cards": [9, 1, 2, 3, 4], "elixir": 7},
        {"time": 5, "cards": [9, 1, 2, 3, 4], "elixir": 8},
        {"time": 6, "cards": [9, 5, 2, 3, 4], "elixir": 5},
    ]
    actions = [
        {"card_id": 0, "xy": None},
        {"card_id": 1, "xy": np.array([8.0, 20.0])},  # played slot 1 (hand id 1)
        {"card_id": 0, "xy": None},
    ]
    plays = plays_from_episode(states, actions)
    assert plays == [(1, 1, 1)]  # (state index, slot, hand id at that moment)
