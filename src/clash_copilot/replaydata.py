"""Reading KataCR replay episodes (wty-yy/Clash-Royale-Replay-Dataset, MIT).

Episodes are xz-compressed numpy object arrays holding one dict:
  state:  per-step {time, unit_infos, cards, elixir} -- the RECORDER's view;
          `cards` are deck-local integer ids [next, slot1..slot4], `elixir`
          is the ground-truth bar reading, `time` is elapsed seconds (OCR).
  action: per-step {xy, card_id} -- card_id is the HAND SLOT (1-4) played,
          xy is None when no card was played that step.
  reward: float array.

Object arrays require unpickling; pickles from the internet can execute
code, so loading goes through a restricted unpickler that only permits
numpy reconstruction.
"""

import io
import lzma
import pickle
from pathlib import Path

_ALLOWED = {
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy", "ndarray"),
    ("numpy", "dtype"),
}


class _NumpyOnlyUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if (module, name) in _ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"blocked pickle global: {module}.{name}")


def load_episode(path: str | Path) -> dict:
    raw = lzma.decompress(Path(path).read_bytes())
    buf = io.BytesIO(raw)
    magic = buf.read(6)
    if magic != b"\x93NUMPY":
        raise ValueError(f"not an .npy payload: {path}")
    major, _minor = buf.read(2)
    header_len = int.from_bytes(buf.read(2 if major == 1 else 4), "little")
    buf.read(header_len)
    return _NumpyOnlyUnpickler(buf).load().item()


def plays_from_episode(states: list[dict], actions: list[dict]) -> list[tuple[int, int, int]]:
    """(state index, hand slot 1-4, deck-local card id) for each real play."""
    plays = []
    for index, action in enumerate(actions):
        if action["xy"] is None:
            continue
        slot = action["card_id"]
        plays.append((index, slot, states[index]["cards"][slot]))
    return plays
