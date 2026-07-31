"""Fetch sprites for post-2024 units and write them in KataCR's format.

KataCR's sprite set (and therefore its detector's vocabulary) is frozen at
~May 2024, which is a measured source of missed detections: an audit of
2026 footage found Suspicious Bush (released 2024-07) present in half the
sampled frames and never detected, because the model has no such class.

Sprites come from a decoded game-asset dump that stores per-animation RGBA
frames (mirsella/clash-royale, sprites/sc/decoded/<workspace>/exports/...).
Individual files are fetched over HTTP -- no 3.8 GB clone needed.

Output: data/footage/katacr-dataset/images/segment/<class>/<class>_<bel>_<n>.png,
alpha-cropped RGBA, matching the format the generator consumes.

Assets belong to Supercell; under their Fan Content Policy this is
non-commercial research use. Sprites are NOT redistributed -- this script
refetches them, and the output directory is gitignored.

Run:  .venv/bin/python scripts/fetch_unit_sprites.py --unit suspicious-bush
      .venv/bin/python scripts/fetch_unit_sprites.py --list
"""

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np

API = "https://api.github.com/repos/mirsella/clash-royale/contents/sprites/sc/decoded"
RAW = "https://raw.githubusercontent.com/mirsella/clash-royale/main/sprites/sc/decoded"
SEGMENT = Path("data/footage/katacr-dataset/images/segment")

# katacr class name -> (asset workspace, substrings identifying in-arena poses)
UNITS = {
    "suspicious-bush": ("chr_goblin_bush", ("idle", "walk", "run")),
    "berserker": ("chr_berserker", ("idle", "walk", "run", "attack")),
    "boss-bandit": ("chr_boss_bandit", ("idle", "walk", "run", "attack")),
    "goblin-machine": ("chr_goblin_machine", ("idle", "walk", "run", "attack")),
    "goblin-demolisher": ("chr_goblin_demolisher", ("idle", "walk", "run")),
    "rune-giant": ("chr_giant_buffing", ("idle", "walk", "run")),
    "royal-chef": ("chr_royal_chef", ("idle", "walk", "run")),
}
MIN_SIDE = 12  # ignore particle/fragment exports
MAX_PER_UNIT = 40


def get_json(url: str):
    with urllib.request.urlopen(url) as response:
        return json.load(response)


def alpha_crop(png_bytes: bytes) -> np.ndarray | None:
    image = cv2.imdecode(np.frombuffer(png_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim != 3 or image.shape[2] != 4:
        return None
    ys, xs = np.where(image[:, :, 3] > 8)
    if len(ys) == 0:
        return None
    cropped = image[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    if min(cropped.shape[:2]) < MIN_SIDE:
        return None
    return cropped


def fetch_unit(name: str, workspace: str, keep: tuple[str, ...]) -> int:
    exports = [e["name"] for e in get_json(f"{API}/{workspace}/exports")]
    wanted = [e for e in exports if any(k in e.lower() for k in keep)]
    if not wanted:
        wanted = exports
    print(f"{name}: {len(exports)} exports, {len(wanted)} match {keep}")

    out_dir = SEGMENT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for export in wanted:
        if saved >= MAX_PER_UNIT:
            break
        url = f"{RAW}/{workspace}/exports/{export}/frame_0000.png"
        try:
            with urllib.request.urlopen(url) as response:
                sprite = alpha_crop(response.read())
        except Exception:
            continue
        if sprite is None:
            continue
        # KataCR keys sprites by belligerence; unit art is side-independent
        # (side is conveyed by the HP bar the generator attaches), so the
        # same sprite serves both sides.
        for bel in (0, 1):
            cv2.imwrite(str(out_dir / f"{name}_{bel}_{saved:07d}.png"), sprite)
        saved += 1
    print(f"  saved {saved} sprites -> {out_dir}")
    return saved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit", action="append", help="katacr class name (repeatable)")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    if args.list or not args.unit:
        print("available:", ", ".join(sorted(UNITS)))
        return
    for name in args.unit:
        if name not in UNITS:
            sys.exit(f"unknown unit {name}; use --list")
        workspace, keep = UNITS[name]
        fetch_unit(name, workspace, keep)


if __name__ == "__main__":
    main()
