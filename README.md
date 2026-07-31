# ClashCopilot

A computer-vision study project that watches recorded Clash Royale footage and tracks the opponent's state in real time: which cards they've played, their exact hand and next draw once the deck is fully revealed, and a simulated estimate of their elixir.

**Scope and ethics.** This is a learning/portfolio project in the spirit of building a chess engine: legitimate to build and study, not to be pointed at rated games. Using a real-time assist in live ladder play violates Supercell's Terms of Service (the "unauthorized third-party software" clause). Accordingly, the project is designed around **recorded replay footage and private/casual test matches**, not live-ladder deployment, and makes no attempt at stealth. Game assets fall under the [Supercell Fan Content Policy](https://supercell.com/en/fan-content-policy/) (non-commercial, with disclaimer): *This material is unofficial and is not endorsed by Supercell.*

## What works today (v0 slice)

An end-to-end pipeline on **synthetic footage**: a scripted opponent's plays are rendered as frames, and the real pipeline — template detection → play events → cycle + elixir tracking — produces a running state readout:

```
[t=  3.2s] Ice Spirit  (1.00 conf) | opp elixir ≈  6.1 | revealed 1/8 | unknown slots: 7
[t= 48.2s] The Log     (1.00 conf) | opp elixir ≈  2.2 | revealed 8/8 | hand: Cannon, Hog Rider, Ice Spirit, Skeletons | next: Musketeer
```

Deck slots read **unknown until each card is revealed** — no guessing. Once all 8 are seen, the hand and next draw are *computed, not predicted* (see "cycle rule" below).

```bash
uv venv -p 3.12 && uv pip install -e ".[dev]"
.venv/bin/python -m pytest              # 43 tests
.venv/bin/python scripts/demo_synthetic.py
```

**Arena detection overlay** (`scripts/annotate_video.py`): renders a match recording with boxes around troops/spells — unit name, ally/enemy side, confidence — using the MIT-licensed pretrained models from [ClashRoyaleBuildABot](https://github.com/Pbatch/ClashRoyaleBuildABot) (97-class YOLO ONNX + ally/enemy side classifier), ~2 fps processing on CPU:

```bash
.venv/bin/python scripts/annotate_video.py match.mp4 --start 60 --duration 10
```

Boxes are stabilized across frames: ByteTrack association (`supervision`, Apache-2.0) plus a per-track majority vote over name/side and windowed-mean confidence — measured on real footage, this cuts frame-to-frame label/side flicker from 11.9% to 5.4% of matched box pairs (`--no-track` reverts to raw per-frame output). Caveats: the detector is 2024-era (post-2024 units like Berserker are missing from its 97 classes), and match-end celebration effects produce spurious boxes.

The same pipeline also runs on **video files** via the CLI, with a per-setup layout config (all regions normalized to frame size — no hard-coded pixels). Generate a synthetic clip to try it:

```bash
.venv/bin/python scripts/make_synthetic_video.py
.venv/bin/python -m clash_copilot data/synthetic/clip.avi \
    --layout data/synthetic/layout.json --templates data/synthetic/templates
```

## Research findings

Four questions were researched before any code was written (July 2026).

### 1. Detection: template matching where you can, learned detection only where you must

- The strongest prior art is **[KataCR](https://github.com/wty-yy/katacr)** ([paper](https://arxiv.org/abs/2504.04783), [dataset](https://github.com/wty-yy/Clash-Royale-Detection-Dataset)): a 150-class Clash Royale detector trained **entirely on synthetic composites** (segmented unit sprites pasted onto empty arena backgrounds with augmentation), validated on ~7k hand-labeled real frames — YOLOv8-x reached **AP50 85.3** on real footage. Synthetic-first data generation is proven for exactly this game.
- For **fixed UI regions** (card portraits, elixir bar, timer), template matching / tiny classifiers need zero training data and are what successful bots use ([py-clash-bot](https://github.com/pyclashbot/py-clash-bot) is pure template matching). Learned detection is only needed for **animated arena troops**.
- Detector landscape (mid-2026): Ultralytics **YOLO26** is the easiest to use but AGPL-3.0 (fine for an open portfolio repo, poisons closed reuse); **RF-DETR** (Apache-2.0, DINOv2 backbone) is the strongest fine-tuning detector with clean licensing, with known rough edges on Apple-Silicon inference; **DEIMv2** (Apache-2.0) is the lightweight fallback.

### 2. Prior art: what worked and what didn't

| Project | Approach | Lesson |
|---|---|---|
| [AmarSaini/Clash-Royale-AI-Card-Tracker](https://github.com/AmarSaini/Clash-Royale-AI-Card-Tracker) | Two LeNet CNNs on the **spectator view** deck strip | Card *identity* needs only official card art + augmentation; card *timing* needs a separate class-agnostic trigger; N-consecutive-frame debouncing is mandatory |
| [JustinAngara/…-Enemy-Elixir-Tracker](https://github.com/JustinAngara/Clash-Royale-Enemy-Elixir-Tracker) | Pixel sampling + manual keys | "Enemy elixir detection" in 1v1 is a misnomer — **the opponent's bar is never on screen**; every credible tracker simulates it |
| [HummingNerd/Clash_Royale_Enemy_Card_Tracker](https://github.com/HummingNerd/Clash_Royale_Enemy_Card_Tracker) (Dec 2025 YOLO demo) | Spawn-dust template match as *event* trigger, YOLO/ResNet for *identity* | Separating "something was played" from "what was played" survives simultaneous plays; auto-labeled data collection via two synced clients |
| [NezarAli/ElixirCounter](https://github.com/NezarAli/ElixirCounter) | Manual-input overlay | Clean cycle bookkeeping model (queues, Mirror special case) |

Cross-cutting: hard-coded pixel coordinates killed most of these projects (normalize ROIs to window geometry), and none report quantitative accuracy — replay/spectator view reveals the opponent's true deck strip, which is free ground truth for evaluation.

### 3. Game mechanics (the deterministic core)

- **Cycle rule**: deck = 8 cards, hand = 4. A played card goes to the back of the 8-card queue and cannot return until 4 other cards are played. Therefore **the last 4 plays are exactly the out-of-hand queue**: once all 8 cards have been seen, the current hand is `seen − last 4 plays` and the next draw is the 4th-most-recent play. This is bookkeeping, not ML.
- **Elixir**: both players start at 5, cap 10; regen is 1 per **2.8s**, doubled after 120s of match time, tripled after 240s (last overtime minute). Opponent elixir = regen clock − cost of observed plays.
- Edge cases deferred: Mirror (variable cost, never in the starting hand), Elixir Collector, Elixir Golem death payouts, champion ability costs. Note the **October 2025 champion rework** removed champions' special cycle behavior — they now cycle like normal cards, which simplifies tracking.

### 4. Official API: offline value only

- There is **no play-by-play data anywhere public** — battle logs are end-of-match summaries. CV on footage is the only way to get play sequences.
- The API can't identify a live opponent mid-match (no "current battle" endpoint), so it plays **zero live role**.
- It *is* useful offline: `/cards` provides the full roster with elixir costs and downloadable icon URLs (`scripts/fetch_cards.py`); battle logs contain both players' full 8-card decks, so crawling top-ladder players could build deck-frequency datasets if ever needed.

### Researched but deliberately cut: probabilistic deck inference

Early-game deck *prediction* (inferring unrevealed cards from meta priors) was researched — the Hearthstone literature ([Bursztein, DEF CON 22 / IEEE CIG 2016](https://elie.net/blog/hearthstone/predicting-hearthstone-opponent-deck-using-machine-learning)) shows simple co-occurrence statistics over harvested decks beat sequence models, and the official API could supply the deck corpus. **This project intentionally does not predict**: unrevealed slots display as unknown, and the hand is derived only once the full deck has been observed. The findings are recorded here in case that scope ever reopens.

## Architecture decisions

1. **Recorded footage first, live capture later.** Reproducible, ToS-friendly, and replays show ground truth. Everything downstream consumes a `FrameSource` iterator, so live capture (`mss`) is a drop-in addition, not a redesign.
2. **Layered pipeline with strict boundaries**: `capture` (frames) → `detection` (play events) → `state` (deterministic simulation) → readout. Vision only *feeds events* into the simulation; all game knowledge lives in `state`, which is fully unit-tested with no CV dependency.
3. **v0 detection = template matching with temporal debouncing** (no training data, no GPU). v1 = class-agnostic play trigger + crop classifier; v2 = synthetic-composite-trained detector (KataCR recipe, RF-DETR for licensing) only if arena-level tracking is ever needed.
4. **Deterministic state only.** Elixir is a simulation (start 5, 2.8/1.4/0.93s regen, cap 10, minus observed costs) with `leaked`/`underflow` counters as drift telemetry. Hand/next-card facts come from the cycle rule alone.
5. **Stack**: Python 3.12, OpenCV + NumPy, pytest; stdlib-only API script. No ML framework until a learned detector is actually needed.

## Repo layout

```
src/clash_copilot/
  capture/source.py      FrameSource protocol; video file + in-memory sources
  detection/template.py  Template matcher + debounce -> PlayEvent
  state/elixir.py        Opponent elixir simulation
  state/cycle.py         Cycle bookkeeping: seen / hand / next card / anomalies
  pipeline.py            OpponentTracker: frames -> GameState snapshots
  geometry.py            Normalized ROI regions + per-setup layout JSON
  report.py              Terminal formatting of GameState
  synthetic.py           Scripted-match frame rendering for demos/tests
  classify/              Learned card identity: augmentation, CNN, benchmark
                         (torch via the optional [ml] extra)
  __main__.py            CLI: python -m clash_copilot VIDEO --layout ... --templates ...
  cards.py, crapi.py     Card metadata (bundled sample; official API helpers)
scripts/
  demo_synthetic.py      End-to-end demo on rendered synthetic footage
  make_synthetic_video.py  Synthetic clip + templates + layout for the CLI
  fetch_cards.py         Full roster + icons from the official API
  eval_cards.py          Benchmark: card identity vs real footage crops
  train_classifier.py    Train the CNN on augmented portraits + real crops ([ml] extra)
  harvest_crops.py       Weak-label real slot crops from match recordings
  validate_elixir.py     Score the elixir simulation vs replay ground truth
  annotate_video.py      Arena overlay: unit boxes + name/side/confidence
tests/                   43 tests; state, geometry, and detection logic covered
```

## Current limitations

- **The demo is synthetic.** Real footage needs: ROI calibration for a chosen recording setup, real card art as templates, and an event trigger that doesn't assume card portraits appear in a fixed zone (real options: spectator-view deck-strip diffing, or spawn-dust detection).
- **Card identity on real footage is measured, not solved.** Benchmark (`scripts/eval_cards.py`, 158 real hand-slot crops from the MIT-licensed [KataCR dataset](https://github.com/wty-yy/Clash-Royale-Dataset), reference exemplars excluded). Open 122-card roster: template matching from API icons **63.3%** (after fixing an icon alpha-channel bug worth +11 points), from in-game exemplars with the shared card frame cropped off **66.5%** (`--templates origin`; uncropped, the frame dominates correlation and scores 24%), tiny CNN trained on augmented art + exemplars **51.3%**. **Deployment mode** (`--restrict`: candidates limited to the deck's 8 cards, as the tracker knows once cards are revealed): **82.3%** per frame with origin templates, before multi-frame debouncing votes across a card's screen time. The CNN masters synthetic data but doesn't transfer; real labeled training data is the fix, and available public data cannot cleanly provide it (see roadmap).
- One play at a time: overlapping/simultaneous plays within the debounce window would be missed.
- Elixir edge cases unmodeled (Mirror, Elixir Collector, champion abilities); the estimate drifts if any play is missed — `underflows` and `anomalies` surface that.
- **Elixir simulation validated against real matches** (`scripts/validate_elixir.py`, KataCR replay ground truth): given play costs decoded from the true bar, regen + accounting tracks the real bar at **MAE ≈ 1.1–1.2 elixir in double-elixir phases of both matches** (~1.5 single-phase on the clean episode; the other episode carries a persistent ~2.5 offset traceable to plays missing from the dataset's own action labels — the spend audit closes the books otherwise — not to the model). Found along the way: **regen starts during the pre-match countdown**, so a tracker that starts at 5 when the timer starts reads ~1–2 low; prior manual trackers initializing at 6 were compensating for exactly this. The tracker now starts at 6 at timer-zero (`TIMER_START`, overridable via `OpponentTracker(start_elixir=...)`).
- Timestamps come from frame index / fps, not the in-game timer (OCR on the match clock would be more robust to dropped frames).

## Roadmap

1. **Scale the real-crop harvest.** The weak-label pipeline is built and validated end-to-end (`scripts/harvest_crops.py`: per-match deck voting from slot reads, deck-restricted matching, temporal smoothing, hue verification against current icons, and a visual QA pass that caught three systematically mislabeled classes — including 2025-era cards absent from the 2024 exemplar set being force-fit to old identities). Two YouTube videos yielded **3,402 QA'd crops across 20 classes**, and retraining with them moved the benchmark 51.3% → **55.1%**, with per-class gains concentrated exactly where real data exists (Cannon 58→79%, Fireball 43→79%). The path to ≥95% is more footage covering more classes — harvesting is ~30 min/video of compute with QA montages as the gate. (KataCR hand-bar sequences were rejected as a source: same episode as the benchmark crops — test-set leakage.)
2. **Better event trigger**: spawn-dust/elixir-droplet detection as the class-agnostic "a card was played" signal, with card identity classified from the surrounding crop.
3. **Match-clock OCR** to replace frame-index time and handle double/triple elixir transitions exactly.
4. **Evaluation harness**: replay footage + hand-logged play sequences → precision/recall for detection, mean absolute error for elixir.
5. **Live capture source** (`mss`) + a minimal overlay/TUI readout — for private/casual test matches only.
6. (If ever needed) synthetic-composite arena detector per the KataCR recipe.
