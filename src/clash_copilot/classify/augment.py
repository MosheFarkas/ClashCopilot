"""Synthesize realistic hand-slot crops from official card portraits.

Every transform here reproduces a variance mode observed in real footage
(see scripts/eval_cards.py): grey-out when unaffordable, gold/grey border
states, the elixir-cost badge, the "next" slot countdown text rendered over
the art, plus capture noise (blur, compression, brightness).

All randomness flows through the passed numpy Generator, so samples are
reproducible per seed. No torch here -- pure numpy/cv2, independently
testable.
"""

import cv2
import numpy as np

SIZE = (64, 80)  # (w, h) of a synthesized crop; matches the eval harness

_GOLD = (0, 200, 255)
_GREY = (150, 150, 150)
_ELIXIR_PINK = (196, 64, 226)


def grey_out(image: np.ndarray, strength: float) -> np.ndarray:
    """Desaturate and darken, like an unaffordable card in the hand."""
    grey3 = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    blended = image.astype(np.float32) * (1 - strength) + grey3.astype(np.float32) * strength
    return np.clip(blended * (1 - 0.25 * strength), 0, 255).astype(np.uint8)


def make_empty_slot(rng: np.random.Generator) -> np.ndarray:
    """An unoccupied slot: flat bluish-grey fill, border, mild noise."""
    w, h = SIZE
    base = np.array(rng.integers(90, 150, 3), dtype=np.uint8)
    image = np.full((h, w, 3), base, dtype=np.uint8)
    noise = rng.normal(0, rng.uniform(2, 8), (h, w, 3))
    image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    cv2.rectangle(image, (0, 0), (w - 1, h - 1), _GREY, int(rng.integers(2, 5)))
    return image


def random_negative(patches: list[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    """A random SIZE crop from real non-card imagery (arena, UI panels).

    Real "empty" benchmark crops are arbitrary background/UI content, not
    tidy grey slots -- domain negatives cover that far better than synthesis.
    """
    w, h = SIZE
    patch = patches[int(rng.integers(0, len(patches)))]
    if patch.shape[0] < h or patch.shape[1] < w:
        patch = cv2.resize(patch, (max(w, patch.shape[1]), max(h, patch.shape[0])))
    x = int(rng.integers(0, patch.shape[1] - w + 1))
    y = int(rng.integers(0, patch.shape[0] - h + 1))
    image = patch[y : y + h, x : x + w].copy()
    gain = float(rng.uniform(0.8, 1.2))
    return np.clip(image.astype(np.float32) * gain, 0, 255).astype(np.uint8)


def augment_portrait(icon: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One synthetic hand-slot crop (SIZE, BGR uint8) from a card portrait."""
    w, h = SIZE
    scale = rng.uniform(0.8, 1.25)
    if scale < 1.0:  # card smaller than crop: hand-bar background shows around it
        background = np.array(
            [rng.integers(150, 220), rng.integers(70, 130), rng.integers(20, 70)],
            dtype=np.uint8,
        )  # BGR blues like the real bar
        image = np.full((h, w, 3), background, dtype=np.uint8)
        cw, ch = int(w * scale), int(h * scale)
        x = int(rng.integers(0, w - cw + 1))
        y = int(rng.integers(0, h - ch + 1))
        image[y : y + ch, x : x + cw] = cv2.resize(icon, (cw, ch))
    else:
        big = cv2.resize(icon, (int(w * scale), int(h * scale)))
        x = int(rng.integers(0, big.shape[1] - w + 1))
        y = int(rng.integers(0, big.shape[0] - h + 1))
        image = big[y : y + h, x : x + w].copy()

    border = rng.random()
    if border < 0.4:
        color = _GOLD if border < 0.2 else _GREY
        cv2.rectangle(image, (0, 0), (w - 1, h - 1), color, int(rng.integers(2, 5)))

    if rng.random() < 0.4:
        image = grey_out(image, float(rng.uniform(0.6, 1.0)))

    if rng.random() < 0.15:  # dark purple frame state (evolution/charged rendering)
        tint = np.zeros_like(image)
        tint[:] = (90, 20, 60)
        image = cv2.addWeighted(image, 0.65, tint, 0.35, 0)
        cv2.rectangle(image, (0, 0), (w - 1, h - 1), (160, 60, 120), int(rng.integers(2, 5)))

    if rng.random() < 0.8:  # elixir-cost badge, bottom-left
        radius = int(rng.integers(8, 13))
        center = (radius + 2, h - radius - 2)
        cv2.circle(image, center, radius, _ELIXIR_PINK, -1)
        cv2.putText(
            image, str(rng.integers(1, 10)), (center[0] - 5, center[1] + 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2,
        )

    if rng.random() < 0.3:  # "next" slot countdown text over the art
        overlay = image.copy()
        cv2.putText(
            overlay, f"{rng.integers(1, 6)}sec", (int(rng.integers(-10, 6)), int(rng.integers(h // 2, h))),
            cv2.FONT_HERSHEY_DUPLEX, rng.uniform(0.8, 1.2), (255, 255, 255), 3,
        )
        alpha = float(rng.uniform(0.7, 1.0))
        image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    gain = float(rng.uniform(0.7, 1.3))
    bias = float(rng.uniform(-25, 25))
    image = np.clip(image.astype(np.float32) * gain + bias, 0, 255).astype(np.uint8)

    if rng.random() < 0.5:  # hue/gamma drift from screen recording + video codecs
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.int16)
        hsv[:, :, 0] = (hsv[:, :, 0] + rng.integers(-8, 9)) % 180
        image = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
        gamma = float(rng.uniform(0.7, 1.4))
        table = np.clip(((np.arange(256) / 255.0) ** gamma) * 255, 0, 255).astype(np.uint8)
        image = table[image]

    if rng.random() < 0.7:  # real crops are ~60px sources: crush the resolution
        factor = float(rng.uniform(0.4, 0.9))
        small = cv2.resize(image, (max(8, int(w * factor)), max(8, int(h * factor))))
        image = cv2.resize(small, (w, h))

    if rng.random() < 0.3:
        image = cv2.GaussianBlur(image, (3, 3), 0)

    if rng.random() < 0.5:  # capture/compression artifacts
        quality = int(rng.integers(40, 90))
        _, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    return image


def training_batch(
    icons: dict[str, np.ndarray | list[np.ndarray]],
    per_class: int,
    rng: np.random.Generator,
    negatives: list[np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """(images, integer labels, class names). Classes: sorted cards + 'empty' last.

    Each card may have one source image or a list of views (e.g. official
    portrait + in-game exemplar + evolution variant); a view is drawn at
    random per sample. When `negatives` (real non-card imagery) is given,
    half the empty-class samples are random crops of it instead of
    synthetic slots.
    """
    names = sorted(icons) + ["empty"]
    images, labels = [], []
    for label, name in enumerate(names):
        views = icons[name] if name != "empty" else None
        if isinstance(views, np.ndarray):
            views = [views]
        for _ in range(per_class):
            if views is not None:
                source = views[int(rng.integers(0, len(views)))]
                images.append(augment_portrait(source, rng))
            elif negatives and rng.random() < 0.5:
                images.append(random_negative(negatives, rng))
            else:
                images.append(make_empty_slot(rng))
            labels.append(label)
    return np.stack(images), np.array(labels, dtype=np.int64), names
