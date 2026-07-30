import numpy as np
import pytest

from clash_copilot.classify.augment import (
    SIZE,
    augment_portrait,
    grey_out,
    make_empty_slot,
    training_batch,
)


def fake_icon(seed=0):
    return np.random.default_rng(seed).integers(0, 255, (120, 100, 3), dtype=np.uint8)


def test_augment_portrait_shape_and_dtype():
    out = augment_portrait(fake_icon(), np.random.default_rng(1))
    assert out.shape == (SIZE[1], SIZE[0], 3)
    assert out.dtype == np.uint8


def test_augment_portrait_is_deterministic_per_seed():
    a = augment_portrait(fake_icon(), np.random.default_rng(7))
    b = augment_portrait(fake_icon(), np.random.default_rng(7))
    assert (a == b).all()


def test_augment_portrait_varies_across_draws():
    rng = np.random.default_rng(7)
    a = augment_portrait(fake_icon(), rng)
    b = augment_portrait(fake_icon(), rng)
    assert not (a == b).all()


def test_grey_out_reduces_saturation():
    import cv2

    icon = fake_icon()
    greyed = grey_out(icon, strength=1.0)
    sat = lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2HSV)[:, :, 1].mean()
    assert sat(greyed) < sat(icon) * 0.2
    assert greyed.shape == icon.shape


def test_make_empty_slot_shape():
    out = make_empty_slot(np.random.default_rng(3))
    assert out.shape == (SIZE[1], SIZE[0], 3)
    assert out.dtype == np.uint8
    assert out.std() > 0  # not a flat fill; empty slots still have borders/noise


def test_training_batch_layout():
    icons = {"Knight": fake_icon(1), "Archers": fake_icon(2)}
    images, labels, names = training_batch(icons, per_class=4, rng=np.random.default_rng(0))
    assert names == ["Archers", "Knight", "empty"]  # sorted cards + empty last
    assert images.shape == (12, SIZE[1], SIZE[0], 3)
    assert labels.shape == (12,)
    assert set(labels.tolist()) == {0, 1, 2}
    assert (np.bincount(labels) == 4).all()


def test_training_batch_uses_real_negatives_for_empty_class():
    icons = {"Knight": fake_icon(1)}
    black_patch = np.zeros((200, 300, 3), dtype=np.uint8)
    images, labels, names = training_batch(
        icons, per_class=8, rng=np.random.default_rng(0), negatives=[black_patch]
    )
    empty_label = names.index("empty")
    empty_images = images[labels == empty_label]
    # with a black negative source, some empty samples must be mostly dark
    assert any(image.mean() < 30 for image in empty_images)
