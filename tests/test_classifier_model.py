import numpy as np
import pytest

torch = pytest.importorskip("torch")

from clash_copilot.classify.augment import SIZE, training_batch  # noqa: E402
from clash_copilot.classify.model import CardClassifier  # noqa: E402


def fake_icons(n=3):
    # coarse block patterns: low-frequency structure like real card art
    # (pure pixel noise averages out under pooling and is unlearnable)
    return {
        f"card{i}": np.kron(
            np.random.default_rng(i).integers(0, 255, (12, 10, 3), dtype=np.uint8),
            np.ones((10, 10, 1), dtype=np.uint8),
        )
        for i in range(n)
    }


def test_predict_returns_name_and_probability():
    clf = CardClassifier.new(["a", "b", "c"])
    images = np.random.default_rng(1).integers(
        0, 255, (2, SIZE[1], SIZE[0], 3), dtype=np.uint8
    )
    out = clf.predict(images)
    assert len(out) == 2
    for name, prob in out:
        assert name in {"a", "b", "c"}
        assert 0.0 < prob <= 1.0


def test_fit_overfits_a_tiny_batch():
    icons = fake_icons()
    images, labels, names = training_batch(icons, per_class=16, rng=np.random.default_rng(2))
    clf = CardClassifier.new(names)
    clf.fit(images, labels, epochs=8, lr=1e-3)
    predicted = [name for name, _ in clf.predict(images)]
    accuracy = np.mean([p == names[l] for p, l in zip(predicted, labels)])
    assert accuracy > 0.9


def test_save_load_roundtrip(tmp_path):
    clf = CardClassifier.new(["a", "b"])
    images = np.random.default_rng(3).integers(
        0, 255, (2, SIZE[1], SIZE[0], 3), dtype=np.uint8
    )
    before = clf.predict(images)
    path = tmp_path / "model.pt"
    clf.save(path)
    loaded = CardClassifier.load(path)
    assert loaded.class_names == ["a", "b"]
    assert loaded.predict(images) == before
