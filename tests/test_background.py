import numpy as np

from clash_copilot.classify.background import median_background


def test_median_removes_transient_objects():
    # static scene of value 100; a "unit" of value 255 appears in a
    # different spot in each frame -> the median must recover the scene
    frames = []
    for i in range(7):
        f = np.full((20, 20, 3), 100, dtype=np.uint8)
        f[i * 2 : i * 2 + 3, 5:8] = 255
        frames.append(f)
    bg = median_background(frames)
    assert bg.shape == (20, 20, 3)
    assert (bg == 100).all()


def test_median_keeps_static_structures():
    frames = []
    for i in range(5):
        f = np.full((10, 10, 3), 50, dtype=np.uint8)
        f[0:3, 0:3] = 200  # a tower: present in every frame
        f[7, i] = 255  # a moving unit
        frames.append(f)
    bg = median_background(frames)
    assert (bg[0:3, 0:3] == 200).all()
    assert bg[7].max() == 50


def test_median_requires_frames():
    try:
        median_background([])
    except ValueError:
        return
    raise AssertionError("expected ValueError on empty input")
