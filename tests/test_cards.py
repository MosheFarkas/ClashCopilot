import cv2
import numpy as np

from clash_copilot.cards import load_card_icon


def write_rgba_icon(path):
    canvas = np.zeros((80, 100, 4), dtype=np.uint8)
    canvas[20:60, 30:90] = (10, 200, 30, 255)  # opaque green block, BGRA
    cv2.imwrite(str(path), canvas)


def test_load_card_icon_crops_to_opaque_region_and_drops_alpha(tmp_path):
    path = tmp_path / "icon.png"
    write_rgba_icon(path)
    icon = load_card_icon(path)
    assert icon.shape == (40, 60, 3)  # alpha bounding box, BGR only
    assert (icon[0, 0] == (10, 200, 30)).all()  # art, not black fill


def test_load_card_icon_passes_through_opaque_images(tmp_path):
    path = tmp_path / "plain.png"
    plain = np.full((50, 40, 3), 90, dtype=np.uint8)
    cv2.imwrite(str(path), plain)
    icon = load_card_icon(path)
    assert icon.shape == (50, 40, 3)
    assert (icon == plain).all()
