import json

import pytest

from clash_copilot.geometry import Layout, Region


def test_region_to_pixels_scales_to_frame():
    region = Region(x=0.25, y=0.1, w=0.5, h=0.2)
    # frame shape is (height, width, channels) as numpy gives it
    assert region.to_pixels((200, 400, 3)) == (100, 20, 200, 40)


def test_region_rejects_out_of_range_values():
    with pytest.raises(ValueError):
        Region(x=0.8, y=0.0, w=0.5, h=0.1)  # x + w > 1
    with pytest.raises(ValueError):
        Region(x=-0.1, y=0.0, w=0.5, h=0.1)


def test_layout_loads_from_json(tmp_path):
    path = tmp_path / "layout.json"
    path.write_text(
        json.dumps(
            {
                "play_zone": {"x": 0.3, "y": 0.2, "w": 0.4, "h": 0.5},
                "card_size": {"w": 0.1, "h": 0.15},
                "threshold": 0.8,
            }
        )
    )
    layout = Layout.from_json(path)
    assert layout.play_zone == Region(0.3, 0.2, 0.4, 0.5)
    assert layout.card_size == (0.1, 0.15)
    assert layout.threshold == 0.8


def test_layout_threshold_defaults_when_omitted(tmp_path):
    path = tmp_path / "layout.json"
    path.write_text(
        json.dumps(
            {
                "play_zone": {"x": 0.3, "y": 0.2, "w": 0.4, "h": 0.5},
                "card_size": {"w": 0.1, "h": 0.15},
            }
        )
    )
    assert Layout.from_json(path).threshold == 0.9


def test_layout_card_size_in_pixels():
    layout = Layout(
        play_zone=Region(0.3, 0.2, 0.4, 0.5), card_size=(0.1, 0.15), threshold=0.9
    )
    assert layout.card_size_pixels((200, 400, 3)) == (40, 30)  # (w_px, h_px)
