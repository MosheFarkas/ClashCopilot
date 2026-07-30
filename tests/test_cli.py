import cv2
import numpy as np

from clash_copilot.__main__ import build_tracker, load_templates
from clash_copilot.geometry import Layout, Region


def test_load_templates_reads_pngs_keyed_by_stem(tmp_path):
    knight = np.full((40, 40, 3), 90, dtype=np.uint8)
    cv2.imwrite(str(tmp_path / "Knight.png"), knight)
    cv2.imwrite(str(tmp_path / "Hog Rider.png"), np.zeros((40, 40, 3), dtype=np.uint8))
    templates = load_templates(tmp_path)
    assert set(templates) == {"Knight", "Hog Rider"}
    assert (templates["Knight"] == knight).all()


def test_build_tracker_configures_detector_from_layout():
    layout = Layout(
        play_zone=Region(0.25, 0.1, 0.5, 0.5), card_size=(0.1, 0.2), threshold=0.85
    )
    templates = {"Knight": np.zeros((80, 80, 3), dtype=np.uint8)}
    tracker = build_tracker(
        layout, templates, frame_shape=(200, 400, 3), card_costs={"Knight": 3}
    )
    detector = tracker.detector
    assert detector.roi == (100, 20, 200, 100)
    assert detector.threshold == 0.85
    # card_size pixels: w = 0.1 * 400 = 40, h = 0.2 * 200 = 40
    assert detector.templates["Knight"].shape == (40, 40, 3)
