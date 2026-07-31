import numpy as np
import pytest

pytest.importorskip("onnxruntime")

from clash_copilot.detection.arena import ArenaDetector, ArenaDetection, load_unit_names


class StubSession:
    """Mimics the units ONNX session: fp16 NCHW in, [1, N, 6] out."""

    def __init__(self, rows):
        self.rows = rows
        self.received = None

    def run(self, _outputs, feeds):
        self.received = feeds["images"]
        return [np.array([self.rows], dtype=np.float32)]


class StubSide:
    def __init__(self, side="enemy"):
        self.side = side

    def run(self, _outputs, feeds):
        onehot = [1.0, 0.0] if self.side == "ally" else [0.0, 1.0]
        return [np.array([onehot], dtype=np.float32)]


def make_detector(rows, side="enemy"):
    detector = ArenaDetector.__new__(ArenaDetector)
    detector.session = StubSession(rows)
    detector.side_session = StubSide(side)
    detector.unit_names = ["archer", "balloon", "giant"]
    detector.min_conf = 0.3
    detector.units_input = "images"
    detector.side_input = "input"
    return detector


def test_load_unit_names_bundled():
    names = load_unit_names()
    assert len(names) == 97
    assert names[0] == "archer"


def test_detect_maps_boxes_back_to_frame_coords():
    # one detection filling the whole model canvas maps back to the arena band
    rows = [[0, 0, 352, 480, 0.9, 1]]
    detector = make_detector(rows)
    frame = np.zeros((1280, 590, 3), dtype=np.uint8)
    detections = detector.detect(frame)
    assert len(detections) == 1
    d = detections[0]
    assert d.name == "balloon"
    assert d.side == "enemy"
    assert d.conf == pytest.approx(0.9, abs=1e-3)
    x0, y0, x1, y1 = d.bbox
    assert (x0, x1) == (0, 590)
    # arena band: 5%..80% of frame height
    assert y0 == pytest.approx(0.05 * 1280, abs=25)
    assert y1 == pytest.approx(0.80 * 1280, abs=25)


def test_detect_filters_low_confidence():
    rows = [[0, 0, 352, 480, 0.1, 0]]
    detector = make_detector(rows)
    assert detector.detect(np.zeros((1280, 590, 3), dtype=np.uint8)) == []


def test_detect_feeds_fp16_nchw():
    detector = make_detector([[0, 0, 352, 480, 0.9, 0]])
    detector.detect(np.zeros((1280, 590, 3), dtype=np.uint8))
    fed = detector.session.received
    assert fed.shape == (1, 3, 480, 352)
    assert fed.dtype == np.float16


def test_side_classifier_controls_side_label():
    rows = [[100, 100, 200, 300, 0.8, 2]]
    assert make_detector(rows, side="ally").detect(
        np.zeros((1280, 590, 3), dtype=np.uint8)
    )[0].side == "ally"
