import cv2
import numpy as np

from clash_copilot.classify.augment import SIZE
from clash_copilot.classify.benchmark import load_labeled_crops


def write_jpg(path, seed=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(
        str(path),
        np.random.default_rng(seed).integers(0, 255, (73, 60, 3), dtype=np.uint8),
    )


def test_load_labeled_crops_maps_dirs_to_classes(tmp_path):
    write_jpg(tmp_path / "hog-rider" / "a.jpg")
    write_jpg(tmp_path / "ice-spirit-evolution" / "b.jpg")  # maps to base card
    write_jpg(tmp_path / "empty" / "c.jpg")
    write_jpg(tmp_path / "_augmentation" / "d.jpg")  # skipped

    crops = load_labeled_crops(tmp_path, roster=["Hog Rider", "Ice Spirit"])
    labels = sorted(expected for _, expected in crops)
    assert labels == ["Hog Rider", "Ice Spirit", "empty"]
    for crop, _ in crops:
        assert crop.shape == (SIZE[1], SIZE[0], 3)


def test_load_labeled_crops_skips_classes_outside_roster(tmp_path):
    write_jpg(tmp_path / "hog-rider" / "a.jpg")
    write_jpg(tmp_path / "not-a-card" / "b.jpg")
    crops = load_labeled_crops(tmp_path, roster=["Hog Rider"])
    assert [expected for _, expected in crops] == ["Hog Rider"]


def test_load_labeled_crops_excludes_reference_exemplars(tmp_path):
    # each KataCR class dir contains a clean reference copy named after the
    # dir (e.g. hog-rider/hog-rider.jpg == card_classification_origin) --
    # those are training exemplars, not captured crops, and must stay out
    write_jpg(tmp_path / "hog-rider" / "00030_2.jpg")
    write_jpg(tmp_path / "hog-rider" / "hog-rider.jpg")
    crops = load_labeled_crops(tmp_path, roster=["Hog Rider"])
    assert len(crops) == 1
