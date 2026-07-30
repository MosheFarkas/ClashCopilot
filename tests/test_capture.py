import cv2
import numpy as np
import pytest

from clash_copilot.capture.source import ArraySource, Frame, VideoFileSource


def test_array_source_yields_frames_in_order():
    frames = [
        Frame(image=np.zeros((4, 4, 3), dtype=np.uint8), t=0.0),
        Frame(image=np.ones((4, 4, 3), dtype=np.uint8), t=0.2),
    ]
    out = list(ArraySource(frames).frames())
    assert [f.t for f in out] == [0.0, 0.2]
    assert (out[1].image == 1).all()


def test_video_file_source_reads_frames_with_timestamps(tmp_path):
    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), 5.0, (64, 48)
    )
    for _ in range(10):
        writer.write(np.random.default_rng(0).integers(0, 255, (48, 64, 3), dtype=np.uint8))
    writer.release()

    frames = list(VideoFileSource(path).frames())
    assert len(frames) == 10
    assert frames[0].t == pytest.approx(0.0)
    assert frames[5].t == pytest.approx(1.0)  # frame 5 at 5 fps
    assert frames[0].image.shape == (48, 64, 3)
