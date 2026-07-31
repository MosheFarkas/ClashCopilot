"""Cross-frame stabilization of arena detections.

ByteTrack (supervision, Apache-2.0) associates boxes across frames into
tracks; TrackSmoother then votes name/side per track over a sliding
window so a single-frame misread cannot flip a label, and reports a
windowed mean confidence instead of a jittery per-frame score.
"""

from collections import Counter, defaultdict, deque
from dataclasses import dataclass

import numpy as np

from clash_copilot.detection.arena import ArenaDetection, ArenaDetector


class TrackSmoother:
    def __init__(self, window: int = 9):
        self.window = window
        self._history: dict[int, deque] = defaultdict(lambda: deque(maxlen=self.window))

    def update(self, track_id: int, name: str, side: str, conf: float) -> tuple[str, str, float]:
        history = self._history[track_id]
        history.append((name, side, conf))
        names = Counter(entry[0] for entry in history)
        sides = Counter(entry[1] for entry in history)
        mean_conf = sum(entry[2] for entry in history) / len(history)
        return names.most_common(1)[0][0], sides.most_common(1)[0][0], mean_conf


@dataclass(frozen=True)
class TrackedDetection:
    track_id: int
    name: str
    side: str
    conf: float
    bbox: tuple[int, int, int, int]


class TrackedArenaDetector:
    """ArenaDetector + ByteTrack association + per-track label smoothing."""

    def __init__(self, detector: ArenaDetector, fps: float = 30.0, window: int = 9):
        from supervision.tracker.byte_tracker.core import ByteTrack

        self.detector = detector
        self.tracker = ByteTrack(frame_rate=int(round(fps)))
        self.smoother = TrackSmoother(window=window)
        self._names = detector.unit_names

    def detect(self, frame: np.ndarray) -> list[TrackedDetection]:
        import supervision as sv

        raw = self.detector.detect(frame)
        if not raw:
            self.tracker.update_with_detections(sv.Detections.empty())
            return []
        detections = sv.Detections(
            xyxy=np.array([d.bbox for d in raw], dtype=np.float32),
            confidence=np.array([d.conf for d in raw], dtype=np.float32),
            class_id=np.array([self._names.index(d.name) for d in raw]),
            data={"side": np.array([d.side for d in raw])},
        )
        tracked = self.tracker.update_with_detections(detections)

        out = []
        for i in range(len(tracked)):
            track_id = int(tracked.tracker_id[i])
            name, side, conf = self.smoother.update(
                track_id,
                self._names[int(tracked.class_id[i])],
                str(tracked.data["side"][i]),
                float(tracked.confidence[i]),
            )
            x0, y0, x1, y1 = tracked.xyxy[i]
            out.append(
                TrackedDetection(
                    track_id=track_id,
                    name=name,
                    side=side,
                    conf=conf,
                    bbox=(round(x0), round(y0), round(x1), round(y1)),
                )
            )
        return out
