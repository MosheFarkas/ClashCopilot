"""Arena unit detection: boxes + name + side + confidence per frame.

Uses the MIT-licensed pretrained models from Pbatch/ClashRoyaleBuildABot:
  units_M_480x352.onnx -- YOLO-style detector, 97 unit classes, NMS
      embedded (output rows: left, top, right, bottom, conf, class);
  side.onnx -- tiny ally/enemy classifier on a 16x16 crop.
Preprocessing mirrors their pipeline: crop the arena band (5%..80% of
frame height), letterbox to 352x480 with value-114 padding, RGB fp16.
"""

import json
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import cv2
import numpy as np

MODEL_W, MODEL_H = 352, 480
BAND_TOP, BAND_BOTTOM = 0.05, 0.80
SIDE_SIZE = 16


def load_unit_names() -> list[str]:
    raw = resources.files("clash_copilot.data").joinpath("arena_units.json").read_text()
    return json.loads(raw)["units"]


@dataclass(frozen=True)
class ArenaDetection:
    name: str
    side: str  # "ally" | "enemy"
    conf: float
    bbox: tuple[int, int, int, int]  # x0, y0, x1, y1 in frame pixels


class ArenaDetector:
    def __init__(self, units_model: str | Path, side_model: str | Path, min_conf: float = 0.3):
        import onnxruntime as ort

        self.session = ort.InferenceSession(str(units_model), providers=["CPUExecutionProvider"])
        self.side_session = ort.InferenceSession(str(side_model), providers=["CPUExecutionProvider"])
        self.unit_names = load_unit_names()
        self.min_conf = min_conf
        self.units_input = self.session.get_inputs()[0].name
        self.side_input = self.side_session.get_inputs()[0].name

    def detect(self, frame: np.ndarray) -> list[ArenaDetection]:
        """Detections for one BGR frame."""
        height, width = frame.shape[:2]
        band_y0 = int(BAND_TOP * height)
        band = frame[band_y0 : int(BAND_BOTTOM * height), :]
        tensor, padding, scale = self._preprocess(band)
        rows = self.session.run(None, {self.units_input: tensor})[0][0]

        detections = []
        for left, top, right, bottom, conf, cls in rows:
            if conf < self.min_conf:
                continue
            x0 = (left - padding[0]) / scale
            x1 = (right - padding[0]) / scale
            y0 = (top - padding[2]) / scale + band_y0
            y1 = (bottom - padding[2]) / scale + band_y0
            x0, y0 = max(0, round(x0)), max(0, round(y0))
            x1, y1 = min(width, round(x1)), min(height, round(y1))
            if x1 <= x0 or y1 <= y0:
                continue
            bbox = (x0, y0, x1, y1)
            detections.append(
                ArenaDetection(
                    name=self.unit_names[int(cls)],
                    side=self._side(frame, bbox),
                    conf=float(conf),
                    bbox=bbox,
                )
            )
        return detections

    def _preprocess(self, band: np.ndarray):
        height, width = band.shape[:2]
        scale = min(MODEL_W / width, MODEL_H / height)
        resized = cv2.resize(band, (round(width * scale), round(height * scale)))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        dx, dy = MODEL_W - rgb.shape[1], MODEL_H - rgb.shape[0]
        pad_right, pad_bottom = dx // 2, dy // 2
        pad_left, pad_top = dx - pad_right, dy - pad_bottom
        padded = np.pad(
            rgb,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode="constant",
            constant_values=114,
        )
        tensor = (padded.astype(np.float16) / 255.0).transpose(2, 0, 1)[None]
        return tensor, (pad_left, pad_right, pad_top, pad_bottom), scale

    def _side(self, frame: np.ndarray, bbox: tuple[int, int, int, int]) -> str:
        x0, y0, x1, y1 = bbox
        crop = frame[y0:y1, x0:x1]
        rgb = cv2.cvtColor(
            cv2.resize(crop, (SIDE_SIZE, SIDE_SIZE), interpolation=cv2.INTER_CUBIC),
            cv2.COLOR_BGR2RGB,
        )
        pred = self.side_session.run(
            None, {self.side_input: (rgb.astype(np.float32) / 255.0)[None]}
        )
        return ("ally", "enemy")[int(np.argmax(pred[0]))]
