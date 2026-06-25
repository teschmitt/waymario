"""Steering brain.

A ``Steerer`` turns a frame into a ``SteeringDecision``. ``OpenCVSteerer`` uses
classical image processing: Rainbow Road is a bright ribbon over a black
starfield, so thresholding the brightness inside a look-ahead region-of-interest
and finding the lit centroid gives a robust "where is the track" signal. The
horizontal offset of that centroid from the frame center, scaled by a gain,
becomes the steering command.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import cv2
import numpy as np

from .config import Config


@dataclass
class SteeringDecision:
    steering: float
    """Desired steering, -1 (full left) .. +1 (full right)."""
    confidence: float
    """Fraction of the ROI that read as track (0..1)."""
    centroid_x: float | None = None
    """Normalized lateral indicator within the sub-frame, -1..1 (None if no track seen).
    OpenCVSteerer: track centroid offset. HSVSteerer: cross-track error e_y."""
    hue: float | None = None
    """Median OpenCV hue (0..179) sampled by HSVSteerer (None for other steerers)."""


def _clamp(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _subframe(frame: np.ndarray, cfg: Config) -> np.ndarray:
    """Crop a frame to this player's screen quadrant (fractions from config)."""
    height, width = frame.shape[:2]
    px0, py0, px1, py1 = cfg.player_region()
    return frame[int(height * py0):int(height * py1), int(width * px0):int(width * px1)]


class Steerer(ABC):
    @abstractmethod
    def decide(self, frame: np.ndarray) -> SteeringDecision:
        """Decide how to steer for a single BGR frame."""

    @abstractmethod
    def roi_box(self, sub_h: int, sub_w: int) -> tuple[int, int, int, int]:
        """Return the (x0, y0, x1, y1) sub-frame rectangle this steerer samples."""


class OpenCVSteerer(Steerer):
    def __init__(self, config: Config) -> None:
        self._config = config

    def roi_box(self, sub_h: int, sub_w: int) -> tuple[int, int, int, int]:
        cfg = self._config
        return (0, int(sub_h * cfg.roi_top), sub_w, int(sub_h * cfg.roi_bottom))

    def decide(self, frame: np.ndarray) -> SteeringDecision:
        cfg = self._config

        # Crop to this player's screen quadrant first.
        subframe = _subframe(frame, cfg)
        sub_h, sub_w = subframe.shape[:2]

        # ROI is relative to the player's sub-frame.
        top = int(sub_h * cfg.roi_top)
        bottom = int(sub_h * cfg.roi_bottom)
        roi = subframe[top:bottom, :]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, cfg.bright_threshold, 255, cv2.THRESH_BINARY)

        lit = int(np.count_nonzero(mask))
        confidence = lit / mask.size if mask.size else 0.0

        if confidence < cfg.min_confidence:
            return SteeringDecision(steering=0.0, confidence=confidence, centroid_x=None)

        moments = cv2.moments(mask, binaryImage=True)
        cx = moments["m10"] / moments["m00"]

        offset = (cx - sub_w / 2) / (sub_w / 2)
        steering = _clamp(offset * cfg.steering_gain)

        return SteeringDecision(steering=steering, confidence=confidence, centroid_x=offset)
