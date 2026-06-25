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
    lateral: float | None = None
    """Normalized track-centroid offset within the sub-frame, -1..1 (None if no track seen). Both OpenCVSteerer and HSVSteerer report the centroid's horizontal offset; they differ only in how they segment the track (brightness threshold vs. HSV colour mask)."""
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
            return SteeringDecision(steering=0.0, confidence=confidence, lateral=None)

        moments = cv2.moments(mask, binaryImage=True)
        cx = moments["m10"] / moments["m00"]

        offset = (cx - sub_w / 2) / (sub_w / 2)
        steering = _clamp(offset * cfg.steering_gain)

        return SteeringDecision(steering=steering, confidence=confidence, lateral=offset)


class HSVSteerer(Steerer):
    """Steer from the rainbow ribbon's position, located by colour.

    Rainbow Road is a vividly *coloured* ribbon over a black starfield. We build a
    mask of saturated, bright pixels across a full-width look-ahead ROI — that picks
    out the ribbon and, unlike a plain brightness threshold, rejects the desaturated
    HUD text and white boost flames — then steer from the horizontal offset of that
    mask's centroid, exactly as ``OpenCVSteerer`` does with its brightness mask.

    (An earlier version mapped the ribbon's *hue* to a cross-track error. On Rainbow
    Road, though, hue varies with look-ahead *depth* — which stripe you are looking
    at — not with lateral position, so the median hue of a fixed patch was nearly
    constant and the kart drove in circles. The near→far hue gradient is still a real
    signal, but it tells forward from reversed, not left from right — see
    ``stuck.StuckDetector._direction_gradient``.)
    """

    def __init__(self, config: Config) -> None:
        self._config = config

    def roi_box(self, sub_h: int, sub_w: int) -> tuple[int, int, int, int]:
        cfg = self._config
        return (0, int(sub_h * cfg.roi_top), sub_w, int(sub_h * cfg.roi_bottom))

    def decide(self, frame: np.ndarray) -> SteeringDecision:
        cfg = self._config
        subframe = _subframe(frame, cfg)
        sub_h, sub_w = subframe.shape[:2]

        top = int(sub_h * cfg.roi_top)
        bottom = int(sub_h * cfg.roi_bottom)
        roi = subframe[top:bottom, :]
        if roi.size == 0:
            return SteeringDecision(steering=0.0, confidence=0.0, lateral=None, hue=None)

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        mask = (s >= cfg.hue_min_sat) & (v >= cfg.hue_min_val)

        lit = int(np.count_nonzero(mask))
        confidence = lit / mask.size if mask.size else 0.0
        if confidence < cfg.min_confidence:
            # No trustworthy coloured track ahead — coast straight.
            return SteeringDecision(steering=0.0, confidence=confidence, lateral=None, hue=None)

        moments = cv2.moments(mask.astype(np.uint8), binaryImage=True)
        cx = moments["m10"] / moments["m00"]
        offset = (cx - sub_w / 2) / (sub_w / 2)
        steering = _clamp(offset * cfg.steering_gain)

        hue = float(np.median(h[mask]))  # diagnostic only (HUD); steering is centroid-based
        return SteeringDecision(steering=steering, confidence=confidence, lateral=offset, hue=hue)


class StraightSteerer(Steerer):
    """Debug steerer: always drives straight ahead, no vision needed."""

    def __init__(self, config: Config) -> None:
        self._config = config

    def roi_box(self, sub_h: int, sub_w: int) -> tuple[int, int, int, int]:
        return (0, 0, sub_w, sub_h)

    def decide(self, frame: np.ndarray) -> SteeringDecision:
        return SteeringDecision(steering=0.0, confidence=1.0, lateral=None, hue=None)


def build_steerer(config: Config) -> Steerer:
    """Construct the steerer named by ``config.steerer``."""
    if config.steerer == "hsv":
        return HSVSteerer(config)
    if config.steerer == "brightness":
        return OpenCVSteerer(config)
    if config.steerer == "straight":
        return StraightSteerer(config)
    raise ValueError(
        f"unknown steerer {config.steerer!r}; expected 'hsv', 'brightness', or 'straight'"
    )
