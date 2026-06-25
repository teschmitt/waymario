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
    """Normalized lateral indicator within the sub-frame, -1..1 (None if no track seen). OpenCVSteerer: track-centroid offset. HSVSteerer: cross-track error e_y."""
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
    """Read lateral position from the track's red->purple hue gradient.

    Sample one small look-ahead patch, take the median hue of on-track
    (saturated, bright) pixels, map it linearly to a cross-track error, and
    steer against that error.
    """

    def __init__(self, config: Config) -> None:
        self._config = config

    def _patch_bounds(self, sub_h: int, sub_w: int) -> tuple[int, int, int, int]:
        cfg = self._config
        cx = cfg.hue_patch_cx * sub_w
        cy = cfg.hue_patch_cy * sub_h
        half_w = cfg.hue_patch_w * sub_w / 2.0
        half_h = cfg.hue_patch_h * sub_h / 2.0
        x0 = max(0, int(cx - half_w))
        y0 = max(0, int(cy - half_h))
        x1 = min(sub_w, int(cx + half_w))
        y1 = min(sub_h, int(cy + half_h))
        return x0, y0, x1, y1

    def roi_box(self, sub_h: int, sub_w: int) -> tuple[int, int, int, int]:
        return self._patch_bounds(sub_h, sub_w)

    def decide(self, frame: np.ndarray) -> SteeringDecision:
        cfg = self._config
        subframe = _subframe(frame, cfg)
        sub_h, sub_w = subframe.shape[:2]

        x0, y0, x1, y1 = self._patch_bounds(sub_h, sub_w)
        patch = subframe[y0:y1, x0:x1]
        if patch.size == 0:
            return SteeringDecision(steering=0.0, confidence=0.0, lateral=None, hue=None)

        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        h = hsv[:, :, 0]
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]

        gate = (s >= cfg.hue_min_sat) & (v >= cfg.hue_min_val)
        passed = h[gate]
        total = h.size
        confidence = float(passed.size) / float(total) if total else 0.0

        if passed.size == 0 or confidence < cfg.min_confidence:
            # No trustworthy colored track in the patch — coast straight.
            return SteeringDecision(steering=0.0, confidence=confidence, lateral=None, hue=None)

        # Plain median is correct only because the gradient stays within H~0..160 and
        # never crosses the OpenCV red/magenta hue wrap (179->0). Off-track pixels are
        # already removed by the S/V gate above.
        hue = float(np.median(passed))
        span = cfg.hue_right - cfg.hue_left
        if span == 0:
            # Misconfigured edge hues — can't form an error; coast (keep the hue diagnostic).
            return SteeringDecision(steering=0.0, confidence=confidence, lateral=None, hue=hue)
        e_y = _clamp(2.0 * (hue - cfg.hue_left) / span - 1.0)
        steering = _clamp(-cfg.hue_gain * e_y)
        return SteeringDecision(steering=steering, confidence=confidence, lateral=e_y, hue=hue)


def build_steerer(config: Config) -> Steerer:
    """Construct the steerer named by ``config.steerer``."""
    if config.steerer == "hsv":
        return HSVSteerer(config)
    if config.steerer == "brightness":
        return OpenCVSteerer(config)
    raise ValueError(
        f"unknown steerer {config.steerer!r}; expected 'hsv' or 'brightness'"
    )
