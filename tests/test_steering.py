"""Steering decisions from synthetic frames."""

from __future__ import annotations

import numpy as np

from waymario.config import Config
from waymario.steering import OpenCVSteerer, SteeringDecision


def test_steering_decision_has_hue_field_defaulting_none() -> None:
    d = SteeringDecision(steering=0.0, confidence=0.0)
    assert d.hue is None


def test_opencv_roi_box_is_full_width_band() -> None:
    steerer = OpenCVSteerer(Config())
    # Config defaults: roi_top=0.45, roi_bottom=0.95
    assert steerer.roi_box(sub_h=100, sub_w=200) == (0, 45, 200, 95)


def _frame_with_bright_strip(width: int, x0: int, x1: int, height: int = 200) -> np.ndarray:
    """Black frame with a bright vertical strip in [x0, x1) inside the ROI band."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, x0:x1] = 255
    return frame


def test_track_on_right_steers_right() -> None:
    steerer = OpenCVSteerer(Config())
    frame = _frame_with_bright_strip(width=640, x0=500, x1=560)
    decision = steerer.decide(frame)
    assert decision.steering > 0
    assert decision.confidence > 0


def test_track_on_left_steers_left() -> None:
    steerer = OpenCVSteerer(Config())
    frame = _frame_with_bright_strip(width=640, x0=80, x1=140)
    decision = steerer.decide(frame)
    assert decision.steering < 0


def test_centered_track_goes_straight() -> None:
    steerer = OpenCVSteerer(Config())
    frame = _frame_with_bright_strip(width=640, x0=300, x1=340)
    decision = steerer.decide(frame)
    assert abs(decision.steering) < 0.05


def test_black_frame_coasts_straight() -> None:
    steerer = OpenCVSteerer(Config())
    frame = np.zeros((200, 640, 3), dtype=np.uint8)
    decision = steerer.decide(frame)
    assert decision.steering == 0.0
    assert decision.centroid_x is None
