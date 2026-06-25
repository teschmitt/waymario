"""Steering decisions from synthetic frames."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from waymario.config import Config
from waymario.steering import HSVSteerer, OpenCVSteerer, SteeringDecision, build_steerer


def test_steering_decision_has_hue_field_defaulting_none() -> None:
    d = SteeringDecision(steering=0.0, confidence=0.0)
    assert d.hue is None
    assert SteeringDecision(steering=0.0, confidence=0.0, hue=42.0).hue == 42.0


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
    assert decision.lateral is None


def _bgr_for_hue(hue: int, sat: int = 200, val: int = 200) -> tuple[int, int, int]:
    """BGR tuple for a single OpenCV HSV color."""
    px = np.uint8([[[hue, sat, val]]])
    b, g, r = cv2.cvtColor(px, cv2.COLOR_HSV2BGR)[0, 0]
    return int(b), int(g), int(r)


def _colored_strip_frame(width: int, x0: int, x1: int, hue: int = 10,
                         height: int = 200) -> np.ndarray:
    """Black frame with a saturated coloured strip in [x0, x1), spanning the ROI band."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, x0:x1] = _bgr_for_hue(hue, sat=200, val=200)
    return frame


def test_hsv_colored_track_on_right_steers_right() -> None:
    # Coloured ribbon on the right of the ROI => centroid offset > 0 => steer right.
    steerer = HSVSteerer(Config())
    decision = steerer.decide(_colored_strip_frame(640, 500, 560))
    assert decision.steering > 0
    assert decision.lateral is not None and decision.lateral > 0
    assert decision.confidence > 0
    assert decision.hue is not None


def test_hsv_colored_track_on_left_steers_left() -> None:
    steerer = HSVSteerer(Config())
    decision = steerer.decide(_colored_strip_frame(640, 80, 140))
    assert decision.steering < 0
    assert decision.lateral is not None and decision.lateral < 0


def test_hsv_centered_track_goes_straight() -> None:
    steerer = HSVSteerer(Config())
    decision = steerer.decide(_colored_strip_frame(640, 300, 340))
    assert abs(decision.steering) < 0.05


def test_hsv_desaturated_frame_coasts_straight() -> None:
    # All-black frame: every pixel fails the S/V gate => coast.
    steerer = HSVSteerer(Config())
    frame = np.zeros((200, 640, 3), dtype=np.uint8)
    decision = steerer.decide(frame)
    assert decision.steering == 0.0
    assert decision.hue is None
    assert decision.lateral is None


def test_hsv_rejects_desaturated_bright_frame() -> None:
    # White is bright but unsaturated: the saturation gate rejects it, so HSV coasts
    # where a brightness threshold would chase it. This is HSV's edge over brightness.
    steerer = HSVSteerer(Config())
    frame = np.full((200, 640, 3), 255, dtype=np.uint8)
    decision = steerer.decide(frame)
    assert decision.confidence < Config().min_confidence
    assert decision.steering == 0.0
    assert decision.lateral is None


def test_hsv_partial_roi_has_fractional_confidence() -> None:
    # Left half coloured, right half black: centroid sits left => steer left.
    frame = np.zeros((200, 640, 3), dtype=np.uint8)
    frame[:, :320] = _bgr_for_hue(10)
    steerer = HSVSteerer(Config())
    decision = steerer.decide(frame)
    assert 0.2 < decision.confidence < 0.8
    assert decision.steering < 0


def test_hsv_roi_box_within_subframe() -> None:
    steerer = HSVSteerer(Config())
    x0, y0, x1, y1 = steerer.roi_box(sub_h=200, sub_w=640)
    assert 0 <= x0 < x1 <= 640
    assert 0 <= y0 < y1 <= 200


def test_config_default_steerer_is_hsv() -> None:
    assert Config().steerer == "hsv"


def test_build_steerer_selects_hsv() -> None:
    assert isinstance(build_steerer(Config(steerer="hsv")), HSVSteerer)


def test_build_steerer_selects_brightness() -> None:
    assert isinstance(build_steerer(Config(steerer="brightness")), OpenCVSteerer)


def test_build_steerer_rejects_unknown() -> None:
    with pytest.raises(ValueError):
        build_steerer(Config(steerer="rainbow"))
