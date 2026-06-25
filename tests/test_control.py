"""Controller state encoding and the steering->controls policy."""

from __future__ import annotations

from waymario.config import Config
from waymario.control import Button, ControllerState, drive_policy
from waymario.steering import SteeringDecision


def test_n64_bytes_layout() -> None:
    state = ControllerState(stick_x=80, stick_y=-1, buttons=Button.A | Button.START)
    btn_hi, btn_lo, sx, sy = state.to_n64_bytes()
    assert btn_hi == 0x90  # A (0x80) | START (0x10)
    assert btn_lo == 0x00
    assert sx == 80
    assert sy == 0xFF  # -1 as unsigned byte


def test_stick_is_clamped_to_signed_byte() -> None:
    assert ControllerState(stick_x=999).to_n64_bytes()[2] == 127
    assert ControllerState(stick_x=-999).to_n64_bytes()[2] == (-128 & 0xFF)  # 0x80


def test_policy_always_accelerates() -> None:
    state = drive_policy(SteeringDecision(steering=0.0, confidence=0.5), Config())
    assert Button.A in state.buttons


def test_policy_scales_steering_to_stick() -> None:
    config = Config()
    right = drive_policy(SteeringDecision(steering=1.0, confidence=0.5), config)
    left = drive_policy(SteeringDecision(steering=-1.0, confidence=0.5), config)
    assert right.stick_x == config.max_stick
    assert left.stick_x == -config.max_stick
