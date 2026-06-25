"""N64 controller state and the steering->controls policy.

``Button`` bit positions match the N64 controller status word so a
``ControllerState`` serializes straight into the two button bytes the joybus
poll response expects. ``drive_policy`` turns a ``SteeringDecision`` into a
concrete controller state (hold A to accelerate, steer with the analog stick).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntFlag

from .config import Config
from .steering import SteeringDecision


class Button(IntFlag):
    """N64 buttons laid out as a 16-bit word: high byte = status byte 0,
    low byte = status byte 1 (matching the standard joybus poll response)."""

    # status byte 0
    A = 0x8000
    B = 0x4000
    Z = 0x2000
    START = 0x1000
    D_UP = 0x0800
    D_DOWN = 0x0400
    D_LEFT = 0x0200
    D_RIGHT = 0x0100
    # status byte 1
    L = 0x0020
    R = 0x0010
    C_UP = 0x0008
    C_DOWN = 0x0004
    C_LEFT = 0x0002
    C_RIGHT = 0x0001


def _to_signed_byte(value: int) -> int:
    return max(-128, min(127, int(value)))


@dataclass
class ControllerState:
    stick_x: int = 0  # -128..127
    stick_y: int = 0  # -128..127
    buttons: Button = field(default=Button(0))

    def to_n64_bytes(self) -> bytes:
        """The 4 canonical N64 status bytes: button hi, button lo, stick X, stick Y."""
        b = int(self.buttons)
        return bytes(
            (
                (b >> 8) & 0xFF,
                b & 0xFF,
                _to_signed_byte(self.stick_x) & 0xFF,
                _to_signed_byte(self.stick_y) & 0xFF,
            )
        )


def drive_policy(decision: SteeringDecision, config: Config) -> ControllerState:
    """Always accelerate (hold A); steer the analog stick from the decision."""
    stick_x = _to_signed_byte(round(decision.steering * config.max_stick))
    return ControllerState(stick_x=stick_x, stick_y=0, buttons=Button.A)
