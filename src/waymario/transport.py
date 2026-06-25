"""Controller output transport.

Serializes a ``ControllerState`` into the ASCII text frame the Pi Pico expects
and sends it over serial. ``SerialLink`` writes to the Pico over USB/UART;
``NullLink`` just records frames so the full brain can run with no Pico attached.

Wire protocol (text, newline-terminated)::

    <buttons>,<stick_x>,<stick_y>\n

    buttons : any combination of  a=A  b=B  z=Z  r=R  l=L  s=Start  (empty=none)
    stick_x : -80..+80  (negative=left,  positive=right)
    stick_y : -80..+80  (negative=down,  positive=up)

Examples::

    a,0,0       # A pressed, stick centred
    ar,80,0     # A + R, full right
    ,0,0        # no buttons, stick centred (neutral)
    ,0,-80      # no buttons, full reverse (MK64 reverses with the stick, not B)
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .control import Button, ControllerState

# Map Button flags -> single-character token the Pico expects.
_BUTTON_CHARS: list[tuple[Button, str]] = [
    (Button.A,     "a"),
    (Button.B,     "b"),
    (Button.Z,     "z"),
    (Button.R,     "r"),
    (Button.L,     "l"),
    (Button.START, "s"),
]


def encode(state: ControllerState) -> bytes:
    """Encode a ControllerState into the ASCII text frame the Pico expects."""
    btn_str = "".join(ch for flag, ch in _BUTTON_CHARS if flag in state.buttons)
    line = f"{btn_str},{state.stick_x},{state.stick_y}\n"
    return line.encode()


class ControllerLink(ABC):
    @abstractmethod
    def send(self, state: ControllerState) -> None:
        """Transmit one controller state."""

    def close(self) -> None:  # noqa: B027 - optional override
        pass

    def __enter__(self) -> ControllerLink:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class SerialLink(ControllerLink):
    """Send controller frames to the Pi Pico over serial."""

    def __init__(self, port: str, baud: int = 115200) -> None:
        import serial  # local import so the brain runs without pyserial present

        self._serial = serial.Serial(port, baud, timeout=0)

    def send(self, state: ControllerState) -> None:
        self._serial.write(encode(state))

    def close(self) -> None:
        self._serial.close()


class NullLink(ControllerLink):
    """No-hardware sink: keeps the last frame sent for inspection/tests."""

    def __init__(self) -> None:
        self.last_state: ControllerState | None = None
        self.last_frame: bytes | None = None
        self.count = 0

    def send(self, state: ControllerState) -> None:
        self.last_state = state
        self.last_frame = encode(state)
        self.count += 1
