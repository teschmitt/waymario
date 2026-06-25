"""Controller output transport.

Serializes a ``ControllerState`` into the fixed 6-byte wire frame the Pi Pico
expects, and sends it. ``SerialLink`` writes to the Pico over USB/UART;
``NullLink`` just records frames so the full brain can run with no Pico attached.

Wire frame (see firmware/README.md for the authoritative contract):

    [0xA5][btn_hi][btn_lo][stick_x][stick_y][xor]

where ``xor`` is the XOR of the four payload bytes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .control import ControllerState

FRAME_HEADER = 0xA5
FRAME_SIZE = 6


def encode(state: ControllerState) -> bytes:
    """Encode a controller state into the 6-byte wire frame."""
    payload = state.to_n64_bytes()
    checksum = 0
    for byte in payload:
        checksum ^= byte
    return bytes((FRAME_HEADER, *payload, checksum))


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
