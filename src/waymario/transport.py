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
    b,0,-80     # B only, full reverse

The Pico **holds the last state** until a new line arrives, and it talks back: a
boot banner, a syntax help block, and ``dbg:``/``Ready.`` lines. ``SerialLink``
runs a background reader that prints everything the Pico sends as ``[pico] ...``.
"""

from __future__ import annotations

import sys
import threading
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
    """Send controller frames to the Pi Pico over serial.

    Also reads back everything the Pico prints (boot banner, ``dbg:`` lines, …)
    on a background thread and echoes it to stdout as ``[pico] …``.
    """

    def __init__(
        self,
        port: str,
        baud: int = 115200,
        *,
        echo: bool = True,
        serial_obj: object | None = None,
    ) -> None:
        if serial_obj is not None:
            # Injected (tests / a pre-opened port): skip the pyserial dependency.
            self._serial = serial_obj
        else:
            import serial  # local import so the brain runs without pyserial present

            # Small read timeout so the reader's readline() blocks briefly rather
            # than busy-spinning; writes are unaffected.
            self._serial = serial.Serial(port, baud, timeout=0.1)

        self._stop = threading.Event()
        self._reader: threading.Thread | None = None
        if echo:
            self._reader = threading.Thread(
                target=self._read_loop, name="pico-reader", daemon=True
            )
            self._reader.start()

    def _read_loop(self) -> None:
        """Print every line the Pico sends until close() sets the stop flag."""
        while not self._stop.is_set():
            try:
                line = self._serial.readline()
            except Exception:  # port closed out from under us, etc.
                break
            if line:
                text = line.decode("utf-8", errors="replace").rstrip("\r\n")
                sys.stdout.write(f"[pico] {text}\n")
                sys.stdout.flush()

    def send(self, state: ControllerState) -> None:
        self._serial.write(encode(state))

    def close(self) -> None:
        self._stop.set()
        if self._reader is not None:
            self._reader.join(timeout=1.0)
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
