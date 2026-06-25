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

The Pico **holds the last state** until a new line arrives, and it talks back: a
boot banner, a syntax help block, and ``dbg:``/``Ready.`` lines. ``SerialLink``
runs a background reader that prints everything the Pico sends as ``[pico] ...``.
"""

from __future__ import annotations

import sys
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable

from .control import Button, ControllerState

# Map Button flags -> single-character token the Pico expects.
_BUTTON_CHARS: list[tuple[Button, str]] = [
    (Button.A,     "a"),
    (Button.B,     "b"),
    (Button.Z,     "z"),
    (Button.R,     "r"),
    (Button.L,     "l"),
    (Button.START, "s"),
    (Button.U,     "u"),
]

# Inverse lookup for decode(): single character -> Button flag.
_CHAR_BUTTONS: dict[str, Button] = {ch: flag for flag, ch in _BUTTON_CHARS}


def encode(state: ControllerState) -> bytes:
    """Encode a ControllerState into the ASCII text frame the Pico expects."""
    btn_str = "".join(ch for flag, ch in _BUTTON_CHARS if flag in state.buttons)
    line = f"{btn_str},{state.stick_x},{state.stick_y}\n"
    return line.encode()


def decode(line: str | bytes) -> ControllerState:
    """Parse one ASCII wire frame back into a ControllerState (inverse of encode).

    Accepts ``<buttons>,<stick_x>,<stick_y>`` with optional trailing newline.
    Raises ValueError on anything malformed so callers can drop garbage rather
    than forward it to the Pico.
    """
    if isinstance(line, bytes):
        line = line.decode("utf-8", errors="replace")
    parts = line.strip().split(",")
    if len(parts) != 3:
        raise ValueError(f"expected 'buttons,x,y', got {line!r}")
    btn_str, x_str, y_str = parts
    buttons = Button(0)
    for ch in btn_str:
        flag = _CHAR_BUTTONS.get(ch)
        if flag is None:
            raise ValueError(f"unknown button char {ch!r} in {line!r}")
        buttons |= flag
    try:
        stick_x, stick_y = int(x_str), int(y_str)
    except ValueError as exc:
        raise ValueError(f"non-integer stick value in {line!r}") from exc
    return ControllerState(stick_x=stick_x, stick_y=stick_y, buttons=buttons)


def _default_print(text: str) -> None:
    """Default reader sink: echo a line the device sent as ``[pico] …`` on stdout."""
    sys.stdout.write(f"[pico] {text}\n")
    sys.stdout.flush()


def _read_lines(
    read_line: Callable[[], bytes],
    on_line: Callable[[str], None],
    stop: threading.Event,
) -> None:
    """Pump newline-delimited bytes from ``read_line`` into ``on_line`` until stopped.

    ``read_line`` is expected to return ``b""`` on timeout/EOF (so the loop can
    check the stop flag) and raise on a closed underlying stream.
    """
    while not stop.is_set():
        try:
            line = read_line()
        except Exception:  # stream closed out from under us, etc.
            break
        if line:
            on_line(line.decode("utf-8", errors="replace").rstrip("\r\n"))


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
        on_line: Callable[[str], None] | None = None,
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

        self._on_line = on_line if on_line is not None else _default_print
        self._stop = threading.Event()
        self._reader: threading.Thread | None = None
        if echo:
            self._reader = threading.Thread(
                target=_read_lines,
                args=(self._serial.readline, self._on_line, self._stop),
                name="pico-reader",
                daemon=True,
            )
            self._reader.start()

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


class TcpLink(ControllerLink):
    """Send controller frames to a ``waymario daemon`` over TCP.

    The daemon speaks the same line protocol as the Pico, so this is just
    ``SerialLink`` over a socket: ``send()`` writes the encoded frame, and a
    background reader prints everything the daemon broadcasts back (the Pico's
    own output, multiplexed to every client) as ``[pico] …`` on **stderr** -- so
    it doesn't fight a ``\\r`` status line some clients keep on stdout.
    """

    def __init__(
        self,
        host: str,
        port: int,
        *,
        echo: bool = True,
        on_line: Callable[[str], None] | None = None,
    ) -> None:
        import socket

        self._sock = socket.create_connection((host, port))
        # Buffered line reader over the socket; close() unblocks readline().
        self._rfile = self._sock.makefile("rb")
        self._on_line = on_line if on_line is not None else _default_print_err
        self._stop = threading.Event()
        self._reader: threading.Thread | None = None
        if echo:
            self._reader = threading.Thread(
                target=_read_lines,
                args=(self._rfile.readline, self._on_line, self._stop),
                name="daemon-reader",
                daemon=True,
            )
            self._reader.start()

    def send(self, state: ControllerState) -> None:
        self._sock.sendall(encode(state))

    def close(self) -> None:
        import socket

        self._stop.set()
        try:
            self._sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self._sock.close()
        if self._reader is not None:
            self._reader.join(timeout=1.0)


def _default_print_err(text: str) -> None:
    """Reader sink for TcpLink: echo a broadcast line as ``[pico] …`` on stderr."""
    sys.stderr.write(f"[pico] {text}\n")
    sys.stderr.flush()
