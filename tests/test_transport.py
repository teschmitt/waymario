"""ASCII wire frame encoding and the NullLink sink."""

from __future__ import annotations

import queue
import time

import pytest

from waymario.control import Button, ControllerState
from waymario.transport import NullLink, SerialLink, decode, encode


def test_encode_is_newline_terminated() -> None:
    frame = encode(ControllerState())
    assert frame.endswith(b"\n")


def test_encode_neutral() -> None:
    # No buttons, no stick -> ",0,0\n"
    assert encode(ControllerState()) == b",0,0\n"


def test_encode_a_button() -> None:
    state = ControllerState(buttons=Button.A)
    assert encode(state) == b"a,0,0\n"


def test_encode_multiple_buttons() -> None:
    state = ControllerState(buttons=Button.A | Button.R)
    frame = encode(state).decode()
    # both chars must be present, order is a then r
    assert "a" in frame
    assert "r" in frame
    assert frame.endswith(",0,0\n")


def test_encode_stick() -> None:
    state = ControllerState(stick_x=80, stick_y=-40, buttons=Button.A)
    assert encode(state) == b"a,80,-40\n"


def test_encode_b_reverse() -> None:
    state = ControllerState(stick_x=0, stick_y=-80, buttons=Button.B)
    assert encode(state) == b"b,0,-80\n"


@pytest.mark.parametrize(
    "state",
    [
        ControllerState(),
        ControllerState(buttons=Button.A),
        ControllerState(stick_x=80, stick_y=-40, buttons=Button.A | Button.R),
        ControllerState(stick_x=-80, stick_y=80, buttons=Button.B | Button.Z | Button.L | Button.START),
    ],
)
def test_decode_round_trips_encode(state: ControllerState) -> None:
    assert decode(encode(state)) == state


def test_decode_accepts_bytes_and_str() -> None:
    assert decode(b"a,80,0\n") == ControllerState(stick_x=80, buttons=Button.A)
    assert decode("a,80,0") == ControllerState(stick_x=80, buttons=Button.A)


@pytest.mark.parametrize("line", ["", "a,0", "a,0,0,0", "a,x,0", "q,0,0"])
def test_decode_rejects_malformed(line: str) -> None:
    with pytest.raises(ValueError):
        decode(line)


def test_null_link_records_last_frame() -> None:
    link = NullLink()
    state = ControllerState(stick_x=42, buttons=Button.A)
    link.send(state)
    link.send(ControllerState())
    assert link.count == 2
    assert link.last_state == ControllerState()
    assert link.last_frame == encode(ControllerState())


class FakeSerial:
    """Stand-in for pyserial's Serial: records writes, replays queued read lines."""

    def __init__(self, lines: list[bytes] | None = None) -> None:
        self.writes: list[bytes] = []
        self.closed = False
        self._lines: "queue.Queue[bytes]" = queue.Queue()
        for line in lines or []:
            self._lines.put(line)

    def write(self, data: bytes) -> None:
        self.writes.append(data)

    def feed_line(self, line: bytes) -> None:
        """Queue another line for the reader to pick up after construction."""
        self._lines.put(line)

    def readline(self) -> bytes:
        # Mimic a timeout: return b"" when nothing is queued instead of blocking.
        try:
            return self._lines.get_nowait()
        except queue.Empty:
            return b""

    def close(self) -> None:
        self.closed = True


def test_serial_link_writes_encoded_frame() -> None:
    fake = FakeSerial()
    link = SerialLink("ignored", echo=False, serial_obj=fake)
    link.send(ControllerState(stick_x=80, buttons=Button.A))
    link.close()
    assert fake.writes == [encode(ControllerState(stick_x=80, buttons=Button.A))]
    assert fake.closed


def test_serial_link_echoes_pico_output(capsys) -> None:
    fake = FakeSerial(lines=[b"N64 serial controller ready.\n", b"dbg: joybus_enable\r\n"])
    link = SerialLink("ignored", serial_obj=fake)
    # Give the reader thread a moment to drain the queued lines.
    deadline = time.monotonic() + 2.0
    while fake._lines.qsize() and time.monotonic() < deadline:
        time.sleep(0.01)
    link.close()
    out = capsys.readouterr().out
    assert "[pico] N64 serial controller ready." in out
    assert "[pico] dbg: joybus_enable" in out


def test_serial_link_close_stops_reader() -> None:
    fake = FakeSerial()
    link = SerialLink("ignored", serial_obj=fake)
    link.close()  # must not hang
    assert fake.closed
    assert link._reader is not None and not link._reader.is_alive()
