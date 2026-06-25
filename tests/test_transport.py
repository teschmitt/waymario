"""ASCII wire frame encoding and the NullLink sink."""

from __future__ import annotations

from waymario.control import Button, ControllerState
from waymario.transport import NullLink, encode


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


def test_null_link_records_last_frame() -> None:
    link = NullLink()
    state = ControllerState(stick_x=42, buttons=Button.A)
    link.send(state)
    link.send(ControllerState())
    assert link.count == 2
    assert link.last_state == ControllerState()
    assert link.last_frame == encode(ControllerState())
