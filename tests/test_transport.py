"""Wire frame encoding and the NullLink sink."""

from __future__ import annotations

from waymario.control import Button, ControllerState
from waymario.transport import FRAME_HEADER, FRAME_SIZE, NullLink, encode


def test_frame_shape_and_checksum() -> None:
    frame = encode(ControllerState(stick_x=10, stick_y=-5, buttons=Button.A))
    assert len(frame) == FRAME_SIZE
    assert frame[0] == FRAME_HEADER
    payload = frame[1:5]
    checksum = 0
    for byte in payload:
        checksum ^= byte
    assert frame[5] == checksum


def test_null_link_records_last_frame() -> None:
    link = NullLink()
    state = ControllerState(stick_x=42, buttons=Button.A)
    link.send(state)
    link.send(ControllerState())
    assert link.count == 2
    assert link.last_state == ControllerState()
    assert link.last_frame == encode(ControllerState())
