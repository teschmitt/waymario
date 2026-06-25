"""Keyboard -> ControllerState parsing and per-action decay."""

from __future__ import annotations

from waymario.config import Config
from waymario.control import Button, ControllerState
from waymario.keyboard import _BTN_DEBOUNCE, _HOLD_SECONDS, KeyboardDriver, _format_buttons


def _driver() -> KeyboardDriver:
    return KeyboardDriver(Config())


def test_steer_right_with_letter() -> None:
    d = _driver()
    d.feed(b"d", 0.0)
    assert d.state(0.0).stick_x == Config().max_stick


def test_steer_left_with_arrow() -> None:
    d = _driver()
    d.feed(b"\x1b[D", 0.0)
    assert d.state(0.0).stick_x == -Config().max_stick


def test_arrow_up_sets_positive_stick_y() -> None:
    d = _driver()
    d.feed(b"\x1b[A", 0.0)
    assert d.state(0.0).stick_y == Config().max_stick


def test_space_presses_a() -> None:
    d = _driver()
    d.feed(b" ", 0.0)
    assert d.state(0.0).buttons & Button.A


def test_enter_presses_start() -> None:
    d = _driver()
    d.feed(b"\r", 0.0)
    assert d.state(0.0).buttons & Button.START


def test_letters_are_case_insensitive() -> None:
    d = _driver()
    d.feed(b"B", 0.0)
    assert d.state(0.0).buttons & Button.B


def test_action_decays_to_neutral_after_hold_window() -> None:
    d = _driver()
    d.feed(b"d", 0.0)
    assert d.state(_HOLD_SECONDS - 0.001).stick_x != 0
    assert d.state(_HOLD_SECONDS + 0.001) == ControllerState()


def test_repeat_refreshes_hold() -> None:
    d = _driver()
    d.feed(b"d", 0.0)
    d.feed(b"d", _HOLD_SECONDS - 0.001)  # auto-repeat keeps it alive
    assert d.state(_HOLD_SECONDS + 0.001).stick_x != 0


def test_combined_inputs_in_one_buffer() -> None:
    d = _driver()
    d.feed(b" \x1b[C", 0.0)  # A + steer right
    state = d.state(0.0)
    assert state.buttons & Button.A
    assert state.stick_x == Config().max_stick


def test_opposite_directions_cancel() -> None:
    d = _driver()
    d.feed(b"ad", 0.0)  # left + right
    assert d.state(0.0).stick_x == 0


def test_button_latches_on_then_off() -> None:
    d = _driver()
    d.feed(b" ", 0.0)  # tap A on
    assert d.state(0.0).buttons & Button.A
    # Still latched long after, with no further input (no auto-repeat needed).
    assert d.state(5.0).buttons & Button.A
    d.feed(b" ", 5.0)  # tap A off
    assert not d.state(5.0).buttons & Button.A


def test_button_repeat_within_debounce_toggles_once() -> None:
    d = _driver()
    d.feed(b" ", 0.0)  # tap on
    d.feed(b" ", _BTN_DEBOUNCE / 2)  # auto-repeat burst, ignored
    assert d.state(1.0).buttons & Button.A  # still on, not flipped back off


def test_latched_button_holds_while_stick_decays() -> None:
    # The parallel case: A stays held while you steer with a momentary arrow.
    d = _driver()
    d.feed(b" ", 0.0)  # latch A
    d.feed(b"\x1b[C", 1.0)  # steer right a second later
    state = d.state(1.0)
    assert state.buttons & Button.A
    assert state.stick_x == Config().max_stick
    # Stick decays; A remains latched.
    later = d.state(1.0 + _HOLD_SECONDS + 0.001)
    assert later.stick_x == 0
    assert later.buttons & Button.A


def test_format_buttons() -> None:
    assert _format_buttons(Button(0)) == "-"
    assert _format_buttons(Button.A) == "A"
    assert _format_buttons(Button.A | Button.B) == "A B"


def test_q_sets_quit() -> None:
    d = _driver()
    d.feed(b"q", 0.0)
    assert d.quit


def test_ctrl_c_sets_quit() -> None:
    d = _driver()
    d.feed(b"\x03", 0.0)
    assert d.quit


def test_unbound_key_is_ignored() -> None:
    d = _driver()
    d.feed(b"x", 0.0)
    assert d.state(0.0) == ControllerState()
    assert not d.quit
