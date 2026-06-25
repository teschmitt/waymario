"""Manual keyboard control: drive the console by hand for debugging.

Reads keypresses from the terminal in raw mode and turns them into
``ControllerState``s, bypassing capture and steering entirely. Useful for
sanity-checking the transport/Pico/console link without trusting the vision.

A terminal in raw mode only delivers key *press* bytes -- there is no key
*release* event, and the OS only auto-repeats the **last** key held. That makes
"hold two keys at once" impossible if both rely on auto-repeat. So we split the
two kinds of input:

- **Buttons latch**: a tap toggles the button on; tap again to toggle it off. No
  auto-repeat needed, so a latched button (e.g. A to accelerate) stays on while
  you steer. Tap, don't hold -- holding a button key re-toggles it.
- **The stick is momentary**: hold an arrow / WASD key to steer. It is the one
  key actually held, so its auto-repeat streams fine; it decays back to centre a
  fraction of a second after you let go.

Key bindings::

    left / right  or  a / d     steer stick left / right  (momentary)
    up   / down   or  w / s      stick up / down           (momentary)
    space                        toggle A (accelerate)
    b                            toggle B (brake / reverse / item)
    z                            toggle Z
    l / r                        toggle L / R
    enter                        toggle Start
    q / Ctrl-C                   quit (controller released to neutral)
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from enum import Enum, auto

from .config import Config
from .control import Button, ControllerState
from .transport import ControllerLink

_HOLD_SECONDS = 0.15
"""How long a stick direction stays active after its last keypress. Must
comfortably exceed the terminal's auto-repeat period so a held key never flickers
off."""

_BTN_DEBOUNCE = 0.25
"""Ignore repeated presses of a button key within this window, so a slightly long
tap (or a burst of auto-repeat) toggles the button once rather than many times."""


class Axis(Enum):
    """Stick directions, kept separate from buttons since they map to analog X/Y."""

    LEFT = auto()
    RIGHT = auto()
    UP = auto()
    DOWN = auto()


# Single-character keys -> action. Buttons map to a ``Button`` flag (toggled);
# stick directions map to an ``Axis`` (momentary). Letters match case-insensitively.
_KEY_BINDINGS: dict[str, Axis | Button] = {
    "a": Axis.LEFT,
    "d": Axis.RIGHT,
    "w": Axis.UP,
    "s": Axis.DOWN,
    " ": Button.A,
    "b": Button.B,
    "z": Button.Z,
    "l": Button.L,
    "r": Button.R,
    "\r": Button.START,
    "\n": Button.START,
}

# Arrow-key escape sequences (ESC [ A/B/C/D) -> action.
_ARROW_BINDINGS: dict[str, Axis] = {
    "\x1b[A": Axis.UP,
    "\x1b[B": Axis.DOWN,
    "\x1b[C": Axis.RIGHT,
    "\x1b[D": Axis.LEFT,
}

_QUIT_KEYS = frozenset({"q", "\x03"})  # q or Ctrl-C


class KeyboardDriver:
    """Turns raw terminal bytes into a ``ControllerState``.

    Buttons latch (tap to toggle on/off); stick directions are momentary and
    decay after ``hold_seconds``. Pure and clock-injected so it can be
    unit-tested without a real terminal: feed byte buffers with an explicit
    ``now`` and read back ``state(now)``.
    """

    def __init__(
        self,
        config: Config,
        hold_seconds: float = _HOLD_SECONDS,
        btn_debounce: float = _BTN_DEBOUNCE,
    ) -> None:
        self._config = config
        self._hold = hold_seconds
        self._debounce = btn_debounce
        self._axis_last: dict[Axis, float] = {}
        self._btn_last: dict[Button, float] = {}
        self._latched = Button(0)
        self.quit = False

    def feed(self, data: bytes, now: float) -> None:
        """Parse a read buffer: refresh stick directions, toggle buttons.

        Arrow escape sequences are matched first, then remaining bytes are read
        one character at a time against the single-key bindings.
        """
        text = data.decode("latin-1")
        i = 0
        while i < len(text):
            seq = text[i : i + 3]
            if seq in _ARROW_BINDINGS:
                self._axis_last[_ARROW_BINDINGS[seq]] = now
                i += 3
                continue
            ch = text[i]
            if ch in _QUIT_KEYS:
                self.quit = True
            else:
                action = _KEY_BINDINGS.get(ch.lower() if ch.isalpha() else ch)
                if isinstance(action, Axis):
                    self._axis_last[action] = now
                elif isinstance(action, Button):
                    self._toggle_button(action, now)
            i += 1

    def _toggle_button(self, btn: Button, now: float) -> None:
        """Flip a latched button, debouncing auto-repeat / over-long taps."""
        if now - self._btn_last.get(btn, float("-inf")) > self._debounce:
            self._latched ^= btn
        self._btn_last[btn] = now

    def state(self, now: float) -> ControllerState:
        """Latched buttons plus the stick from directions still within the hold."""
        active = {a for a, t in self._axis_last.items() if now - t < self._hold}
        stick_x = 0
        if Axis.RIGHT in active:
            stick_x += self._config.max_stick
        if Axis.LEFT in active:
            stick_x -= self._config.max_stick
        stick_y = 0
        if Axis.UP in active:
            stick_y += self._config.max_stick
        if Axis.DOWN in active:
            stick_y -= self._config.max_stick

        return ControllerState(stick_x=stick_x, stick_y=stick_y, buttons=self._latched)


class LatestLine:
    """Thread-safe holder for the most recent line the daemon/Pico sent.

    The network reader runs on its own thread; the keyboard loop reads the latest
    line to render it in place, so device output never scrolls past the controls.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._text = ""

    def set(self, text: str) -> None:
        with self._lock:
            self._text = text

    def get(self) -> str:
        with self._lock:
            return self._text


def _format_buttons(buttons: Button) -> str:
    """Render the pressed buttons by name (e.g. ``A B Z``), ``-`` when none."""
    names = [b.name for b in Button if b in buttons and b.name is not None]
    return " ".join(names) if names else "-"


def run_keyboard(
    link: ControllerLink,
    config: Config,
    status: Callable[[], str] | None = None,
) -> None:
    """Read the keyboard and send controller frames until the user quits.

    Puts stdin into cbreak mode, reads any pending bytes each tick, and emits a
    ``ControllerState`` paced to ``config.target_fps``. ``status`` (if given)
    returns the latest line from the device, shown in place on the status line.
    On exit (quit, EOF, or Ctrl-C) the terminal is restored and the controller
    released to neutral.
    """
    import os
    import select
    import sys
    import termios
    import time
    import tty

    fd = sys.stdin.fileno()
    if not os.isatty(fd):
        raise RuntimeError("keyboard control needs an interactive terminal (a TTY)")

    frame_period = 1.0 / config.target_fps if config.target_fps > 0 else 0.0
    driver = KeyboardDriver(config)
    old_attrs = termios.tcgetattr(fd)
    print(__doc__.split("Key bindings::", 1)[-1].strip(), flush=True)
    print("\nDriving -- press q to quit.\n", flush=True)
    try:
        tty.setcbreak(fd)
        last_print = 0.0
        while not driver.quit:
            start = time.monotonic()
            # Drain everything available this tick (non-blocking).
            while select.select([fd], [], [], 0)[0]:
                chunk = os.read(fd, 64)
                if not chunk:  # EOF
                    driver.quit = True
                    break
                driver.feed(chunk, start)

            state = driver.state(start)
            link.send(state)

            if start - last_print >= 0.1:
                line = (
                    f"stick=({state.stick_x:+4d},{state.stick_y:+4d})  "
                    f"buttons=[{_format_buttons(state.buttons)}]"
                )
                if status is not None and (pico := status()):
                    line += f"   pico: {pico}"
                # \x1b[K clears to end of line so a shorter line leaves no residue.
                print(f"\r{line}\x1b[K", end="", flush=True)
                last_print = start

            if frame_period:
                elapsed = time.monotonic() - start
                if elapsed < frame_period:
                    time.sleep(frame_period - elapsed)
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_attrs)
        link.send(ControllerState())  # neutral
        print("\nReleased controller to neutral.", flush=True)
