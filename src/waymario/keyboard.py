"""Manual keyboard control: drive the console by hand for debugging.

Reads keypresses from the terminal in raw mode and turns them into
``ControllerState``s, bypassing capture and steering entirely. Useful for
sanity-checking the transport/Pico/console link without trusting the vision.

A terminal in raw mode only delivers key *press* bytes -- there is no key
*release* event. We lean on the OS key auto-repeat instead: while a key is held
the terminal streams repeated bytes, and they stop on release. So each action is
treated as active for a short window after its last byte (``hold_seconds``).
Holding a key keeps refreshing it; releasing lets it decay back to neutral.

Key bindings::

    left / right  or  a / d    steer stick left / right
    up   / down   or  w / s     stick up / down
    space                       A (accelerate)
    b                           B (brake / reverse / item)
    z                           Z
    l / r                       L / R
    enter                       Start
    q / Ctrl-C                  quit (controller released to neutral)
"""

from __future__ import annotations

from enum import Enum, auto

from .config import Config
from .control import Button, ControllerState
from .transport import ControllerLink

_HOLD_SECONDS = 0.15
"""How long an action stays active after its last keypress. Must comfortably
exceed the terminal's auto-repeat period so a held key never flickers off."""


class Axis(Enum):
    """Stick directions, kept separate from buttons since they map to analog X/Y."""

    LEFT = auto()
    RIGHT = auto()
    UP = auto()
    DOWN = auto()


# Single-character keys -> action. Buttons map to a ``Button`` flag; stick
# directions map to an ``Axis``. Letters are matched case-insensitively below.
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
    """Turns raw terminal bytes into a ``ControllerState`` with per-action decay.

    Pure and clock-injected so it can be unit-tested without a real terminal:
    feed byte buffers with an explicit ``now`` and read back ``state(now)``.
    """

    def __init__(self, config: Config, hold_seconds: float = _HOLD_SECONDS) -> None:
        self._config = config
        self._hold = hold_seconds
        self._last: dict[Axis | Button, float] = {}
        self.quit = False

    def feed(self, data: bytes, now: float) -> None:
        """Parse a read buffer, stamping each recognized action with ``now``.

        Arrow escape sequences are matched first, then remaining bytes are read
        one character at a time against the single-key bindings.
        """
        text = data.decode("latin-1")
        i = 0
        while i < len(text):
            seq = text[i : i + 3]
            if seq in _ARROW_BINDINGS:
                self._last[_ARROW_BINDINGS[seq]] = now
                i += 3
                continue
            ch = text[i]
            if ch in _QUIT_KEYS:
                self.quit = True
            else:
                action = _KEY_BINDINGS.get(ch.lower() if ch.isalpha() else ch)
                if action is not None:
                    self._last[action] = now
            i += 1

    def _active(self, now: float) -> set[Axis | Button]:
        return {a for a, t in self._last.items() if now - t < self._hold}

    def state(self, now: float) -> ControllerState:
        """Build the controller state from actions still within their hold window."""
        active = self._active(now)
        buttons = Button(0)
        for action in active:
            if isinstance(action, Button):
                buttons |= action

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

        return ControllerState(stick_x=stick_x, stick_y=stick_y, buttons=buttons)


def run_keyboard(link: ControllerLink, config: Config) -> None:
    """Read the keyboard and send controller frames until the user quits.

    Puts stdin into cbreak mode, reads any pending bytes each tick, and emits a
    ``ControllerState`` paced to ``config.target_fps``. On exit (quit, EOF, or
    Ctrl-C) the terminal is restored and the controller released to neutral.
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
    print("Driving -- press q to quit.\n", flush=True)
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
                btns = str(state.buttons).removeprefix("Button.")
                print(
                    f"\rstick=({state.stick_x:+4d},{state.stick_y:+4d}) buttons={btns:<24}",
                    end="",
                    flush=True,
                )
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
