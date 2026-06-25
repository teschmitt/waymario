"""Orchestration loop: capture -> steer -> policy -> send."""

from __future__ import annotations

import time

from .capture import FrameSource
from .config import Config
from .control import ControllerState, drive_policy
from .steering import Steerer
from .transport import ControllerLink


def run(
    source: FrameSource,
    steerer: Steerer,
    link: ControllerLink,
    config: Config,
) -> None:
    """Drive until the source is exhausted or interrupted.

    On exit (including Ctrl-C) the controller is released to neutral so the kart
    doesn't keep its last command.
    """
    frame_period = 1.0 / config.target_fps if config.target_fps > 0 else 0.0
    try:
        for frame in source.frames():
            start = time.monotonic()

            decision = steerer.decide(frame)
            state = drive_policy(decision, config)
            link.send(state)

            if frame_period:
                elapsed = time.monotonic() - start
                if elapsed < frame_period:
                    time.sleep(frame_period - elapsed)
    except KeyboardInterrupt:
        pass
    finally:
        link.send(ControllerState())  # neutral
