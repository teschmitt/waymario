"""Orchestration loop: capture -> steer -> policy -> send."""

from __future__ import annotations

import time

from .capture import FrameSource
from .config import Config
from .control import ControllerState, drive_policy
from .steering import Steerer
from .stuck import StuckDetector
from .transport import ControllerLink

_DEBUG_EVERY = 10  # print one line every N frames to avoid flooding the terminal


def run(
    source: FrameSource,
    steerer: Steerer,
    link: ControllerLink,
    config: Config,
    debug: bool = False,
) -> None:
    """Drive until the source is exhausted or interrupted.

    On exit (including Ctrl-C) the controller is released to neutral so the kart
    doesn't keep its last command.
    """
    frame_period = 1.0 / config.target_fps if config.target_fps > 0 else 0.0
    stuck = StuckDetector(config)
    frame_no = 0
    try:
        for frame in source.frames():
            start = time.monotonic()
            frame_no += 1

            decision = steerer.decide(frame)
            # Let the stuck detector override the normal policy if recovering.
            recovery = stuck.update(frame, decision)
            state = recovery if recovery is not None else drive_policy(decision, config)
            link.send(state)

            if debug and frame_no % _DEBUG_EVERY == 0:
                mode = "RECOVER" if recovery is not None else "DRIVE  "
                phase = stuck._phase.name
                print(
                    f"[{frame_no:6d}] {mode} | "
                    f"conf={decision.confidence:.3f} "
                    f"steer={decision.steering:+.2f} "
                    f"stick_x={state.stick_x:+4d} stick_y={state.stick_y:+4d} "
                    f"phase={phase}",
                    flush=True,
                )

            if frame_period:
                elapsed = time.monotonic() - start
                if elapsed < frame_period:
                    time.sleep(frame_period - elapsed)
    except KeyboardInterrupt:
        pass
    finally:
        link.send(ControllerState())  # neutral
