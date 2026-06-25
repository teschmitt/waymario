"""End-to-end smoke test: synthetic frame source -> steer -> NullLink."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from waymario.capture import FrameSource
from waymario.config import Config
from waymario.control import Button
from waymario.drive import run
from waymario.steering import OpenCVSteerer
from waymario.transport import NullLink


class _StubSource(FrameSource):
    def __init__(self, frames: list[np.ndarray]) -> None:
        self._frames = frames

    def frames(self) -> Iterator[np.ndarray]:
        yield from self._frames


def test_pipeline_runs_and_neutralizes_on_exit() -> None:
    # Track on the right -> should command a positive (right) stick while driving.
    frame = np.zeros((200, 640, 3), dtype=np.uint8)
    frame[:, 500:560] = 255

    config = Config(target_fps=0)  # no sleep
    link = NullLink()
    run(_StubSource([frame, frame, frame]), OpenCVSteerer(config), link, config)

    # Final send is the neutral release in the finally block.
    assert link.count == 4
    assert link.last_state.stick_x == 0
    assert link.last_state.buttons == Button(0)
