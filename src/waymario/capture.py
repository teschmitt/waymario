"""Frame sources.

A ``FrameSource`` yields BGR frames. ``CaptureDeviceSource`` reads the live HDMI
capture dongle; ``VideoFileSource`` replays a recording so the rest of the
pipeline runs with no hardware attached.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator

import cv2
import numpy as np

Frame = np.ndarray


class FrameSource(ABC):
    @abstractmethod
    def frames(self) -> Iterator[Frame]:
        """Yield BGR frames until the source is exhausted or closed."""

    def close(self) -> None:  # noqa: B027 - optional override
        pass

    def __enter__(self) -> FrameSource:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class CaptureDeviceSource(FrameSource):
    """Live frames from a V4L2 HDMI capture device via OpenCV."""

    def __init__(self, device: int = 0, width: int | None = None, height: int | None = None) -> None:
        self._cap = cv2.VideoCapture(device)
        if not self._cap.isOpened():
            raise RuntimeError(f"could not open capture device {device!r}")
        if width:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def frames(self) -> Iterator[Frame]:
        while True:
            ok, frame = self._cap.read()
            if not ok:
                break
            yield frame

    def close(self) -> None:
        self._cap.release()


class VideoFileSource(FrameSource):
    """Replay a recorded clip; useful for development and tests."""

    def __init__(self, path: str, loop: bool = False) -> None:
        self._path = path
        self._loop = loop
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise RuntimeError(f"could not open video file {path!r}")

    def frames(self) -> Iterator[Frame]:
        while True:
            ok, frame = self._cap.read()
            if not ok:
                if self._loop:
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break
            yield frame

    def close(self) -> None:
        self._cap.release()
