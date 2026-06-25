"""Stuck detection and recovery.

``StuckDetector`` watches two signals every frame:

1. **Frame diff** — the mean absolute difference between the current and
   previous frame (cropped to the player's subframe).  If the image barely
   changes for ``stuck_frames`` consecutive frames the kart is probably not
   moving.

2. **Low confidence streak** — if the steerer can't find the track for
   ``stuck_frames`` consecutive frames the kart is likely facing a wall or
   has fallen off.

Either condition triggers a *recovery sequence*:

    Phase 1 — REVERSE:  hold A + stick-Y down for ``recovery_reverse_frames``
    Phase 2 — TURN:     keep backing up + full stick-X for ``recovery_turn_frames``
    Phase 3 — RESUME:   hand back to normal drive_policy

Mario Kart 64 has no reverse button — backing up is done by holding A (the
gas must stay pressed) and pulling the analog stick full down, so recovery
never touches B.  The turn direction alternates each recovery so the bot
doesn't spin forever on the same side.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

import cv2
import numpy as np

from .config import Config
from .control import Button, ControllerState
from .steering import SteeringDecision


class _Phase(Enum):
    NORMAL = auto()
    REVERSE = auto()
    TURN = auto()


@dataclass
class StuckDetector:
    config: Config
    _phase: _Phase = field(default=_Phase.NORMAL, init=False)
    _phase_frames: int = field(default=0, init=False)
    _low_conf_streak: int = field(default=0, init=False)
    _diff_streak: int = field(default=0, init=False)
    _prev_gray: np.ndarray | None = field(default=None, init=False)
    _recovery_count: int = field(default=0, init=False)  # how many recoveries so far
    _turn_direction: int = field(default=1, init=False)  # +1 right, -1 left

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, frame: np.ndarray, decision: SteeringDecision) -> ControllerState | None:
        """Feed one frame + steering decision.

        Returns:
            A recovery ``ControllerState`` while stuck, or ``None`` to let
            the normal ``drive_policy`` take over.
        """
        cfg = self.config
        subframe = self._crop_subframe(frame)
        gray = cv2.cvtColor(subframe, cv2.COLOR_BGR2GRAY)

        # --- update streak counters (only in NORMAL phase) ---
        if self._phase is _Phase.NORMAL:
            # frame diff streak
            if self._prev_gray is not None and self._prev_gray.shape == gray.shape:
                diff = float(np.mean(np.abs(gray.astype(np.int16) - self._prev_gray.astype(np.int16))))
                if diff < cfg.stuck_frame_diff_threshold:
                    self._diff_streak += 1
                else:
                    self._diff_streak = 0
            # low-confidence streak
            if decision.confidence < cfg.min_confidence:
                self._low_conf_streak += 1
            else:
                self._low_conf_streak = 0

            if self._diff_streak >= cfg.stuck_frames or self._low_conf_streak >= cfg.stuck_frames:
                self._start_recovery()

        self._prev_gray = gray

        # --- drive recovery phases ---
        if self._phase is _Phase.NORMAL:
            return None  # normal driving

        if self._phase is _Phase.REVERSE:
            # No reverse button on the N64 pad — back up by holding A (the
            # gas, which must stay pressed to reverse) plus the analog stick
            # straight down.
            state = ControllerState(stick_x=0, stick_y=-cfg.max_stick, buttons=Button.A)
            self._phase_frames += 1
            if self._phase_frames > cfg.recovery_reverse_frames:
                self._phase = _Phase.TURN
                self._phase_frames = 0
            return state

        if self._phase is _Phase.TURN:
            # Keep reversing (A + stick-Y down) while steering hard to one side
            # so the kart backs away from the wall at an angle.
            state = ControllerState(
                stick_x=self._turn_direction * cfg.max_stick,
                stick_y=-cfg.max_stick,
                buttons=Button.A,
            )
            self._phase_frames += 1
            if self._phase_frames > cfg.recovery_turn_frames:
                self._end_recovery()
            return state

        return None  # should never reach here

    @property
    def is_recovering(self) -> bool:
        return self._phase is not _Phase.NORMAL

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _crop_subframe(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        px0, py0, px1, py1 = self.config.player_region()
        return frame[int(h * py0):int(h * py1), int(w * px0):int(w * px1)]

    def _start_recovery(self) -> None:
        self._phase = _Phase.REVERSE
        self._phase_frames = 0
        self._diff_streak = 0
        self._low_conf_streak = 0
        self._recovery_count += 1
        # alternate turn direction each recovery
        self._turn_direction = 1 if self._recovery_count % 2 == 1 else -1

    def _end_recovery(self) -> None:
        self._phase = _Phase.NORMAL
        self._phase_frames = 0
        self._prev_gray = None  # reset diff so we don't immediately re-trigger