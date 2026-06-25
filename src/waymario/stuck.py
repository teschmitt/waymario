"""Stuck detection and recovery.

Mario Kart 64's Rainbow Road is a ribbon of brightly colored stripes — a *spectrum*
of hues, so no single color dominates the view ahead. Its guard rails are a single
color (a green / gold-star barrier). When the kart rams a rail head-on, that one
color fills the look-ahead view, so the patch in front of Mario stops looking like a
rainbow and becomes dominated by one hue.

``StuckDetector`` inspects a small front look-ahead box every frame and measures, of
the colored (saturated, bright) pixels, the fraction that fall in the single most
common hue bucket:

* **Track ahead** — the rainbow spreads across many hue buckets, so the dominant
  bucket holds only ~20-25%.
* **Rail ahead**  — one color fills the box, so the dominant bucket holds ~50-100%
  (and bare space off the edge has almost no colored pixels at all).

Counting *distinct* colors is not enough: a stuck scene still contains several
colors (the green rail, gold stars, the kart, the red "REVERSE" prompt). What sets
the rainbow apart is that its colors are spread *evenly* — no one of them dominates.

If the rainbow stays missing for ``stuck_frames`` consecutive frames the kart is
wedged against a rail, and recovery kicks in:

    RECOVER:  hold A (keep driving) + full **right** stick, until the rainbow
              reappears for ``recovery_clear_frames`` consecutive frames.

We deliberately do **not** reverse out of the rail — backing off a Rainbow Road rail
tends to drop the kart off the edge. Powering forward while steering hard right
scrubs the kart along the rail and back onto the ribbon.
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
    RECOVER = auto()


@dataclass
class StuckDetector:
    config: Config
    _phase: _Phase = field(default=_Phase.NORMAL, init=False)
    _absent_streak: int = field(default=0, init=False)   # consecutive frames rainbow missing
    _present_streak: int = field(default=0, init=False)  # consecutive frames rainbow back

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, frame: np.ndarray, decision: SteeringDecision) -> ControllerState | None:
        """Feed one frame. ``decision`` is accepted for API symmetry but unused —
        the model is purely the color of the road ahead.

        Returns a recovery ``ControllerState`` while wedged against a rail, or
        ``None`` to let the normal ``drive_policy`` take over.
        """
        cfg = self.config
        rainbow_ahead = self._rainbow_ahead(frame)

        if self._phase is _Phase.NORMAL:
            if rainbow_ahead:
                self._absent_streak = 0
            else:
                self._absent_streak += 1
            if self._absent_streak >= max(1, cfg.stuck_frames):
                self._enter_recover()
                return self._recovery_state()
            return None

        # _Phase.RECOVER — keep driving + hard right until the rainbow is back.
        if rainbow_ahead:
            self._present_streak += 1
        else:
            self._present_streak = 0
        if self._present_streak >= max(1, cfg.recovery_clear_frames):
            self._exit_recover()
            return None
        return self._recovery_state()

    @property
    def is_recovering(self) -> bool:
        return self._phase is _Phase.RECOVER

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _recovery_state(self) -> ControllerState:
        """Keep accelerating (A) and steer hard right; never reverse."""
        return ControllerState(stick_x=self.config.max_stick, stick_y=0, buttons=Button.A)

    def _rainbow_ahead(self, frame: np.ndarray) -> bool:
        """True if the front box shows the rainbow track (a many-hued spectrum),
        False if a single color (a guard rail) or empty space fills the view."""
        cfg = self.config
        colored_frac, dominant_frac = self._front_color_stats(frame)
        if colored_frac < cfg.stuck_min_colored_frac:
            return False  # almost no track color ahead — void off the edge / too dim
        return dominant_frac <= cfg.stuck_max_dominant_frac

    def _front_color_stats(self, frame: np.ndarray) -> tuple[float, float]:
        """Measure the front look-ahead box and return ``(colored_frac, dominant_frac)``.

        ``colored_frac`` is the fraction of the box that is saturated/bright enough
        to be track rather than the black starfield. ``dominant_frac`` is, of those
        colored pixels, the fraction in the single most-populated hue bucket — low
        for the rainbow's spectrum, high for a one-color rail.
        """
        cfg = self.config
        roi = self._front_roi(frame)
        if roi.size == 0:
            return 0.0, 1.0
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        colored = hue[(sat >= cfg.stuck_min_sat) & (val >= cfg.stuck_min_val)]
        total = hue.size
        if total == 0 or colored.size == 0:
            return 0.0, 1.0
        hist, _ = np.histogram(colored, bins=cfg.stuck_hue_bins, range=(0, 180))
        return colored.size / total, float(hist.max()) / colored.size

    def _front_roi(self, frame: np.ndarray) -> np.ndarray:
        """Crop to this player's quadrant, then to the front look-ahead box."""
        cfg = self.config
        h, w = frame.shape[:2]
        px0, py0, px1, py1 = cfg.player_region()
        sub = frame[int(h * py0):int(h * py1), int(w * px0):int(w * px1)]
        sh, sw = sub.shape[:2]
        y0, y1 = int(sh * cfg.stuck_roi_top), int(sh * cfg.stuck_roi_bottom)
        x0, x1 = int(sw * cfg.stuck_roi_left), int(sw * cfg.stuck_roi_right)
        return sub[y0:y1, x0:x1]

    def _enter_recover(self) -> None:
        self._phase = _Phase.RECOVER
        self._absent_streak = 0
        self._present_streak = 0

    def _exit_recover(self) -> None:
        self._phase = _Phase.NORMAL
        self._absent_streak = 0
        self._present_streak = 0
