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

A single colour filling the box is *necessary* but not sufficient: a kart driving
forward over a uniform-coloured stretch (or rainbow seen up close, where one stripe
fills the near box) trips the dominant-hue test too, yet it is not stuck. The tell is
**motion** — a wedged kart makes no forward progress, so its view freezes, while a
driving kart's view keeps flowing. So the rail/void verdict is gated on the front box
being *frozen* frame-to-frame (mean abs grayscale diff below ``stuck_static_max_diff``).

If the rainbow stays missing **and the view stays frozen** for ``stuck_frames``
consecutive frames the kart is wedged against a rail, and recovery kicks in:

    RECOVER:  hold A (keep driving) + full **right** stick, until the rainbow
              reappears (or the view starts flowing) for ``recovery_clear_frames``
              consecutive frames.

We deliberately do **not** reverse out of the rail — backing off a Rainbow Road rail
tends to drop the kart off the edge. Powering forward while steering hard right
scrubs the kart along the rail and back onto the ribbon.

The same detector also catches driving the **wrong way along the ribbon**. Rainbow
Road's stripes run across the track, so driven forward their colors climb the hue
circle from near to far — blue → violet → red → orange → yellow → green. Driven
backwards that order flips. ``_direction_gradient`` sums the signed near→far hue
steps; a clearly negative sum (with the rainbow still present) means Mario is facing
backwards, which feeds the *same* forward + hard-right recovery — never reverse off
Rainbow Road. Crucially, a reversed rainbow is still *present*, so recovery cannot end
just because "the rainbow is back": it ends only once the colors read forward again.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

import cv2
import numpy as np

from .config import Config
from .control import Button, ControllerState
from .steering import SteeringDecision


def _circ_delta(a: float, b: float) -> float:
    """Signed shortest step from hue ``a`` to hue ``b`` on OpenCV's 0..180 hue circle,
    in (-90, +90]. Positive = up the circle (the forward driving direction), negative
    = back. Wraps across the 179/0 (red) seam the short way."""
    return (b - a + 90.0) % 180.0 - 90.0


class _Phase(Enum):
    NORMAL = auto()
    RECOVER = auto()
    TURN_AROUND = auto()


@dataclass
class StuckDetector:
    config: Config
    _phase: _Phase = field(default=_Phase.NORMAL, init=False)
    _absent_streak: int = field(default=0, init=False)   # consecutive frames front view bad
    _present_streak: int = field(default=0, init=False)  # consecutive frames front view ok
    _last_gradient: float = field(default=0.0, init=False)   # last near->far hue gradient
    _last_dir_valid: bool = field(default=False, init=False)  # was that gradient readable
    _last_reversed: bool = field(default=False, init=False)   # did it read wrong-way
    _prev_front: np.ndarray | None = field(default=None, init=False)  # last front-box gray
    _last_motion: float = field(default=0.0, init=False)      # last front-box frame diff
    _recover_streak: int = field(default=0, init=False)         # frames spent in RECOVER
    _turnaround_frames_left: int = field(default=0, init=False)  # frames remaining in 180
    _prev_floor_hue: float | None = field(default=None, init=False)
    _frozen_hue_streak: int = field(default=0, init=False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, frame: np.ndarray, decision: SteeringDecision) -> ControllerState | None:
        """Feed one frame. ``decision`` is accepted for API symmetry but unused —
        the model is purely the color of the road ahead.

        Returns a recovery ``ControllerState`` while wedged against a rail or facing
        the wrong way, or ``None`` to let the normal ``drive_policy`` take over.
        """
        cfg = self.config
        wedged, good_road = self._assess(frame)

        # Track floor hue freeze in all phases.
        self._update_floor_hue(frame)

        if self._phase is _Phase.NORMAL:
            # Arm only on a real wedge: a frozen rail/void, or facing backwards.
            self._absent_streak = self._absent_streak + 1 if wedged else 0
            if self._absent_streak >= max(1, cfg.stuck_frames):
                self._enter_recover()
                return self._recovery_state()
            # Trigger 180 if floor hue has been frozen for too long (kart going in circles).
            if self._frozen_hue_streak >= max(1, cfg.turnaround_hue_frames):
                self._enter_turnaround()
                return self._turnaround_state()
            return None

        if self._phase is _Phase.TURN_AROUND:
            self._turnaround_frames_left -= 1
            if self._turnaround_frames_left <= 0:
                self._exit_turnaround()
                return None
            return self._turnaround_state()

        # _Phase.RECOVER — keep driving + hard right until the road clears.
        # If recovery has been going too long without clearing, escalate to 180.
        self._recover_streak += 1
        if self._recover_streak >= max(1, cfg.turnaround_after_recovery_frames):
            self._enter_turnaround()
            return self._turnaround_state()
        self._present_streak = self._present_streak + 1 if good_road else 0
        if self._present_streak >= max(1, cfg.recovery_clear_frames):
            self._exit_recover()
            return None
        return self._recovery_state()

    @property
    def is_recovering(self) -> bool:
        return self._phase is _Phase.RECOVER

    @property
    def last_gradient(self) -> float:
        """Most recent near→far hue gradient (signed; +forward, -reversed)."""
        return self._last_gradient

    @property
    def last_motion(self) -> float:
        """Most recent front-box frame-to-frame mean abs diff (low = frozen/wedged)."""
        return self._last_motion

    @property
    def last_direction(self) -> str:
        """Compact tag for HUD/debug: ``FWD`` / ``REV`` / ``--`` (unreadable)."""
        if not self._last_dir_valid:
            return "--"
        if self._last_reversed:
            return "REV"
        if self._last_gradient >= self.config.wrong_way_min_gradient:
            return "FWD"
        return "--"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _recovery_state(self) -> ControllerState:
        """Keep accelerating (A) and steer right at recovery_stick strength."""
        return ControllerState(stick_x=self.config.recovery_stick, stick_y=0, buttons=Button.A)

    def _turnaround_state(self) -> ControllerState:
        """Reverse (negative stick_y) and steer hard right to spin 180 degrees."""
        return ControllerState(stick_x=self.config.turnaround_stick, stick_y=-self.config.turnaround_stick, buttons=Button.A)

    def _update_floor_hue(self, frame: np.ndarray) -> None:
        """Increment frozen_hue_streak if the floor median hue hasn't moved."""
        cfg = self.config
        roi = self._front_roi(frame)
        if roi.size == 0:
            return
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        colored = h[(s >= cfg.stuck_min_sat) & (v >= cfg.stuck_min_val)]
        if colored.size == 0:
            self._prev_floor_hue = None
            self._frozen_hue_streak = 0
            return
        hue = float(np.median(colored))
        if self._prev_floor_hue is not None:
            diff = abs(_circ_delta(self._prev_floor_hue, hue))
            if diff < cfg.turnaround_hue_diff:
                self._frozen_hue_streak += 1
            else:
                self._frozen_hue_streak = 0
        self._prev_floor_hue = hue

    def _enter_turnaround(self) -> None:
        self._phase = _Phase.TURN_AROUND
        self._recover_streak = 0
        self._present_streak = 0
        self._frozen_hue_streak = 0
        self._turnaround_frames_left = self.config.turnaround_frames

    def _exit_turnaround(self) -> None:
        self._phase = _Phase.NORMAL
        self._absent_streak = 0
        self._recover_streak = 0
        self._frozen_hue_streak = 0

    def _assess(self, frame: np.ndarray) -> tuple[bool, bool]:
        """Read the road ahead once and return ``(wedged, good_road)``.

        ``wedged`` (recovery *enter* condition): a missing rainbow (rail/void) while the
        view is **frozen**, or a clearly reversed rainbow. The freeze test is what tells
        a real wedge from driving over a uniform-coloured stretch, which trips the
        dominant-hue test but keeps the image flowing.

        ``good_road`` (recovery *exit* condition): the rainbow is present and running
        forward. Deliberately asymmetric with ``wedged`` — recovery should end only when
        the real track is back, not merely because the kart is sliding along the rail.

        Records the direction and motion readings for the HUD as a side effect.
        """
        cfg = self.config
        rainbow_ahead = self._rainbow_ahead(frame)
        moving = self._update_motion(frame)
        gradient, dir_valid = self._direction_gradient(frame)
        reversed_ahead = (
            rainbow_ahead and dir_valid and gradient <= -cfg.wrong_way_min_gradient
        )
        self._last_gradient = gradient
        self._last_dir_valid = dir_valid
        self._last_reversed = reversed_ahead

        wedged = ((not rainbow_ahead) and not moving) or reversed_ahead
        good_road = rainbow_ahead and not reversed_ahead
        return wedged, good_road

    def _update_motion(self, frame: np.ndarray) -> bool:
        """True if the front look-ahead box is changing frame-to-frame (the kart is
        making forward progress), False if it is frozen (wedged). The first frame has
        no predecessor so it reads as frozen — safe, because a real wedge still needs
        ``stuck_frames`` consecutive frozen frames to arm recovery."""
        gray = cv2.cvtColor(self._front_roi(frame), cv2.COLOR_BGR2GRAY)
        prev = self._prev_front
        self._prev_front = gray
        if prev is None or prev.shape != gray.shape or gray.size == 0:
            self._last_motion = 0.0
            return False
        self._last_motion = float(cv2.absdiff(gray, prev).mean())
        return self._last_motion > self.config.stuck_static_max_diff

    def _direction_gradient(self, frame: np.ndarray) -> tuple[float, bool]:
        """Sum the signed near→far circular hue steps across the wrong-way strip.

        Returns ``(gradient, valid)``. ``gradient`` walks band medians from the near
        edge (bottom) to the far edge (top): positive climbs the hue circle (forward),
        negative descends it (reversed). Steps are taken only between *adjacent*
        contributing bands, so a dropped (under-colored) band never bridges two
        far-apart hues whose shortest-path step could point the wrong way. ``valid``
        is False when too few bands carry enough color to judge a direction at all.
        """
        cfg = self.config
        roi = self._wrong_way_roi(frame)
        bands = max(2, cfg.wrong_way_bands)
        h = roi.shape[0]
        if roi.size == 0 or h < bands:
            return 0.0, False

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

        # Band medians indexed near (0, bottom rows) -> far (bands-1, top rows).
        band_hue: list[float | None] = []
        for i in range(bands):
            y1 = h - (i * h) // bands
            y0 = h - ((i + 1) * h) // bands
            gate = (sat[y0:y1] >= cfg.stuck_min_sat) & (val[y0:y1] >= cfg.stuck_min_val)
            if gate.sum() < cfg.wrong_way_min_band_frac * gate.size:
                band_hue.append(None)
            else:
                band_hue.append(float(np.median(hue[y0:y1][gate])))

        gradient = 0.0
        steps = 0
        for near, far in zip(band_hue, band_hue[1:]):
            if near is None or far is None:
                continue
            gradient += _circ_delta(near, far)
            steps += 1
        if steps < max(1, cfg.wrong_way_min_bands - 1):
            return 0.0, False
        return gradient, True

    def _wrong_way_roi(self, frame: np.ndarray) -> np.ndarray:
        """Crop to this player's quadrant, then to the near look-ahead strip whose
        vertical hue gradient encodes the driving direction."""
        cfg = self.config
        h, w = frame.shape[:2]
        px0, py0, px1, py1 = cfg.player_region()
        sub = frame[int(h * py0):int(h * py1), int(w * px0):int(w * px1)]
        sh, sw = sub.shape[:2]
        y0, y1 = int(sh * cfg.wrong_way_roi_top), int(sh * cfg.wrong_way_roi_bottom)
        x0, x1 = int(sw * cfg.wrong_way_roi_left), int(sw * cfg.wrong_way_roi_right)
        return sub[y0:y1, x0:x1]

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
        self._recover_streak = 0

    def _exit_recover(self) -> None:
        self._phase = _Phase.NORMAL
        self._absent_streak = 0
        self._present_streak = 0
