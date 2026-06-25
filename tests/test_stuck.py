"""Tests for StuckDetector — the rail-vs-rainbow color model, no hardware needed.

The model: the rainbow track is a spectrum (no single hue dominates the front box),
while a guard rail is one dominant color. See ``stuck.StuckDetector``.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from waymario.config import Config
from waymario.control import Button
from waymario.steering import SteeringDecision
from waymario.stuck import StuckDetector, _circ_delta

_REPO = Path(__file__).resolve().parent.parent
_SAMPLE_NOT_STUCK = _REPO / "sample-not-stuck.png"
_SAMPLE_STUCK = _REPO / "sample-stuck.png"
_SAMPLE2 = _REPO / "sample2.png"


def _config(**kwargs) -> Config:
    """Config with fast timings and whole sub-frame boxes (both the rail box and the
    wrong-way strip), so the synthetic rainbow/rail frames below exercise the detector
    independent of ROI placement."""
    defaults = dict(
        stuck_frames=5,
        recovery_clear_frames=3,
        max_stick=80,
        stuck_roi_left=0.0,
        stuck_roi_right=1.0,
        stuck_roi_top=0.0,
        stuck_roi_bottom=1.0,
        wrong_way_roi_left=0.0,
        wrong_way_roi_right=1.0,
        wrong_way_roi_top=0.0,
        wrong_way_roi_bottom=1.0,
    )
    defaults.update(kwargs)
    return Config(**defaults)


def _decision() -> SteeringDecision:
    # The color model ignores the decision; any value works.
    return SteeringDecision(steering=0.0, confidence=0.5, lateral=0.0)


def _rainbow_frame(h: int = 96, w: int = 96) -> np.ndarray:
    """A saturated hue sweep that climbs near->far (bottom row low hue, top row high)
    = the rainbow track driven the *correct* way (a spectrum where no single hue
    dominates, and whose near->far gradient reads forward). The hue *histogram* is
    the same whichever way the sweep runs, so the rail/dominant-hue tests are
    orientation-independent; only the direction tests care which way it climbs."""
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[:, :, 0] = np.linspace(179, 0, h).astype(np.uint8)[:, None]
    hsv[:, :, 1] = 220
    hsv[:, :, 2] = 200
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _reversed_rainbow_frame(h: int = 96, w: int = 96) -> np.ndarray:
    """The rainbow track driven the *wrong* way: the same spectrum, but its hue
    descends near->far (bottom row high hue, top row low), so the gradient reads
    reversed. Same histogram as ``_rainbow_frame`` -> still 'rainbow present', just
    backwards."""
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[:, :, 0] = np.linspace(0, 179, h).astype(np.uint8)[:, None]
    hsv[:, :, 1] = 220
    hsv[:, :, 2] = 200
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _rail_frame(hue: int = 60, h: int = 96, w: int = 96) -> np.ndarray:
    """A flat single-hue saturated wall = a guard rail filling the view."""
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[:, :, 0] = hue
    hsv[:, :, 1] = 220
    hsv[:, :, 2] = 200
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _void_frame(h: int = 96, w: int = 96) -> np.ndarray:
    """Bare space off the edge of the track."""
    return np.zeros((h, w, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Rainbow track ahead — never recovers
# ---------------------------------------------------------------------------

def test_rainbow_track_never_recovers() -> None:
    cfg = _config()
    det = StuckDetector(cfg)
    frame = _rainbow_frame()
    result: object = "sentinel"
    for _ in range(cfg.stuck_frames * 3):
        result = det.update(frame, _decision())
    assert result is None
    assert not det.is_recovering


# ---------------------------------------------------------------------------
# One color dominating the view ahead triggers recovery
# ---------------------------------------------------------------------------

def test_rail_in_front_triggers_recovery() -> None:
    cfg = _config()
    det = StuckDetector(cfg)
    result = None
    for _ in range(cfg.stuck_frames):
        result = det.update(_rail_frame(), _decision())
    assert det.is_recovering
    assert result is not None


@pytest.mark.parametrize("hue", [10, 25, 60, 120, 160])
def test_any_flat_color_triggers(hue: int) -> None:
    """Whatever color the rail is, a single dominant hue means stuck."""
    cfg = _config()
    det = StuckDetector(cfg)
    for _ in range(cfg.stuck_frames):
        det.update(_rail_frame(hue=hue), _decision())
    assert det.is_recovering


def test_void_ahead_also_triggers() -> None:
    cfg = _config()
    det = StuckDetector(cfg)
    for _ in range(cfg.stuck_frames):
        det.update(_void_frame(), _decision())
    assert det.is_recovering


def test_sparse_color_reads_as_stuck() -> None:
    """A near-black box with a few scattered bright pixels has no real track:
    it must fail safe to stuck, not be fooled into reading 'track ahead'."""
    cfg = _config()
    det = StuckDetector(cfg)
    sparse = _void_frame()
    for i, hue in enumerate(range(0, 170, 17)):  # ~10 distinct-hue bright pixels
        sparse[i, i] = cv2.cvtColor(
            np.array([[[hue, 220, 200]]], np.uint8), cv2.COLOR_HSV2BGR
        )[0, 0]
    for _ in range(cfg.stuck_frames):
        det.update(sparse, _decision())
    assert det.is_recovering


def test_multi_colored_scene_is_not_track_if_one_color_dominates() -> None:
    """A stuck scene can still contain several colors (rail + stars + kart). As
    long as one color dominates the box, it is not the rainbow's even spectrum."""
    cfg = _config()
    det = StuckDetector(cfg)
    frame = _rail_frame(hue=60)          # mostly green wall
    frame[:, :12] = _rail_frame(hue=25)[:, :12]  # a gold sliver on the side
    for _ in range(cfg.stuck_frames):
        det.update(frame, _decision())
    assert det.is_recovering


# ---------------------------------------------------------------------------
# Recovery behaviour: keep driving + hard right, never reverse
# ---------------------------------------------------------------------------

def test_recovery_keeps_driving_and_steers_hard_right() -> None:
    cfg = _config()
    det = StuckDetector(cfg)
    state = None
    for _ in range(cfg.stuck_frames):
        state = det.update(_rail_frame(), _decision())
    assert det.is_recovering
    assert state is not None
    assert Button.A in state.buttons       # keep driving
    assert Button.B not in state.buttons    # do NOT back out
    assert state.stick_x == cfg.max_stick   # steer hard right
    assert state.stick_y == 0               # nothing on the reverse axis


def test_recovery_persists_while_rail_present() -> None:
    cfg = _config()
    det = StuckDetector(cfg)
    for _ in range(cfg.stuck_frames):
        det.update(_rail_frame(), _decision())
    assert det.is_recovering
    # keep ramming the rail: stays recovering, keeps commanding hard right
    for _ in range(cfg.stuck_frames * 2):
        state = det.update(_rail_frame(), _decision())
        assert det.is_recovering
        assert state is not None
        assert state.stick_x == cfg.max_stick
        assert Button.B not in state.buttons


# ---------------------------------------------------------------------------
# Recovery exit hysteresis
# ---------------------------------------------------------------------------

def test_recovery_exit_hysteresis_and_holds_command() -> None:
    """Exit takes exactly recovery_clear_frames good frames, and the hard-right
    command keeps holding through the clearing window."""
    cfg = _config(recovery_clear_frames=4)
    det = StuckDetector(cfg)
    for _ in range(cfg.stuck_frames):
        det.update(_rail_frame(), _decision())
    assert det.is_recovering

    # clearing window: clear_frames-1 good frames -> still recovering, still hard right
    for _ in range(cfg.recovery_clear_frames - 1):
        state = det.update(_rainbow_frame(), _decision())
        assert det.is_recovering
        assert state is not None
        assert state.stick_x == cfg.max_stick
        assert state.stick_y == 0
        assert Button.B not in state.buttons
        assert Button.A in state.buttons
    # the final good frame ends recovery
    result = det.update(_rainbow_frame(), _decision())
    assert not det.is_recovering
    assert result is None


def test_recovery_clear_needs_consecutive_frames() -> None:
    """A rail frame interrupting an almost-complete clear streak resets it, so the
    exit genuinely depends on the configured consecutive count."""
    cfg = _config(recovery_clear_frames=4)
    det = StuckDetector(cfg)
    for _ in range(cfg.stuck_frames):
        det.update(_rail_frame(), _decision())
    assert det.is_recovering

    for _ in range(cfg.recovery_clear_frames - 1):   # almost cleared
        det.update(_rainbow_frame(), _decision())
    assert det.is_recovering
    det.update(_rail_frame(), _decision())           # interruption resets the streak
    for _ in range(cfg.recovery_clear_frames - 1):   # not enough on its own
        det.update(_rainbow_frame(), _decision())
    assert det.is_recovering


# ---------------------------------------------------------------------------
# Transient color loss does not trigger
# ---------------------------------------------------------------------------

def test_brief_rainbow_gap_does_not_trigger() -> None:
    cfg = _config()
    det = StuckDetector(cfg)
    for _ in range(cfg.stuck_frames - 1):
        det.update(_rail_frame(), _decision())
    assert not det.is_recovering
    det.update(_rainbow_frame(), _decision())  # rainbow resets the absent streak
    for _ in range(cfg.stuck_frames - 1):
        det.update(_rail_frame(), _decision())
    assert not det.is_recovering


def test_stuck_frames_zero_does_not_trigger_on_track() -> None:
    """A nonsensical stuck_frames=0 must not drop a healthy track into recovery."""
    cfg = _config(stuck_frames=0)
    det = StuckDetector(cfg)
    result: object = "sentinel"
    for _ in range(5):
        result = det.update(_rainbow_frame(), _decision())
    assert not det.is_recovering
    assert result is None


# ---------------------------------------------------------------------------
# Multiplayer: the detector reads its own quadrant, not another player's
# ---------------------------------------------------------------------------

def _quad_frame(p4: np.ndarray, rest: np.ndarray, h: int = 200, w: int = 240) -> np.ndarray:
    """Full frame filled with ``rest``, with player 4's quadrant (bottom-right,
    per config _PLAYER_REGIONS[4][4]) overwritten by ``p4``."""
    frame = rest if rest.shape == (h, w, 3) else cv2.resize(rest, (w, h))
    frame = frame.copy()
    qh, qw = h - h // 2, w - w // 2
    frame[h // 2:, w // 2:] = cv2.resize(p4, (qw, qh))
    return frame


def test_player4_reads_own_rainbow_quadrant() -> None:
    cfg = _config(players=4, player=4)
    det = StuckDetector(cfg)
    frame = _quad_frame(p4=_rainbow_frame(), rest=_rail_frame())  # P4 sees rainbow
    for _ in range(cfg.stuck_frames * 2):
        det.update(frame, _decision())
    assert not det.is_recovering  # rails in the other quadrants must be ignored


def test_player4_reads_own_rail_quadrant() -> None:
    cfg = _config(players=4, player=4)
    det = StuckDetector(cfg)
    frame = _quad_frame(p4=_rail_frame(), rest=_rainbow_frame())  # P4 sees a wall
    for _ in range(cfg.stuck_frames):
        det.update(frame, _decision())
    assert det.is_recovering  # others' rainbows must not mask P4's rail


# ---------------------------------------------------------------------------
# Grounded on the real game frames, with the default front box + thresholds
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _SAMPLE_NOT_STUCK.exists(), reason="sample not present")
def test_real_not_stuck_sample_reads_as_track() -> None:
    img = cv2.imread(str(_SAMPLE_NOT_STUCK))
    assert img is not None
    cfg = Config(stuck_frames=5)  # default ROI + thresholds
    det = StuckDetector(cfg)
    result: object = "sentinel"
    for _ in range(20):
        result = det.update(img, _decision())
    assert not det.is_recovering
    assert result is None
    # margin guard: the live track's dominant hue sits well below the cutoff, so a
    # later threshold bump toward the live value trips this instead of passing.
    colored, dominant = det._front_color_stats(img)
    assert colored >= cfg.stuck_min_colored_frac
    assert dominant <= cfg.stuck_max_dominant_frac - 0.10


@pytest.mark.skipif(not _SAMPLE_STUCK.exists(), reason="sample not present")
def test_real_stuck_sample_triggers_recovery() -> None:
    img = cv2.imread(str(_SAMPLE_STUCK))
    assert img is not None
    cfg = Config(stuck_frames=5)  # default ROI + thresholds
    det = StuckDetector(cfg)
    state = None
    for _ in range(cfg.stuck_frames):
        state = det.update(img, _decision())
    assert det.is_recovering
    assert state is not None
    assert state.stick_x == cfg.max_stick      # hard right
    assert state.stick_y == 0                  # no reverse on the stick
    assert Button.B not in state.buttons        # not backing out
    assert Button.A in state.buttons            # still driving
    # margin guard: the rail's dominant hue sits clearly above the cutoff.
    _colored, dominant = det._front_color_stats(img)
    assert dominant >= cfg.stuck_max_dominant_frac + 0.05


# ===========================================================================
# Wrong-way (reversed-rainbow) detection
#
# Driven forward, the rainbow's stripe colours climb the hue circle near->far
# (blue -> violet -> red -> orange -> yellow -> green); driven backwards they
# descend. The detector reads the sign of that near->far hue gradient and, when
# clearly reversed, triggers the same forward + hard-right recovery as a rail-ram.
# ===========================================================================

# --- the circular hue-step helper ---------------------------------------------

def test_circ_delta_small_steps_keep_sign() -> None:
    assert _circ_delta(10, 30) == 20      # forward step
    assert _circ_delta(30, 10) == -20     # backward step


def test_circ_delta_wraps_the_short_way() -> None:
    # 170 -> 10 is +20 across the 179/0 seam (red wrap), not -160.
    assert _circ_delta(170, 10) == 20
    assert _circ_delta(10, 170) == -20


def test_circ_delta_is_antisymmetric() -> None:
    for a, b in [(0, 45), (120, 5), (89, 175), (60, 60)]:
        assert _circ_delta(a, b) == -_circ_delta(b, a)


# --- the near->far gradient ----------------------------------------------------

def test_forward_rainbow_gradient_is_positive() -> None:
    det = StuckDetector(_config())
    g, valid = det._direction_gradient(_rainbow_frame())
    assert valid
    assert g >= det.config.wrong_way_min_gradient   # clearly forward


def test_reversed_rainbow_gradient_is_negative() -> None:
    det = StuckDetector(_config())
    g, valid = det._direction_gradient(_reversed_rainbow_frame())
    assert valid
    assert g <= -det.config.wrong_way_min_gradient   # clearly reversed


def test_flat_color_gradient_is_ambiguous_not_reversed() -> None:
    """A single-hue wall has no near->far gradient: |G| stays inside the deadband,
    so the wrong-way path never fires on it (the rail/dominant path handles walls)."""
    det = StuckDetector(_config())
    g, _valid = det._direction_gradient(_rail_frame(hue=60))
    assert abs(g) < det.config.wrong_way_min_gradient


def test_void_gradient_is_invalid() -> None:
    det = StuckDetector(_config())
    _g, valid = det._direction_gradient(_void_frame())
    assert not valid   # no coloured bands -> direction unreadable


# --- triggering & recovery -----------------------------------------------------

def test_reversed_rainbow_triggers_recovery() -> None:
    cfg = _config()
    det = StuckDetector(cfg)
    state = None
    for _ in range(cfg.stuck_frames):
        state = det.update(_reversed_rainbow_frame(), _decision())
    assert det.is_recovering
    assert state is not None
    assert Button.A in state.buttons        # keep driving
    assert Button.B not in state.buttons     # never reverse
    assert state.stick_x == cfg.max_stick    # hard right
    assert state.stick_y == 0


def test_forward_rainbow_is_not_wrong_way() -> None:
    cfg = _config()
    det = StuckDetector(cfg)
    result: object = "sentinel"
    for _ in range(cfg.stuck_frames * 3):
        result = det.update(_rainbow_frame(), _decision())
    assert not det.is_recovering
    assert result is None


def test_reversed_recovery_does_not_clear_while_still_reversed() -> None:
    """The key correctness check: a reversed rainbow is still 'present', so the old
    'rainbow reappeared' exit would wrongly end recovery. Recovery must persist as
    long as the colours stay backwards."""
    cfg = _config()
    det = StuckDetector(cfg)
    for _ in range(cfg.stuck_frames):
        det.update(_reversed_rainbow_frame(), _decision())
    assert det.is_recovering
    for _ in range(cfg.recovery_clear_frames * 3):
        state = det.update(_reversed_rainbow_frame(), _decision())
        assert det.is_recovering            # never clears while reversed
        assert state is not None
        assert state.stick_x == cfg.max_stick


def test_reversed_recovery_clears_once_forward_again() -> None:
    cfg = _config()
    det = StuckDetector(cfg)
    for _ in range(cfg.stuck_frames):
        det.update(_reversed_rainbow_frame(), _decision())
    assert det.is_recovering
    # turning back to the correct direction clears recovery after the hysteresis
    result: object = "sentinel"
    for _ in range(cfg.recovery_clear_frames):
        result = det.update(_rainbow_frame(), _decision())
    assert not det.is_recovering
    assert result is None


def test_brief_wrong_way_blip_does_not_trigger() -> None:
    """A single reversed frame inside forward driving must not arm recovery."""
    cfg = _config()
    det = StuckDetector(cfg)
    for _ in range(cfg.stuck_frames - 1):
        det.update(_reversed_rainbow_frame(), _decision())
    assert not det.is_recovering
    det.update(_rainbow_frame(), _decision())          # forward resets the streak
    for _ in range(cfg.stuck_frames - 1):
        det.update(_reversed_rainbow_frame(), _decision())
    assert not det.is_recovering


def test_wrong_way_reads_own_quadrant() -> None:
    """Player 4 reading a reversed rainbow recovers even while other quadrants drive
    forward — direction is judged per-quadrant like everything else."""
    cfg = _config(players=4, player=4)
    det = StuckDetector(cfg)
    frame = _quad_frame(p4=_reversed_rainbow_frame(), rest=_rainbow_frame())
    for _ in range(cfg.stuck_frames):
        det.update(frame, _decision())
    assert det.is_recovering


# --- grounded on the real forward sample --------------------------------------

@pytest.mark.skipif(not _SAMPLE2.exists(), reason="sample not present")
def test_real_forward_sample_reads_forward() -> None:
    """sample2.png is Mario driving the correct way; its near->far hue gradient must
    read forward (positive) with the default ROI/thresholds. No real reversed-rainbow
    frame exists yet, so the reversed direction is covered only synthetically above."""
    img = cv2.imread(str(_SAMPLE2))
    assert img is not None
    det = StuckDetector(Config())   # default ROI + thresholds
    g, valid = det._direction_gradient(img)
    assert valid
    assert g > 0
