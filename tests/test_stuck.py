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
from waymario.stuck import StuckDetector

_REPO = Path(__file__).resolve().parent.parent
_SAMPLE_NOT_STUCK = _REPO / "sample-not-stuck.png"
_SAMPLE_STUCK = _REPO / "sample-stuck.png"


def _config(**kwargs) -> Config:
    """Config with fast timings and a whole sub-frame box, so the synthetic
    rainbow/rail frames below exercise the detector independent of ROI placement."""
    defaults = dict(
        stuck_frames=5,
        recovery_clear_frames=3,
        max_stick=80,
        stuck_roi_left=0.0,
        stuck_roi_right=1.0,
        stuck_roi_top=0.0,
        stuck_roi_bottom=1.0,
    )
    defaults.update(kwargs)
    return Config(**defaults)


def _decision() -> SteeringDecision:
    # The color model ignores the decision; any value works.
    return SteeringDecision(steering=0.0, confidence=0.5, lateral=0.0)


def _rainbow_frame(h: int = 96, w: int = 96) -> np.ndarray:
    """A saturated full 0..179 hue sweep down the rows = the rainbow track
    (a spectrum where no single hue dominates)."""
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
