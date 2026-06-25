"""Tunable settings for the waymario pipeline.

One flat dataclass so every knob lives in one place and can be overridden from
the CLI or a future config file without threading args through the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass

# Multiplayer screen layout (1-indexed, matches the controller port):
#
#   1-player  : full screen
#   2-player  : P1 left half, P2 right half
#   3/4-player: P1 top-left, P2 top-right, P3 bot-left, P4 bot-right
#
# Each entry is (x_start, y_start, x_end, y_end) as fractions of the full frame.
_PLAYER_REGIONS: dict[int, dict[int, tuple[float, float, float, float]]] = {
    1: {1: (0.0, 0.0, 1.0, 1.0)},
    2: {
        1: (0.0, 0.0, 0.5, 1.0),  # left half
        2: (0.5, 0.0, 1.0, 1.0),  # right half
    },
    3: {
        1: (0.0, 0.0, 0.5, 0.5),  # top-left
        2: (0.5, 0.0, 1.0, 0.5),  # top-right
        3: (0.0, 0.5, 1.0, 1.0),  # bottom full-width strip (real MK64 3p layout)
    },
    4: {
        1: (0.0, 0.0, 0.5, 0.5),  # top-left
        2: (0.5, 0.0, 1.0, 0.5),  # top-right
        3: (0.0, 0.5, 0.5, 1.0),  # bot-left
        4: (0.5, 0.5, 1.0, 1.0),  # bot-right
    },
}


@dataclass
class Config:
    # --- multiplayer ---
    players: int = 1
    """Total number of players (1-4). Determines the screen-split layout."""
    player: int = 1
    """Which player slot this bot occupies (1-based). Selects the screen quadrant."""

    # --- capture ---
    device: int = 0
    """V4L2 index of the HDMI capture dongle."""
    width: int | None = None
    height: int | None = None

    # --- steering / vision ---
    steerer: str = "hsv"
    """Which steering algorithm to use: "hsv" (color-band) or "brightness" (centroid)."""
    roi_top: float = 0.45
    """Fraction of the *player's sub-frame* height where the ROI starts (look ahead,
    ignore the sky/HUD above)."""
    roi_bottom: float = 0.95
    bright_threshold: int = 60
    """Grayscale value above which a pixel counts as 'track' rather than the
    black starfield."""
    steering_gain: float = 2.5
    """Maps normalized centroid offset (-1..1 across the ROI) to a steering
    command, before clamping to [-1, 1]."""
    min_confidence: float = 0.01
    """Minimum fraction of ROI lit up to trust the centroid; below this we coast
    straight rather than chase noise."""

    # --- HSV color-band steering ---
    hue_left: float = 5.0
    """OpenCV hue (0..179) at the track's left/red edge -> e_y = -1."""
    hue_right: float = 140.0
    """OpenCV hue at the track's right/purple edge -> e_y = +1."""
    hue_gain: float = 1.0
    """Maps normalized cross-track error e_y (-1..1) to a steering command."""
    hue_min_sat: int = 60
    """Minimum saturation for a pixel to count as colored track."""
    hue_min_val: int = 60
    """Minimum value/brightness for a pixel to count as colored track."""
    hue_patch_cx: float = 0.5
    """Look-ahead patch center x, fraction of the player sub-frame width."""
    hue_patch_cy: float = 0.62
    """Look-ahead patch center y, fraction of sub-frame height (above the tires)."""
    hue_patch_w: float = 0.12
    """Patch width, fraction of sub-frame width."""
    hue_patch_h: float = 0.10
    """Patch height, fraction of sub-frame height."""

    # --- control ---
    max_stick: int = 80
    """Peak N64 analog magnitude to command (full deflection is ~80-84)."""

    # --- transport ---
    serial_port: str = "/dev/ttyUSB0"
    baud: int = 115200
    daemon_port: int = 9999
    """Default TCP port for the controller daemon (``waymario daemon``) and the
    keyboard client that connects to it."""

    # --- stuck detection & recovery ---
    stuck_frames: int = 90
    """Consecutive frames of no motion *or* no track before recovery triggers.
    At 60 fps this is 1.5 s."""
    stuck_frame_diff_threshold: float = 2.0
    """Mean absolute pixel difference below which a frame counts as 'no motion'.
    0 = pixel-perfect identical; ~2 tolerates compression noise."""
    recovery_reverse_frames: int = 60
    """Frames to hold B (reverse) at the start of recovery (~1 s at 60 fps)."""
    recovery_turn_frames: int = 45
    """Frames to hold B + full stick while turning out of the wall (~0.75 s)."""

    # --- loop ---
    target_fps: float = 60.0

    def player_region(self) -> tuple[float, float, float, float]:
        """Return (x0, y0, x1, y1) as fractions of the full frame for this player's quadrant."""
        try:
            return _PLAYER_REGIONS[self.players][self.player]
        except KeyError:
            raise ValueError(
                f"Invalid combination: players={self.players}, player={self.player}. "
                f"Valid players values are 1-4 and player must be within that range."
            ) from None
