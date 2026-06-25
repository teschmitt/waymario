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

    # --- control ---
    max_stick: int = 80
    """Peak N64 analog magnitude to command (full deflection is ~80-84)."""

    # --- transport ---
    serial_port: str = "/dev/ttyACM0"
    baud: int = 115200

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
            )
