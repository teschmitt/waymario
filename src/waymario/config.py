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
    """Which steering algorithm to use. "hsv" (default: centroid of the colour-masked
    rainbow ribbon — saturation gating rejects the HUD/flames a brightness threshold
    catches), "brightness" (centroid of the lit track), or "straight" (debug: no
    steering). Both hsv and brightness steer from a track-centroid offset; they differ
    only in how they segment the track."""
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
    # The HSV steerer reuses roi_top/roi_bottom (the look-ahead band) and
    # steering_gain (centroid offset -> steering); these two gates define which
    # pixels count as coloured rainbow track for its centroid.
    hue_min_sat: int = 60
    """Minimum HSV saturation for a pixel to count as coloured rainbow track —
    rejects the desaturated HUD text / white boost flames a brightness threshold
    would otherwise pull the centroid toward."""
    hue_min_val: int = 60
    """Minimum HSV value/brightness for a pixel to count as coloured track."""

    # --- control ---
    max_stick: int = 80
    """Peak N64 analog magnitude to command (full deflection is ~80-84)."""

    # --- transport ---
    serial_port: str = "/dev/ttyUSB0"
    baud: int = 115200
    daemon_port: int = 9999
    """Default TCP port for the controller daemon (``waymario daemon``) and the
    keyboard client that connects to it."""

    # --- stuck detection (rail vs rainbow) & recovery ---
    stuck_frames: int = 30
    """Consecutive frames with the rainbow track missing from the front look-ahead
    box (a guard rail filling the view) before recovery triggers. ~0.5 s at 60 fps."""
    stuck_roi_left: float = 0.35
    stuck_roi_right: float = 0.65
    stuck_roi_top: float = 0.55
    stuck_roi_bottom: float = 0.82
    """Front look-ahead box (fractions of the player sub-frame) inspected for the
    rainbow track. When Mario rams a rail head-on this box fills with the rail's
    single color instead of the rainbow's many-hued stripes."""
    stuck_min_sat: int = 60
    """Minimum HSV saturation for a pixel to count as colored track rather than
    the black starfield / dim background."""
    stuck_min_val: int = 60
    """Minimum HSV value/brightness for a pixel to count as colored track."""
    stuck_hue_bins: int = 12
    """Hue buckets across OpenCV's 0..179 range. The rainbow spreads its colored
    pixels across many buckets; a single-colored rail piles them into one."""
    stuck_max_dominant_frac: float = 0.40
    """If the single most-populated hue bucket holds more than this fraction of the
    colored pixels in the front box, one color dominates the view (a guard rail)
    rather than the rainbow's spectrum -> stuck. Rainbow Road measures ~0.2-0.25;
    a rail measures ~0.5-1.0, so 0.4 sits in the gap."""
    stuck_min_colored_frac: float = 0.12
    """The front box must be at least this fraction saturated/bright 'track' color
    before its hue spread is judged at all; below it there's no track ahead (a dark
    void off the edge, or noise too sparse to trust), which also counts as stuck."""
    stuck_static_max_diff: float = 5.0
    """Mean absolute grayscale frame-to-frame difference in the front box below which
    the view counts as *frozen*. A rail/void ahead only arms recovery when the view is
    also this still — a kart wedged against a wall makes no forward progress so its
    image stops changing, whereas normal driving keeps the image flowing even over a
    uniform-coloured stretch that would otherwise trip the dominant-hue test. This is
    what stops rainbow-up-close (one stripe filling the near box) from false-triggering.
    Measured on live footage: wedged <=~4, driving median ~13 (resolution-independent,
    it's a per-pixel mean). The reversed/wrong-way path is NOT motion-gated — the kart
    is still moving when it faces backwards."""
    recovery_clear_frames: int = 8
    """Consecutive frames the rainbow track must be visible again before recovery
    ends and normal steering resumes (hysteresis, ~0.13 s at 60 fps)."""

    # --- wrong-way (reversed-rainbow) detection ---
    # Rainbow Road's stripes run across the track; driven forward, their colours
    # climb the hue circle near->far: blue -> violet -> red -> orange -> yellow ->
    # green. Driven backwards the order flips. We read the sign of that near->far
    # hue gradient and, when it is clearly reversed, trigger the same forward +
    # hard-right recovery as a rail-ram (never reverse off Rainbow Road).
    wrong_way_bands: int = 8
    """Horizontal bands the look-ahead strip is split into, near->far. Each band
    contributes one median hue; consecutive bands give the gradient steps."""
    wrong_way_roi_top: float = 0.45
    wrong_way_roi_bottom: float = 0.92
    wrong_way_roi_left: float = 0.40
    wrong_way_roi_right: float = 0.60
    """Near look-ahead strip (fractions of the player sub-frame) whose vertical hue
    gradient is read. Kept central and near so the ribbon hasn't curved off-frame."""
    wrong_way_min_gradient: float = 30.0
    """Minimum magnitude of the summed near->far circular hue gradient (OpenCV hue
    units) to trust a direction. Below this the reading is ambiguous (coast/keep
    going); at or below -this with the rainbow present it reads as wrong-way."""
    wrong_way_min_band_frac: float = 0.15
    """A band needs at least this fraction of saturated/bright pixels to contribute
    a hue; sparser bands are dropped so off-track noise doesn't skew the gradient."""
    wrong_way_min_bands: int = 3
    """Minimum number of contributing bands before a direction is judged at all;
    fewer than this and the gradient is treated as unreadable (not wrong-way)."""

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
