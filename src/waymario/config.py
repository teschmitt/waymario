"""Tunable settings for the waymario pipeline.

One flat dataclass so every knob lives in one place and can be overridden from
the CLI or a future config file without threading args through the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Config:
    # --- capture ---
    device: int = 0
    """V4L2 index of the HDMI capture dongle."""
    width: int | None = None
    height: int | None = None

    # --- steering / vision ---
    roi_top: float = 0.45
    """Fraction of frame height where the region-of-interest starts (look ahead,
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
