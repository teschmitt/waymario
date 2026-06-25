"""waymario — autonomous Mario Kart 64 Rainbow Road driver.

Pipeline: HDMI capture -> classical-CV steering -> N64 controller state ->
serial -> Pi Pico (joybus) -> console.
"""

from __future__ import annotations

from .cli import main

__all__ = ["main"]
