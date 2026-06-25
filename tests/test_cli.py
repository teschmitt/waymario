"""CLI argument parsing."""

from __future__ import annotations

import pytest

from waymario.cli import _build_parser


def test_run_steerer_defaults_to_hsv() -> None:
    args = _build_parser().parse_args(["run"])
    assert args.steerer == "hsv"


def test_preview_steerer_defaults_to_hsv() -> None:
    args = _build_parser().parse_args(["preview"])
    assert args.steerer == "hsv"


def test_run_accepts_brightness_steerer() -> None:
    args = _build_parser().parse_args(["run", "--steerer", "brightness"])
    assert args.steerer == "brightness"


def test_invalid_steerer_is_rejected() -> None:
    with pytest.raises(SystemExit):
        _build_parser().parse_args(["run", "--steerer", "rainbow"])
