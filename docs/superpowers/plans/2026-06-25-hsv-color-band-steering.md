# HSV Color-Band Steering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a selectable `HSVSteerer` that reads lateral position from the track's red→purple hue gradient and steers by proportional cross-track error, with HSV as the new default steerer.

**Architecture:** A new `Steerer` implementation samples one small look-ahead patch in the player's sub-frame, takes the median hue of on-track (saturated, bright) pixels, maps it linearly to a normalized cross-track error `e_y ∈ [-1,+1]`, and commands `steering = -hue_gain·e_y`. A `build_steerer(config)` factory and a `--steerer {hsv,brightness}` flag select between the new `HSVSteerer` and the existing `OpenCVSteerer`. The preview overlay asks the active steerer for its ROI rectangle so it draws correctly for either.

**Tech Stack:** Python (uv-managed), OpenCV (`cv2`), NumPy, argparse, pytest, ruff.

## Global Constraints

- Run tests with `uv run pytest`; lint with `uv run ruff check`. Both must stay green.
- Every module begins with `from __future__ import annotations` (existing convention).
- OpenCV HSV ranges: `H ∈ [0,179]`, `S ∈ [0,255]`, `V ∈ [0,255]`.
- `OpenCVSteerer` behavior must remain unchanged (its existing tests stay green).
- Commit messages: terse, lowercase, no conventional-commit prefix, no attribution (match existing history: "stuck detection", "playback speed").
- Steering values are always clamped to `[-1, 1]` via the existing `_clamp` helper in `steering.py`.

## File Structure

| File | Responsibility |
|------|----------------|
| `src/waymario/steering.py` | `SteeringDecision` (+`hue`), `Steerer` ABC (+`roi_box`), `_subframe` helper, `OpenCVSteerer`, **new** `HSVSteerer`, **new** `build_steerer` |
| `src/waymario/config.py` | tunables: **new** `steerer` + `hue_*` knobs |
| `src/waymario/cli.py` | **new** `--steerer` flag, `build_steerer` wiring, steerer-aware preview ROI/HUD, `_build_parser` refactor |
| `tests/test_steering.py` | `HSVSteerer` behavior + `build_steerer` factory tests |
| `tests/test_cli.py` | **new** parser tests for `--steerer` |
| `README.md` | note the HSV steerer + `--steerer` flag |

---

## Task 1: Shared steering interface — `hue` field, `roi_box`, `_subframe` helper

Foundation both steerers and the preview depend on: a diagnostic `hue` field, a `roi_box` method so the preview can draw whichever ROI is active, and a shared sub-frame crop helper. `OpenCVSteerer` is refactored onto the helper and gains `roi_box` with **no behavior change**.

**Files:**
- Modify: `src/waymario/steering.py`
- Test: `tests/test_steering.py`

**Interfaces:**
- Produces:
  - `SteeringDecision(steering: float, confidence: float, centroid_x: float | None = None, hue: float | None = None)`
  - `Steerer.roi_box(self, sub_h: int, sub_w: int) -> tuple[int, int, int, int]` (abstract) — returns `(x0, y0, x1, y1)` in sub-frame pixel coords.
  - `_subframe(frame: np.ndarray, cfg: Config) -> np.ndarray` (module-level helper).
  - `OpenCVSteerer.roi_box` returns the full-width band `(0, int(sub_h*roi_top), sub_w, int(sub_h*roi_bottom))`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_steering.py`:

```python
from waymario.steering import OpenCVSteerer, SteeringDecision


def test_steering_decision_has_hue_field_defaulting_none() -> None:
    d = SteeringDecision(steering=0.0, confidence=0.0)
    assert d.hue is None


def test_opencv_roi_box_is_full_width_band() -> None:
    steerer = OpenCVSteerer(Config())
    # Config defaults: roi_top=0.45, roi_bottom=0.95
    assert steerer.roi_box(sub_h=100, sub_w=200) == (0, 45, 200, 95)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_steering.py::test_opencv_roi_box_is_full_width_band tests/test_steering.py::test_steering_decision_has_hue_field_defaulting_none -v`
Expected: FAIL — `SteeringDecision.__init__() got an unexpected keyword`/`AttributeError: 'OpenCVSteerer' object has no attribute 'roi_box'`.

- [ ] **Step 3: Add the `hue` field to `SteeringDecision`**

In `src/waymario/steering.py`, extend the dataclass:

```python
@dataclass
class SteeringDecision:
    steering: float
    """Desired steering, -1 (full left) .. +1 (full right)."""
    confidence: float
    """Fraction of the ROI that read as track (0..1)."""
    centroid_x: float | None = None
    """Normalized lateral indicator within the sub-frame, -1..1 (None if no track seen).
    OpenCVSteerer: track centroid offset. HSVSteerer: cross-track error e_y."""
    hue: float | None = None
    """Median OpenCV hue (0..179) sampled by HSVSteerer (None for other steerers)."""
```

- [ ] **Step 4: Add the `_subframe` helper and the abstract `roi_box`**

In `src/waymario/steering.py`, add the helper above the `Steerer` class:

```python
def _subframe(frame: np.ndarray, cfg: Config) -> np.ndarray:
    """Crop a frame to this player's screen quadrant (fractions from config)."""
    height, width = frame.shape[:2]
    px0, py0, px1, py1 = cfg.player_region()
    return frame[int(height * py0):int(height * py1), int(width * px0):int(width * px1)]
```

Add the abstract method to `Steerer`:

```python
class Steerer(ABC):
    @abstractmethod
    def decide(self, frame: np.ndarray) -> SteeringDecision:
        """Decide how to steer for a single BGR frame."""

    @abstractmethod
    def roi_box(self, sub_h: int, sub_w: int) -> tuple[int, int, int, int]:
        """Return the (x0, y0, x1, y1) sub-frame rectangle this steerer samples."""
```

- [ ] **Step 5: Refactor `OpenCVSteerer` onto `_subframe` and implement `roi_box`**

Replace `OpenCVSteerer.decide`'s inline crop with the helper, and add `roi_box`. The body below is behavior-identical to the current implementation:

```python
class OpenCVSteerer(Steerer):
    def __init__(self, config: Config) -> None:
        self._config = config

    def roi_box(self, sub_h: int, sub_w: int) -> tuple[int, int, int, int]:
        cfg = self._config
        return (0, int(sub_h * cfg.roi_top), sub_w, int(sub_h * cfg.roi_bottom))

    def decide(self, frame: np.ndarray) -> SteeringDecision:
        cfg = self._config

        # Crop to this player's screen quadrant first.
        subframe = _subframe(frame, cfg)
        sub_h, sub_w = subframe.shape[:2]

        # ROI is relative to the player's sub-frame.
        top = int(sub_h * cfg.roi_top)
        bottom = int(sub_h * cfg.roi_bottom)
        roi = subframe[top:bottom, :]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, cfg.bright_threshold, 255, cv2.THRESH_BINARY)

        lit = int(np.count_nonzero(mask))
        confidence = lit / mask.size if mask.size else 0.0

        if confidence < cfg.min_confidence:
            return SteeringDecision(steering=0.0, confidence=confidence, centroid_x=None)

        moments = cv2.moments(mask, binaryImage=True)
        cx = moments["m10"] / moments["m00"]

        offset = (cx - sub_w / 2) / (sub_w / 2)
        steering = _clamp(offset * cfg.steering_gain)

        return SteeringDecision(steering=steering, confidence=confidence, centroid_x=offset)
```

- [ ] **Step 6: Run the full steering suite to verify pass + no regression**

Run: `uv run pytest tests/test_steering.py tests/test_drive.py -v`
Expected: PASS — the two new tests plus all four existing `OpenCVSteerer` tests and the drive smoke test.

- [ ] **Step 7: Lint**

Run: `uv run ruff check`
Expected: no errors.

- [ ] **Step 8: Commit**

```bash
git add src/waymario/steering.py tests/test_steering.py
git commit -m "steerer roi_box + hue diagnostic"
```

---

## Task 2: `HSVSteerer` + hue config knobs

The hue-gradient steerer and the tunables it reads. Config knobs are folded in here because the steerer is their first consumer.

**Files:**
- Modify: `src/waymario/config.py`
- Modify: `src/waymario/steering.py`
- Test: `tests/test_steering.py`

**Interfaces:**
- Consumes: `_subframe`, `_clamp`, `SteeringDecision`, `Steerer` (Task 1).
- Produces: `HSVSteerer(config: Config)` with `decide(frame) -> SteeringDecision` and `roi_box(sub_h, sub_w) -> tuple[int,int,int,int]`. New `Config` fields: `hue_left`, `hue_right`, `hue_gain`, `hue_min_sat`, `hue_min_val`, `hue_patch_cx`, `hue_patch_cy`, `hue_patch_w`, `hue_patch_h`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_steering.py` (imports: `import cv2`, and extend the steering import with `HSVSteerer`):

```python
import cv2

from waymario.steering import HSVSteerer


def _bgr_for_hue(hue: int, sat: int = 200, val: int = 200) -> tuple[int, int, int]:
    """BGR tuple for a single OpenCV HSV color."""
    px = np.uint8([[[hue, sat, val]]])
    b, g, r = cv2.cvtColor(px, cv2.COLOR_HSV2BGR)[0, 0]
    return int(b), int(g), int(r)


def _solid_hue_frame(hue: int, width: int = 640, height: int = 200,
                     sat: int = 200, val: int = 200) -> np.ndarray:
    """Frame filled with one HSV color, so the centered patch samples that hue."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = _bgr_for_hue(hue, sat, val)
    return frame


def test_hsv_red_patch_steers_right() -> None:
    # Red (low hue) => too far left (e_y<0) => steer right (steering>0).
    steerer = HSVSteerer(Config())
    decision = steerer.decide(_solid_hue_frame(hue=10))
    assert decision.steering > 0
    assert decision.centroid_x is not None and decision.centroid_x < 0
    assert decision.confidence > 0.9
    assert decision.hue is not None


def test_hsv_purple_patch_steers_left() -> None:
    # Purple (high hue) => too far right (e_y>0) => steer left (steering<0).
    steerer = HSVSteerer(Config())
    decision = steerer.decide(_solid_hue_frame(hue=135))
    assert decision.steering < 0
    assert decision.centroid_x is not None and decision.centroid_x > 0


def test_hsv_center_hue_goes_straight() -> None:
    # Midpoint of hue_left=5 and hue_right=140 is 72.5 => e_y ~ 0.
    steerer = HSVSteerer(Config())
    decision = steerer.decide(_solid_hue_frame(hue=72))
    assert abs(decision.steering) < 0.05


def test_hsv_desaturated_frame_coasts_straight() -> None:
    # All-black frame: every pixel fails the S/V gate => coast.
    steerer = HSVSteerer(Config())
    frame = np.zeros((200, 640, 3), dtype=np.uint8)
    decision = steerer.decide(frame)
    assert decision.steering == 0.0
    assert decision.hue is None
    assert decision.centroid_x is None


def test_hsv_partial_patch_has_fractional_confidence() -> None:
    # Left half colored, right half black; centered patch straddles the seam.
    frame = np.zeros((200, 640, 3), dtype=np.uint8)
    frame[:, :320] = _bgr_for_hue(10)
    steerer = HSVSteerer(Config())
    decision = steerer.decide(frame)
    assert 0.2 < decision.confidence < 0.8


def test_hsv_roi_box_within_subframe() -> None:
    steerer = HSVSteerer(Config())
    x0, y0, x1, y1 = steerer.roi_box(sub_h=200, sub_w=640)
    assert 0 <= x0 < x1 <= 640
    assert 0 <= y0 < y1 <= 200
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_steering.py -k hsv -v`
Expected: FAIL — `ImportError: cannot import name 'HSVSteerer'`.

- [ ] **Step 3: Add the hue config knobs**

In `src/waymario/config.py`, inside `@dataclass class Config`, add this block immediately after the `min_confidence` field and before the `# --- control ---` section (keep `min_confidence`; it is reused as the coast threshold). The `steerer` selector field is added separately in Task 3 — add only these hue knobs now:

```python
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
```

- [ ] **Step 4: Implement `HSVSteerer`**

In `src/waymario/steering.py`, add after `OpenCVSteerer`:

```python
class HSVSteerer(Steerer):
    """Read lateral position from the track's red->purple hue gradient.

    Sample one small look-ahead patch, take the median hue of on-track
    (saturated, bright) pixels, map it linearly to a cross-track error, and
    steer against that error.
    """

    def __init__(self, config: Config) -> None:
        self._config = config

    def _patch_bounds(self, sub_h: int, sub_w: int) -> tuple[int, int, int, int]:
        cfg = self._config
        cx = cfg.hue_patch_cx * sub_w
        cy = cfg.hue_patch_cy * sub_h
        half_w = cfg.hue_patch_w * sub_w / 2.0
        half_h = cfg.hue_patch_h * sub_h / 2.0
        x0 = max(0, int(cx - half_w))
        y0 = max(0, int(cy - half_h))
        x1 = min(sub_w, int(cx + half_w))
        y1 = min(sub_h, int(cy + half_h))
        return x0, y0, x1, y1

    def roi_box(self, sub_h: int, sub_w: int) -> tuple[int, int, int, int]:
        return self._patch_bounds(sub_h, sub_w)

    def decide(self, frame: np.ndarray) -> SteeringDecision:
        cfg = self._config
        subframe = _subframe(frame, cfg)
        sub_h, sub_w = subframe.shape[:2]

        x0, y0, x1, y1 = self._patch_bounds(sub_h, sub_w)
        patch = subframe[y0:y1, x0:x1]
        if patch.size == 0:
            return SteeringDecision(steering=0.0, confidence=0.0, centroid_x=None, hue=None)

        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        h = hsv[:, :, 0]
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]

        gate = (s >= cfg.hue_min_sat) & (v >= cfg.hue_min_val)
        passed = h[gate]
        total = h.size
        confidence = float(passed.size) / float(total) if total else 0.0

        if passed.size == 0 or confidence < cfg.min_confidence:
            # No trustworthy colored track in the patch — coast straight.
            return SteeringDecision(steering=0.0, confidence=confidence, centroid_x=None, hue=None)

        hue = float(np.median(passed))
        e_y = _clamp(2.0 * (hue - cfg.hue_left) / (cfg.hue_right - cfg.hue_left) - 1.0)
        steering = _clamp(-cfg.hue_gain * e_y)
        return SteeringDecision(steering=steering, confidence=confidence, centroid_x=e_y, hue=hue)
```

- [ ] **Step 5: Run the HSV tests to verify they pass**

Run: `uv run pytest tests/test_steering.py -v`
Expected: PASS — all HSV tests plus the Task 1 and original tests.

- [ ] **Step 6: Lint**

Run: `uv run ruff check`
Expected: no errors.

- [ ] **Step 7: Commit**

```bash
git add src/waymario/config.py src/waymario/steering.py tests/test_steering.py
git commit -m "hsv color-band steerer"
```

---

## Task 3: `build_steerer` factory + `Config.steerer`

A single place that maps the `steerer` string to a `Steerer`, so the CLI never hard-codes a class.

**Files:**
- Modify: `src/waymario/config.py`
- Modify: `src/waymario/steering.py`
- Test: `tests/test_steering.py`

**Interfaces:**
- Consumes: `OpenCVSteerer`, `HSVSteerer`, `Config` (Tasks 1–2).
- Produces: `Config.steerer: str = "hsv"`; `build_steerer(config: Config) -> Steerer` (raises `ValueError` for unknown names).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_steering.py` (extend the steering import with `build_steerer`, add `import pytest`):

```python
import pytest

from waymario.steering import build_steerer


def test_config_default_steerer_is_hsv() -> None:
    assert Config().steerer == "hsv"


def test_build_steerer_selects_hsv() -> None:
    assert isinstance(build_steerer(Config(steerer="hsv")), HSVSteerer)


def test_build_steerer_selects_brightness() -> None:
    assert isinstance(build_steerer(Config(steerer="brightness")), OpenCVSteerer)


def test_build_steerer_rejects_unknown() -> None:
    with pytest.raises(ValueError):
        build_steerer(Config(steerer="rainbow"))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_steering.py -k "steerer and (build or default_steerer)" -v`
Expected: FAIL — `ImportError: cannot import name 'build_steerer'` / `TypeError: ... unexpected keyword argument 'steerer'`.

- [ ] **Step 3: Add the `steerer` selector field to `Config`**

In `src/waymario/config.py`, add at the top of the `# --- steering / vision ---` section (just before `roi_top`):

```python
    steerer: str = "hsv"
    """Which steering algorithm to use: "hsv" (color-band) or "brightness" (centroid)."""
```

- [ ] **Step 4: Add the `build_steerer` factory**

In `src/waymario/steering.py`, add at the end of the module:

```python
def build_steerer(config: Config) -> Steerer:
    """Construct the steerer named by ``config.steerer``."""
    if config.steerer == "hsv":
        return HSVSteerer(config)
    if config.steerer == "brightness":
        return OpenCVSteerer(config)
    raise ValueError(
        f"unknown steerer {config.steerer!r}; expected 'hsv' or 'brightness'"
    )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_steering.py -v`
Expected: PASS — all factory tests plus the full steering suite.

- [ ] **Step 6: Lint**

Run: `uv run ruff check`
Expected: no errors.

- [ ] **Step 7: Commit**

```bash
git add src/waymario/config.py src/waymario/steering.py tests/test_steering.py
git commit -m "build_steerer factory + steerer config"
```

---

## Task 4: CLI `--steerer` flag, steerer-aware preview, README

Wire selection into both subcommands, make the preview overlay correct for either steerer (right ROI box + hue in the HUD), and document the flag. The `--debug` mosaic stays brightness-specific (trimmed-preview decision).

**Files:**
- Modify: `src/waymario/cli.py`
- Modify: `README.md`
- Test: `tests/test_cli.py` (create)

**Interfaces:**
- Consumes: `build_steerer` (Task 3), `Config.steerer` (Task 3), `Steerer.roi_box` (Task 1), `SteeringDecision.hue` (Task 1).
- Produces: `_build_parser() -> argparse.ArgumentParser`; `--steerer {hsv,brightness}` (default `hsv`) on `run` and `preview`.

- [ ] **Step 1: Write the failing parser tests**

Create `tests/test_cli.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cli.py -v`
Expected: FAIL — `ImportError: cannot import name '_build_parser'`.

- [ ] **Step 3: Add the `--steerer` arg helper and refactor `main` into `_build_parser`**

In `src/waymario/cli.py`, update the import and add a steerer-arg helper:

```python
from .steering import OpenCVSteerer, build_steerer
```

Add this helper next to `_add_player_args`:

```python
def _add_steerer_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--steerer",
        choices=["hsv", "brightness"],
        default="hsv",
        help="steering algorithm: hsv (color-band, default) or brightness (centroid)",
    )
```

Replace `main()` with a `_build_parser()` + thin `main()`:

```python
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="waymario", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="drive live: capture -> steer -> controller")
    _add_source_args(p_run)
    _add_player_args(p_run)
    _add_steerer_arg(p_run)
    p_run.add_argument("--port", help="serial port to the Pi Pico")
    p_run.add_argument("--no-serial", action="store_true", help="use NullLink (no Pico)")
    p_run.add_argument("--debug", action="store_true", help="print confidence/steering/phase every 10 frames")
    p_run.set_defaults(func=_cmd_run)

    p_preview = sub.add_parser("preview", help="show the CV overlay for tuning (no output)")
    _add_source_args(p_preview)
    _add_player_args(p_preview)
    _add_steerer_arg(p_preview)
    p_preview.add_argument(
        "--stream",
        action="store_true",
        help="serve the overlay as an MJPEG stream instead of opening a local window",
    )
    p_preview.add_argument(
        "--stream-port",
        type=int,
        default=8080,
        metavar="PORT",
        help="port for the MJPEG HTTP server (default: 8080)",
    )
    p_preview.add_argument(
        "--stream-fps",
        type=float,
        default=15.0,
        metavar="FPS",
        help="max frames per second to push to the stream (default: 15)",
    )
    p_preview.add_argument(
        "--debug",
        action="store_true",
        help="show 2x2 mosaic of preprocessing steps (brightness steerer only)",
    )
    p_preview.set_defaults(func=_cmd_preview)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    return args.func(args)
```

- [ ] **Step 4: Run parser tests to verify they pass**

Run: `uv run pytest tests/test_cli.py -v`
Expected: PASS.

- [ ] **Step 5: Wire `build_steerer` into `_cmd_run`**

In `_cmd_run`, set the steerer on the config and build it via the factory:

```python
def _cmd_run(args: argparse.Namespace) -> int:
    config = Config()
    if args.port:
        config.serial_port = args.port
    config.steerer = args.steerer
    _apply_player_args(args, config)
    with _build_source(args, config) as source, _build_link(args, config) as link:
        run(source, build_steerer(config), link, config, debug=args.debug)
    return 0
```

- [ ] **Step 6: Make the preview steerer-aware**

In `_cmd_preview`, set `config.steerer = args.steerer` right after `_apply_player_args(args, config)`, and build the steerer via the factory:

```python
    config = Config()
    _apply_player_args(args, config)
    config.steerer = args.steerer
    steerer = build_steerer(config)
    stuck = StuckDetector(config)
```

Replace `_process_frame` so it draws the active steerer's ROI box and shows hue:

```python
    def _process_frame(frame: "cv2.typing.MatLike") -> "cv2.typing.MatLike":
        """Annotate frame with the active steerer's ROI box, position line, HUD."""
        decision = steerer.decide(frame)
        recovery = stuck.update(frame, decision)
        state = recovery if recovery is not None else drive_policy(decision, config)
        phase = stuck._phase.name
        sub, x0, y0 = _subframe(frame)
        sub_h, sub_w = sub.shape[:2]
        rx0, ry0, rx1, ry1 = steerer.roi_box(sub_h, sub_w)
        cv2.rectangle(frame, (x0 + rx0, y0 + ry0), (x0 + rx1, y0 + ry1), (0, 255, 0), 1)
        if decision.centroid_x is not None:
            cx = x0 + int((decision.centroid_x + 1) / 2 * sub_w)
            cv2.line(frame, (cx, y0 + ry0), (cx, y0 + ry1), (0, 0, 255), 2)
        _hud(frame, decision, state, phase)
        return frame
```

Update `_hud`'s text line to append hue when present. Replace the `text = (...)` assignment inside `_hud` with:

```python
        hue_txt = f"  hue={decision.hue:.0f}" if decision.hue is not None else ""
        text = (
            f"P{config.player} {mode_tag}  "
            f"conf={decision.confidence:.3f}  "
            f"steer={decision.steering:+.2f}{hue_txt}  "
            f"stick=({state.stick_x:+d},{state.stick_y:+d})"
        )
```

Restrict the `--debug` mosaic to the brightness steerer (it visualizes grayscale/threshold). Replace the `process = ...` line with:

```python
    process = _debug_mosaic if (args.debug and config.steerer == "brightness") else _process_frame
```

- [ ] **Step 7: Run the whole suite + lint**

Run: `uv run pytest -v && uv run ruff check`
Expected: PASS, no lint errors.

- [ ] **Step 8: Manual smoke (visual / no automated assertion)**

The preview overlay is visual; verify wiring with whatever clip you have:

Run: `uv run waymario preview --video <clip.mp4> --loop` (default `--steerer hsv`)
Expected: a small green ROI box centered ~62% down the screen, a red position line, and `hue=<n>` in the HUD bar. Then `uv run waymario preview --video <clip.mp4> --steerer brightness --debug` still shows the original 2×2 grayscale/threshold mosaic.
If no clip is available, this step is skipped — the parser tests + full suite already cover the wiring; note the skip.

- [ ] **Step 9: Update the README**

In `README.md`, under "Usage", add a line after the existing dry-run example:

```sh
# Pick the steering algorithm (default: hsv color-band)
uv run waymario run --video clips/rainbow_road.mp4 --no-serial --steerer brightness
```

And in the "How it works" steer line, change:

```
               [steer]   → SteeringDecision   (classical OpenCV)
```
to:
```
               [steer]   → SteeringDecision   (hsv color-band, or brightness centroid)
```

- [ ] **Step 10: Commit**

```bash
git add src/waymario/cli.py tests/test_cli.py README.md
git commit -m "steerer selection flag + steerer-aware preview"
```

---

## Self-Review

**1. Spec coverage:**
- Selectable steerer, HSV default → Tasks 3 (`Config.steerer="hsv"`, `build_steerer`) + 4 (`--steerer`). ✓
- Single look-ahead patch + continuous linear map → Task 2 (`_patch_bounds`, `e_y`/`steering` formula). ✓
- Confidence gating + median hue + coast → Task 2. ✓
- All config knobs → Tasks 2 (`hue_*`) + 3 (`steerer`). ✓
- `SteeringDecision.hue`, `Steerer.roi_box`, `build_steerer` → Tasks 1 + 3. ✓
- CLI flag + steerer-aware preview ROI/HUD, trimmed (no HSV mosaic) → Task 4. ✓
- Tests for red/purple/center/coast/mixed/factory → Tasks 2 + 3. ✓
- `OpenCVSteerer` unchanged → Task 1 refactor is behavior-identical; existing tests guard it. ✓
- README note → Task 4. ✓

**2. Placeholder scan:** No "TBD"/"implement later"; every code step shows full, type-this code.

**3. Type consistency:** `build_steerer`, `roi_box(sub_h, sub_w)`, `_subframe(frame, cfg)`, `_patch_bounds`, and `SteeringDecision(... centroid_x, hue)` names/signatures match across Tasks 1–4. `centroid_x` carries `e_y` for HSV (documented in Task 1 Step 3). `config.steerer` set in both `_cmd_run` and `_cmd_preview`.
