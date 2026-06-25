# waymario — HSV Color-Band Steering

**Date:** 2026-06-25
**Status:** Approved
**Branch:** `feature/use-hsv-color-bands`
**Parent design:** [`2026-06-25-waymario-autonomous-rainbow-road-design.md`](2026-06-25-waymario-autonomous-rainbow-road-design.md)

## Goal

Add a second `Steerer` that reads the kart's lateral position from the track's
**color** instead of bright-pixel geometry. The track carries a continuous
lateral hue gradient — red (far left) → orange → yellow → green → blue → purple
(far right) — so the hue of a small patch in front of the kart is a direct
read-out of cross-track position. We map that hue to a normalized cross-track
error and steer against it (proportional control).

This is an **alternative** to the existing `OpenCVSteerer` (brightness centroid),
not a replacement. Both implement the same `Steerer` interface and are selectable
at runtime; HSV is the new default.

## Decisions

1. **Selectable steerer, HSV default.** Keep `OpenCVSteerer`; add `HSVSteerer`.
   A `--steerer {hsv,brightness}` flag (default `hsv`) and a `Config.steerer`
   field choose between them. `OpenCVSteerer` is untouched behaviorally.
2. **Single look-ahead patch.** Sample one small, horizontally-centered patch a
   short distance *ahead* of the tires (not right at them), giving slight
   anticipation. Pure proportional control on the hue error.
3. **Continuous linear hue→steer map.** Smooth proportional response, faithful to
   "hue maps to your exact lateral position." (Not discrete bands.)
4. **Trimmed preview.** The `preview` overlay draws whichever ROI the *active*
   steerer uses and shows the measured hue in the HUD. We do **not** build a
   dedicated 2×2 HSV debug mosaic; the existing brightness mosaic stays for
   `--steerer brightness`, and HSV uses the simple overlay.

## Algorithm (`HSVSteerer.decide`)

OpenCV HSV ranges: `H ∈ [0,179]`, `S ∈ [0,255]`, `V ∈ [0,255]`. The gradient runs
red `H≈0` (track left) → purple `H≈140` (track right), monotonic and never
crossing the magenta/red wrap.

1. Crop to the player's screen quadrant (same `Config.player_region()` logic as
   `OpenCVSteerer`).
2. Cut a small patch from the sub-frame, centered at
   `(hue_patch_cx, hue_patch_cy)` with size `(hue_patch_w, hue_patch_h)` — all
   fractions of the sub-frame. `hue_patch_cy` sits above the tires for look-ahead.
3. Convert the patch BGR→HSV. Keep only pixels with `S ≥ hue_min_sat` **and**
   `V ≥ hue_min_val` — the "on the colored track" gate (rejects the black
   starfield and dark HUD).
4. `confidence = passed_pixels / patch_pixels`. If `confidence < min_confidence`,
   coast straight: return `steering=0.0, hue=None, centroid_x=None`.
5. Take the **median** hue of the passed pixels (robust to stray pixels; avoids
   any hue-wrap math since the gradient is unimodal and bounded).
6. Map to cross-track error and steering:

   ```
   e_y      = clamp( 2·(hue − hue_left) / (hue_right − hue_left) − 1 , −1, +1 )
   steering = clamp( −hue_gain · e_y )
   ```

   - Red patch  → `e_y = −1` (too far left)  → `steering = +1` (steer right).
   - Purple patch → `e_y = +1` (too far right) → `steering = −1` (steer left).
   - Green/blue center → `e_y ≈ 0` → straight.

   `centroid_x` carries `e_y` (normalized lateral indicator) so the preview's
   existing position line keeps working; `hue` carries the raw median hue.

## Config (`config.py`)

New tunables (defaults are sensible starting points, tuned in `preview`):

| Knob           | Default | Meaning                                             |
|----------------|---------|-----------------------------------------------------|
| `steerer`      | `"hsv"` | `"hsv"` or `"brightness"` — selects the `Steerer`.  |
| `hue_left`     | `5.0`   | OpenCV hue at the track's left/red edge → `e_y=-1`. |
| `hue_right`    | `140.0` | OpenCV hue at the track's right/purple edge → `e_y=+1`. |
| `hue_gain`     | `1.0`   | `e_y` → steering scale (`e_y` is already normalized). |
| `hue_min_sat`  | `60`    | Minimum S for a pixel to count as track.            |
| `hue_min_val`  | `60`    | Minimum V for a pixel to count as track.            |
| `hue_patch_cx` | `0.5`   | Patch center x (fraction of player sub-frame).      |
| `hue_patch_cy` | `0.62`  | Patch center y — short look-ahead, above the tires. |
| `hue_patch_w`  | `0.12`  | Patch width (fraction of sub-frame).                |
| `hue_patch_h`  | `0.10`  | Patch height (fraction of sub-frame).               |

`min_confidence` is reused for the coast-straight threshold.

## Interface changes

- **`SteeringDecision`** gains `hue: float | None = None` (diagnostic, default
  keeps it backward-compatible). `centroid_x` is reused to carry `e_y`.
- **`Steerer` ABC** gains `roi_box(sub_h, sub_w) → (x0, y0, x1, y1)` returning the
  sub-frame pixel rectangle the steerer samples, so the preview can draw the right
  ROI without hard-coding either shape. `OpenCVSteerer` returns its full-width
  band; `HSVSteerer` returns its small patch.
- **`build_steerer(config) → Steerer`** factory in `steering.py`:
  `"hsv"→HSVSteerer`, `"brightness"→OpenCVSteerer`, error on anything else.

## CLI changes (`cli.py`)

- `--steerer {hsv,brightness}` (default `hsv`) on both `run` and `preview`; sets
  `config.steerer`.
- `run` and `preview` build the steerer via `build_steerer(config)` instead of
  hard-coding `OpenCVSteerer`.
- Preview's simple overlay draws `steerer.roi_box(...)` and shows `hue=` in the
  HUD when present. The `--debug` 2×2 mosaic remains brightness-specific; with
  `--steerer hsv` the preview uses the simple overlay.

## Testing strategy (`tests/test_steering.py`)

TDD with synthetic BGR frames carrying a known-hue patch at the configured patch
location inside the player sub-frame (a small helper builds these):

- Red patch → `steering > 0` and `e_y < 0` (steer right).
- Purple patch → `steering < 0` (steer left).
- Green/blue (center) patch → `|steering|` ≈ 0.
- Desaturated / black patch → `confidence < min_confidence` → `steering == 0.0`,
  `hue is None` (coast).
- Mixed patch (part colored, part black) → fractional `confidence`.
- `build_steerer` returns `HSVSteerer` for `"hsv"`, `OpenCVSteerer` for
  `"brightness"`, raises for unknown values.

Existing `OpenCVSteerer` tests stay green (behavior unchanged).

## Files touched

| File                          | Change                                              |
|-------------------------------|-----------------------------------------------------|
| `src/waymario/steering.py`    | `HSVSteerer`, `build_steerer`, `roi_box`, `hue` field |
| `src/waymario/config.py`      | `steerer` + `hue_*` knobs                           |
| `src/waymario/cli.py`         | `--steerer` flag, `build_steerer`, steerer-aware ROI |
| `tests/test_steering.py`      | `HSVSteerer` + factory tests                        |
| `README.md`                   | note the HSV steerer + `--steerer` flag             |

## Non-goals

- Discrete color-band classification (chose continuous mapping).
- A dedicated HSV debug mosaic in `preview` (trimmed).
- Curvature/derivative (PD) control from a second far patch — single look-ahead
  patch only for now; the interface leaves room to add it later.
- Auto-calibrating the edge hues from footage — they are manual tunables.
