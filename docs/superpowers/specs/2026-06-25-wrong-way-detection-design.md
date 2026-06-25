# Wrong-way (reversed-rainbow) detection — design

## Problem

`StuckDetector` (`stuck.py`) already recovers from ramming a guard rail: hold A,
full-right stick, never reverse, until the rainbow reappears. But it only *triggers*
when the rainbow is **missing** from the front box (a single hue dominates, or a void).
When Mario drives the **wrong way along the ribbon**, the rainbow is still fully
visible — just in reverse colour order — so the current detector sees a healthy track
and never corrects. This adds that missing trigger, feeding the existing recovery.

The correct driving direction (user-supplied, confirmed by measuring the samples):

```
near (by the kart)  →  far (horizon)
blue → violet → red → orange → yellow → green
```

## The signal

Measured median hue down the centre look-ahead column of `sample2.png`, near→far:

| near→far | blue | violet | red | orange | yellow | green |
|----------|------|--------|-----|--------|--------|-------|
| OpenCV H | 116  | 150    | 3   | 18     | 29     | 57    |

This is **monotonically increasing on the hue circle** (…150 → wrap 179→0 → 18 → 29
→ 57). `sample-not-stuck.png` shows the same rising pattern. So:

- **Forward (correct):** sum of consecutive circular hue-deltas, read near→far, is **positive**.
- **Reversed (wrong way):** that sum is **negative**.

Endpoint-only comparison fails — near (blue 116) and far (green 57) sit ~half a circle
apart, so a single near-vs-far shortest-path delta is ambiguous. Summing *adjacent*
band steps (each ~15–40°, well under 90°) unwraps the cycle correctly. Steps are only
summed between **immediately adjacent** valid bands, so a dropped band never bridges
two distant hues (which could mis-wrap).

## Algorithm (`OpenCVSteerer`-free, lives in `StuckDetector`)

Per frame, over a near look-ahead strip (`wrong_way_roi_*`, default y 0.45–0.92,
x 0.40–0.60 of the player sub-frame):

1. Split the strip into `wrong_way_bands` (=8) horizontal bands.
2. Each band's hue = median of pixels passing the `stuck_min_sat/val` gate; bands with
   fewer than `wrong_way_min_band_frac` (=0.15) coloured pixels are dropped (hue = None).
3. `G = Σ circ_delta(band[k], band[k+1])` over **consecutive non-dropped** pairs,
   near→far. `circ_delta(a,b) = ((b−a+90) mod 180) − 90` (signed shortest hue step).
4. Direction is *valid* only if there are ≥ `wrong_way_min_bands − 1` (=2) such steps.
5. `reversed = rainbow_present AND valid AND (G ≤ −wrong_way_min_gradient)` (=30).

## Integration into the state machine (minimal)

The rail/void test (`_rainbow_ahead`, dominant-hue histogram) is **unchanged**. The
only change is broadening the per-frame verdict:

```
front_ok = rainbow_ahead and not reversed     # reversed is guarded by rainbow_ahead
```

- NORMAL: count frames where `not front_ok`; at `stuck_frames` → RECOVER.
- RECOVER: count frames where `front_ok`; at `recovery_clear_frames` → NORMAL.

Both rail-ram and reversed-rainbow share the same RECOVER phase, the same forward+
hard-right `_recovery_state()`, and the same hysteresis/timing (`stuck_frames`,
`recovery_clear_frames`). No new state, no new recovery action.

**Why this is safe for the existing rail path:** when the rainbow is absent,
`rainbow_ahead` is False so `reversed` is forced False and `front_ok` is False — exactly
today's behaviour. The recovery-exit predicate now also requires *not reversed*, which
is strictly better: you never resume normal steering while still pointed backwards.

## Config (new knobs; reuses `stuck_min_sat/val`, `stuck_frames`, `recovery_clear_frames`)

`wrong_way_bands=8`, `wrong_way_roi_top=0.45`, `wrong_way_roi_bottom=0.92`,
`wrong_way_roi_left=0.40`, `wrong_way_roi_right=0.60`, `wrong_way_min_gradient=30.0`,
`wrong_way_min_band_frac=0.15`, `wrong_way_min_bands=3`.

## Debug / preview (tuning aid — no clean reversed sample exists yet)

`drive.py --debug` line and the `preview` HUD gain the signed gradient `G` and a
`FWD/REV/--` tag; preview also draws the wrong-way strip box. Needed because the only
real reversed frame would have to be captured live — thresholds will want live tuning
(consistent with the 2-frame-thresholds caveat in project memory).

## Testing (TDD, no hardware)

- Unit: `circ_delta` sign/wrap; `_direction_gradient` on synthetic forward/reversed sweeps.
- Behaviour: reversed rainbow → recovers after `stuck_frames`; forward rainbow → never;
  rail/void → existing behaviour preserved; recovery exits only once forward again.
- The shared `_rainbow_frame()` fixture is flipped to forward orientation (its hue
  histogram, and thus every rail test, is unchanged); a `_reversed_rainbow_frame()` is added.
- Grounding: `sample2.png` reads forward (G > 0, never recovers) with default config.
- Known gap: no real reversed-rainbow frame; reversed behaviour is covered only
  synthetically until live footage is captured.
```
