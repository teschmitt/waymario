---
name: run-waymario
description: Build, run, test, and screenshot the waymario autonomous Mario Kart 64 driver. Use when asked to run/start/launch waymario, drive or preview the vision pipeline, screenshot or capture the CV overlay, evaluate changes to the preview/steering code, or smoke-test the capture→steer→controller loop. No N64/Pico/HDMI hardware needed.
---

# Run waymario

waymario reads Mario Kart 64 over HDMI, steers with classical OpenCV, and emits a
real N64 controller signal. To evaluate it headless you (a) run the tests and
(b) need to *see* what the vision code renders — but the on-screen overlay normally
only goes to an OpenCV window or an MJPEG stream, neither of which an agent can see.

The agent path is the **`preview --capture-frames` flag built into the app**: it
runs the real `preview` renderer over a clip and writes processed frames to PNGs you
can Read, printing one metadata line per frame so you can pick which to open. No
display, no capture hardware, no Pico, no helper script — it's the actual code, so
edits to `preview`/`steering` show up directly in the captured PNG.

All paths are relative to the repo root; run every `uv` command from there.
**Verified this session on macOS (darwin)** with the bundled 4-player clip.

## Prerequisites

```sh
uv sync     # installs deps incl. the prebuilt opencv-python wheel
```

Nix flake + uv project (Python 3.13). On **Linux** the opencv wheel `dlopen()`s
native libs (libGL, X11, libstdc++…) that the nix devShell puts on
`LD_LIBRARY_PATH` — work inside `nix develop` / the `direnv` shell or `import cv2`
fails. On macOS the wheel is self-contained and `uv run` works directly. A clip
must sit in the repo root (a `*.mkv` is bundled).

## Test

```sh
uv run pytest        # 17 passed in <1s, no hardware
```

(`uv run ruff check` currently reports 8 pre-existing style warnings unrelated to
running the app — don't treat them as a regression.)

## See the vision output — agent path (`--capture-frames`)

```sh
mkv=$(ls *.mkv | head -1)   # bundled clip's filename has a fullwidth colon; glob avoids typing it

# Save every 150th frame's debug mosaic; default cap is 12 frames, output to a temp dir.
uv run waymario preview --video "$mkv" --players 4 --player 4 --debug --capture-frames 150

# The normal in-game overlay instead of the mosaic; fewer frames; a chosen dir.
uv run waymario preview --video "$mkv" --players 4 --player 4 --capture-frames 200 \
  --capture-count 4 --capture-dir /tmp/wm
```

It prints a `Capture dir:` line then one line per saved frame, e.g.:

```
Capture dir: /var/folders/.../waymario-capture-xxxx
frame_000150.png  conf=0.593 steer=+0.15 stick=(+12,+0) centroid=+0.059 phase=NORMAL
frame_000300.png  conf=0.969 steer=+0.03 stick=(+2,+0) centroid=+0.012 phase=NORMAL
```

**Use the metadata to choose which PNGs to Read** — e.g. low `conf=`, a large
`steer=`, or `phase=RECOVER` mark the interesting frames. Then Read
`<capture-dir>/frame_NNNNNN.png`.

Flags:
- `--capture-frames N` — save every Nth frame (the stride). Required to enter capture mode.
- `--capture-count K` — stop after K saved frames (default 12). Bounds runtime; reads stop early.
- `--capture-dir DIR` — output dir (default: a fresh temp dir, printed as `Capture dir:`).
- `--debug` — capture the 2×2 mosaic (subframe / ROI / grayscale / threshold) instead of the
  in-game overlay. Panel ④ is the brightness-threshold mask with the `lit=%` driving confidence.
- `--players N --player M` — split-screen layout and which quadrant to read (`M` ≤ `N`, 1–4).

Verified: `--debug --capture-frames 150 --capture-count 4` wrote 4 readable mosaics;
the plain overlay form wrote frames showing all four quadrants + the `P4 [DRIVE]` HUD.

## Smoke-test the full loop — agent path (text)

Runs the real `run` pipeline (capture→steer→StuckDetector→controller) with no Pico
via `NullLink`, one line per 10 frames. It plays the whole clip (~8400 frames @
60fps ≈ 2.3 min), so cap it with `timeout`:

```sh
mkv=$(ls *.mkv | head -1)
timeout 8 uv run waymario run --video "$mkv" --no-serial --debug --players 4 --player 4
```

Expected: `[    50] DRIVE   | conf=0.675 steer=-0.03 stick_x=  -2 stick_y=  +0 phase=NORMAL`
lines, then `timeout` kills it (exit 124 is fine). `phase` flips to `RECOVER` when
the StuckDetector takes over. (When the serial path lands, drop `--no-serial` and add
`--port /dev/ttyACM0` to drive a real Pico.)

## Human path (needs a display/browser; skip when headless)

```sh
uv run waymario preview --players 4 --player 4 --debug --video "<clip>.mkv"            # OpenCV window
uv run waymario preview --players 4 --player 4 --debug --video "<clip>.mkv" \
  --stream --stream-port 1234                                                          # browser: http://localhost:1234/
```

An agent should use `--capture-frames` instead — same renderer, written to files.

## Gotchas

- **`uv run` can fail under the Claude Code command sandbox** with
  `failed to open .../.cache/uv/...: Operation not permitted` (uv needs to write its
  cache). Re-run with the sandbox disabled. Same for creating files under
  `.claude/skills/`.
- **The bundled clip filename has a fullwidth colon** (`：`, U+FF1A) and `[ ]`, so it
  must be quoted — use `mkv=$(ls *.mkv | head -1)` and pass `"$mkv"`.
- **`--capture-frames` reads frames in order; it can't seek.** With stride `N` and
  count `K` it stops around frame `N*K`. Raise the stride to reach later in the clip.
- **`waymario run` without `--loop` plays the whole clip** (~2.3 min); always wrap a
  smoke in `timeout`.
- **High `conf=`/`lit=%` ≠ tracking the road.** The grayscale `bright_threshold`
  (default 60) also lights up bright infield, sprites and HUD, so judge tuning from
  the mosaic's panel-④ mask, not the confidence number alone.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Operation not permitted` on `.cache/uv` | Disable the command sandbox for that `uv` call. |
| `ImportError`/libGL on `import cv2` (Linux) | Run inside the nix devShell so `LD_LIBRARY_PATH` is set. |
| `could not open video file …` | Quote the clip path / check it exists, or pass `--video PATH`. |
| `ValueError: Invalid combination: players=…, player=…` | `--player` must be ≤ `--players` (1–4). |
| Capture wrote 0 frames | Stride larger than the clip; lower `--capture-frames`. |
