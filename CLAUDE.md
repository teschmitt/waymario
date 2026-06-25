# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

waymario autonomously drives Mario Kart 64's Rainbow Road on a **real N64 console**.
It reads the game over an HDMI capture device, decides how to steer with classical
computer vision, and emits a real N64 controller signal — no human input. It runs on
a Raspberry Pi, with a Pi Pico (RP2040) handling the timing-critical joybus signal.

## Commands

```sh
uv sync                  # install deps (uses the nix-provided Python; see below)
uv run waymario --help   # CLI entry point (waymario = waymario:main)

uv run pytest            # all tests — no hardware needed
uv run pytest tests/test_steering.py::<name>   # single test
uv run ruff check        # don't bother with linting. It fails and doesn't matter for this type of project
```

Running the bot:

```sh
# Live: Pi + Pico + console wired up
uv run waymario run --port /dev/ttyACM0

# Dry run on a recording, no Pico
uv run waymario run --video clips/rainbow_road.mp4 --loop --no-serial

# Tune vision with a debug overlay (q to quit); --stream serves MJPEG instead of a window
uv run waymario preview --video clips/rainbow_road.mp4 --loop --debug
uv run waymario preview --video clips/rainbow_road.mp4 --loop --stream  # http://<ip>:8080/
```

Multiplayer: `--players N` (1-4) sets the split-screen layout, `--player N` picks which
quadrant this bot reads. Both `run` and `preview` accept them.

Current dev loop — tuning vision on the bundled 4-player clip, reading player 4:

```sh
# window with the 2x2 debug mosaic
uv run waymario preview --players 4 --player 4 --debug \
  --video "Mario Kart 64 Netplay： Rainbow Road 4 player race [J1vBhAEKe30].mkv"

# same, in a browser at http://localhost:1234/
uv run waymario preview --players 4 --player 4 --debug \
  --video "Mario Kart 64 Netplay： Rainbow Road 4 player race [J1vBhAEKe30].mkv" \
  --stream --stream-port 1234
```

## Environment

This is a Nix flake + uv project targeting Python 3.13. `uv sync` installs the
**prebuilt opencv-python wheel**, which `dlopen()`s native libs at runtime
(libstdc++, libGL, X11, …). The nix devShell (`flake.nix`) puts those on
`LD_LIBRARY_PATH` and pins `UV_PYTHON` to the nix interpreter — outside the
devShell, opencv import may fail. `direnv` auto-enters the shell (`.envrc`).

## Architecture

A linear pipeline of small units behind ABC interfaces, wired by one loop
(`drive.run`). Every hardware touchpoint has a no-hardware stand-in, so the whole
pipeline runs and tests on a laptop.

```
HDMI capture → [capture] → frame
               [steer]    → SteeringDecision   (classical OpenCV)
               [control]  → ControllerState     (stick + buttons)
               [transport]→ serial → Pi Pico → joybus → N64
```

- **`capture.py`** — `FrameSource` yields BGR frames. `CaptureDeviceSource` (live
  V4L2) vs `VideoFileSource` (replay a clip, the no-hardware path).
- **`steering.py`** — `OpenCVSteerer.decide(frame) → SteeringDecision`. Crops to the
  player's quadrant, takes a look-ahead ROI, thresholds brightness (Rainbow Road is a
  bright ribbon on a black starfield), and steers from the lit-pixel centroid's
  horizontal offset. Below `min_confidence` lit fraction it coasts straight.
- **`control.py`** — `drive_policy(decision, config) → ControllerState` (always hold A,
  steer the analog stick). `Button` is an `IntFlag` whose bits match the N64 joybus
  status word, so `ControllerState.to_n64_bytes()` serializes directly to the 4 status
  bytes.
- **`transport.py`** — `ControllerLink.send(state)`. `SerialLink` writes to the Pico;
  `NullLink` records the last frame for tests. `encode()` builds the 6-byte wire frame.
- **`config.py`** — one flat `Config` dataclass holding every tunable (ROI bounds,
  brightness threshold, steering gain, max stick, baud, fps) plus `_PLAYER_REGIONS`,
  the split-screen quadrant table. New knobs go here, not threaded through calls.
- **`drive.py`** — the loop: capture → steer → policy → send, paced to `target_fps`.
  Always neutralizes the controller in a `finally` (including on Ctrl-C) so the kart
  doesn't keep its last command.
- **`stream.py`** — `MJPEGServer`, a threaded MJPEG-over-HTTP broadcaster for watching
  the `preview` overlay headlessly (e.g. over SSH).
- **`cli.py`** — argparse front end; `_build_source`/`_build_link` select real vs
  stand-in implementations from the flags.

## Pi ↔ Pico contract

`firmware/` holds only the **protocol contract** — the PIO/joybus firmware is future
work. The wire frame is `[0xA5][btn_hi][btn_lo][stick_x][stick_y][xor]` (xor of the 4
payload bytes), default 115200 baud. `waymario.transport.encode()` is the reference
encoder; keep it and `firmware/README.md` in sync if the frame changes.
