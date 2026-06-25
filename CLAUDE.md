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

The Pico is owned by the **daemon**; `run` and `keyboard` are TCP clients that
connect to it (so they can run off-Pi and coexist — last-writer-wins).

```sh
# Live: start the daemon on the Pi (owns the Pico), then drive through it.
uv run waymario daemon --port /dev/ttyACM0          # binds 0.0.0.0:9999, logs both directions
uv run waymario run --daemon <pi-ip>:9999           # the brain, as a client
uv run waymario keyboard --daemon <pi-ip>:9999      # manual control, as a client

# Dry run on a recording, no Pico: point run at a --no-serial daemon.
uv run waymario daemon --no-serial &                # network path only, no hardware
uv run waymario run --video clips/rainbow_road.mp4 --loop --daemon 127.0.0.1:9999

# Tune vision with a debug overlay (q to quit); --stream serves MJPEG instead of a window.
# preview sends NO controller output -- it's vision-only, so it has no --daemon.
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
  `TcpLink` writes to the daemon over the network; `NullLink` records the last frame
  for tests. `encode()`/`decode()` are the wire-frame codec.
- **`daemon.py`** — `ControllerDaemon`, a threaded TCP server (`waymario daemon`) that
  owns the one serial link to the Pico and relays the same line protocol over the
  network: client frames → Pico, Pico output → all clients (multiplexed), both
  directions logged to stderr (`[tx …]`/`[rx]`).
- **`config.py`** — one flat `Config` dataclass holding every tunable (ROI bounds,
  brightness threshold, steering gain, max stick, baud, fps) plus `_PLAYER_REGIONS`,
  the split-screen quadrant table. New knobs go here, not threaded through calls.
- **`drive.py`** — the loop: capture → steer → policy → send, paced to `target_fps`.
  Always neutralizes the controller in a `finally` (including on Ctrl-C) so the kart
  doesn't keep its last command.
- **`stream.py`** — `MJPEGServer`, a threaded MJPEG-over-HTTP broadcaster for watching
  the `preview` overlay headlessly (e.g. over SSH).
- **`cli.py`** — argparse front end; `_build_source` selects capture vs video, and
  `_connect_daemon` builds the `TcpLink` for the daemon clients. Subcommands:
  `daemon` (owns the Pico), `run` and `keyboard` (both connect to the daemon over
  TCP via `--daemon HOST:PORT`), and `preview` (vision-only, sends nothing).

## Pi ↔ Pico contract

`firmware/` holds only the **protocol contract** — the PIO/joybus firmware is future
work. Pi → Pico is an **ASCII line protocol**: `<buttons>,<stick_x>,<stick_y>\n`
(buttons `a b z r l s`, sticks `-80..+80`, `,0,0` = neutral), default 115200 baud.
The Pico holds the last state until a new line arrives, and prints status/`dbg:`
lines back — `SerialLink` echoes those as `[pico] …`. `waymario.transport.encode()`
is the reference encoder; keep it and `firmware/README.md` in sync if the frame
changes. (Separately, Pico → N64 is the joybus status word; its bit layout matches
`control.Button` and `ControllerState.to_n64_bytes()`.)
