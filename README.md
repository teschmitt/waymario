# waymario

Autonomous **Mario Kart 64 Rainbow Road** driver. Reads the game over an HDMI
capture device, decides how to steer with computer vision, and emits a **real N64
controller signal** — no human input.

## How it works

```
HDMI capture → [capture] → frame
               [steer]   → SteeringDecision   (hsv colour-mask centroid, or brightness centroid)
               [control] → ControllerState     (stick + buttons)
               [transport] → serial → Pi Pico → joybus → N64
```

Runs on a Raspberry Pi. The timing-critical N64 joybus signal is handled by a Pi
Pico (RP2040); the Pi sends high-level button/stick state over serial. See
[`firmware/README.md`](firmware/README.md) for the Pi↔Pico wire protocol.

Every hardware touchpoint has a no-hardware stand-in — replay a video file
instead of the capture device (`--video`), and use `NullLink` (`--no-serial`)
instead of a Pico — so the whole pipeline runs and tests on a laptop.

## Setup

```sh
uv sync
```

## Usage

```sh
# Drive live (Pi + Pico + console wired up)
uv run waymario run --port /dev/ttyACM0

# Dry run on a recording, no Pico
uv run waymario run --video clips/rainbow_road.mp4 --loop --no-serial

# Pick the steering algorithm (default: hsv colour-mask centroid)
uv run waymario run --video clips/rainbow_road.mp4 --no-serial --steerer brightness

# Tune the vision with a debug overlay (press q to quit)
uv run waymario preview --video clips/rainbow_road.mp4 --loop
```

## Develop

```sh
uv run pytest        # tests (no hardware needed)
uv run ruff check    # lint
```

## Layout

| Path                   | Purpose                                             |
|------------------------|-----------------------------------------------------|
| `src/waymario/capture.py`   | `FrameSource`: HDMI device or video file       |
| `src/waymario/steering.py`  | `Steerer` + `OpenCVSteerer`: frame → steering  |
| `src/waymario/control.py`   | `ControllerState` + steering→controls policy   |
| `src/waymario/transport.py` | `ControllerLink`: serial to Pico, or `NullLink`|
| `src/waymario/drive.py`     | the orchestration loop                         |
| `src/waymario/config.py`    | all tunables                                   |
| `firmware/`            | Pi Pico controller-emulator contract                |
| `docs/superpowers/specs/`   | design doc                                     |
