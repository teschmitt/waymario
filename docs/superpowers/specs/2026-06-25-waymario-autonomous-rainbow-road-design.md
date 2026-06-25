# waymario — Autonomous Mario Kart 64 Rainbow Road Driver

**Date:** 2026-06-25
**Status:** Approved

## Goal

Steer a player through Mario Kart 64's Rainbow Road on a **real N64 console** with no
human input. Software runs on a Raspberry Pi: it reads the game's video over an HDMI
capture device, decides how to steer using computer vision, and emits a **real N64
controller signal** to the console's controller port.

## Hardware Context

- **Console:** real Nintendo 64, video out via HDMI mod/upscaler.
- **Video in:** USB HDMI capture device on the Pi (V4L2).
- **Compute:** Raspberry Pi running the Python brain.
- **Controller out:** Pi → **Pi Pico (RP2040)** over USB/UART serial. The Pico emulates
  an N64 controller, bit-banging the timing-critical joybus protocol via PIO. Python
  stays pure and only sends high-level button/stick state — it never touches joybus
  timing directly (CPython on stock Pi OS cannot hit ~1µs precision reliably).

## Architecture

A linear pipeline of four small, independently-testable units behind clean interfaces,
wired by one orchestration loop:

```
HDMI capture → [capture] → frame
               [steer]   → SteeringDecision
               [control] → ControllerState
               [transport] → serial → Pi Pico → joybus → N64
```

### Components

1. **`capture.py` — `FrameSource`** (interface): yields BGR frames.
   - `CaptureDeviceSource`: wraps OpenCV `VideoCapture` over the V4L2 HDMI device.
   - `VideoFileSource`: replays a recorded clip — lets the whole pipeline run with no
     capture hardware.

2. **`steering.py` — `Steerer`** (interface): `frame → SteeringDecision`.
   - `OpenCVSteerer` (classical CV): crop a region-of-interest, threshold the bright
     track against the black starfield, find the road centroid, compute the horizontal
     offset from frame center, convert to a proportional steering value in `[-1, 1]`.
   - `SteeringDecision`: `steering: float` (-1 left … +1 right), plus diagnostic fields
     (e.g. confidence / centroid) for tuning.

3. **`control.py` — `ControllerState`** (dataclass): `stick_x`, `stick_y`, and buttons
   (`a`, `b`, `start`, …). A `policy` function maps `SteeringDecision → ControllerState`
   (hold A to accelerate; `stick_x` from steering).

4. **`transport.py` — `ControllerLink`** (interface): serialize `ControllerState` to the
   Pico over a tiny fixed wire protocol.
   - `SerialLink`: `pyserial` to the Pico.
   - `NullLink`: logs only — runs the full brain with no Pico attached.

### Orchestration & support

- **`drive.py`**: the loop — capture → steer → policy → send — at the capture frame
  rate, with graceful Ctrl-C shutdown.
- **`config.py`**: tunables (serial port, ROI, threshold, steering gain, target fps).
- **`cli.py`**: `waymario run` (live) and `waymario preview` (CV debug overlay for tuning).

### Out of Python scope, but scaffolded

- **`firmware/README.md`**: pins the **serial wire protocol** (the Pi↔Pico contract) so
  the Pico's PIO/joybus firmware can be built against a stable interface.

## Project Shape (uv, src layout)

```
pyproject.toml          # uv-managed
src/waymario/           # capture, steering, control, transport, drive, config, cli
tests/                  # unit tests per module
firmware/README.md      # serial protocol spec for the Pico
```

- **Runtime deps:** `opencv-python`, `numpy`, `pyserial`.
- **Dev deps:** `pytest`, `ruff`.
- **Console entry point:** `waymario`.

## Design Principle: no-hardware stand-ins

Every hardware touchpoint has a software substitute — `VideoFileSource` for the camera,
`NullLink` for the serial output. The entire pipeline is runnable and testable on a
laptop before the Pi, Pico, or console are wired up.

## Testing Strategy

- `steering`: feed synthetic / recorded frames, assert steering sign and magnitude.
- `control`: assert the policy maps decisions to expected controller states.
- `transport`: assert `ControllerState` serializes to the expected wire bytes (`NullLink`
  captures output).
- `drive`: integration smoke test with `VideoFileSource` + `NullLink`.

## Non-Goals (initial scaffold)

- ML / learned steering policy (interfaces leave room; not built now).
- Pico firmware implementation (only the protocol contract is specified).
- Lap logic, item use, recovery from falling off — pure track-following first.
