# waymario firmware — Pi Pico N64 controller emulator

This directory holds the RP2040 (Pi Pico) firmware that turns serial commands
from the Pi into a real N64 controller signal. **Only the contract lives here for
now** — the Python brain is built against this protocol; the PIO/joybus
implementation is future work.

## Why a Pico?

The N64 controller port speaks the **joybus** protocol with ~1µs bit timing.
CPython on stock Raspberry Pi OS (non-realtime, GC pauses) cannot meet that. The
Pico's PIO state machines can, so the Pi sends *intent* (button + stick state)
and the Pico handles the wire timing.

There are **two links**, do not confuse them:

- **Pi → Pico** — an ASCII text line protocol (below). This is what `waymario`
  emits over serial.
- **Pico → N64** — the joybus status word on the controller port. Its bit layout
  matches `waymario.control.Button`, and `ControllerState.to_n64_bytes()` is the
  canonical 4-byte status word the Pico fills in for each poll.

## Serial wire protocol (Pi → Pico)

ASCII, **one newline-terminated line per controller state**:

```
<buttons>,<stick_x>,<stick_y>\n
```

| Field      | Meaning                                                          |
|------------|------------------------------------------------------------------|
| `buttons`  | any combination of `a`=A `b`=B `z`=Z `r`=R `l`=L `s`=Start (empty = none) |
| `stick_x`  | `-80..+80`  (negative = left,  positive = right)                 |
| `stick_y`  | `-80..+80`  (negative = down,  positive = up)                    |

Examples:

```
a,0,0       # A pressed, stick centred
ar,80,0     # A + R, full right
az,0,0      # A + Z
,0,0        # no buttons, stick centred (neutral)
b,0,-80     # B only, full reverse
```

- Default baud: **115200**.
- The Pico **holds the last state** until a new line arrives — resend on change is
  enough, though `waymario` currently sends every frame.
- The Pico talks back: a boot banner, a syntax help block, and `dbg:`/`Ready.`
  status lines. `waymario.transport.SerialLink` reads these on a background thread
  and prints them as `[pico] …`.

Reference encoder: `waymario.transport.encode()` (decoder: `decode()`).

## Network layer (Pi ↔ clients)

`waymario daemon` runs on the Pi, holds the single serial link to the Pico, and
relays this **same line protocol over TCP** so you can drive from any machine:

```
waymario daemon --port /dev/ttyACM0      # binds 0.0.0.0:9999 by default
waymario keyboard --daemon <pi-ip>:9999  # reference client
```

Any number of clients connect; each sends the identical `<buttons>,<stick_x>,<stick_y>`
frames. The daemon forwards them to the Pico (last-writer-wins), **multiplexes the
Pico's output back to every connected client**, and logs both directions to stderr
(`[tx …]` for frames it sends to the device, `[rx] …` for what the Pico replies).
Client link: `waymario.transport.TcpLink`.

## Pico responsibilities (to implement)

1. Read newline-terminated ASCII frames over USB CDC / UART; parse buttons + sticks.
2. Hold the latest valid state.
3. On each joybus poll from the console, emit the standard 0x01 (status) and
   0x00/0xFF (identify) responses, filling the 4 data bytes from the held state
   (the `ControllerState.to_n64_bytes()` layout).
4. Fail safe to neutral (`,0,0`) if no valid frame arrives within a timeout.
