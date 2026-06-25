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

## Serial wire protocol (Pi → Pico)

Fixed **6-byte frame**, little ceremony, easy to parse on the MCU:

| Offset | Byte       | Meaning                                            |
|--------|------------|----------------------------------------------------|
| 0      | `0xA5`     | frame header / resync marker                       |
| 1      | `btn_hi`   | N64 status byte 0 (A, B, Z, Start, D-pad)          |
| 2      | `btn_lo`   | N64 status byte 1 (L, R, C-buttons)                |
| 3      | `stick_x`  | analog X, signed int8 (two's complement)           |
| 4      | `stick_y`  | analog Y, signed int8 (two's complement)           |
| 5      | `xor`      | XOR of bytes 1–4 (checksum)                         |

- Default baud: **115200**.
- `btn_hi`/`btn_lo` map **directly** onto the joybus poll response button bytes,
  so the Pico can forward them with no remapping. Bit layout matches
  `waymario.control.Button`.
- Bytes 1–4 are exactly the 4 canonical N64 status bytes produced by
  `ControllerState.to_n64_bytes()`.

Reference encoder: `waymario.transport.encode()`.

## Pico responsibilities (to implement)

1. Read 6-byte frames over USB CDC / UART; resync on `0xA5`; verify `xor`.
2. Hold the latest valid state.
3. On each joybus poll from the console, emit the standard 0x01 (status) and
   0x00/0xFF (identify) responses, filling the 4 data bytes from the held state.
4. Fail safe to neutral (all zero) if no valid frame arrives within a timeout.
