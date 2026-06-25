"""Command-line entry point: ``waymario run`` and ``waymario preview``."""

from __future__ import annotations

import argparse
import sys

from .capture import CaptureDeviceSource, FrameSource, VideoFileSource
from .config import Config
from .control import Button, ControllerState, drive_policy
from .drive import run
from .steering import build_steerer
from .stuck import StuckDetector
from .transport import NullLink, SerialLink, TcpLink

# Button chips drawn in the preview HUD, in controller-ish order. Each lights up
# when the corresponding bit is set in the controller state.
_BUTTON_DISPLAY: list[tuple[str, Button]] = [
    ("A", Button.A),
    ("B", Button.B),
    ("Z", Button.Z),
    ("L", Button.L),
    ("R", Button.R),
    ("ST", Button.START),
    ("C^", Button.C_UP),
    ("Cv", Button.C_DOWN),
    ("C<", Button.C_LEFT),
    ("C>", Button.C_RIGHT),
]


def _draw_buttons(img, state: ControllerState, x: int, y: int) -> None:
    """Draw a row of button chips at image-local (x, y); pressed ones light up green."""
    import cv2

    cw, ch, gap = 32, 22, 4
    for i, (label, bit) in enumerate(_BUTTON_DISPLAY):
        pressed = bool(state.buttons & bit)
        bx = x + i * (cw + gap)
        cv2.rectangle(img, (bx, y), (bx + cw, y + ch), (0, 200, 0) if pressed else (45, 45, 45), -1)
        cv2.rectangle(img, (bx, y), (bx + cw, y + ch), (210, 210, 210), 1)
        cv2.putText(img, label, (bx + 4, y + ch - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255) if pressed else (140, 140, 140), 1)


def _build_source(args: argparse.Namespace, config: Config) -> FrameSource:
    if args.video:
        return VideoFileSource(args.video, loop=args.loop)
    return CaptureDeviceSource(config.device, config.width, config.height)


def _connect_daemon(
    daemon_arg: str, config: Config, *, on_line=None
) -> TcpLink | None:
    """Connect a TcpLink to the daemon, or print an error and return None."""
    host, port = _parse_host_port(daemon_arg, config.daemon_port)
    try:
        return TcpLink(host, port, on_line=on_line)
    except (ConnectionError, OSError) as exc:
        print(f"error: couldn't reach daemon at {host}:{port}: {exc}", file=sys.stderr)
        return None


def _apply_player_args(args: argparse.Namespace, config: Config) -> None:
    """Copy --players / --player from CLI args into config."""
    config.players = args.players
    config.player = args.player


def _cmd_run(args: argparse.Namespace) -> int:
    config = Config()
    config.steerer = args.steerer
    if args.device is not None:
        config.device = args.device
    _apply_player_args(args, config)
    link = _connect_daemon(args.daemon, config)
    if link is None:
        return 1
    with link, _build_source(args, config) as source:
        run(source, build_steerer(config), link, config, debug=args.debug)
    return 0


def _parse_host_port(value: str, default_port: int) -> tuple[str, int]:
    """Parse a ``HOST`` or ``HOST:PORT`` string, falling back to ``default_port``."""
    host, sep, port_str = value.rpartition(":")
    if not sep:
        return value, default_port
    return host, int(port_str)


def _cmd_keyboard(args: argparse.Namespace) -> int:
    """Manually drive the console from the keyboard, via the controller daemon."""
    from .keyboard import LatestLine, run_keyboard

    config = Config()
    # Capture the daemon's broadcasts into the status line instead of letting them
    # scroll past the controls.
    latest = LatestLine()
    link = _connect_daemon(args.daemon, config, on_line=latest.set)
    if link is None:
        return 1
    try:
        with link:
            run_keyboard(link, config, status=latest.get)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def _cmd_daemon(args: argparse.Namespace) -> int:
    """Run the controller daemon: own the Pico, expose it over TCP."""
    from .daemon import ControllerDaemon

    config = Config()
    if args.port:
        config.serial_port = args.port

    def link_factory(on_line):
        if args.no_serial:
            return NullLink()
        return SerialLink(config.serial_port, config.baud, on_line=on_line)

    daemon = ControllerDaemon(args.host, args.listen_port, link_factory)
    try:
        daemon.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        daemon.shutdown()
    return 0


def _cmd_preview(args: argparse.Namespace) -> int:
    """Show the CV overlay (ROI box, centroid, steering) for tuning. No output sent."""
    import cv2

    from .stream import MJPEGServer

    config = Config()
    if args.device is not None:
        config.device = args.device
    _apply_player_args(args, config)
    config.steerer = args.steerer
    steerer = build_steerer(config)
    stuck = StuckDetector(config)

    use_stream = args.stream

    def _subframe(frame: "cv2.typing.MatLike"):
        """Return (subframe, x0, y0) — the player's screen quadrant and its offset."""
        h, w = frame.shape[:2]
        px0, py0, px1, py1 = config.player_region()
        x0, y0 = int(w * px0), int(h * py0)
        x1, y1 = int(w * px1), int(h * py1)
        return frame[y0:y1, x0:x1], x0, y0

    def _hud(frame: "cv2.typing.MatLike", decision, state, phase: str) -> None:
        """Draw status bar at the top of the player's subframe."""
        sub, x0, y0 = _subframe(frame)
        sh, sw = sub.shape[:2]
        bar_color = (0, 0, 180) if phase != "NORMAL" else (0, 80, 0)
        cv2.rectangle(frame, (x0, y0), (x0 + sw, y0 + 36), bar_color, -1)
        mode_tag = f"[{phase}]" if phase != "NORMAL" else "[DRIVE]"
        hue_txt = f"  hue={decision.hue:.0f}" if decision.hue is not None else ""
        text = (
            f"P{config.player} {mode_tag}  "
            f"conf={decision.confidence:.3f}  "
            f"steer={decision.steering:+.2f}{hue_txt}  "
            f"stick=({state.stick_x:+d},{state.stick_y:+d})  "
            f"dir={stuck.last_direction}({stuck.last_gradient:+.0f})"
        )
        cv2.putText(frame, text, (x0 + 8, y0 + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        _draw_buttons(frame, state, x0 + 8, y0 + 42)

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
        # Wrong-way strip: the near look-ahead window whose near->far hue gradient
        # tells forward from reversed (drawn orange to distinguish from the steer ROI).
        wx0 = x0 + int(sub_w * config.wrong_way_roi_left)
        wx1 = x0 + int(sub_w * config.wrong_way_roi_right)
        wy0 = y0 + int(sub_h * config.wrong_way_roi_top)
        wy1 = y0 + int(sub_h * config.wrong_way_roi_bottom)
        cv2.rectangle(frame, (wx0, wy0), (wx1, wy1), (0, 165, 255), 1)
        if decision.lateral is not None:
            cx = x0 + int((decision.lateral + 1) / 2 * sub_w)
            cv2.line(frame, (cx, y0 + ry0), (cx, y0 + ry1), (0, 0, 255), 2)
        _hud(frame, decision, state, phase)
        return frame, decision, state, phase

    def _debug_mosaic(frame: "cv2.typing.MatLike") -> "cv2.typing.MatLike":
        """Build a 2x2 mosaic of the player's subframe showing every preprocessing step."""
        import numpy as np
        decision = steerer.decide(frame)  # run CV once, reuse below
        recovery = stuck.update(frame, decision)
        state = recovery if recovery is not None else drive_policy(decision, config)
        phase = stuck._phase.name
        sub, _x0, _y0 = _subframe(frame)
        sub_h, sub_w = sub.shape[:2]
        top = int(sub_h * config.roi_top)
        bottom = int(sub_h * config.roi_bottom)

        # Panel 1 — player subframe with overlay
        p1 = sub.copy()
        cv2.rectangle(p1, (0, top), (sub_w, bottom), (0, 255, 0), 2)
        if decision.lateral is not None:
            cx = int((decision.lateral + 1) / 2 * sub_w)
            cv2.line(p1, (cx, top), (cx, bottom), (0, 0, 255), 2)
        bar_color = (0, 0, 180) if phase != "NORMAL" else (0, 80, 0)
        cv2.rectangle(p1, (0, 0), (sub_w, 36), bar_color, -1)
        cv2.putText(p1,
                    f"1:[{phase}] conf={decision.confidence:.3f} steer={decision.steering:+.2f} stick=({state.stick_x:+d},{state.stick_y:+d})",
                    (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        _draw_buttons(p1, state, 8, 42)

        # Panel 2 — ROI crop (padded back to subframe size)
        roi = sub[top:bottom, :]
        p2 = np.zeros_like(sub)
        p2[top:bottom, :] = roi
        cv2.rectangle(p2, (0, top), (sub_w, bottom), (0, 255, 0), 2)
        cv2.putText(p2, f"2: ROI  top={config.roi_top:.2f} bot={config.roi_bottom:.2f}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)

        # Panel 3 — grayscale
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray_full = np.zeros((sub_h, sub_w), dtype=np.uint8)
        gray_full[top:bottom, :] = gray_roi
        p3 = cv2.cvtColor(gray_full, cv2.COLOR_GRAY2BGR)
        cv2.putText(p3, "3: grayscale",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)

        # Panel 4 — binary threshold mask
        _, mask_roi = cv2.threshold(gray_roi, config.bright_threshold, 255, cv2.THRESH_BINARY)
        mask_full = np.zeros((sub_h, sub_w), dtype=np.uint8)
        mask_full[top:bottom, :] = mask_roi
        p4 = cv2.cvtColor(mask_full, cv2.COLOR_GRAY2BGR)
        lit_pct = mask_roi.mean() / 255 * 100
        cv2.putText(p4, f"4: threshold>{config.bright_threshold}  lit={lit_pct:.1f}%",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
        if decision.lateral is not None:
            cx = int((decision.lateral + 1) / 2 * sub_w)
            cv2.line(p4, (cx, top), (cx, bottom), (0, 0, 255), 2)

        # Stack into 2x2 grid
        top_row = np.hstack([p1, p2])
        bot_row = np.hstack([p3, p4])
        return np.vstack([top_row, bot_row]), decision, state, phase

    process = _debug_mosaic if (args.debug and config.steerer == "brightness") else _process_frame

    if args.capture_frames:
        return _capture(args, source_builder=lambda: _build_source(args, config), process=process)

    import time
    frame_period = 1.0 / args.stream_fps

    if use_stream:
        with _build_source(args, config) as source, MJPEGServer(port=args.stream_port) as server:
            for frame in source.frames():
                t0 = time.monotonic()
                server.push(process(frame)[0])
                elapsed = time.monotonic() - t0
                if elapsed < frame_period:
                    time.sleep(frame_period - elapsed)
    else:
        with _build_source(args, config) as source:
            for frame in source.frames():
                t0 = time.monotonic()
                cv2.imshow("waymario preview", process(frame)[0])
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                elapsed = time.monotonic() - t0
                if elapsed < frame_period:
                    time.sleep(frame_period - elapsed)
        cv2.destroyAllWindows()
    return 0


def _capture(args: argparse.Namespace, source_builder, process) -> int:
    """Headless: process and save every Nth frame to a directory, with a metadata line
    per saved frame so a reader can pick which PNGs to open. No window, no stream."""
    import os
    import tempfile

    import cv2

    out_dir = args.capture_dir or tempfile.mkdtemp(prefix="waymario-capture-")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Capture dir: {out_dir}", flush=True)

    saved = 0
    with source_builder() as source:
        for i, frame in enumerate(source.frames()):
            if i % args.capture_frames:
                continue
            img, decision, state, phase = process(frame)
            name = f"frame_{i:06d}.png"
            cv2.imwrite(os.path.join(out_dir, name), img)
            lateral = "none" if decision.lateral is None else f"{decision.lateral:+.3f}"
            print(
                f"{name}  conf={decision.confidence:.3f} steer={decision.steering:+.2f} "
                f"stick=({state.stick_x:+d},{state.stick_y:+d}) lateral={lateral} phase={phase}",
                flush=True,
            )
            saved += 1
            if saved >= args.capture_count:
                break
    print(f"Saved {saved} frame(s) to {out_dir}", flush=True)
    return 0


def _add_source_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--video", help="replay a video file instead of the capture device")
    parser.add_argument("--loop", action="store_true", help="loop the video file")


def _add_player_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--players",
        type=int,
        default=1,
        choices=[1, 2, 3, 4],
        metavar="N",
        help="total number of players in the game (1-4), sets the split-screen layout",
    )
    parser.add_argument(
        "--player",
        type=int,
        default=1,
        metavar="N",
        help="which player slot this bot is (1-based, must be <= --players)",
    )


def _add_steerer_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--steerer",
        choices=["hsv", "brightness"],
        default="hsv",
        help="steering algorithm: hsv (color-band, default) or brightness (centroid)",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="waymario", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="drive live: capture -> steer -> controller (via the daemon)")
    _add_source_args(p_run)
    _add_player_args(p_run)
    _add_steerer_arg(p_run)
    p_run.add_argument(
        "--daemon",
        default="127.0.0.1",
        metavar="HOST[:PORT]",
        help="address of the controller daemon to send frames to (default: 127.0.0.1:9999)",
    )
    p_run.add_argument("--debug", action="store_true", help="print confidence/steering/phase every 10 frames")
    p_run.add_argument("--device", type=int, default=None, metavar="N", help="V4L2 video device index (default: 1)")
    p_run.set_defaults(func=_cmd_run)

    p_keyboard = sub.add_parser(
        "keyboard",
        help="manually drive from the keyboard for debugging (connects to the daemon)",
        description=(
            "Drive the console by hand from the terminal, through a running "
            "'waymario daemon'. Keys: arrows or WASD steer the stick, space=A, b=B, "
            "z=Z, l/r=L/R, enter=Start, q/Ctrl-C to quit."
        ),
    )
    p_keyboard.add_argument(
        "--daemon",
        default="127.0.0.1",
        metavar="HOST[:PORT]",
        help="address of the controller daemon (default: 127.0.0.1:9999)",
    )
    p_keyboard.set_defaults(func=_cmd_keyboard)

    p_daemon = sub.add_parser(
        "daemon",
        help="run the controller daemon: own the Pico, expose it over TCP",
        description=(
            "Hold the serial link to the Pico and relay controller frames from any "
            "number of TCP clients, multiplexing the Pico's output back to all of "
            "them. Logs every frame sent and received to stderr."
        ),
    )
    p_daemon.add_argument("--port", help="serial port to the Pi Pico")
    p_daemon.add_argument("--no-serial", action="store_true", help="use NullLink (no Pico)")
    p_daemon.add_argument(
        "--host",
        default="0.0.0.0",
        metavar="ADDR",
        help="address to bind the TCP listener to (default: 0.0.0.0)",
    )
    p_daemon.add_argument(
        "--listen-port",
        type=int,
        default=Config.daemon_port,
        metavar="PORT",
        help="TCP port to listen on (default: 9999)",
    )
    p_daemon.set_defaults(func=_cmd_daemon)

    p_preview = sub.add_parser("preview", help="show the CV overlay for tuning (no output)")
    _add_source_args(p_preview)
    _add_player_args(p_preview)
    _add_steerer_arg(p_preview)
    p_preview.add_argument("--device", type=int, default=None, metavar="N", help="V4L2 video device index (default: 1)")
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
    p_preview.add_argument(
        "--capture-frames",
        type=int,
        metavar="N",
        help="headless: process and save every Nth frame as a PNG (no window/stream), then exit",
    )
    p_preview.add_argument(
        "--capture-count",
        type=int,
        default=12,
        metavar="K",
        help="stop after saving K frames in --capture-frames mode (default: 12)",
    )
    p_preview.add_argument(
        "--capture-dir",
        metavar="DIR",
        help="output directory for --capture-frames (default: a fresh temp dir)",
    )
    p_preview.set_defaults(func=_cmd_preview)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    return args.func(args)
