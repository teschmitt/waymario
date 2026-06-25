"""Command-line entry point: ``waymario run`` and ``waymario preview``."""

from __future__ import annotations

import argparse

import cv2

from .capture import CaptureDeviceSource, FrameSource, VideoFileSource
from .config import Config
from .control import ControllerState, drive_policy
from .drive import run
from .steering import OpenCVSteerer, SteeringDecision
from .stuck import StuckDetector
from .transport import ControllerLink, NullLink, SerialLink


def _build_source(args: argparse.Namespace, config: Config) -> FrameSource:
    if args.video:
        return VideoFileSource(args.video, loop=args.loop)
    return CaptureDeviceSource(config.device, config.width, config.height)


def _build_link(args: argparse.Namespace, config: Config) -> ControllerLink:
    if args.no_serial:
        return NullLink()
    return SerialLink(config.serial_port, config.baud)


def _apply_player_args(args: argparse.Namespace, config: Config) -> None:
    """Copy --players / --player from CLI args into config."""
    config.players = args.players
    config.player = args.player


# ---------------------------------------------------------------------------
# Shared annotation helpers (used by both `run --stream` and `preview`)
# ---------------------------------------------------------------------------

def _subframe(frame: cv2.typing.MatLike, config: Config):
    """Return (subframe, x0, y0) for the player's screen quadrant."""
    h, w = frame.shape[:2]
    px0, py0, px1, py1 = config.player_region()
    x0, y0 = int(w * px0), int(h * py0)
    x1, y1 = int(w * px1), int(h * py1)
    return frame[y0:y1, x0:x1], x0, y0


def _annotate_frame(
    frame: cv2.typing.MatLike,
    decision: SteeringDecision,
    state: ControllerState,
    phase: str,
    config: Config,
) -> cv2.typing.MatLike:
    """Draw ROI box, centroid line and HUD onto a copy of *frame*."""
    frame = frame.copy()
    sub, x0, y0 = _subframe(frame, config)
    sub_h, sub_w = sub.shape[:2]
    top = y0 + int(sub_h * config.roi_top)
    bottom = y0 + int(sub_h * config.roi_bottom)

    cv2.rectangle(frame, (x0, top), (x0 + sub_w, bottom), (0, 255, 0), 1)
    if decision.centroid_x is not None:
        cx = x0 + int((decision.centroid_x + 1) / 2 * sub_w)
        cv2.line(frame, (cx, top), (cx, bottom), (0, 0, 255), 2)

    bar_color = (0, 0, 180) if phase != "NORMAL" else (0, 80, 0)
    cv2.rectangle(frame, (x0, y0), (x0 + sub_w, y0 + 36), bar_color, -1)
    mode_tag = f"[{phase}]" if phase != "NORMAL" else "[DRIVE]"
    text = (
        f"P{config.player} {mode_tag}  "
        f"conf={decision.confidence:.3f}  "
        f"steer={decision.steering:+.2f}  "
        f"stick=({state.stick_x:+d},{state.stick_y:+d})"
    )
    cv2.putText(frame, text, (x0 + 8, y0 + 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return frame


# ---------------------------------------------------------------------------
# run command
# ---------------------------------------------------------------------------

def _cmd_run(args: argparse.Namespace) -> int:
    from .stream import MJPEGServer

    config = Config()
    if args.port:
        config.serial_port = args.port
    if args.device is not None:
        config.device = args.device
    _apply_player_args(args, config)

    on_frame = None
    server: MJPEGServer | None = None
    if args.stream:
        server = MJPEGServer(port=args.stream_port)
        server.start()

        def on_frame(frame, decision, state, phase):  # type: ignore[misc]
            annotated = _annotate_frame(frame, decision, state, phase, config)
            server.push(annotated)  # type: ignore[union-attr]

    try:
        with _build_source(args, config) as source, _build_link(args, config) as link:
            run(source, OpenCVSteerer(config), link, config, debug=args.debug, on_frame=on_frame)
    finally:
        if server is not None:
            server.stop()
    return 0



def _cmd_preview(args: argparse.Namespace) -> int:
    """Show the CV overlay (ROI box, centroid, steering) for tuning. No output sent."""
    from .stream import MJPEGServer

    config = Config()
    if args.device is not None:
        config.device = args.device
    _apply_player_args(args, config)
    steerer = OpenCVSteerer(config)
    stuck = StuckDetector(config)

    use_stream = args.stream

    def _process_frame(frame: cv2.typing.MatLike):
        """Annotate frame with ROI box, centroid line, HUD and stuck state."""
        decision = steerer.decide(frame)
        recovery = stuck.update(frame, decision)
        state = recovery if recovery is not None else drive_policy(decision, config)
        phase = stuck._phase.name
        annotated = _annotate_frame(frame, decision, state, phase, config)
        return annotated, decision, state, phase

    def _debug_mosaic(frame: cv2.typing.MatLike):
        """Build a 2x2 mosaic of the player's subframe showing every preprocessing step."""
        import numpy as np
        decision = steerer.decide(frame)  # run CV once, reuse below
        recovery = stuck.update(frame, decision)
        state = recovery if recovery is not None else drive_policy(decision, config)
        phase = stuck._phase.name
        sub, _x0, _y0 = _subframe(frame, config)
        sub_h, sub_w = sub.shape[:2]
        top = int(sub_h * config.roi_top)
        bottom = int(sub_h * config.roi_bottom)

        # Panel 1 — player subframe with overlay
        p1 = sub.copy()
        cv2.rectangle(p1, (0, top), (sub_w, bottom), (0, 255, 0), 2)
        if decision.centroid_x is not None:
            cx = int((decision.centroid_x + 1) / 2 * sub_w)
            cv2.line(p1, (cx, top), (cx, bottom), (0, 0, 255), 2)
        bar_color = (0, 0, 180) if phase != "NORMAL" else (0, 80, 0)
        cv2.rectangle(p1, (0, 0), (sub_w, 36), bar_color, -1)
        cv2.putText(p1,
                    f"1:[{phase}] conf={decision.confidence:.3f} steer={decision.steering:+.2f} stick=({state.stick_x:+d},{state.stick_y:+d})",
                    (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

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
        if decision.centroid_x is not None:
            cx = int((decision.centroid_x + 1) / 2 * sub_w)
            cv2.line(p4, (cx, top), (cx, bottom), (0, 0, 255), 2)

        # Stack into 2x2 grid
        top_row = np.hstack([p1, p2])
        bot_row = np.hstack([p3, p4])
        return np.vstack([top_row, bot_row]), decision, state, phase

    process = _debug_mosaic if args.debug else _process_frame

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
            centroid = "none" if decision.centroid_x is None else f"{decision.centroid_x:+.3f}"
            print(
                f"{name}  conf={decision.confidence:.3f} steer={decision.steering:+.2f} "
                f"stick=({state.stick_x:+d},{state.stick_y:+d}) centroid={centroid} phase={phase}",
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


def main() -> int:
    parser = argparse.ArgumentParser(prog="waymario", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="drive live: capture -> steer -> controller")
    _add_source_args(p_run)
    _add_player_args(p_run)
    p_run.add_argument("--port", help="serial port to the Pi Pico")
    p_run.add_argument("--no-serial", action="store_true", help="use NullLink (no Pico)")
    p_run.add_argument("--debug", action="store_true", help="print confidence/steering/phase every 10 frames")
    p_run.add_argument("--device", type=int, default=None, metavar="N", help="V4L2 video device index (default: 1)")
    p_run.add_argument(
        "--stream",
        action="store_true",
        help="serve a live annotated MJPEG stream while driving (same overlay as 'preview --stream')",
    )
    p_run.add_argument(
        "--stream-port",
        type=int,
        default=8080,
        metavar="PORT",
        help="port for the MJPEG HTTP server when --stream is set (default: 8080)",
    )
    p_run.set_defaults(func=_cmd_run)

    p_preview = sub.add_parser("preview", help="show the CV overlay for tuning (no output)")
    _add_source_args(p_preview)
    _add_player_args(p_preview)
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
        help="show 2x2 mosaic of preprocessing steps (original / ROI / grayscale / threshold)",
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

    args = parser.parse_args()
    return args.func(args)
