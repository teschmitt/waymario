"""Command-line entry point: ``waymario run`` and ``waymario preview``."""

from __future__ import annotations

import argparse

from .capture import CaptureDeviceSource, FrameSource, VideoFileSource
from .config import Config
from .control import drive_policy
from .drive import run
from .steering import OpenCVSteerer
from .transport import ControllerLink, NullLink, SerialLink


def _build_source(args: argparse.Namespace, config: Config) -> FrameSource:
    if args.video:
        return VideoFileSource(args.video, loop=args.loop)
    return CaptureDeviceSource(config.device, config.width, config.height)


def _build_link(args: argparse.Namespace, config: Config) -> ControllerLink:
    if args.no_serial:
        return NullLink()
    return SerialLink(config.serial_port, config.baud)


def _cmd_run(args: argparse.Namespace) -> int:
    config = Config()
    if args.port:
        config.serial_port = args.port
    with _build_source(args, config) as source, _build_link(args, config) as link:
        run(source, OpenCVSteerer(config), link, config)
    return 0


def _cmd_preview(args: argparse.Namespace) -> int:
    """Show the CV overlay (ROI box, centroid, steering) for tuning. No output sent."""
    import cv2

    config = Config()
    steerer = OpenCVSteerer(config)
    with _build_source(args, config) as source:
        for frame in source.frames():
            decision = steerer.decide(frame)
            state = drive_policy(decision, config)

            height, width = frame.shape[:2]
            top = int(height * config.roi_top)
            bottom = int(height * config.roi_bottom)
            cv2.rectangle(frame, (0, top), (width, bottom), (0, 255, 0), 1)
            if decision.centroid_x is not None:
                cx = int((decision.centroid_x + 1) / 2 * width)
                cv2.line(frame, (cx, top), (cx, bottom), (0, 0, 255), 2)
            cv2.putText(
                frame,
                f"steer={decision.steering:+.2f} conf={decision.confidence:.3f} stick={state.stick_x:+d}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            cv2.imshow("waymario preview", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cv2.destroyAllWindows()
    return 0


def _add_source_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--video", help="replay a video file instead of the capture device")
    parser.add_argument("--loop", action="store_true", help="loop the video file")


def main() -> int:
    parser = argparse.ArgumentParser(prog="waymario", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="drive live: capture -> steer -> controller")
    _add_source_args(p_run)
    p_run.add_argument("--port", help="serial port to the Pi Pico")
    p_run.add_argument("--no-serial", action="store_true", help="use NullLink (no Pico)")
    p_run.set_defaults(func=_cmd_run)

    p_preview = sub.add_parser("preview", help="show the CV overlay for tuning (no output)")
    _add_source_args(p_preview)
    p_preview.set_defaults(func=_cmd_preview)

    args = parser.parse_args()
    return args.func(args)
