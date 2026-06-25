"""MJPEG-over-HTTP streaming server.

Serves a live Motion JPEG stream on ``http://<host>:<port>/`` so the CV overlay
can be watched from any browser on the same network — no display needed on the
Pi, works fine over SSH from Windows.

Usage::

    with MJPEGServer(host="0.0.0.0", port=8080) as server:
        for frame in source.frames():
            server.push(frame)

The server runs its own daemon thread; ``push()`` is thread-safe and
non-blocking — if no client is connected the frame is just dropped.
"""

from __future__ import annotations

import logging
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import cv2
import numpy as np

log = logging.getLogger(__name__)

_BOUNDARY = b"--waymarioframe"
_CONTENT_TYPE = b"image/jpeg"


class MJPEGServer:
    """Lightweight MJPEG broadcast server.

    One shared latest-frame slot: every connected client gets the most recent
    frame available.  Slow clients don't block the capture loop.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 8080) -> None:
        self._host = host
        self._port = port
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._latest: bytes | None = None
        self._running = False
        self._server: HTTPServer | None = None
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def push(self, frame: np.ndarray, quality: int = 70) -> None:
        """Encode *frame* as JPEG and notify waiting client threads."""
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            return
        data = buf.tobytes()
        with self._condition:
            self._latest = data
            self._condition.notify_all()

    def start(self) -> None:
        """Start the HTTP server in a background daemon thread."""
        server = _make_server(self._host, self._port, self)
        self._server = server
        self._running = True
        self._thread = threading.Thread(target=server.serve_forever, daemon=True, name="mjpeg-server")
        self._thread.start()
        host_display = self._host if self._host != "0.0.0.0" else _local_ip()
        log.info("MJPEG stream at http://%s:%d/", host_display, self._port)
        print(f"[waymario] MJPEG stream → http://{host_display}:{self._port}/  (open in browser)")

    def stop(self) -> None:
        if self._server:
            self._server.shutdown()
            self._server = None
        self._running = False

    def __enter__(self) -> MJPEGServer:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Internal helpers used by the request handler
    # ------------------------------------------------------------------

    def _wait_for_frame(self, last: bytes | None) -> bytes | None:
        """Block until a frame newer than *last* is available, or server stops."""
        with self._condition:
            while self._running and self._latest is last:
                self._condition.wait(timeout=1.0)
            return self._latest


# ---------------------------------------------------------------------------
# HTTP plumbing
# ---------------------------------------------------------------------------

def _make_server(host: str, port: int, mjpeg: MJPEGServer) -> HTTPServer:
    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: object) -> None:  # silence access log
            pass

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/":
                self._serve_stream()
            else:
                self.send_error(404)

        def _serve_stream(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={_BOUNDARY.decode()}")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()

            last: bytes | None = None
            try:
                while mjpeg._running:
                    frame_bytes = mjpeg._wait_for_frame(last)
                    if frame_bytes is None:
                        continue
                    last = frame_bytes
                    header = (
                        _BOUNDARY + b"\r\n"
                        + b"Content-Type: " + _CONTENT_TYPE + b"\r\n"
                        + b"Content-Length: " + str(len(frame_bytes)).encode() + b"\r\n"
                        + b"\r\n"
                    )
                    try:
                        self.wfile.write(header + frame_bytes + b"\r\n")
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        break  # client disconnected
            except Exception:
                pass

    return HTTPServer((host, port), _Handler)


def _local_ip() -> str:
    """Best-effort: the LAN IP the Pi would use to reach a router."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "localhost"