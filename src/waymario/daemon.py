"""Network controller daemon: own the Pico, expose it over TCP.

``waymario daemon`` runs on the Pi, holds the single serial link to the Pico, and
listens on ``0.0.0.0`` so any number of clients can drive it over the network.
Clients send the exact same ASCII frames they would have written to serial
(``<buttons>,<stick_x>,<stick_y>\\n``); the daemon:

- forwards every client frame to the Pico (last-writer-wins across clients),
- reads everything the Pico sends back and **multiplexes it to all clients**,
- logs **both directions to stderr** -- every input it sends to the device
  (``[tx …]``) and everything it gets back (``[rx] …``) -- so the operator can
  watch the link live.

The link is injected via a factory so the daemon stays hardware-free in tests:
``link_factory(on_line)`` builds the device link (a ``SerialLink`` in production,
a ``NullLink`` for ``--no-serial``, or a ``FakeSerial``-backed link under test)
with the daemon's Pico-line sink already wired in.
"""

from __future__ import annotations

import socketserver
import sys
import threading
from collections.abc import Callable

from .control import ControllerState
from .transport import ControllerLink, decode


class _ClientHandler(socketserver.StreamRequestHandler):
    """One thread per connected client: relay its frames to the device."""

    def handle(self) -> None:
        daemon: ControllerDaemon = self.server.daemon  # type: ignore[attr-defined]
        addr = f"{self.client_address[0]}:{self.client_address[1]}"
        daemon._add_client(self.wfile, addr)
        try:
            for raw in self.rfile:  # blocks until a newline or disconnect
                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                try:
                    state = decode(line)
                except ValueError as exc:
                    daemon._log(f"[bad {addr}] {exc}")
                    continue
                daemon._log(f"[tx {addr}] {line}")
                daemon._send(state)
        except OSError:
            pass  # client vanished mid-read
        finally:
            daemon._remove_client(self.wfile, addr)


class _Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


class ControllerDaemon:
    """Multiplex TCP clients onto one Pico link, fanning device output back out."""

    def __init__(
        self,
        host: str,
        port: int,
        link_factory: Callable[[Callable[[str], None]], ControllerLink],
    ) -> None:
        # Wire our broadcast+log sink into the link's reader before serving.
        self._link = link_factory(self._on_pico_line)
        self._clients: set = set()
        self._clients_lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._server = _Server((host, port), _ClientHandler)
        self._server.daemon = self  # type: ignore[attr-defined]

    @property
    def address(self) -> tuple[str, int]:
        """The bound ``(host, port)`` -- resolves port 0 to the OS-chosen port."""
        return self._server.server_address  # type: ignore[return-value]

    @property
    def client_count(self) -> int:
        with self._clients_lock:
            return len(self._clients)

    def serve_forever(self) -> None:
        host, port = self.address
        self._log(f"[daemon] listening on {host}:{port}")
        self._server.serve_forever()

    def shutdown(self) -> None:
        """Stop serving, release the kart to neutral, and close everything."""
        self._server.shutdown()
        self._server.server_close()
        with self._write_lock:
            self._link.send(ControllerState())  # neutral
        self._link.close()
        with self._clients_lock:
            self._clients.clear()
        self._log("[daemon] stopped (controller released to neutral)")

    # --- internal plumbing ------------------------------------------------

    def _send(self, state: ControllerState) -> None:
        with self._write_lock:
            self._link.send(state)

    def _on_pico_line(self, text: str) -> None:
        """Called by the link's reader thread for each line the Pico sends."""
        self._log(f"[rx] {text}")
        data = (text + "\n").encode()
        with self._clients_lock:
            dead = []
            for wfile in self._clients:
                try:
                    wfile.write(data)
                    wfile.flush()
                except OSError:
                    dead.append(wfile)
            for wfile in dead:
                self._clients.discard(wfile)

    def _add_client(self, wfile, addr: str) -> None:
        with self._clients_lock:
            self._clients.add(wfile)
        self._log(f"[client +{addr}]")

    def _remove_client(self, wfile, addr: str) -> None:
        with self._clients_lock:
            self._clients.discard(wfile)
        self._log(f"[client -{addr}]")

    def _log(self, message: str) -> None:
        sys.stderr.write(message + "\n")
        sys.stderr.flush()
