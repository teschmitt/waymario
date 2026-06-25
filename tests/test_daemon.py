"""The controller daemon: relay frames to the device, multiplex output to clients."""

from __future__ import annotations

import socket
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager

from waymario.control import Button, ControllerState
from waymario.daemon import ControllerDaemon
from waymario.transport import SerialLink, TcpLink, encode

from test_transport import FakeSerial


@contextmanager
def running_daemon(fake: FakeSerial) -> Iterator[ControllerDaemon]:
    """Start a daemon on an ephemeral port backed by ``fake``; yield the daemon."""
    daemon = ControllerDaemon(
        "127.0.0.1",
        0,
        lambda on_line: SerialLink("ignored", serial_obj=fake, on_line=on_line),
    )
    thread = threading.Thread(target=daemon.serve_forever, daemon=True)
    thread.start()
    try:
        yield daemon
    finally:
        daemon.shutdown()
        thread.join(timeout=2.0)


def _wait_until(predicate, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_client_frame_reaches_device() -> None:
    fake = FakeSerial()
    with running_daemon(fake) as daemon:
        with socket.create_connection(daemon.address) as sock:
            sock.sendall(b"a,80,0\n")
            want = encode(ControllerState(stick_x=80, buttons=Button.A))
            assert _wait_until(lambda: want in fake.writes)


def test_malformed_frame_is_dropped() -> None:
    fake = FakeSerial()
    with running_daemon(fake) as daemon:
        with socket.create_connection(daemon.address) as sock:
            sock.sendall(b"garbage\n")
            sock.sendall(b"b,0,-80\n")
            want = encode(ControllerState(stick_y=-80, buttons=Button.B))
            assert _wait_until(lambda: want in fake.writes)
            assert all(b"garbage" not in w for w in fake.writes)


def test_device_output_multiplexed_to_all_clients() -> None:
    fake = FakeSerial()
    with running_daemon(fake) as daemon:
        a = socket.create_connection(daemon.address)
        b = socket.create_connection(daemon.address)
        a.settimeout(2.0)
        b.settimeout(2.0)
        try:
            assert _wait_until(lambda: daemon.client_count == 2)
            fake.feed_line(b"dbg: hello\n")
            assert a.makefile("rb").readline() == b"dbg: hello\n"
            assert b.makefile("rb").readline() == b"dbg: hello\n"
        finally:
            a.close()
            b.close()


def test_tcp_link_drives_daemon() -> None:
    fake = FakeSerial()
    with running_daemon(fake) as daemon:
        with TcpLink(*daemon.address, echo=False) as link:
            link.send(ControllerState(stick_x=-80, buttons=Button.A))
            want = encode(ControllerState(stick_x=-80, buttons=Button.A))
            assert _wait_until(lambda: want in fake.writes)


def test_shutdown_sends_neutral() -> None:
    fake = FakeSerial()
    with running_daemon(fake):
        pass  # context exit triggers shutdown()
    assert encode(ControllerState()) in fake.writes
    assert fake.closed
