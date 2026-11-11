"""Test helpers for a daemon on a Unix socket: a blocking client and an event-driven file wait."""

import os
import select
import socket
import time
from pathlib import Path

from voxlektor.contracts import protocol as p
from voxlektor.contracts.codec import decode_reply

TIMEOUT = 10.0


def request(path: Path, line: bytes) -> p.Reply:
    """Send one frame on a new connection and return the decoded reply."""
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(TIMEOUT)
        sock.connect(str(path))
        sock.sendall(line)
        data = b""
        while not data.endswith(b"\n"):
            piece = sock.recv(65536)
            if not piece:
                break
            data += piece
    return decode_reply(data)


def wait_for_file(path: Path, timeout: float = TIMEOUT, *, present: bool = True) -> None:
    """Block until ``path`` exists (or, with ``present=False``, is gone).

    Woken by kqueue events on the directory, so there is no polling.
    """
    deadline = time.monotonic() + timeout
    fd = os.open(path.parent, os.O_RDONLY)
    kq = select.kqueue()
    try:
        watch = select.kevent(
            fd,
            filter=select.KQ_FILTER_VNODE,
            flags=select.KQ_EV_ADD | select.KQ_EV_CLEAR,
            fflags=select.KQ_NOTE_WRITE,
        )
        kq.control([watch], 0, 0)  # watch first, then look: no event can be missed
        while path.exists() is not present:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                change = "appear" if present else "go away"
                raise TimeoutError(f"{path} did not {change} within {timeout} s")
            kq.control(None, 1, remaining)
    finally:
        kq.close()
        os.close(fd)
