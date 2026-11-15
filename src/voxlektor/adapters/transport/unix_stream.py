"""Blocking protocol v1 event stream (plain ``socket``, no asyncio): ``subscribe``, then events.

``open_stream`` connects, sends ``subscribe`` and waits for its ``ok`` reply, all within
``timeout``; it then returns an iterator over the event frames that follow, without a timeout
(events come whenever the daemon has them). Each item is the frame's ``data`` plus ``"event"``
(the event name); unknown events and fields pass through untouched. The iterator ends when the
daemon closes the connection (EOF, or a reset). The write side stays open for the whole stream:
the daemon treats a client that half-closes after ``subscribe`` as gone.

Failures: nobody listening on ``path`` (missing, ``ECONNREFUSED``, not a socket) is
``NotRunning``, raised by ``open_stream`` itself. A missing, late, undecodable or error reply is
a plain ``ServiceUnavailable`` (an error reply carries its message, e.g. a daemon too old to
know ``subscribe``); so is an event line that is not a valid frame or is larger than
``MAX_FRAME``, raised while iterating.
"""

import errno
import socket
import time
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path

import msgspec

from voxlektor.contracts import protocol as p
from voxlektor.contracts.codec import ProtocolError, decode_reply
from voxlektor.domain.errors import NotRunning, ServiceUnavailable

_NO_LISTENER = frozenset({errno.ENOENT, errno.ECONNREFUSED, errno.ENOTSOCK})
_RECV_BYTES = 1 << 16
_EVENT = "event"


class _Lines:
    """Splits what a socket receives into lines of at most ``MAX_FRAME`` bytes."""

    def __init__(self, sock: socket.socket) -> None:
        self._sock = sock
        self._buffer = bytearray()

    def next_line(self, before_recv: Callable[[], None] | None = None) -> bytes | None:
        """The next line (with its newline); ``None`` at a clean EOF between lines.

        Bytes left without a newline at EOF are returned as they are (the decoder rejects them).
        """
        while (end := self._buffer.find(b"\n")) < 0:
            if len(self._buffer) > p.MAX_FRAME:
                raise ServiceUnavailable(f"frame from voxlektor larger than {p.MAX_FRAME} bytes")
            if before_recv is not None:
                before_recv()
            piece = self._sock.recv(_RECV_BYTES)
            if not piece:
                rest = bytes(self._buffer)
                self._buffer.clear()
                return rest or None
            self._buffer += piece
        line = bytes(self._buffer[: end + 1])
        del self._buffer[: end + 1]
        return line


def open_stream(path: Path, timeout: float) -> Iterator[dict[str, object]]:
    req_id = uuid.uuid4().hex
    line = msgspec.json.encode({"cmd": "subscribe", "v": p.PROTOCOL_VERSION, "id": req_id})
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    reader = _Lines(sock)
    try:
        raw = _handshake(sock, reader, path, line + b"\n", time.monotonic() + timeout)
        _check_reply(raw, req_id, path)
    except TimeoutError:
        sock.close()
        raise ServiceUnavailable(f"voxlektor did not answer within {timeout:g} s") from None
    except OSError as exc:
        sock.close()
        if exc.errno in _NO_LISTENER:
            raise NotRunning() from None
        raise ServiceUnavailable(f"cannot reach voxlektor at {path}: {exc}") from None
    except BaseException:
        sock.close()
        raise
    sock.settimeout(None)
    return _events(sock, reader)


def _handshake(
    sock: socket.socket, reader: _Lines, path: Path, line: bytes, deadline: float
) -> bytes | None:
    """Connect, send ``line`` and read the reply line, all before ``deadline``."""
    sock.settimeout(_remaining(deadline))
    sock.connect(str(path))
    sock.settimeout(_remaining(deadline))
    sock.sendall(line)
    return reader.next_line(lambda: sock.settimeout(_remaining(deadline)))


def _check_reply(raw: bytes | None, req_id: str, path: Path) -> None:
    if raw is None:
        raise ServiceUnavailable("voxlektor closed the connection without a reply")
    try:
        reply = decode_reply(raw)
    except ProtocolError as err:
        raise ServiceUnavailable(f"invalid reply from {path}: {err.message}") from None
    if reply.id not in (req_id, None):
        raise ServiceUnavailable(f"reply from {path} belongs to another request")
    if not reply.ok:
        raise ServiceUnavailable(reply.error.message if reply.error else "subscribe failed")


def _events(sock: socket.socket, reader: _Lines) -> Iterator[dict[str, object]]:
    with sock:
        while True:
            try:
                raw = reader.next_line()
            except ConnectionError:  # a reset is the daemon going away, like EOF
                return
            if raw is None:
                return
            try:
                frame = msgspec.json.decode(raw, type=p.Event)
            except (msgspec.DecodeError, msgspec.ValidationError) as exc:
                raise ServiceUnavailable(f"invalid event from voxlektor: {exc}") from None
            data = {key: value for key, value in frame.data.items() if key != _EVENT}
            yield {_EVENT: frame.event, **data}


def _remaining(deadline: float) -> float:
    left = deadline - time.monotonic()
    if left <= 0:
        raise TimeoutError
    return left
