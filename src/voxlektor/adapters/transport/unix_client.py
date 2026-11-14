"""Blocking protocol v1 client (plain ``socket``, no asyncio): one request per connection.

``request`` adds ``v`` and a fresh ``id``, sends one line and returns the decoded reply as a
dict (``{"v", "id", "ok", "data"}`` or ``{"v", "id", "ok", "error": {"code", "message"}}``;
an error code this version does not know is already ``internal``). ``timeout`` bounds the
whole exchange (connect, send and the full reply), not each socket operation.

Failures: nobody listening on ``path`` (missing, ``ECONNREFUSED``, not a socket) is
``NotRunning``. Anything else that keeps the reply from arriving (the deadline, a connection
closed early, an undecodable, oversized or foreign reply, other socket errors) is a plain
``ServiceUnavailable``: something may be alive there. A request larger than ``MAX_FRAME`` is
refused before connecting with ``ProtocolError(bad_frame)``.

``probe`` asks whatever listens on ``path`` for its status and answers in three states: ``None``
(nobody listens: a leftover file at most), ``True`` (a daemon answered ``status`` with ok),
``False`` (something listens but is not ready, e.g. shutting down, or answers oddly).
"""

import errno
import socket
import time
import uuid
from pathlib import Path

import msgspec

from voxlektor.contracts import protocol as p
from voxlektor.contracts.codec import ProtocolError, decode_reply
from voxlektor.domain.errors import NotRunning, ServiceUnavailable

_NO_LISTENER = frozenset({errno.ENOENT, errno.ECONNREFUSED, errno.ENOTSOCK})
_RECV_BYTES = 1 << 16


def request(path: Path, req: dict[str, object], timeout: float = 5.0) -> dict[str, object]:
    req_id = uuid.uuid4().hex
    line = msgspec.json.encode({**req, "v": p.PROTOCOL_VERSION, "id": req_id}) + b"\n"
    if len(line) > p.MAX_FRAME:
        raise ProtocolError(
            p.ErrorCode.bad_frame, f"request larger than {p.MAX_FRAME} bytes", req_id
        )
    try:
        raw = _exchange(path, line, time.monotonic() + timeout)
    except TimeoutError:
        raise ServiceUnavailable(f"voxlektor did not answer within {timeout:g} s") from None
    except OSError as exc:
        if exc.errno in _NO_LISTENER:
            raise NotRunning() from None
        raise ServiceUnavailable(f"cannot reach voxlektor at {path}: {exc}") from None
    if not raw:
        raise ServiceUnavailable("voxlektor closed the connection without a reply")
    try:
        reply = decode_reply(raw)
    except ProtocolError as err:
        raise ServiceUnavailable(f"invalid reply from {path}: {err.message}") from None
    if reply.id not in (req_id, None):
        raise ServiceUnavailable(f"reply from {path} belongs to another request")
    return msgspec.to_builtins(reply)


def probe(path: Path, timeout: float = 1.0) -> bool | None:
    """``None``: nobody listens; ``True``: a daemon is ready; ``False``: alive but not ready."""
    try:
        reply = request(path, {"cmd": "status"}, timeout)
    except NotRunning:  # ENOENT, ECONNREFUSED, ENOTSOCK (_NO_LISTENER)
        return None
    except ServiceUnavailable:
        return False
    return reply.get("ok") is True


def _exchange(path: Path, line: bytes, deadline: float) -> bytes:
    """Connect, send ``line`` and read one reply line, all before ``deadline``.

    Reading stops at the first newline or once more than ``MAX_FRAME`` bytes have arrived (the
    codec then rejects the oversized frame).
    """
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(_remaining(deadline))
        sock.connect(str(path))
        sock.settimeout(_remaining(deadline))
        sock.sendall(line)  # the timeout bounds the whole sendall
        data = bytearray()
        while len(data) <= p.MAX_FRAME:
            sock.settimeout(_remaining(deadline))
            piece = sock.recv(_RECV_BYTES)
            if not piece:
                break
            data += piece
            end = data.find(b"\n")
            if end >= 0:
                return bytes(data[: end + 1])
        return bytes(data)


def _remaining(deadline: float) -> float:
    left = deadline - time.monotonic()
    if left <= 0:
        raise TimeoutError
    return left
