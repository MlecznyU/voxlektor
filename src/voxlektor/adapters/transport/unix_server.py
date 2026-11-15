"""Unix socket server for protocol v1: framing, and event streams after ``subscribe``.

One JSON object per line, several requests per connection, answered in order. ``handler`` turns
one request line into one reply line (decoding and dispatch are the caller's business). A line
longer than the reader limit (``MAX_FRAME + 1``) cannot be resynchronised: it gets a
``bad_frame`` reply and the connection is closed. The socket is ``chmod 0600`` before
``on_ready`` runs. Runs on the loop thread.

Events: with a ``subscribe`` factory, a ``subscribe`` request turns its connection into a
stream (without one, the line goes to ``handler`` like any other). The server calls the factory
and takes the snapshot in the same loop step, so no event falls between the snapshot and the
stream; then it writes the ``ok`` reply, the ``snapshot`` event frame and every queued event in
order (``{"v":1,"event":...,"data":{...}}``; lifecycle names uppercased as in ``status``). Lines
the client sends afterwards are ignored. The stream ends, and the subscription is closed, when
the client closes its side or on overflow: the queue yields ``None``, or it is full while a
client that does not read blocks the writes (checked every ``_STALL_POLL_S``). On overflow the
connection is aborted, so the client sees the connection end rather than a gap.
"""

import asyncio
import contextlib
import logging
import os
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from typing import Protocol

from voxlektor.adapters.transport.mapping import error_reply, to_wire_reply
from voxlektor.app.effects import Reply as AppReply
from voxlektor.contracts import protocol as p
from voxlektor.contracts.codec import ProtocolError, decode_request, encode_line

_log = logging.getLogger(__name__)

_STALL_POLL_S = 0.1  # how often a blocked write checks its subscriber's queue for overflow
_READ_CHUNK = 65536

type _Handler = Callable[[bytes], Awaitable[bytes]]
type _Queue = asyncio.Queue[tuple[str, Mapping[str, object]] | None]


class Subscription(Protocol):
    """One subscriber's view of the events: the status now, then the events after it."""

    def snapshot(self) -> Mapping[str, object]: ...

    def queue(self) -> asyncio.Queue[tuple[str, Mapping[str, object]] | None]: ...

    def close(self) -> None: ...


async def serve_unix(
    path: Path,
    handler: _Handler,
    *,
    on_ready: Callable[[], None] | None,
    subscribe: Callable[[], Subscription] | None = None,
) -> asyncio.Server:
    async def connection(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            await _serve_connection(reader, writer, handler, subscribe)
        except ConnectionError:
            pass  # the client went away
        except Exception:
            _log.exception("serving a connection failed")
        finally:
            writer.close()
            with contextlib.suppress(ConnectionError):
                await writer.wait_closed()

    server = await asyncio.start_unix_server(connection, path=str(path), limit=p.MAX_FRAME + 1)
    try:
        os.chmod(path, 0o600)
        if on_ready is not None:
            on_ready()
    except BaseException:
        server.close()
        path.unlink(missing_ok=True)
        raise
    return server


async def _serve_connection(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    handler: _Handler,
    subscribe: Callable[[], Subscription] | None,
) -> None:
    while True:
        try:
            line = await reader.readline()
        except ValueError:  # longer than the reader limit
            message = f"frame larger than {p.MAX_FRAME} bytes"
            writer.write(encode_line(error_reply(ProtocolError(p.ErrorCode.bad_frame, message))))
            await writer.drain()
            return
        if not line:
            return
        req_id = None if subscribe is None else _subscribe_id(line)
        if subscribe is not None and req_id is not None:
            await _stream(req_id, reader, writer, subscribe())
            return
        writer.write(await handler(line))
        await writer.drain()


def _subscribe_id(line: bytes) -> str | None:
    """The id of a valid ``subscribe`` request; ``None`` for anything else."""
    if b'"subscribe"' not in line:  # cheap: most lines are no subscribe, skip a second decode
        return None
    try:
        req = decode_request(line)
    except ProtocolError:
        return None  # the handler answers it
    return req.id if isinstance(req, p.SubscribeRequest) else None


async def _stream(
    req_id: str, reader: asyncio.StreamReader, writer: asyncio.StreamWriter, sub: Subscription
) -> None:
    try:
        snapshot = sub.snapshot()  # same loop step as the subscription: nothing in between
        queue = sub.queue()
        writer.write(encode_line(p.Reply(v=p.PROTOCOL_VERSION, id=req_id, ok=True, data={})))
        writer.write(_event_line("snapshot", snapshot))
        gone = asyncio.ensure_future(_until_eof(reader))
        try:
            if not await _pump(writer, queue, gone):
                writer.transport.abort()
        finally:
            gone.cancel()
    finally:
        sub.close()


async def _pump(writer: asyncio.StreamWriter, queue: _Queue, gone: asyncio.Future[None]) -> bool:
    """Write events until the client is gone (``True``) or the queue overflows (``False``)."""
    while True:
        drained = await _drain(writer, queue, gone)
        if drained is not None:
            return drained
        get = asyncio.ensure_future(queue.get())
        await asyncio.wait({get, gone}, return_when=asyncio.FIRST_COMPLETED)
        if not get.done():
            get.cancel()
            return True
        item = get.result()
        if item is None:
            return False
        writer.write(_event_line(*item))


async def _drain(
    writer: asyncio.StreamWriter, queue: _Queue, gone: asyncio.Future[None]
) -> bool | None:
    """``None`` once the writes are flushed; else ``True`` = client gone, ``False`` = overflow."""
    drain = asyncio.ensure_future(writer.drain())
    while True:
        await asyncio.wait(
            {drain, gone}, timeout=_STALL_POLL_S, return_when=asyncio.FIRST_COMPLETED
        )
        if drain.done():
            drain.result()  # a ConnectionError means the client went away
            return None
        if gone.done() or queue.full():
            drain.cancel()
            return gone.done()


async def _until_eof(reader: asyncio.StreamReader) -> None:
    """Read and ignore whatever the client sends until it closes its side (or resets)."""
    with contextlib.suppress(ConnectionError):
        while await reader.read(_READ_CHUNK):
            pass


def _event_line(event: str, data: Mapping[str, object]) -> bytes:
    wire = to_wire_reply("", AppReply(ok=True, data=data, error=None)).data  # same rules as status
    return encode_line(p.Event(v=p.PROTOCOL_VERSION, event=event, data=wire or {}))
