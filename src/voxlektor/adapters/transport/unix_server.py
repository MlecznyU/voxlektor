"""Unix socket server for protocol v1: framing only.

One JSON object per line, several requests per connection, answered in order. ``handler`` turns
one request line into one reply line (decoding and dispatch are the caller's business). A line
longer than the reader limit (``MAX_FRAME + 1``) cannot be resynchronised: it gets a
``bad_frame`` reply and the connection is closed. The socket is ``chmod 0600`` before
``on_ready`` runs. Runs on the loop thread.
"""

import asyncio
import contextlib
import logging
import os
from collections.abc import Awaitable, Callable
from pathlib import Path

from voxlektor.adapters.transport.mapping import error_reply
from voxlektor.contracts import protocol as p
from voxlektor.contracts.codec import ProtocolError, encode_line

_log = logging.getLogger(__name__)

type _Handler = Callable[[bytes], Awaitable[bytes]]


async def serve_unix(
    path: Path,
    handler: _Handler,
    *,
    on_ready: Callable[[], None] | None,
) -> asyncio.Server:
    async def connection(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            await _serve_connection(reader, writer, handler)
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
    reader: asyncio.StreamReader, writer: asyncio.StreamWriter, handler: _Handler
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
        writer.write(await handler(line))
        await writer.drain()
