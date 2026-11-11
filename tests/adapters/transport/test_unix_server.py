"""Framing over the Unix socket. The handlers here are small stand-ins built from the codec and
the mapping; the daemon's own handler is exercised end to end in ``tests/e2e``."""

import asyncio
import contextlib
import stat
from collections.abc import Awaitable, Callable
from pathlib import Path

import pytest

from voxlektor.adapters.transport.mapping import error_reply
from voxlektor.adapters.transport.unix_server import serve_unix
from voxlektor.contracts import protocol as p
from voxlektor.contracts.codec import ProtocolError, decode_reply, decode_request, encode_line

TIMEOUT = 5.0
STATUS = b'{"v":1,"id":"s1","cmd":"status"}\n'

type _Handler = Callable[[bytes], Awaitable[bytes]]


async def _status_handler(line: bytes) -> bytes:
    """Answers ``status`` with a fixed state; protocol errors become error replies."""
    try:
        req = decode_request(line)
    except ProtocolError as err:
        return encode_line(error_reply(err))
    return encode_line(p.Reply(v=1, id=req.id, ok=True, data={"state": "COLD"}))


async def _exchange(path: Path, *lines: bytes) -> list[p.Reply]:
    reader, writer = await asyncio.open_unix_connection(str(path))
    try:
        replies: list[p.Reply] = []
        for line in lines:
            writer.write(line)
            await writer.drain()
            replies.append(decode_reply(await asyncio.wait_for(reader.readline(), TIMEOUT)))
        return replies
    finally:
        writer.close()
        await writer.wait_closed()


def _serve(path: Path, handler: _Handler, test: Callable[[], Awaitable[None]]) -> None:
    async def main() -> None:
        server = await serve_unix(path, handler, on_ready=None)
        try:
            await test()
        finally:
            server.close()
            await asyncio.wait_for(server.wait_closed(), TIMEOUT)

    asyncio.run(main())


def test_socket_is_private_and_on_ready_runs_once_listening(short_tmp: Path) -> None:
    path = short_tmp / "s.sock"
    seen: list[int] = []

    def ready() -> None:
        seen.append(stat.S_IMODE(path.stat().st_mode))

    async def main() -> None:
        server = await serve_unix(path, _status_handler, on_ready=ready)
        try:
            assert stat.S_ISSOCK(path.stat().st_mode)
            assert stat.S_IMODE(path.stat().st_mode) == 0o600
            assert (await _exchange(path, STATUS))[0].ok
        finally:
            server.close()
            await server.wait_closed()

    asyncio.run(main())
    assert seen == [0o600]


def test_a_failing_on_ready_closes_the_server_and_removes_the_socket(short_tmp: Path) -> None:
    path = short_tmp / "s.sock"

    def broken() -> None:
        raise OSError("cannot write the pid file")

    async def main() -> None:
        with pytest.raises(OSError, match="pid file"):
            await serve_unix(path, _status_handler, on_ready=broken)

    asyncio.run(main())
    assert not path.exists()


def test_status_round_trip_and_several_requests_per_connection(short_tmp: Path) -> None:
    path = short_tmp / "s.sock"

    async def test() -> None:
        first, second = await _exchange(path, STATUS, b'{"v":1,"id":"s2","cmd":"status"}\n')
        assert (first.id, first.ok, first.data) == ("s1", True, {"state": "COLD"})
        assert second.id == "s2"

    _serve(path, _status_handler, test)


def test_bad_version_gets_an_error_and_a_new_connection_still_works(short_tmp: Path) -> None:
    path = short_tmp / "s.sock"

    async def test() -> None:
        (bad,) = await _exchange(path, b'{"v":9,"id":"x","cmd":"status"}\n')
        assert not bad.ok
        assert bad.error is not None
        assert bad.error.code is p.ErrorCode.bad_version
        assert bad.id == "x"
        (good,) = await _exchange(path, STATUS)
        assert good.ok

    _serve(path, _status_handler, test)


def test_oversize_frame_gets_bad_frame_and_the_connection_closes(short_tmp: Path) -> None:
    path = short_tmp / "s.sock"

    async def test() -> None:
        reader, writer = await asyncio.open_unix_connection(str(path))
        try:
            writer.write(b'{"v":1,"id":"big","cmd":"speak","text":"' + b"a" * p.MAX_FRAME)
            writer.write(b'"}\n')
            with contextlib.suppress(ConnectionError):  # the server may close before reading all
                await writer.drain()
            reply = decode_reply(await asyncio.wait_for(reader.readline(), TIMEOUT))
            assert reply.error is not None
            assert reply.error.code is p.ErrorCode.bad_frame
            assert reply.id is None
            assert await asyncio.wait_for(reader.read(), TIMEOUT) == b""  # closed
        finally:
            writer.close()
        assert (await _exchange(path, STATUS))[0].ok

    _serve(path, _status_handler, test)


def test_a_last_line_without_newline_is_still_served(short_tmp: Path) -> None:
    path = short_tmp / "s.sock"

    async def test() -> None:
        reader, writer = await asyncio.open_unix_connection(str(path))
        writer.write(STATUS.rstrip(b"\n"))
        writer.write_eof()
        reply = decode_reply(await asyncio.wait_for(reader.readline(), TIMEOUT))
        writer.close()
        assert reply.ok

    _serve(path, _status_handler, test)


def test_a_failing_handler_closes_only_that_connection(
    short_tmp: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = short_tmp / "s.sock"
    calls: list[bytes] = []

    async def flaky(line: bytes) -> bytes:
        calls.append(line)
        if len(calls) == 1:
            raise RuntimeError("handler bug")
        return await _status_handler(line)

    async def test() -> None:
        reader, writer = await asyncio.open_unix_connection(str(path))
        writer.write(STATUS)
        assert await asyncio.wait_for(reader.read(), TIMEOUT) == b""
        writer.close()
        assert (await _exchange(path, STATUS))[0].ok

    _serve(path, flaky, test)
    assert "handler bug" in caplog.text


def test_a_client_that_hangs_up_mid_request_is_harmless(short_tmp: Path) -> None:
    path = short_tmp / "s.sock"
    gate = asyncio.Event()

    async def slow(line: bytes) -> bytes:
        await gate.wait()
        return await _status_handler(line)

    async def test() -> None:
        _, writer = await asyncio.open_unix_connection(str(path))
        writer.write(STATUS)
        await writer.drain()
        writer.close()
        await writer.wait_closed()
        gate.set()  # the reply now goes to a closed connection
        (reply,) = await _exchange(path, STATUS)
        assert reply.ok

    _serve(path, slow, test)
