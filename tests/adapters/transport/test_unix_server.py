"""Framing over the Unix socket. The handlers here are small stand-ins built from the codec and
the mapping; the daemon's own handler is exercised end to end in ``tests/e2e``."""

import asyncio
import contextlib
import stat
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path

import msgspec
import pytest

from voxlektor.adapters.transport import unix_server
from voxlektor.adapters.transport.mapping import error_reply
from voxlektor.adapters.transport.unix_server import Subscription, serve_unix
from voxlektor.contracts import protocol as p
from voxlektor.contracts.codec import ProtocolError, decode_reply, decode_request, encode_line
from voxlektor.runtime.events import EventHub

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


# ------------------------------------------------------------------ subscribe

SUBSCRIBE = b'{"v":1,"id":"w1","cmd":"subscribe"}\n'
SNAPSHOT: Mapping[str, object] = {"state": "cold", "groups": [], "tag": None}


class _Subscriptions:
    """A subscribe factory over a real ``EventHub``; each subscription takes the next limit."""

    def __init__(self, hub: EventHub, *limits: int) -> None:
        self._hub = hub
        self._limits = list(limits)
        self.opened = 0
        self.closed: list[int] = []
        self.closed_event = asyncio.Event()

    def __call__(self) -> Subscription:
        number = self.opened
        self.opened += 1
        limit = self._limits[number] if number < len(self._limits) else 1000
        queue, unsubscribe = self._hub.subscribe_bounded(limit)
        owner = self

        class _Sub:
            def snapshot(self) -> Mapping[str, object]:
                return SNAPSHOT

            def queue(self) -> asyncio.Queue[tuple[str, Mapping[str, object]] | None]:
                return queue

            def close(self) -> None:
                unsubscribe()
                owner.closed.append(number)
                owner.closed_event.set()

        return _Sub()


def _serve_events(
    path: Path,
    subscriptions: _Subscriptions,
    test: Callable[[], Awaitable[None]],
    handler: _Handler = _status_handler,
) -> None:
    async def main() -> None:
        server = await serve_unix(path, handler, on_ready=None, subscribe=subscriptions)
        try:
            await test()
        finally:
            server.close()

    asyncio.run(main())


async def _read_json(reader: asyncio.StreamReader) -> dict[str, object]:
    line = await asyncio.wait_for(reader.readline(), TIMEOUT)
    assert line, "the connection closed"
    return msgspec.json.decode(line, type=dict[str, object])


async def _subscribe(path: Path) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    """Connect, subscribe and read the reply and the snapshot frame."""
    reader, writer = await asyncio.open_unix_connection(str(path))
    writer.write(SUBSCRIBE)
    await writer.drain()
    assert await _read_json(reader) == {"v": 1, "id": "w1", "ok": True, "data": {}}
    snapshot = await _read_json(reader)
    assert snapshot["event"] == "snapshot"
    return reader, writer


def test_subscribe_replies_then_snapshot_then_events(short_tmp: Path) -> None:
    path = short_tmp / "s.sock"
    hub = EventHub()
    subscriptions = _Subscriptions(hub)

    async def test() -> None:
        reader, writer = await asyncio.open_unix_connection(str(path))
        writer.write(SUBSCRIBE)
        await writer.drain()
        assert await _read_json(reader) == {"v": 1, "id": "w1", "ok": True, "data": {}}
        snapshot: dict[str, object] = {"state": "COLD", "groups": [], "tag": None}  # uppercased
        assert await _read_json(reader) == {"v": 1, "event": "snapshot", "data": snapshot}
        hub.publish("state", {"owner": None, "group": None, "state": "warm"})
        hub.publish("floor", {"owner": "a", "group": "g1"})
        state = {"owner": None, "group": None, "state": "WARM"}
        assert await _read_json(reader) == {"v": 1, "event": "state", "data": state}
        floor = {"owner": "a", "group": "g1"}
        assert await _read_json(reader) == {"v": 1, "event": "floor", "data": floor}
        writer.close()

    _serve_events(path, subscriptions, test)


def test_events_in_order_to_two_subscribers(short_tmp: Path) -> None:
    path = short_tmp / "s.sock"
    hub = EventHub()

    async def test() -> None:
        first, first_w = await _subscribe(path)
        second, second_w = await _subscribe(path)
        for n in range(3):
            hub.publish("tag_started", {"tag": f"p{n}"})
        for reader in (first, second):
            tags = [(await _read_json(reader))["data"] for _ in range(3)]
            assert tags == [{"tag": "p0"}, {"tag": "p1"}, {"tag": "p2"}]
        first_w.close()
        second_w.close()

    _serve_events(path, _Subscriptions(hub), test)


def test_slow_subscriber_disconnected(short_tmp: Path) -> None:
    path = short_tmp / "s.sock"
    hub = EventHub()
    subscriptions = _Subscriptions(hub, 3, 1000)

    async def test() -> None:
        slow, slow_w = await _subscribe(path)
        other, other_w = await _subscribe(path)
        for n in range(5):  # in one go: the slow queue overflows before anything is sent
            hub.publish("tag_started", {"tag": f"p{n}"})
        rest = await asyncio.wait_for(slow.read(), TIMEOUT)  # read to EOF: closed
        assert rest.count(b"\n") <= 3
        await asyncio.wait_for(subscriptions.closed_event.wait(), TIMEOUT)
        assert subscriptions.closed == [0]
        tags = [(await _read_json(other))["data"] for _ in range(5)]
        assert tags == [{"tag": f"p{n}"} for n in range(5)]
        hub.publish("drained", {"owner": None})  # the other subscriber is intact
        assert (await _read_json(other))["event"] == "drained"
        slow_w.close()
        other_w.close()

    _serve_events(path, subscriptions, test)


def test_a_stalled_subscriber_is_closed_on_overflow(short_tmp: Path) -> None:
    path = short_tmp / "s.sock"
    hub = EventHub()
    subscriptions = _Subscriptions(hub, 3)
    blob = "x" * 60_000

    async def test() -> None:
        _, writer = await _subscribe(path)  # never reads again: the socket buffers fill up
        for n in range(500):
            if subscriptions.closed:
                break
            hub.publish("tag_started", {"tag": f"p{n}", "blob": blob})
            await asyncio.sleep(0.005)
        await asyncio.wait_for(subscriptions.closed_event.wait(), TIMEOUT)
        assert subscriptions.closed == [0]
        writer.close()

    _serve_events(path, subscriptions, test)


def test_unsubscribe_on_client_close(short_tmp: Path) -> None:
    path = short_tmp / "s.sock"
    hub = EventHub()
    subscriptions = _Subscriptions(hub)
    probe, _ = hub.subscribe_bounded(10)

    async def test() -> None:
        _, writer = await _subscribe(path)
        writer.close()
        await writer.wait_closed()
        await asyncio.wait_for(subscriptions.closed_event.wait(), TIMEOUT)
        assert subscriptions.closed == [0]
        hub.publish("drained", {"owner": None})  # reaches only the probe now
        assert probe.qsize() == 1
        assert (await _exchange(path, STATUS))[0].ok  # the server still serves

    _serve_events(path, subscriptions, test)


def test_lines_after_subscribe_are_ignored(short_tmp: Path) -> None:
    path = short_tmp / "s.sock"
    hub = EventHub()

    async def test() -> None:
        reader, writer = await asyncio.open_unix_connection(str(path))
        writer.write(SUBSCRIBE + STATUS)
        await writer.drain()
        assert (await _read_json(reader))["id"] == "w1"
        assert (await _read_json(reader))["event"] == "snapshot"
        hub.publish("drained", {"owner": None})
        assert (await _read_json(reader))["event"] == "drained"  # no status reply in between
        writer.close()

    _serve_events(path, _Subscriptions(hub), test)


def test_with_a_factory_other_lines_still_reach_the_handler(short_tmp: Path) -> None:
    path = short_tmp / "s.sock"
    subscriptions = _Subscriptions(EventHub())

    async def test() -> None:
        status, bad = await _exchange(path, STATUS, b"{bad\n")
        assert status.ok
        assert bad.error is not None
        assert bad.error.code is p.ErrorCode.bad_frame
        assert subscriptions.opened == 0

    _serve_events(path, subscriptions, test)


def test_only_subscribe_lines_are_decoded_for_the_stream_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Other requests skip the subscribe decode (the handler decodes them once)."""
    decoded: list[bytes] = []

    def decode(line: bytes) -> p.Request:
        decoded.append(line)
        return decode_request(line)

    monkeypatch.setattr(unix_server, "decode_request", decode)
    assert unix_server._subscribe_id(STATUS) is None  # pyright: ignore[reportPrivateUsage]
    assert decoded == []
    assert unix_server._subscribe_id(SUBSCRIBE) == "w1"  # pyright: ignore[reportPrivateUsage]
    assert decoded == [SUBSCRIBE]
    broken = b'{"cmd":"subscribe"\n'  # passes the cheap check, then fails the decode
    assert unix_server._subscribe_id(broken) is None  # pyright: ignore[reportPrivateUsage]


def test_without_a_factory_subscribe_goes_to_the_handler(short_tmp: Path) -> None:
    path = short_tmp / "s.sock"
    seen: list[bytes] = []

    async def recording(line: bytes) -> bytes:
        seen.append(line)
        return await _status_handler(line)

    async def test() -> None:
        (reply,) = await _exchange(path, SUBSCRIBE)
        assert reply.id == "w1"

    _serve(path, recording, test)
    assert seen == [SUBSCRIBE]


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
