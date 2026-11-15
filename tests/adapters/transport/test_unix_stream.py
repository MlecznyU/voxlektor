"""The blocking event-stream reader: ``subscribe``, then one event per line until EOF."""

import contextlib
import json
import socket
import threading
from collections.abc import Callable, Mapping
from pathlib import Path

import pytest

from voxlektor.adapters.transport.unix_stream import open_stream
from voxlektor.contracts import protocol as p
from voxlektor.domain.errors import NotRunning, ServiceUnavailable

TIMEOUT = 5.0
HALF_CLOSE_PROBE_S = 0.2
PIECE_GAP_S = 0.02

type Lines = Callable[[dict[str, object]], list[bytes]]


def frame(event: str, data: Mapping[str, object]) -> bytes:
    return json.dumps({"v": 1, "event": event, "data": dict(data)}).encode() + b"\n"


def ok(req: dict[str, object]) -> bytes:
    return json.dumps({"v": 1, "id": req["id"], "ok": True, "data": {}}).encode() + b"\n"


def stream_of(*frames: bytes) -> Lines:
    """The ``ok`` reply to the subscribe request, then ``frames``."""
    return lambda req: [ok(req), *frames]


class EventServer:
    """Accepts one connection on a thread, reads the request line and sends ``lines(request)``.

    Then ``then()`` runs (if given), the server checks whether the client half-closed its write
    side (``half_closed``), and closes the connection unless ``hold`` is set; with ``hold`` it
    waits for ``release`` first.
    """

    def __init__(
        self,
        path: Path,
        lines: Lines,
        *,
        then: Callable[[], None] | None = None,
        hold: bool = False,
    ) -> None:
        self.received: list[dict[str, object]] = []
        self.half_closed: bool | None = None
        self.release = threading.Event()
        self._lines = lines
        self._then = then
        self._hold = hold
        self._listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._listener.bind(str(path))
        self._listener.listen(1)
        self._thread = threading.Thread(target=self._serve)
        self._thread.start()

    def _serve(self) -> None:
        with self._listener:
            conn, _ = self._listener.accept()
            with conn, conn.makefile("rb") as stream, contextlib.suppress(OSError):
                req = json.loads(stream.readline())
                self.received.append(req)
                for piece in self._lines(req):  # piece by piece: a frame may span reads
                    conn.sendall(piece)
                    self.release.wait(PIECE_GAP_S)
                if self._then is not None:
                    self._then()
                conn.settimeout(HALF_CLOSE_PROBE_S)
                try:
                    self.half_closed = conn.recv(1) == b""
                except TimeoutError:
                    self.half_closed = False
                if self._hold:
                    self.release.wait(TIMEOUT)

    def join(self) -> None:
        self.release.set()
        self._thread.join(TIMEOUT)
        assert not self._thread.is_alive()


@pytest.fixture
def path(short_tmp: Path) -> Path:
    return short_tmp / "s.sock"


def test_subscribes_and_yields_snapshot_then_events_until_eof(path: Path) -> None:
    server = EventServer(
        path,
        stream_of(
            frame("snapshot", {"state": "WARM", "owner": None}),
            frame("floor", {"owner": "a", "group": "g"}),
        ),
    )
    events = list(open_stream(path, TIMEOUT))
    server.join()
    (sent,) = server.received
    assert sent["cmd"] == "subscribe"
    assert sent["v"] == p.PROTOCOL_VERSION
    assert isinstance(sent["id"], str)
    assert sent["id"]
    assert events == [
        {"event": "snapshot", "state": "WARM", "owner": None},
        {"event": "floor", "owner": "a", "group": "g"},
    ]


def test_keeps_its_write_side_open_while_reading(path: Path) -> None:
    server = EventServer(path, stream_of(frame("snapshot", {})))
    events = list(open_stream(path, TIMEOUT))
    server.join()
    assert events == [{"event": "snapshot"}]
    assert server.half_closed is False


def test_frames_split_across_reads_are_reassembled(path: Path) -> None:
    whole = frame("state", {"state": "LOADING", "owner": None, "group": None})
    reply = ok

    def lines(req: dict[str, object]) -> list[bytes]:
        answer = reply(req)
        return [answer[:5], answer[5:] + whole[:7], whole[7:]]

    server = EventServer(path, lines)
    events = list(open_stream(path, TIMEOUT))
    server.join()
    assert events == [{"event": "state", "state": "LOADING", "owner": None, "group": None}]


def test_unknown_fields_and_events_are_passed_on(path: Path) -> None:
    line = json.dumps({"v": 1, "event": "brand_new", "data": {"x": 1}, "extra": True}).encode()
    server = EventServer(path, stream_of(line + b"\n"))
    events = list(open_stream(path, TIMEOUT))
    server.join()
    assert events == [{"event": "brand_new", "x": 1}]


def test_missing_socket_is_not_running(path: Path) -> None:
    with pytest.raises(NotRunning):
        open_stream(path, TIMEOUT)


def test_regular_file_at_socket_path_is_not_running(path: Path) -> None:
    path.write_text("stale")
    with pytest.raises(NotRunning):
        open_stream(path, TIMEOUT)


def _unavailable(path: Path, match: str, timeout: float = TIMEOUT) -> None:
    with pytest.raises(ServiceUnavailable, match=match) as caught:
        list(open_stream(path, timeout))
    assert not isinstance(caught.value, NotRunning)


def test_error_reply_is_service_unavailable(path: Path) -> None:
    def lines(req: dict[str, object]) -> list[bytes]:
        error = {"code": "unknown_command", "message": "unknown command 'subscribe'"}
        reply = {"v": 1, "id": req["id"], "ok": False, "error": error}
        return [json.dumps(reply).encode() + b"\n"]

    server = EventServer(path, lines)
    _unavailable(path, "unknown command 'subscribe'")
    server.join()


def test_reply_for_another_request_is_service_unavailable(path: Path) -> None:
    def lines(req: dict[str, object]) -> list[bytes]:
        return [json.dumps({"v": 1, "id": "other", "ok": True, "data": {}}).encode() + b"\n"]

    server = EventServer(path, lines)
    _unavailable(path, "another request")
    server.join()


def test_undecodable_reply_is_service_unavailable(path: Path) -> None:
    server = EventServer(path, lambda req: [b"not json\n"])
    _unavailable(path, "invalid reply")
    server.join()


def test_closed_before_the_reply_is_service_unavailable(path: Path) -> None:
    server = EventServer(path, lambda req: [])
    _unavailable(path, "closed the connection")
    server.join()


def test_no_reply_in_time_is_service_unavailable(path: Path) -> None:
    server = EventServer(path, lambda req: [], hold=True)
    _unavailable(path, "did not answer", timeout=0.2)
    server.join()


def test_undecodable_event_is_service_unavailable(path: Path) -> None:
    server = EventServer(path, stream_of(frame("snapshot", {}), b"[1, 2]\n"))
    stream = open_stream(path, TIMEOUT)
    assert next(stream) == {"event": "snapshot"}
    with pytest.raises(ServiceUnavailable, match="invalid event"):
        next(stream)
    server.join()


def test_oversized_event_is_service_unavailable(path: Path) -> None:
    server = EventServer(path, stream_of(b"x" * (p.MAX_FRAME + 2)))
    _unavailable(path, "larger than")
    server.join()


def test_unterminated_last_line_at_eof_is_service_unavailable(path: Path) -> None:
    server = EventServer(path, stream_of(b'{"v": 1, "event": "state"'))
    _unavailable(path, "invalid event")
    server.join()
