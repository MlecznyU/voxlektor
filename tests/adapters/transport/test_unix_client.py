"""The blocking protocol client: one request per connection, unavailability as one error."""

import contextlib
import json
import socket
import threading
from collections.abc import Callable
from pathlib import Path

import pytest

from voxlektor.adapters.transport.unix_client import request
from voxlektor.contracts import protocol as p
from voxlektor.contracts.codec import ProtocolError
from voxlektor.domain.errors import NotRunning, ServiceUnavailable

TIMEOUT = 5.0
TRICKLE_GAP_S = 0.05

type _Answer = Callable[[dict[str, object]], bytes | list[bytes] | None]


class _OneShotServer:
    """Accepts one connection on a thread, reads one line and answers with ``answer(request)``.

    A list answer is sent piece by piece with a short gap between pieces (a slow daemon);
    ``release`` cuts the gaps short once the test is done.
    """

    def __init__(self, path: Path, answer: _Answer) -> None:
        self.received: list[dict[str, object]] = []
        self.release = threading.Event()
        self._answer = answer
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
                reply = self._answer(req)
                pieces = [reply] if isinstance(reply, bytes) else reply or []
                for piece in pieces:
                    conn.sendall(piece)
                    self.release.wait(TRICKLE_GAP_S)

    def join(self) -> None:
        self.release.set()
        self._thread.join(TIMEOUT)
        assert not self._thread.is_alive()


@pytest.fixture
def path(short_tmp: Path) -> Path:
    return short_tmp / "s.sock"


def _ok(req: dict[str, object]) -> bytes:
    reply = {"v": 1, "id": req["id"], "ok": True, "data": {"chunks": 2}}
    return json.dumps(reply).encode() + b"\n"


def test_request_adds_version_and_a_fresh_id_and_returns_the_reply(path: Path) -> None:
    server = _OneShotServer(path, _ok)
    reply = request(path, {"cmd": "speak", "text": "hello"})
    server.join()
    (sent,) = server.received
    assert sent["v"] == p.PROTOCOL_VERSION
    assert isinstance(sent["id"], str)
    assert sent["id"]
    assert sent["cmd"] == "speak"
    assert sent["text"] == "hello"
    assert reply == {"v": 1, "id": sent["id"], "ok": True, "data": {"chunks": 2}}


def test_ids_differ_between_requests(short_tmp: Path) -> None:
    ids: list[object] = []
    for name in ("a.sock", "b.sock"):
        server = _OneShotServer(short_tmp / name, _ok)
        request(short_tmp / name, {"cmd": "status"})
        server.join()
        ids.append(server.received[0]["id"])
    assert ids[0] != ids[1]


def test_error_reply_is_returned_and_unknown_codes_become_internal(path: Path) -> None:
    def answer(req: dict[str, object]) -> bytes:
        error = {"code": "brand_new", "message": "later version"}
        return json.dumps({"v": 1, "id": req["id"], "ok": False, "error": error}).encode() + b"\n"

    server = _OneShotServer(path, answer)
    reply = request(path, {"cmd": "status"})
    server.join()
    assert reply["ok"] is False
    assert reply["error"] == {"code": "internal", "message": "later version"}


def _alive_but_unavailable(path: Path, match: str, timeout: float = 5.0) -> None:
    """``request`` fails with a plain ServiceUnavailable, never NotRunning."""
    with pytest.raises(ServiceUnavailable, match=match) as caught:
        request(path, {"cmd": "status"}, timeout=timeout)
    assert not isinstance(caught.value, NotRunning)


def test_missing_socket_is_not_running(path: Path) -> None:
    with pytest.raises(NotRunning, match="not running"):
        request(path, {"cmd": "status"})


def test_regular_file_at_socket_path_is_not_running(path: Path) -> None:
    path.write_text("stale")  # connect fails with ENOTSOCK: nobody can listen there
    with pytest.raises(NotRunning):
        request(path, {"cmd": "status"})


def test_bound_socket_without_listener_is_not_running(path: Path) -> None:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.bind(str(path))  # bound, never listening: ECONNREFUSED
        with pytest.raises(NotRunning, match="not running"):
            request(path, {"cmd": "status"})


def test_no_answer_within_timeout_is_alive_but_unavailable(path: Path) -> None:
    release = threading.Event()

    def answer(req: dict[str, object]) -> None:
        release.wait(TIMEOUT)

    server = _OneShotServer(path, answer)
    try:
        _alive_but_unavailable(path, "did not answer", timeout=0.2)
    finally:
        release.set()
        server.join()


def test_timeout_bounds_the_whole_exchange_not_each_read(path: Path) -> None:
    """A daemon that trickles a byte every 50 ms never trips a per-read timeout of 0.2 s."""
    reply = b'{"v":1,"id":null,"ok":true,"data":{}}\n'
    server = _OneShotServer(path, lambda req: [bytes([b]) for b in reply])
    try:
        _alive_but_unavailable(path, "did not answer", timeout=0.2)
    finally:
        server.join()


def test_an_exhausted_deadline_stops_before_the_next_step(path: Path) -> None:
    _alive_but_unavailable(path, "did not answer within 0 s", timeout=0.0)


def test_connection_closed_without_reply_is_alive_but_unavailable(path: Path) -> None:
    server = _OneShotServer(path, lambda req: None)
    _alive_but_unavailable(path, "without a reply")
    server.join()


def test_garbage_reply_is_alive_but_unavailable(path: Path) -> None:
    server = _OneShotServer(path, lambda req: b"hello\n")
    _alive_but_unavailable(path, "invalid reply")
    server.join()


def test_oversized_reply_is_alive_but_unavailable(path: Path) -> None:
    server = _OneShotServer(path, lambda req: b"a" * (p.MAX_FRAME + 1))
    _alive_but_unavailable(path, "invalid reply.*larger than")
    server.join()


def test_reply_for_another_request_is_alive_but_unavailable(path: Path) -> None:
    server = _OneShotServer(path, lambda req: b'{"v":1,"id":"other","ok":true,"data":{}}\n')
    _alive_but_unavailable(path, "another request")
    server.join()


def test_other_socket_errors_are_service_unavailable(short_tmp: Path) -> None:
    too_long = short_tmp / ("x" * 120)  # longer than the AF_UNIX path limit
    _alive_but_unavailable(too_long, "cannot reach")


def test_oversized_request_is_refused_before_connecting(path: Path) -> None:
    with pytest.raises(ProtocolError) as caught:
        request(path, {"cmd": "speak", "text": "a" * p.MAX_FRAME})
    assert caught.value.code is p.ErrorCode.bad_frame
