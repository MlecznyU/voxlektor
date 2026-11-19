"""The dialogue-box pattern end to end: one event stream, two owners queueing tagged paragraphs.

A daemon in its own thread (FakeEngine, a memory output, a temporary ``VOXLEKTOR_HOME``), the
CLI's own stream client (``open_stream``) as the one watcher, and blocking requests as the two
clients. The output holds the first chunk until both owners have queued everything, so "in
quick succession" is deterministic: s2's whole reading is queued while s1 holds the floor.
"""

import asyncio
import os
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import msgspec
import pytest

from tests.fixtures.sockets import TIMEOUT, request, wait_for_file
from voxlektor.adapters.audio.memory_out import MemoryCuePlayer, MemoryOutput
from voxlektor.adapters.platform.macos import MacOSPlatform
from voxlektor.adapters.storage.settings_toml import TomlSettingsStore
from voxlektor.adapters.transport.unix_stream import open_stream
from voxlektor.bootstrap.daemon import build_daemon
from voxlektor.domain.settings import merge_layers
from voxlektor.domain.types import ChunkId

pytestmark = pytest.mark.e2e

OWNERS = ("s1", "s2")
PARAGRAPHS = 3


class _HoldFirst(MemoryOutput):
    """Plays every chunk at once, except the very first: that one waits for ``release``."""

    def __init__(self) -> None:
        super().__init__(auto_complete=False)
        self._lock = threading.Lock()
        self._held: ChunkId | None = None
        self._open = False
        self.first_start = threading.Event()

    def start(
        self, chunk: ChunkId, samples: object, sample_rate: int, *, lead_s: float = 0.0
    ) -> None:
        super().start(chunk, samples, sample_rate, lead_s=lead_s)
        with self._lock:
            hold = not self._open and self._held is None
            if hold:
                self._held = chunk
        if hold:
            self.first_start.set()
        else:
            self.complete(chunk)

    def release(self) -> None:
        with self._lock:
            self._open = True
            held = self._held
        if held is not None:
            self.complete(held)


@dataclass
class _Daemon:
    platform: MacOSPlatform
    output: _HoldFirst
    thread: threading.Thread
    errors: list[BaseException]

    @property
    def socket(self) -> Path:
        return self.platform.socket_path()


@pytest.fixture
def daemon(short_tmp: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[_Daemon]:
    monkeypatch.setenv("VOXLEKTOR_HOME", str(short_tmp))
    monkeypatch.setenv("VOXLEKTOR_ENGINE", "fake")
    store = TomlSettingsStore(Path("/nonexistent/config.toml"), os.environ)
    platform = MacOSPlatform()
    output = _HoldFirst()
    built = build_daemon(
        merge_layers(store.read_file(), store.read_env()),
        transport=True,
        platform=platform,
        output=output,
        cues=MemoryCuePlayer(),
    )
    errors: list[BaseException] = []

    def serve() -> None:
        try:
            asyncio.run(built.run())
        except BaseException as exc:  # reported by the fixture, never lost in the thread
            errors.append(exc)

    thread = threading.Thread(target=serve, name="test-daemon")
    thread.start()
    running = _Daemon(platform, output, thread, errors)
    try:
        wait_for_file(platform.pid_path())
        yield running
    finally:
        output.release()
        if thread.is_alive() and running.socket.exists():
            request(running.socket, b'{"v":1,"id":"cleanup","cmd":"shutdown"}\n')
        thread.join(TIMEOUT)
        assert not thread.is_alive()
        assert errors == []


def _speak(path: Path, owner: str, paragraph: int) -> None:
    fields: dict[str, object] = {
        "v": 1,
        "id": f"{owner}p{paragraph}",
        "cmd": "speak",
        "text": f"Paragraph {paragraph} from {owner}.",
        "lang": "en",
        "owner": owner,
        "group": "r1",
        "tag": f"{owner}p{paragraph}",
    }
    if paragraph == 0:
        fields["now"] = True  # dialogue-box starts each reply with --now
    reply = request(path, msgspec.json.encode(fields) + b"\n")
    assert reply.ok, reply.error


def test_two_owners_read_in_turn_and_every_tag_finishes_done(daemon: _Daemon) -> None:
    stream = open_stream(daemon.socket, TIMEOUT)
    snapshot = next(stream)
    assert snapshot["event"] == "snapshot"
    assert snapshot["groups"] == []
    events: list[dict[str, object]] = []  # up to the first daemon-wide drained
    done = threading.Event()

    def collect() -> None:
        for event in stream:  # until the daemon goes; later events (shutdown) are not kept
            if done.is_set():
                continue
            events.append(event)
            if event["event"] == "drained" and event["owner"] is None:
                done.set()

    reader = threading.Thread(target=collect, name="test-watch")
    reader.start()
    try:
        for owner in OWNERS:
            for paragraph in range(PARAGRAPHS):
                _speak(daemon.socket, owner, paragraph)
        assert daemon.output.first_start.wait(TIMEOUT)
        daemon.output.release()
        assert done.wait(TIMEOUT), events
    finally:
        daemon.output.release()
        request(daemon.socket, b'{"v":1,"id":"bye","cmd":"shutdown"}\n')
        daemon.thread.join(TIMEOUT)
        reader.join(TIMEOUT)  # the stream ends when the daemon goes
    assert not reader.is_alive()

    tags = [f"{owner}p{n}" for owner in OWNERS for n in range(PARAGRAPHS)]

    def of(name: str) -> list[dict[str, object]]:
        return [e for e in events if e["event"] == name]

    assert [e["tag"] for e in of("tag_started")] == tags
    assert [(e["tag"], e["reason"]) for e in of("tag_finished")] == [(t, "done") for t in tags]
    assert [(e["owner"], e["group"]) for e in of("floor")] == [("s1", "r1"), ("s2", "r1")]
    assert ("s2", "r1", 1) in [(e["owner"], e["group"], e["position"]) for e in of("waiting")]
    assert [e["owner"] for e in of("drained")] == ["s1", "s2", None]
    assert events[-1] == {"event": "drained", "owner": None, "group": None}
    assert of("error") == []
    for event in events:
        assert {"owner", "group"} <= event.keys(), event
