"""The daemon end to end over its socket: FakeEngine, MemoryOutput, a temporary VOXLEKTOR_HOME.

Runs by default (no model, no audio device). The daemon runs in its own thread with its own
event loop; the test talks to it with a blocking client, as the CLI will.
"""

import asyncio
import os
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

from tests.fixtures.sockets import TIMEOUT, request, wait_for_file
from voxlektor.adapters.audio.memory_out import MemoryCuePlayer, MemoryOutput
from voxlektor.adapters.platform.macos import MacOSPlatform
from voxlektor.adapters.storage.settings_toml import TomlSettingsStore
from voxlektor.bootstrap.daemon import build_daemon
from voxlektor.contracts import protocol as p
from voxlektor.domain.settings import Settings, merge_layers
from voxlektor.domain.types import ChunkId

pytestmark = pytest.mark.e2e


class _SignalOutput(MemoryOutput):
    """MemoryOutput that sets an event when a chunk starts (the test waits on it, no polling)."""

    def __init__(self) -> None:
        super().__init__(auto_complete=True)
        self.first_start = threading.Event()

    def start(self, chunk: ChunkId, samples: object, sample_rate: int) -> None:
        self.first_start.set()
        super().start(chunk, samples, sample_rate)


@dataclass
class _Running:
    platform: MacOSPlatform
    output: _SignalOutput
    thread: threading.Thread
    errors: list[BaseException]

    @property
    def socket(self) -> Path:
        return self.platform.socket_path()

    @property
    def pid(self) -> Path:
        return self.platform.pid_path()


def _settings() -> Settings:
    store = TomlSettingsStore(Path("/nonexistent/config.toml"), os.environ)
    return merge_layers(store.read_file(), store.read_env())


@pytest.fixture
def daemon(short_tmp: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[_Running]:
    monkeypatch.setenv("VOXLEKTOR_HOME", str(short_tmp))
    monkeypatch.setenv("VOXLEKTOR_ENGINE", "fake")
    platform = MacOSPlatform()
    output = _SignalOutput()
    built = build_daemon(
        _settings(), transport=True, platform=platform, output=output, cues=MemoryCuePlayer()
    )
    errors: list[BaseException] = []

    def serve() -> None:
        try:
            asyncio.run(built.run())
        except BaseException as exc:  # reported by the fixture, never lost in the thread
            errors.append(exc)

    before = set(threading.enumerate())
    thread = threading.Thread(target=serve, name="test-daemon")
    thread.start()
    running = _Running(platform, output, thread, errors)
    try:
        wait_for_file(running.pid)  # the pid file is written once the socket listens
        yield running
    finally:
        if thread.is_alive() and running.socket.exists():
            request(running.socket, b'{"v":1,"id":"cleanup","cmd":"shutdown"}\n')
        thread.join(TIMEOUT)
        assert not thread.is_alive()
        assert errors == []
        leaked = [t for t in threading.enumerate() if t not in before and t.is_alive()]
        assert leaked == []


def _line(cmd: str, **fields: object) -> bytes:
    body = ",".join(f'"{key}":{_json(value)}' for key, value in fields.items())
    return f'{{"v":1,"id":"{cmd}-1","cmd":"{cmd}"{"," if body else ""}{body}}}\n'.encode()


def _json(value: object) -> str:
    return f'"{value}"' if isinstance(value, str) else str(value).lower()


def test_speak_then_status_warm_then_shutdown_removes_socket_and_pid(daemon: _Running) -> None:
    assert daemon.pid.read_text().strip() == str(os.getpid())
    speak = request(daemon.socket, _line("speak", text="Hello there, this is a test.", lang="en"))
    assert speak.ok
    assert speak.id == "speak-1"
    assert speak.data == {"chunks": 1}
    assert daemon.output.first_start.wait(TIMEOUT)
    status = request(daemon.socket, _line("status"))
    assert status.ok
    assert status.data is not None
    assert status.data["state"] == "WARM"
    assert status.data["engine"] == "fake"
    bye = request(daemon.socket, _line("shutdown"))
    assert bye.ok
    daemon.thread.join(TIMEOUT)
    assert not daemon.thread.is_alive()
    assert not daemon.socket.exists()
    assert not daemon.pid.exists()


def test_protocol_errors_come_back_as_error_replies(daemon: _Running) -> None:
    bad_version = request(daemon.socket, b'{"v":9,"id":"x","cmd":"status"}\n')
    assert bad_version.error is not None
    assert bad_version.error.code is p.ErrorCode.bad_version
    assert bad_version.id == "x"
    unknown = request(daemon.socket, b'{"v":1,"id":"u","cmd":"dance"}\n')
    assert unknown.error is not None
    assert unknown.error.code is p.ErrorCode.unknown_command
    subscribe = request(daemon.socket, _line("subscribe"))
    assert subscribe.error is not None
    assert subscribe.error.code is p.ErrorCode.unsupported
    voice = request(daemon.socket, _line("speak", text="hello", voice="nobody"))
    assert voice.error is not None
    assert voice.error.code is p.ErrorCode.unknown_voice
    assert "v0-default" in voice.error.message
    assert request(daemon.socket, _line("status")).ok  # the daemon still serves
