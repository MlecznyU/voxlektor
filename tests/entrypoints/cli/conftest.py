"""CLI test kit: a clean ``VOXLEKTOR_HOME`` and a fake wire in place of the socket client."""

import io
import os
import sys
from pathlib import Path

import pytest

from voxlektor.app.ports import PlatformInfo
from voxlektor.bootstrap import client

type Answer = dict[str, object] | BaseException


class FakeWire:
    """Records every request; answers ``ok`` with empty data unless ``replies`` says otherwise."""

    def __init__(self) -> None:
        self.requests: list[dict[str, object]] = []
        self.replies: dict[str, Answer] = {}
        self.ensured = 0

    def request(
        self, path: Path, req: dict[str, object], timeout: float = 5.0
    ) -> dict[str, object]:
        self.requests.append(dict(req))
        answer = self.replies.get(str(req["cmd"]), {"ok": True, "data": {}})
        if isinstance(answer, BaseException):
            raise answer
        return {"v": 1, "id": "test", **answer}

    def ensure(self, platform: PlatformInfo, ping: object) -> None:
        self.ensured += 1

    @property
    def cmds(self) -> list[object]:
        return [req["cmd"] for req in self.requests]

    def fail(self, cmd: str, code: str, message: str) -> None:
        self.replies[cmd] = {"ok": False, "error": {"code": code, "message": message}}


class Tty(io.StringIO):
    def isatty(self) -> bool:
        return True


@pytest.fixture
def home(short_tmp: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    for name in list(os.environ):
        if name.startswith("VOXLEKTOR_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("VOXLEKTOR_HOME", str(short_tmp))
    monkeypatch.setattr(sys, "stdin", Tty())
    return short_tmp


@pytest.fixture
def wire(home: Path, monkeypatch: pytest.MonkeyPatch) -> FakeWire:
    fake = FakeWire()
    monkeypatch.setattr(client, "request", fake.request)
    monkeypatch.setattr(client, "ensure_daemon", fake.ensure)
    return fake
