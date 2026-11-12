"""The CLI's lightweight wiring: layered settings, the protocol client and daemon autostart."""

import os
from collections.abc import Callable
from pathlib import Path

import pytest

from voxlektor.adapters.platform.unsupported import UnsupportedPlatform
from voxlektor.app.ports import PlatformInfo
from voxlektor.bootstrap import client
from voxlektor.bootstrap.client import build_client
from voxlektor.domain.errors import DomainError, NotRunning, PlatformError, ServiceUnavailable
from voxlektor.domain.settings import Settings

type _Request = Callable[[Path, dict[str, object], float], dict[str, object]]


@pytest.fixture
def home(short_tmp: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    for name in list(os.environ):
        if name.startswith("VOXLEKTOR_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("VOXLEKTOR_HOME", str(short_tmp))
    return short_tmp


def _fake_request(
    monkeypatch: pytest.MonkeyPatch, reply: dict[str, object] | Exception
) -> list[tuple[Path, dict[str, object], float]]:
    calls: list[tuple[Path, dict[str, object], float]] = []

    def fake(path: Path, req: dict[str, object], timeout: float = 5.0) -> dict[str, object]:
        calls.append((path, req, timeout))
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr(client, "request", fake)
    return calls


def test_defaults_without_file_env_or_flags(home: Path) -> None:
    ctx = build_client({})
    assert ctx.settings == Settings()
    assert ctx.config_path == home / "config" / "config.toml"
    assert ctx.platform.socket_path() == home / "run" / "voxlektor.sock"


def test_flag_beats_env_beats_file_beats_defaults(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (home / "config").mkdir(parents=True)
    (home / "config" / "config.toml").write_text(
        'voice = "f2-narrator"\nrate = 1.5\nlang = "pl"\n[service]\nidle_timeout = "5m"\n'
    )
    monkeypatch.setenv("VOXLEKTOR_RATE", "1.25")
    monkeypatch.setenv("VOXLEKTOR_LANG", "en")
    from_env = build_client({})
    assert from_env.settings.voice == "f2-narrator"
    assert from_env.settings.rate == 1.25
    assert from_env.settings.lang == "en"
    assert from_env.settings.idle_timeout_s == 300
    flagged = build_client({"rate": 2.0, "lang": None, "idle_timeout": "90s"})
    assert flagged.settings.rate == 2.0
    assert flagged.settings.lang == "en"  # a flag left unset does not override
    assert flagged.settings.idle_timeout_s == 90
    assert flagged.settings.base_lang == "pl"


def test_unsupported_platform_raises_platform_error(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unsupported(*args: str) -> PlatformInfo:
        return UnsupportedPlatform()

    monkeypatch.setattr(client, "select_platform", unsupported)
    with pytest.raises(PlatformError, match="platform not supported yet"):
        build_client({})


def test_send_returns_the_reply_data(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _fake_request(monkeypatch, {"v": 1, "id": "x", "ok": True, "data": {"chunks": 3}})
    ctx = build_client({})
    assert ctx.send("speak", text="hi", now=True) == {"chunks": 3}
    ((path, req, _),) = calls
    assert path == ctx.platform.socket_path()
    assert req == {"cmd": "speak", "text": "hi", "now": True}


def test_send_without_data_returns_an_empty_dict(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_request(monkeypatch, {"v": 1, "id": "x", "ok": True})
    assert build_client({}).send("pause") == {}


def test_error_reply_raises_domain_error_with_the_wire_code(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    error = {"code": "unknown_voice", "message": "unknown voice 'x'; available: a, b"}
    _fake_request(monkeypatch, {"v": 1, "id": "x", "ok": False, "error": error})
    with pytest.raises(DomainError) as caught:
        build_client({}).send("speak", text="hi", voice="x")
    assert caught.value.code == "unknown_voice"
    assert caught.value.message == "unknown voice 'x'; available: a, b"


def test_failed_reply_without_error_is_internal(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_request(monkeypatch, {"v": 1, "id": "x", "ok": False})
    with pytest.raises(DomainError) as caught:
        build_client({}).send("status")
    assert caught.value.code == "internal"


def test_service_unavailable_passes_through(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_request(monkeypatch, ServiceUnavailable())
    with pytest.raises(ServiceUnavailable):
        build_client({}).send("status")


@pytest.mark.parametrize(
    ("reply", "alive"),
    [
        ({"v": 1, "id": "x", "ok": True, "data": {"state": "COLD"}}, True),
        ({"v": 1, "id": "x", "ok": False, "error": {"code": "internal", "message": "bye"}}, False),
        (ServiceUnavailable(), False),
        (NotRunning(), False),
    ],
    ids=["ok", "shutting-down", "unavailable", "not-running"],
)
def test_ensure_autostarts_with_a_short_status_ping(
    home: Path,
    monkeypatch: pytest.MonkeyPatch,
    reply: dict[str, object] | Exception,
    alive: bool,
) -> None:
    calls = _fake_request(monkeypatch, reply)
    seen: list[tuple[PlatformInfo, bool]] = []

    def fake_ensure(platform: PlatformInfo, ping: Callable[[], bool]) -> None:
        seen.append((platform, ping()))

    monkeypatch.setattr(client, "ensure_daemon", fake_ensure)
    ctx = build_client({})
    ctx.ensure()
    assert seen == [(ctx.platform, alive)]
    ((_, req, timeout),) = calls
    assert req == {"cmd": "status"}
    assert timeout <= 1.0


def test_catalog_lists_the_bundled_voices(home: Path) -> None:
    names = build_client({}).catalog().list()
    assert "v0-default" in names
    assert (home / "config" / "voices" / "v0-default").is_dir()
