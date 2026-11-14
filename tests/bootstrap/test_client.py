"""The CLI's lightweight wiring: layered settings, the protocol client and daemon autostart."""

import os
from collections.abc import Callable
from pathlib import Path

import pytest

from voxlektor.adapters.platform.unsupported import UnsupportedPlatform
from voxlektor.app.ports import PlatformInfo
from voxlektor.bootstrap import client
from voxlektor.bootstrap.client import (
    DaemonSamplePlayer,
    build_client,
    build_setup_client,
    model_store,
    setup_prompter,
)
from voxlektor.domain.errors import DomainError, NotRunning, PlatformError, ServiceUnavailable
from voxlektor.domain.settings import Settings
from voxlektor.domain.types import VoiceName

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


class _Daemon:
    """Answers speak with ok and status from ``states`` in turn (the last one repeats)."""

    def __init__(self, states: list[dict[str, object]]) -> None:
        self.states = states
        self.requests: list[dict[str, object]] = []
        self.ensured = 0

    def request(
        self, path: Path, req: dict[str, object], timeout: float = 5.0
    ) -> dict[str, object]:
        self.requests.append(req)
        data: dict[str, object] = {}
        if req["cmd"] == "status":
            data = self.states.pop(0) if len(self.states) > 1 else self.states[0]
        return {"v": 1, "id": "x", "ok": True, "data": data}

    def ensure(self, platform: PlatformInfo, ping: Callable[[], bool]) -> None:
        self.ensured += 1


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    now = [0.0]

    def sleep(seconds: float) -> None:
        now[0] += seconds

    monkeypatch.setattr(client, "_sleep", sleep)
    monkeypatch.setattr(client, "_monotonic", lambda: now[0])
    return now


def _player(monkeypatch: pytest.MonkeyPatch, states: list[dict[str, object]]) -> _Daemon:
    daemon = _Daemon(states)
    monkeypatch.setattr(client, "request", daemon.request)
    monkeypatch.setattr(client, "ensure_daemon", daemon.ensure)
    return daemon


def test_sample_is_spoken_now_and_waited_for(
    home: Path,
    monkeypatch: pytest.MonkeyPatch,
    clock: list[float],
    capsys: pytest.CaptureFixture[str],
) -> None:
    daemon = _player(
        monkeypatch,
        [
            {"state": "cold", "playing": False, "queued": 1},
            {"state": "loading", "playing": False, "queued": 1},
            {"state": "warm", "playing": True, "queued": 0},
            {"state": "warm", "playing": False, "queued": 0},
        ],
    )
    DaemonSamplePlayer(build_client({})).play("Hello.", "f1-warm", "en", 1.25, 70)
    assert daemon.ensured == 1
    speak = daemon.requests[0]
    assert speak == {
        "cmd": "speak",
        "text": "Hello.",
        "voice": "f1-warm",
        "lang": "en",
        "base_lang": "en",
        "rate": 1.25,
        "volume": 70,
        "now": True,
    }
    assert [req["cmd"] for req in daemon.requests[1:]] == ["status"] * 4
    assert clock[0] == pytest.approx(0.6)  # three polls of 0.2 s before the last status
    assert capsys.readouterr().err.count("loading the model…") == 1


def test_sample_waits_for_queued_chunks(
    home: Path, monkeypatch: pytest.MonkeyPatch, clock: list[float]
) -> None:
    daemon = _player(
        monkeypatch,
        [{"state": "warm", "playing": False, "queued": 2}, {"state": "warm", "playing": False}],
    )
    DaemonSamplePlayer(build_client({})).play("x", "v", "pl", 1.0, 50)
    assert len(daemon.requests) == 3


def test_sample_that_never_ends_times_out(
    home: Path, monkeypatch: pytest.MonkeyPatch, clock: list[float]
) -> None:
    _player(monkeypatch, [{"state": "warm", "playing": True, "queued": 0}])
    with pytest.raises(ServiceUnavailable, match="did not finish"):
        DaemonSamplePlayer(build_client({})).play("x", "v", "pl", 1.0, 50)
    assert clock[0] == pytest.approx(120.0)


def test_sample_after_a_failed_model_load_raises_the_daemon_error_at_once(
    home: Path, monkeypatch: pytest.MonkeyPatch, clock: list[float]
) -> None:
    """The daemon falls back to ``cold`` with ``last_error``: that ends the wait, not a timeout."""
    _player(
        monkeypatch,
        [
            {"state": "loading", "playing": False, "queued": 1, "last_error": None},
            {"state": "cold", "playing": False, "queued": 0, "last_error": "model load failed: x"},
        ],
    )
    with pytest.raises(ServiceUnavailable, match=r"^model load failed: x$"):
        DaemonSamplePlayer(build_client({})).play("x", "v", "pl", 1.0, 50)
    assert clock[0] == pytest.approx(0.2)


def test_sample_ends_when_nothing_is_left_even_if_the_engine_is_cold(
    home: Path, monkeypatch: pytest.MonkeyPatch, clock: list[float]
) -> None:
    daemon = _player(monkeypatch, [{"state": "cold", "playing": False, "queued": 0}])
    DaemonSamplePlayer(build_client({})).play("x", "v", "pl", 1.0, 50)
    assert len(daemon.requests) == 2
    assert clock[0] == 0.0


def _write_config(home: Path, text: str) -> None:
    (home / "config").mkdir(parents=True, exist_ok=True)
    (home / "config" / "config.toml").write_text(text)


def test_setup_client_skips_invalid_file_values_with_a_warning(home: Path) -> None:
    _write_config(home, 'voice = "f2-narrator"\nrate = 9.0\n[service]\nidle_timeout = "soon"\n')
    warnings: list[str] = []
    ctx, current = build_setup_client(warnings.append)
    assert current == Settings(voice=VoiceName("f2-narrator"))
    assert ctx.settings == Settings(voice=VoiceName("f2-narrator"))
    assert ctx.config_path == home / "config" / "config.toml"
    assert warnings == [
        "ignoring invalid rate in config.toml: rate must be between 0.5 and 3.0",
        "ignoring invalid idle_timeout in config.toml: "
        "invalid duration 'soon'; use 0, 90s, 10m or 1h",
    ]


def test_setup_client_skips_invalid_env_values_with_a_warning(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_config(home, "volume = 30\n")
    monkeypatch.setenv("VOXLEKTOR_RATE", "fast")
    monkeypatch.setenv("VOXLEKTOR_VOLUME", "101")
    monkeypatch.setenv("VOXLEKTOR_VOICE", "f1-warm")
    warnings: list[str] = []
    ctx, current = build_setup_client(warnings.append)
    assert current == Settings(volume=30)  # the file's values only, never the env's
    assert ctx.settings == Settings(voice=VoiceName("f1-warm"), volume=30)
    assert warnings == [
        "ignoring invalid VOXLEKTOR_RATE in the environment: "
        "VOXLEKTOR_RATE: invalid value 'fast' for rate",
        "ignoring invalid VOXLEKTOR_VOLUME in the environment: "
        "volume must be an integer between 0 and 100",
    ]


def test_setup_client_with_a_valid_config_warns_nothing(home: Path) -> None:
    _write_config(home, 'base_lang = "en"\n')
    warnings: list[str] = []
    _, current = build_setup_client(warnings.append)
    assert current == Settings(base_lang="en")
    assert warnings == []


def test_setup_client_on_an_unsupported_platform_raises_platform_error(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unsupported(*args: str) -> PlatformInfo:
        return UnsupportedPlatform()

    monkeypatch.setattr(client, "select_platform", unsupported)
    with pytest.raises(PlatformError, match="platform not supported yet"):
        build_setup_client(print)


def test_setup_adapters_are_the_hf_store_and_the_questionary_prompter() -> None:
    from voxlektor.adapters.models.hf_cache import HfModelStore  # noqa: PLC0415
    from voxlektor.adapters.terminal.menus import QuestionaryPrompter  # noqa: PLC0415

    assert isinstance(model_store(), HfModelStore)
    assert isinstance(setup_prompter(), QuestionaryPrompter)


def test_catalog_lists_the_bundled_voices(home: Path) -> None:
    names = build_client({}).catalog().list()
    assert "v0-default" in names
    assert (home / "config" / "voices" / "v0-default").is_dir()
