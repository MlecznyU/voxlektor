"""The CLI's lightweight wiring: layered settings, the protocol client and daemon autostart."""

import fcntl
import os
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

import voxlektor
from voxlektor.adapters.platform.unsupported import UnsupportedPlatform
from voxlektor.adapters.procs.lock_holder import LockHolder
from voxlektor.app.ports import PlatformInfo
from voxlektor.bootstrap import client
from voxlektor.bootstrap.client import (
    DaemonSamplePlayer,
    build_client,
    build_profile_client,
    build_setup_client,
    build_tolerant_client,
    model_store,
    resolve_settings,
    setup_prompter,
)
from voxlektor.domain.errors import (
    DomainError,
    InvalidSetting,
    NotRunning,
    PlatformError,
    ServiceUnavailable,
)
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


# ---------------------------------------------------------------- profiles

_PROFILES_TOML = (
    "config_version = 2\n"
    "history = 7\n"
    "[profiles.default]\n"
    'voice = "f2-narrator"\n'
    "rate = 1.5\n"
    "[profiles.claude]\n"
    "rate = 1.25\n"
    "volume = 70\n"
)


def test_resolve_settings_layers_flag_profile_env_default_builtin(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_config(home, _PROFILES_TOML)
    monkeypatch.setenv("VOXLEKTOR_VOLUME", "40")  # claude sets volume: the profile wins
    monkeypatch.setenv("VOXLEKTOR_BASE_LANG", "en")  # claude does not: the env wins
    settings, sources = resolve_settings("claude", {"steps": 8, "lang": None})
    assert settings == Settings(
        voice=VoiceName("f2-narrator"), base_lang="en", rate=1.25, volume=70, steps=8, history=7
    )
    assert sources["steps"] == "flag"
    assert sources["volume"] == "profile"
    assert sources["base_lang"] == "env"
    assert sources["rate"] == "profile"
    assert sources["voice"] == "default"
    assert sources["history"] == "default"
    assert sources["lang"] == "builtin"  # a flag left unset is no flag


@pytest.mark.parametrize("profile", [None, "default"])
def test_resolve_settings_without_a_profile_is_the_default_profile(
    home: Path, profile: str | None
) -> None:
    _write_config(home, _PROFILES_TOML)
    settings, sources = resolve_settings(profile, {})
    assert settings == Settings(voice=VoiceName("f2-narrator"), rate=1.5, history=7)
    assert sources["rate"] == "default"
    assert build_client({}).settings == settings


def test_resolve_settings_of_an_unknown_profile_raises(home: Path) -> None:
    _write_config(home, _PROFILES_TOML)
    with pytest.raises(InvalidSetting, match="unknown profile 'nobody'"):
        resolve_settings("nobody", {})


@pytest.mark.parametrize("name", ["Claude", "-x", "a b", "x" * 33, ""])
def test_resolve_settings_rejects_an_invalid_profile_name(home: Path, name: str) -> None:
    with pytest.raises(InvalidSetting, match="invalid profile name"):
        resolve_settings(name, {})


def test_build_profile_client_carries_the_profile_settings(home: Path) -> None:
    _write_config(home, _PROFILES_TOML)
    ctx = build_profile_client("claude", {"voice": "m1-calm"})
    assert ctx.settings == Settings(voice=VoiceName("m1-calm"), rate=1.25, volume=70, history=7)
    assert ctx.config_path == home / "config" / "config.toml"
    with pytest.raises(InvalidSetting, match="unknown profile 'nobody'"):
        build_profile_client("nobody", {})


# ---------------------------------------------------------------- tolerant resolution

_BROKEN_TOML = (
    "config_version = 2\n"
    "history = 0\n"
    "engine = 'omnivoice'\n"
    "[profiles.default]\n"
    'voice = "f2-narrator"\n'
    "rate = 9.0\n"
    "[profiles.zeta]\n"
    "volume = 500\n"
    "steps = 8\n"
    "[profiles.claude]\n"
    "rate = 1.25\n"
    'lang = "de"\n'
)


def test_tolerant_resolution_skips_each_invalid_value_once_with_a_warning(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_config(home, _BROKEN_TOML)
    monkeypatch.setenv("VOXLEKTOR_STEPS", "many")
    monkeypatch.setenv("VOXLEKTOR_VOLUME", "40")
    warnings: list[str] = []
    ctx, profiles = build_tolerant_client(warnings.append)
    assert warnings == [
        "ignoring invalid rate in config.toml: rate must be between 0.5 and 3.0",
        "ignoring invalid history in config.toml: history must be a positive integer",
        "ignoring invalid lang in profile 'claude' of config.toml: "
        "lang must be one of auto, pl, en",
        "ignoring invalid volume in profile 'zeta' of config.toml: "
        "volume must be an integer between 0 and 100",
        "ignoring invalid VOXLEKTOR_STEPS in the environment: "
        "VOXLEKTOR_STEPS: invalid value 'many' for steps",
    ]
    assert profiles.names == ("default", "claude", "zeta")
    assert ctx.settings == Settings(voice=VoiceName("f2-narrator"), volume=40)
    exists, settings, sources = profiles.resolve("zeta")
    assert exists is True
    assert settings == Settings(voice=VoiceName("f2-narrator"), volume=40, steps=8)
    assert (sources["volume"], sources["steps"], sources["rate"]) == ("env", "profile", "builtin")
    _, claude, claude_sources = profiles.resolve("claude")
    assert (claude.rate, claude.lang, claude_sources["lang"]) == (1.25, "auto", "builtin")
    assert profiles.file_default == Settings(voice=VoiceName("f2-narrator"))  # never the env's
    assert len(warnings) == 5  # resolving again warns nothing new


@pytest.mark.parametrize("name", ["ghost", "Not Valid"])
def test_tolerant_resolution_of_an_unknown_profile_is_default_and_not_existing(
    home: Path, name: str
) -> None:
    _write_config(home, _PROFILES_TOML)
    _, profiles = build_tolerant_client(print)
    exists, settings, _ = profiles.resolve(name)
    assert exists is False
    assert settings == Settings(voice=VoiceName("f2-narrator"), rate=1.5, history=7)
    assert profiles.resolve(None)[:2] == (True, settings)


# ---------------------------------------------------------------- outdated daemon


_IDLE: dict[str, object] = {"playing": False, "queued": 0, "groups": []}


class _Versioned:
    """A daemon reporting ``statuses`` in turn (the last one repeats); ``ensure_daemon`` pings.

    ``shutdown`` answers ok, or the error / exception in ``shutdown``.
    """

    def __init__(
        self, statuses: list[dict[str, object]], shutdown: dict[str, object] | Exception
    ) -> None:
        self.statuses = statuses
        self.shutdown = shutdown
        self.requests: list[dict[str, object]] = []
        self.ensured = 0

    def request(
        self, path: Path, req: dict[str, object], timeout: float = 5.0
    ) -> dict[str, object]:
        self.requests.append(req)
        if req["cmd"] == "status":
            data = self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]
            return {"v": 1, "id": "x", "ok": True, "data": data}
        if isinstance(self.shutdown, Exception):
            raise self.shutdown
        return {"v": 1, "id": "x", **self.shutdown}

    def ensure(self, platform: PlatformInfo, ping: Callable[[], bool]) -> None:
        self.ensured += 1
        assert ping()

    @property
    def cmds(self) -> list[object]:
        return [req["cmd"] for req in self.requests]


def _versioned(
    monkeypatch: pytest.MonkeyPatch,
    statuses: list[dict[str, object]],
    shutdown: dict[str, object] | Exception | None = None,
) -> _Versioned:
    daemon = _Versioned(statuses, {"ok": True, "data": {}} if shutdown is None else shutdown)
    monkeypatch.setattr(client, "request", daemon.request)
    monkeypatch.setattr(client, "ensure_daemon", daemon.ensure)
    return daemon


_BUSY: dict[str, object] = {
    "ok": False,
    "error": {"code": "busy", "message": "the daemon is reading"},
}


def test_outdated_idle_daemon_restarted(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    old: dict[str, object] = {"version": "0.0.1", **_IDLE}
    new: dict[str, object] = {"version": voxlektor.__version__, **_IDLE}
    daemon = _versioned(monkeypatch, [old, new])
    ctx = build_client({})
    ctx.ensure(work=True)
    assert daemon.cmds == ["status", "shutdown", "status"]
    assert daemon.requests[1] == {"cmd": "shutdown", "if_idle": True}
    assert daemon.ensured == 2  # a fresh daemon was started
    assert ctx.daemon_outdated is False


def test_outdated_busy_daemon_is_kept_and_reported(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    daemon = _versioned(monkeypatch, [{"version": "0.0.1", **_IDLE}], _BUSY)
    ctx = build_client({})
    ctx.ensure(work=True)
    assert daemon.cmds == ["status", "shutdown"]
    assert daemon.ensured == 1
    assert ctx.daemon_outdated is True


def test_outdated_daemon_that_reports_a_reading_is_not_asked_to_stop(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A daemon from before ``if_idle`` ignores the field: never send it a plain shutdown.
    reading: dict[str, object] = {"version": "0.0.1", "playing": False, "queued": 2, "groups": []}
    daemon = _versioned(monkeypatch, [reading])
    ctx = build_client({})
    ctx.ensure(work=True)
    assert daemon.cmds == ["status"]
    assert ctx.daemon_outdated is True


def test_daemon_without_a_version_is_outdated(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    daemon = _versioned(monkeypatch, [{"playing": True}])
    ctx = build_client({})
    ctx.ensure(work=True)
    assert daemon.cmds == ["status"]
    assert ctx.daemon_outdated is True


def test_outdated_daemon_gone_before_the_shutdown_is_started_fresh(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    daemon = _versioned(monkeypatch, [{"version": "0.0.1", **_IDLE}], NotRunning())
    ctx = build_client({})
    ctx.ensure(work=True)
    assert daemon.cmds == ["status", "shutdown", "status"]
    assert daemon.ensured == 2
    assert ctx.daemon_outdated is False


def test_outdated_daemon_waits_until_its_socket_is_gone(
    home: Path, monkeypatch: pytest.MonkeyPatch, clock: list[float]
) -> None:
    daemon = _versioned(monkeypatch, [{"version": "0.0.1", **_IDLE}])
    ctx = build_client({})
    sock = ctx.platform.socket_path()
    sock.parent.mkdir(parents=True, exist_ok=True)
    sock.write_text("")
    ticks: list[float] = []

    def sleep(seconds: float) -> None:
        ticks.append(seconds)
        clock[0] += seconds
        if len(ticks) == 3:
            sock.unlink()

    monkeypatch.setattr(client, "_sleep", sleep)
    ctx.ensure(work=True)
    assert len(ticks) == 3
    assert daemon.ensured == 2
    assert ctx.daemon_outdated is False


def test_outdated_daemon_that_does_not_go_away_is_reported(
    home: Path, monkeypatch: pytest.MonkeyPatch, clock: list[float]
) -> None:
    daemon = _versioned(monkeypatch, [{"version": "0.0.1", **_IDLE}])
    ctx = build_client({})
    sock = ctx.platform.socket_path()
    sock.parent.mkdir(parents=True, exist_ok=True)
    sock.write_text("")
    ctx.ensure(work=True)
    assert clock[0] >= 5.0
    assert daemon.ensured == 2
    assert ctx.daemon_outdated is True


def test_current_daemon_is_left_alone(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    daemon = _versioned(monkeypatch, [{"version": voxlektor.__version__, **_IDLE}])
    ctx = build_client({})
    ctx.ensure(work=True)
    assert daemon.cmds == ["status"]
    assert ctx.daemon_outdated is False


def test_ensure_without_work_never_checks_the_version(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    daemon = _versioned(monkeypatch, [{"version": "0.0.1", **_IDLE}])
    ctx = build_client({})
    ctx.ensure()
    assert daemon.cmds == ["status"]
    assert ctx.daemon_outdated is False


# ---------------------------------------------------------------- hung daemon (lock, no answer)

_HUNG = 4242


@pytest.fixture
def held_lock(home: Path) -> Iterator[Path]:
    """``voxlektor.lock`` held by another open file, as a live daemon holds it."""
    run = home / "run"
    run.mkdir(parents=True, exist_ok=True)
    path = run / "voxlektor.lock"
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        yield path
    finally:
        os.close(fd)


def _holder(monkeypatch: pytest.MonkeyPatch, holder: LockHolder) -> list[tuple[Path, Path]]:
    asked: list[tuple[Path, Path]] = []

    def find(lock_path: Path, pid_path: Path) -> LockHolder:
        asked.append((lock_path, pid_path))
        return holder

    monkeypatch.setattr(client, "find_holder", find)
    return asked


def _spawner(monkeypatch: pytest.MonkeyPatch) -> list[bool]:
    """A fake ``ensure_daemon`` that pings once, as the real one does before it spawns."""
    pings: list[bool] = []

    def ensure(platform: PlatformInfo, ping: Callable[[], bool]) -> None:
        pings.append(ping())

    monkeypatch.setattr(client, "ensure_daemon", ensure)
    return pings


def test_ensure_fails_fast_when_a_silent_daemon_holds_the_lock(
    held_lock: Path, monkeypatch: pytest.MonkeyPatch, clock: list[float]
) -> None:
    _fake_request(monkeypatch, NotRunning())
    asked = _holder(monkeypatch, LockHolder(_HUNG, verified=True))
    _spawner(monkeypatch)
    ctx = build_client({})
    with pytest.raises(ServiceUnavailable) as caught:
        ctx.ensure(work=True)
    assert caught.value.message == (
        "a voxlektor daemon (pid 4242) holds the lock but does not answer; stop it with: vxl stop"
    )
    assert type(caught.value) is ServiceUnavailable  # never NotRunning: something is there
    # The full spawn timeout: a cold start or an ordinary stop is never reported as hung.
    assert 15.0 <= clock[0] < 16.0
    assert asked == [(held_lock, ctx.platform.pid_path())]


def test_silent_holder_without_a_known_pid_is_still_reported(
    held_lock: Path, monkeypatch: pytest.MonkeyPatch, clock: list[float]
) -> None:
    _fake_request(monkeypatch, ServiceUnavailable("voxlektor did not answer within 1 s"))
    _holder(monkeypatch, LockHolder(None, verified=False))
    _spawner(monkeypatch)
    with pytest.raises(ServiceUnavailable) as caught:
        build_client({}).ensure()
    assert caught.value.message == (
        "a voxlektor daemon holds the lock but does not answer; stop it with: vxl stop"
    )


def test_holder_that_releases_the_lock_in_the_grace_lets_the_start_go_on(
    held_lock: Path, monkeypatch: pytest.MonkeyPatch, clock: list[float]
) -> None:
    _fake_request(monkeypatch, NotRunning())
    asked = _holder(monkeypatch, LockHolder(_HUNG, verified=True))
    pings = _spawner(monkeypatch)

    def sleep(seconds: float) -> None:
        clock[0] += seconds
        if clock[0] >= 1.0 and not held_lock.exists():
            held_lock.write_text("")  # what the next start finds: a free lock file
        elif clock[0] >= 1.0:
            held_lock.unlink()  # the fixture's lock is on the old file: released for us

    monkeypatch.setattr(client, "_sleep", sleep)
    build_client({}).ensure()
    assert pings == [False]  # not running: ensure_daemon goes on and spawns
    assert asked == []
    assert clock[0] < 2.0


def test_holder_that_starts_answering_in_the_grace_is_used(
    held_lock: Path, monkeypatch: pytest.MonkeyPatch, clock: list[float]
) -> None:
    answers: list[dict[str, object] | Exception] = [
        NotRunning(),
        NotRunning(),
        {"v": 1, "id": "x", "ok": True, "data": {"version": voxlektor.__version__}},
    ]

    def fake(path: Path, req: dict[str, object], timeout: float = 5.0) -> dict[str, object]:
        answer = answers.pop(0) if len(answers) > 1 else answers[0]
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr(client, "request", fake)
    _holder(monkeypatch, LockHolder(_HUNG, verified=True))
    pings = _spawner(monkeypatch)
    ctx = build_client({})
    ctx.ensure(work=True)
    assert pings == [True]
    assert ctx.daemon_outdated is False  # its status (current version) was seen


def test_free_lock_costs_no_wait_and_no_holder_lookup(
    home: Path, monkeypatch: pytest.MonkeyPatch, clock: list[float]
) -> None:
    calls = _fake_request(monkeypatch, NotRunning())
    asked = _holder(monkeypatch, LockHolder(_HUNG, verified=True))
    pings = _spawner(monkeypatch)
    build_client({}).ensure()
    assert pings == [False]
    assert len(calls) == 1
    assert asked == []
    assert clock[0] == 0.0


def test_only_the_first_failed_ping_looks_at_the_lock(
    held_lock: Path, monkeypatch: pytest.MonkeyPatch, clock: list[float]
) -> None:
    # After the spawn the fresh daemon holds the lock while it starts: never report it as hung.
    _fake_request(monkeypatch, NotRunning())
    pings: list[bool] = []

    def ensure(platform: PlatformInfo, ping: Callable[[], bool]) -> None:
        held_lock.unlink()  # free at the first ping: a spawn follows
        pings.append(ping())
        held_lock.write_text("")
        _hold_again(held_lock)
        pings.extend([ping(), ping()])

    monkeypatch.setattr(client, "ensure_daemon", ensure)
    build_client({}).ensure()
    assert pings == [False, False, False]
    assert clock[0] == 0.0


_held: list[int] = []


def _hold_again(lock: Path) -> None:
    fd = os.open(lock, os.O_RDWR)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    _held.append(fd)


@pytest.fixture(autouse=True)
def _close_held() -> Iterator[None]:
    yield
    while _held:
        os.close(_held.pop())


def test_hung_daemon_is_none_when_the_lock_is_free(home: Path) -> None:
    assert build_client({}).hung_daemon() is None


def test_hung_daemon_is_none_when_the_holder_answers(
    held_lock: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_request(monkeypatch, {"v": 1, "id": "x", "ok": True, "data": {}})
    assert build_client({}).hung_daemon() is None


def test_hung_daemon_names_a_silent_holder_at_once_without_a_grace(
    held_lock: Path, monkeypatch: pytest.MonkeyPatch, clock: list[float]
) -> None:
    _fake_request(monkeypatch, NotRunning())
    _holder(monkeypatch, LockHolder(_HUNG, verified=True))
    assert build_client({}).hung_daemon() == LockHolder(_HUNG, verified=True)
    assert clock[0] == 0.0


def _terminate(monkeypatch: pytest.MonkeyPatch, outcome: str) -> list[tuple[int, float]]:
    calls: list[tuple[int, float]] = []

    def terminate(pid: int, grace_s: float = 3.0) -> str:
        calls.append((pid, grace_s))
        return outcome

    monkeypatch.setattr(client, "terminate", terminate)
    return calls


@pytest.mark.parametrize("outcome", ["gone", "terminated", "killed"])
def test_stop_hung_signals_a_verified_holder_and_removes_its_pid_file(
    home: Path, monkeypatch: pytest.MonkeyPatch, outcome: str
) -> None:
    calls = _terminate(monkeypatch, outcome)
    ctx = build_client({})
    pid_file = ctx.platform.pid_path()
    pid_file.write_text(f"{_HUNG}\n")
    assert ctx.stop_hung(LockHolder(_HUNG, verified=True)) == outcome
    assert calls == [(_HUNG, 3.0)]
    assert not pid_file.exists()


def test_stop_hung_keeps_a_pid_file_of_another_process(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _terminate(monkeypatch, "killed")
    ctx = build_client({})
    pid_file = ctx.platform.pid_path()
    pid_file.write_text("777\n")  # a successor already wrote its own
    ctx.stop_hung(LockHolder(_HUNG, verified=True))
    assert pid_file.read_text() == "777\n"


def test_stop_hung_keeps_the_pid_file_of_a_survivor(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _terminate(monkeypatch, "survived")
    ctx = build_client({})
    pid_file = ctx.platform.pid_path()
    pid_file.write_text(f"{_HUNG}\n")
    assert ctx.stop_hung(LockHolder(_HUNG, verified=True)) == "survived"
    assert pid_file.exists()


def test_stop_hung_without_a_pid_file_is_fine(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _terminate(monkeypatch, "terminated")
    assert build_client({}).stop_hung(LockHolder(_HUNG, verified=True)) == "terminated"


def test_stop_hung_refuses_an_unverified_holder(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _terminate(monkeypatch, "killed")
    ctx = build_client({})
    with pytest.raises(ServiceUnavailable) as caught:
        ctx.stop_hung(LockHolder(_HUNG, verified=False))
    lock = ctx.platform.runtime_dir() / "voxlektor.lock"
    assert caught.value.message == (
        f"pid 4242 holds {lock} but is not recognisably a voxlektor daemon; not stopping it"
    )
    assert calls == []


def test_stop_hung_refuses_an_unknown_holder(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _terminate(monkeypatch, "killed")
    ctx = build_client({})
    with pytest.raises(ServiceUnavailable) as caught:
        ctx.stop_hung(LockHolder(None, verified=False))
    lock = ctx.platform.runtime_dir() / "voxlektor.lock"
    assert caught.value.message == (
        f"cannot tell which process holds {lock}; find it with: lsof {lock}"
    )
    assert calls == []
