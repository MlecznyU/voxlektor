"""Lightweight wiring for the CLI: settings, platform, protocol client, autostart, voice catalog,
and the daemon-backed sample player of ``vxl setup``.

Imports nothing heavy (no numpy, MLX, audio libraries, asyncio or runtime), so ``vxl`` starts
fast. Settings precedence: CLI flag > ``VOXLEKTOR_*`` env > ``config.toml`` > defaults; a flag
whose value is ``None`` was not given and does not override anything.

``ClientContext.send`` returns the reply's ``data``; an error reply becomes a ``DomainError``
carrying the wire ``code`` and message; an unreachable daemon is ``ServiceUnavailable``, and
``NotRunning`` (its subclass) when nobody listens on the socket at all.

Hung daemon: ``ensure`` checks, when its first ping fails and before anything is spawned, whether
the daemon lock (``voxlektor.lock``) is held. A holder that neither answers nor releases the lock
within 15 s (the spawn timeout; a daemon starting up or finishing a stop does one of the two) is
reported as ``ServiceUnavailable`` (``hung_message``) instead of fresh starts that each exit
with "already running". ``hung_daemon`` names such a holder; ``stop_hung`` stops a verified one.
"""

import os
import platform as host
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import msgspec

import voxlektor
from voxlektor.adapters.procs.lock_holder import (
    LockHolder,
    Outcome,
    find_holder,
    lock_held,
    lock_path,
    terminate,
)
from voxlektor.adapters.procs.spawn import ensure_daemon
from voxlektor.adapters.storage.settings_toml import TomlSettingsStore
from voxlektor.adapters.storage.voices_fs import FsVoiceRepository
from voxlektor.adapters.transport.unix_client import request
from voxlektor.app.ports import ModelStore, PlatformInfo, Prompter, SettingsStore
from voxlektor.app.voices import VoiceCatalog
from voxlektor.bootstrap.platform_select import select_platform
from voxlektor.contracts import protocol as p
from voxlektor.domain.errors import (
    DomainError,
    InvalidDuration,
    InvalidSetting,
    NotRunning,
    ServiceUnavailable,
)
from voxlektor.domain.profiles import DEFAULT_PROFILE as _DEFAULT_PROFILE
from voxlektor.domain.profiles import Source, resolve, valid_profile_name
from voxlektor.domain.settings import Settings, merge_layers
from voxlektor.domain.types import Lang

_CONFIG_FILE = "config.toml"
_VOICES_DIR = "voices"
_PING_TIMEOUT_S = 1.0
_SAMPLE_POLL_S = 0.2
_SAMPLE_TIMEOUT_S = 120.0
_NOT_WARM = frozenset({"cold", "loading"})
_GONE_WAIT_S = 5.0  # an outdated daemon removing its socket and pid file
_GONE_POLL_S = 0.1
# A lock holder that does not answer gets the full spawn timeout of ``ensure_daemon`` (15 s) to
# answer or release the lock: a cold start (model, socket) or an ordinary stop (engine unload)
# is never reported as hung; then it is, instead of 15 s of respawns that say "already running".
_HUNG_GRACE_S = 15.0
_HUNG_KILL_GRACE_S = 3.0  # ``vxl stop``: SIGTERM, this long, then SIGKILL

# Replaced by tests (fake clock); production uses the real ones.
_sleep = time.sleep
_monotonic = time.monotonic


@dataclass(frozen=True, slots=True)
class ClientContext:
    settings: Settings
    platform: PlatformInfo
    store: SettingsStore
    config_path: Path
    # Set by ``ensure(work=True)``; read through ``daemon_outdated``.
    _outdated: list[bool] = field(default_factory=list[bool], init=False, repr=False, compare=False)

    def send(self, cmd: str, **args: object) -> dict[str, object]:
        """One request to the daemon; the reply's data, or the reply's error raised."""
        raw = request(self.platform.socket_path(), {"cmd": cmd, **args})
        reply = msgspec.convert(raw, p.Reply)
        if not reply.ok:
            if reply.error is None:
                raise DomainError(p.ErrorCode.internal.value, f"{cmd} failed")
            raise DomainError(reply.error.code.value, reply.error.message)
        return dict(reply.data or {})

    def ensure(self, *, work: bool = False) -> None:
        """Make sure a daemon answers, autostarting one if needed.

        With ``work`` (the commands that start work: ``say``, ``warm``, ``watch``) a daemon whose
        status ``version`` differs from this CLI's (or has none) is replaced when it is idle:
        ``shutdown`` with ``if_idle``, wait until its socket and pid file are gone, start a fresh
        one. A daemon that is reading (it reports so, or refuses with ``busy``) is kept and
        ``daemon_outdated`` becomes true, so no other session's speech is ever dropped. The
        version comes from the ping's ``status`` reply, so a current daemon costs no extra
        request. Daemons from before ``if_idle`` ignore the field; they are only asked when their
        status shows nothing queued or playing.
        """
        seen: list[Mapping[str, object]] = []
        ensure_daemon(self.platform, self._first_ping(seen))
        if not work or not seen or seen[-1].get("version") == voxlektor.__version__:
            return
        if self._stop_outdated(seen[-1]):
            ensure_daemon(self.platform, lambda: self._ping([]))

    @property
    def daemon_outdated(self) -> bool:
        """True after ``ensure(work=True)`` kept a daemon older than this CLI (it was busy)."""
        return bool(self._outdated)

    def _stop_outdated(self, status: Mapping[str, object]) -> bool:
        """Stop an outdated daemon when it is idle; True when a fresh one must be started."""
        queued = status.get("queued") or status.get("groups")
        if status.get("playing") is True or bool(queued):
            self._outdated.append(True)
            return False
        try:
            self.send("shutdown", if_idle=True)
        except NotRunning:
            return True  # gone already
        except DomainError:  # ``busy``, or any other refusal: keep it and say so
            self._outdated.append(True)
            return False
        if not _wait_gone(self.platform.socket_path(), self.platform.pid_path()):
            self._outdated.append(True)  # still there: the fresh start may reach the old one
        return True

    def catalog(self) -> VoiceCatalog:
        """The local voice catalog, with the bundled voices installed."""
        catalog = VoiceCatalog(FsVoiceRepository(self.platform.config_dir() / _VOICES_DIR))
        catalog.ensure_bundled()
        return catalog

    def hung_daemon(self, grace_s: float = 0.0) -> LockHolder | None:
        """The holder of the daemon lock when it is held but nobody answers ``status``.

        None when the lock is free or a daemon answers. With ``grace_s`` the holder gets that
        long to answer or to release the lock (it may be starting up or finishing a stop).
        """
        return self._silent_holder([], grace_s)

    def stop_hung(self, holder: LockHolder) -> Outcome:
        """SIGTERM, a 3 s grace, then SIGKILL for a verified holder (``lock_holder.terminate``).

        An unverified or unknown holder is never signalled: ``ServiceUnavailable`` naming it.
        Unless it survived, its pid file (when it still holds its pid) is removed.
        """
        lock = lock_path(self.platform.runtime_dir())
        if holder.pid is None:
            raise ServiceUnavailable(
                f"cannot tell which process holds {lock}; find it with: lsof {lock}"
            )
        if not holder.verified:
            raise ServiceUnavailable(
                f"pid {holder.pid} holds {lock} but is not recognisably a voxlektor daemon; "
                "not stopping it"
            )
        outcome = terminate(holder.pid, _HUNG_KILL_GRACE_S)
        pid_path = self.platform.pid_path()
        if outcome != "survived" and _recorded_pid(pid_path) == str(holder.pid):
            pid_path.unlink(missing_ok=True)
        return outcome

    def _first_ping(self, seen: list[Mapping[str, object]]) -> Callable[[], bool]:
        """``ensure_daemon``'s ping; when the first one fails, a silent lock holder is an error.

        Before anything is spawned: a daemon that holds the lock but does not answer (within a
        short grace) would only make every fresh start exit with "already running".
        """
        first = [True]

        def ping() -> bool:
            if self._ping(seen):
                return True
            if not first:
                return False
            first.clear()
            holder = self._silent_holder(seen, _HUNG_GRACE_S)
            if holder is not None:
                raise ServiceUnavailable(hung_message(holder))
            return bool(seen)  # it answered during the grace

        return ping

    def _silent_holder(self, seen: list[Mapping[str, object]], grace_s: float) -> LockHolder | None:
        lock = lock_path(self.platform.runtime_dir())
        deadline = _monotonic() + grace_s
        while lock_held(lock):
            if self._ping(seen):
                return None
            if _monotonic() >= deadline:
                return find_holder(lock, self.platform.pid_path())
            _sleep(_GONE_POLL_S)
        return None

    def _ping(self, seen: list[Mapping[str, object]]) -> bool:
        """A short ``status``; the data of an answer is appended to ``seen``."""
        try:
            reply = request(self.platform.socket_path(), {"cmd": "status"}, _PING_TIMEOUT_S)
        except ServiceUnavailable:
            return False
        if reply.get("ok") is not True:
            return False
        data = reply.get("data")
        seen.append(msgspec.convert(data, dict[str, object]) if isinstance(data, dict) else {})
        return True


def hung_message(holder: LockHolder) -> str:
    """What to tell the user about a daemon that holds the lock but does not answer."""
    who = "a voxlektor daemon" if holder.pid is None else f"a voxlektor daemon (pid {holder.pid})"
    return f"{who} holds the lock but does not answer; stop it with: vxl stop"


def _recorded_pid(pid_path: Path) -> str | None:
    try:
        return pid_path.read_text(encoding="ascii").strip()
    except (OSError, UnicodeDecodeError):
        return None


def _wait_gone(*paths: Path) -> bool:
    """Wait until none of ``paths`` exists; False when one is still there after a while."""
    deadline = _monotonic() + _GONE_WAIT_S
    while any(path.exists() for path in paths):
        if _monotonic() >= deadline:
            return False
        _sleep(_GONE_POLL_S)
    return True


def build_client(flags: Mapping[str, object]) -> ClientContext:
    platform, config_path, store = _locate()
    given = {key: value for key, value in flags.items() if value is not None}
    settings = merge_layers(store.read_file(), store.read_env(), given)
    return ClientContext(settings, platform, store, config_path)


def resolve_settings(
    profile: str | None, flags: Mapping[str, object]
) -> tuple[Settings, dict[str, Source]]:
    """The effective settings of ``profile`` and the source of every key.

    Precedence: flag > the profile's own keys > ``VOXLEKTOR_*`` env > ``default`` > built-ins
    (see ``domain.profiles.resolve``); ``None`` is the ``default`` profile, and a flag whose
    value is ``None`` was not given. An invalid name (``InvalidSetting`` "invalid profile name
    ...") or a profile missing from ``config.toml`` (``InvalidSetting`` "unknown profile
    'NAME'") is a user error (exit 1).
    """
    _, _, store = _locate()
    return _resolve(store, profile, flags)


def build_profile_client(profile: str | None, flags: Mapping[str, object]) -> ClientContext:
    """``build_client`` whose settings are ``resolve_settings(profile, flags)``."""
    platform, config_path, store = _locate()
    settings, _ = _resolve(store, profile, flags)
    return ClientContext(settings, platform, store, config_path)


def _resolve(
    store: SettingsStore, profile: str | None, flags: Mapping[str, object]
) -> tuple[Settings, dict[str, Source]]:
    name = _DEFAULT_PROFILE if profile is None else profile
    _require_valid_name(name)
    return resolve(
        profile=name,
        profiles=store.read_profiles(),
        globals_=store.read_globals(),
        env=store.read_env(),
        flags={key: value for key, value in flags.items() if value is not None},
    )


def _require_valid_name(name: str) -> None:
    if not valid_profile_name(name):
        raise InvalidSetting(
            f"invalid profile name {name!r}; use a-z, 0-9, '-' and '_' (at most 32, "
            "starting with a letter or digit)"
        )


def build_setup_client(warn: Callable[[str], None]) -> tuple[ClientContext, Settings]:
    """``build_client({})`` for ``vxl setup``, which is how a broken config gets repaired.

    An invalid value in ``config.toml`` or in a ``VOXLEKTOR_*`` variable is skipped (its
    default is used) and reported through ``warn``. Also returns the setup's current values:
    the file's valid values only, never the env's (Enter in setup must not persist an env value).
    """
    ctx, profiles = build_tolerant_client(warn)
    return ctx, profiles.file_default


class TolerantProfiles:
    """Profile resolution over the valid values only (made by ``build_tolerant_client``).

    ``names``: ``default`` first, then the other profiles by name. ``file_default``: the
    ``default`` profile from the file alone (no env). ``resolve(profile)``: whether the profile
    exists, its effective settings (profile > env > default > built-ins) and the source of
    every key; an unknown or invalid name is not an error here: ``False`` and default's values.
    """

    def __init__(
        self,
        profiles: Mapping[str, Mapping[str, object]],
        globals_: Mapping[str, object],
        env: Mapping[str, object],
    ) -> None:
        self._profiles = profiles
        self._globals = globals_
        self._env = env
        others = sorted(name for name in profiles if name != _DEFAULT_PROFILE)
        self.names: tuple[str, ...] = (_DEFAULT_PROFILE, *others)
        self.file_default, _ = resolve(
            profile=_DEFAULT_PROFILE, profiles=profiles, globals_=globals_, env={}, flags={}
        )

    def resolve(self, profile: str | None) -> tuple[bool, Settings, dict[str, Source]]:
        name = _DEFAULT_PROFILE if profile is None else profile
        exists = valid_profile_name(name) and name in self.names
        settings, sources = resolve(
            profile=name if exists else _DEFAULT_PROFILE,
            profiles=self._profiles,
            globals_=self._globals,
            env=self._env,
            flags={},
        )
        return exists, settings, sources

    def resolve_known(self, profile: str | None) -> tuple[Settings, dict[str, Source]]:
        """``resolve`` for a profile that must exist: an invalid name or an unknown profile is
        ``InvalidSetting``, as in ``resolve_settings``."""
        name = _DEFAULT_PROFILE if profile is None else profile
        _require_valid_name(name)
        exists, settings, sources = self.resolve(name)
        if not exists:
            raise InvalidSetting(f"unknown profile {name!r}")
        return settings, sources


def build_tolerant_client(warn: Callable[[str], None]) -> tuple[ClientContext, TolerantProfiles]:
    """The client of the read-only views (``setup``/``setup --check``, ``vxl profiles``,
    ``config show``): one bad value never stops them.

    Every invalid value in ``config.toml`` (globals and any profile) or in a ``VOXLEKTOR_*``
    variable is skipped and reported once through ``warn`` (``ignoring invalid KEY in
    config.toml: …``, ``… in profile 'NAME' of config.toml: …``, ``ignoring invalid NAME in the
    environment: …``); the next layer's value is used. A file that is no valid TOML (or whose
    ``[profiles]`` / ``[service]`` is no table) still raises ``InvalidSetting``. The context's
    settings are the ``default`` profile's (env included, no flags).
    """
    platform, config_path, store = _locate()
    globals_ = store.read_globals()
    tables = store.read_profiles()
    bad: set[tuple[str, str]] = set()  # (profile or "", key) of skipped file values
    for key, value in store.read_file().items():  # the globals and default, in file order
        if (problem := _invalid({key: value})) is not None:
            warn(f"ignoring invalid {key} in {_CONFIG_FILE}: {problem}")
            bad.add(("" if key in globals_ else _DEFAULT_PROFILE, key))
    others = sorted(name for name in tables if name != _DEFAULT_PROFILE)
    for name in others:
        for key, value in tables[name].items():
            if (problem := _invalid({key: value})) is not None:
                warn(f"ignoring invalid {key} in profile {name!r} of {_CONFIG_FILE}: {problem}")
                bad.add((name, key))
    env = _tolerant_env(config_path, warn)
    valid_globals = {k: v for k, v in globals_.items() if ("", k) not in bad}
    valid_tables = {
        name: {k: v for k, v in table.items() if (name, k) not in bad}
        for name, table in tables.items()
    }
    profiles = TolerantProfiles(valid_tables, valid_globals, env)
    _, settings, _ = profiles.resolve(None)
    return ClientContext(settings, platform, store, config_path), profiles


def _tolerant_env(config_path: Path, warn: Callable[[str], None]) -> dict[str, object]:
    """The valid ``VOXLEKTOR_*`` settings; each invalid variable reported and skipped."""
    env_values: dict[str, object] = {}
    for name in sorted(os.environ):
        single = TomlSettingsStore(config_path, {name: os.environ[name]})
        try:
            values = single.read_env()  # empty for names that are no setting
        except InvalidSetting as err:
            warn(f"ignoring invalid {name} in the environment: {err.message}")
            continue
        if (problem := _invalid(values)) is not None:
            warn(f"ignoring invalid {name} in the environment: {problem}")
        else:
            env_values.update(values)
    return env_values


def model_store() -> ModelStore:
    """The model store of ``vxl setup`` (the Hugging Face cache; imports huggingface_hub)."""
    from voxlektor.adapters.models.hf_cache import HfModelStore  # noqa: PLC0415  # heavy

    return HfModelStore()


def setup_prompter() -> Prompter:
    """The menus of the interactive ``vxl setup`` (imports questionary)."""
    from voxlektor.adapters.terminal.menus import QuestionaryPrompter  # noqa: PLC0415  # heavy

    return QuestionaryPrompter()


def _locate() -> tuple[PlatformInfo, Path, TomlSettingsStore]:
    platform = select_platform(sys.platform, host.machine(), host.mac_ver()[0])
    config_path = platform.config_dir() / _CONFIG_FILE
    return platform, config_path, TomlSettingsStore(config_path, os.environ)


def _invalid(values: Mapping[str, object]) -> str | None:
    """Why ``values`` is not a valid settings layer, or ``None`` when it is."""
    try:
        merge_layers(values)
    except (InvalidSetting, InvalidDuration) as err:
        return err.message
    return None


class DaemonSamplePlayer:
    """SamplePlayer for setup: an ordinary ``speak`` (``now``) through the daemon, then wait.

    The daemon is started when needed (also with ``autostart = false``: setup wants samples).
    ``play`` polls ``status`` every 0.2 s until nothing plays or waits, at most 120 s, and says
    ``loading the model…`` once while the engine is still cold or loading. When nothing is left
    and the daemon reports ``last_error`` (e.g. the model failed to load), that error is raised
    as ``ServiceUnavailable``.
    """

    def __init__(self, ctx: ClientContext) -> None:
        self._ctx = ctx
        self._announced = False

    def play(self, text: str, voice: str, lang: Lang, rate: float, volume: int) -> None:
        ctx = self._ctx
        ctx.ensure()
        ctx.send(
            "speak",
            text=text,
            voice=voice,
            lang=lang,
            base_lang=lang,
            rate=rate,
            volume=volume,
            now=True,
        )
        deadline = _monotonic() + _SAMPLE_TIMEOUT_S
        while True:
            status = ctx.send("status")
            if status.get("playing") is not True and not status.get("queued"):
                # Done, whatever the state: a failed model load leaves the daemon ``cold``.
                error = status.get("last_error")
                if isinstance(error, str) and error:
                    raise ServiceUnavailable(error)
                return
            if status.get("state") in _NOT_WARM and not self._announced:
                print("loading the model…", file=sys.stderr, flush=True)
                self._announced = True
            if _monotonic() >= deadline:
                raise ServiceUnavailable("the sample did not finish playing in time")
            _sleep(_SAMPLE_POLL_S)
