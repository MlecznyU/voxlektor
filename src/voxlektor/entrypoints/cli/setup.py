"""``vxl setup``: the guided, re-runnable first run (menus) or its silent flag mode (``--yes``).

Order: platform check (unsupported: exit 3 with where to ask for support), then
``app.setup``. Interactive mode needs a terminal (without one, and without ``--yes``, it refuses
with exit 1 and a hint); it plays samples through the daemon, which it starts even with
``autostart = false``. Flag mode asks nothing, plays nothing and starts no daemon; a model that
has to be downloaded needs ``--accept-model-license``; download progress goes to stderr.

``--check`` (no other options) changes nothing and starts nothing: it prints one JSON line
(``ready``, ``setup_completed``, ``version``, ``protocol``, ``config_path``, ``engine``,
``model`` with ``present`` and its license, the effective ``settings``, the ``features`` list)
and exits 0 when
setup has completed and the model is present, 1 otherwise. It needs no terminal; tools such as
dialogue-box use it (``features`` is the constant ``FEATURES``: what this version offers).

Profiles (S3): ``--profile NAME`` sets up one profile. Interactive with a non-default profile
asks only its voice, base language, rate, volume, reading language and quality steps (each
shows the effective value and its source; "same as default" removes the profile's value),
needs a completed setup and leaves a running daemon alone (it serves other profiles); with
``default`` (or without ``--profile``) it is the full setup. ``--yes --profile NAME`` creates
the profile when missing and stores only the given profile keys (``--voice``, ``--lang``,
``--base-lang``, ``--rate``, ``--volume``, ``--steps``; global options are refused), ensures the
model and marks the setup completed. ``--lang`` / ``--steps`` need ``--profile``.
``--check --profile NAME`` reports that profile (``profile``, ``profile_exists``; unknown: not
ready). ``--progress json`` (with ``--yes``) prints JSON lines on stdout instead of the stderr
progress: ``{"event":"download","done":N,"total":M}``, then ``{"event":"done"}`` or
``{"event":"error","message":…}``; the summary then goes to stderr.

An invalid value in ``config.toml`` or a ``VOXLEKTOR_*`` variable does not stop setup (it is
how a broken config gets repaired): it is reported (``ignoring invalid …``), its default is
used, and setup then writes valid values.

At the end a running daemon is stopped, so its next start picks up the new defaults:
interactive mode always does this (the daemon it started for the samples must not stay with
``autostart = false``), flag mode only when a value changed. A non-default profile never
reaches the daemon, so its runs leave a running daemon alone (only an interactive one with
``autostart = false`` stops the daemon it started for the samples). A usage error (bad or
conflicting options) prints no ``--progress json`` line at all; the exit code is authoritative.
Then the summary and the config path are printed, and interactive mode reads a test sentence
when ``autostart = true`` (through a fresh autostart).

The use case, the engine registry (``importlib.metadata``), the menus (questionary) and the
model store (huggingface_hub) are imported inside the command, never on the CLI control path;
bootstrap picks the concrete menus and model store.
"""

import json
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

import voxlektor
from voxlektor.app.ports import ModelStore
from voxlektor.bootstrap.client import (
    ClientContext,
    DaemonSamplePlayer,
    TolerantProfiles,
    build_tolerant_client,
)
from voxlektor.contracts.engine import ModelInfo
from voxlektor.contracts.protocol import PROTOCOL_VERSION
from voxlektor.domain.errors import DomainError, NotRunning, PlatformError
from voxlektor.domain.profiles import DEFAULT_PROFILE
from voxlektor.domain.settings import Settings
from voxlektor.entrypoints.cli.output import EXIT_OK, EXIT_USER

if TYPE_CHECKING:
    from voxlektor.app.setup import SetupResult

# What this CLI and its daemon offer, for tools such as dialogue-box (``--check`` ``features``);
# sorted. S1: multi-client owners, tags with their events, ``vxl warm``, ``vxl watch``.
# S2: ``steps`` (``--steps``, the ``steps`` setting and the speak field).
# S3: ``profiles`` (``--profile``, ``vxl profiles``, ``setup --check --profile``, ``--progress``).
FEATURES: tuple[str, ...] = ("owners", "profiles", "steps", "tags", "warm", "watch")
ISSUES_URL = "https://github.com/MlecznyU/voxlektor/issues"
PULLS_URL = "https://github.com/MlecznyU/voxlektor/pulls"
NO_TERMINAL = "vxl setup needs a terminal; for tools use: vxl setup --yes [options]"
NEEDS_SETUP = "set up voxlektor first (engine and model): vxl setup"
_DEFAULT_PROFILE = DEFAULT_PROFILE
_GLOBAL_FLAGS = ("engine", "idle_timeout", "autostart")
_TEST_SENTENCES = {
    "pl": "Gotowe. Tak będę czytać.",
    "en": "All set. This is how I will read.",
}
_STOP_WAIT_S = 5.0
_STOP_POLL_S = 0.1
_PROGRESS_STEP = 10

# Replaced by tests (fake clock); production uses the real ones.
_sleep = time.sleep
_monotonic = time.monotonic


def setup(  # noqa: PLR0913, PLR0917  # one parameter per frozen CLI option
    yes: Annotated[
        bool, typer.Option("--yes", help="Ask nothing: use the options, keep the rest.")
    ] = False,
    engine: Annotated[str | None, typer.Option("--engine", help="Engine id.")] = None,
    voice: Annotated[str | None, typer.Option("--voice", help="Voice name.")] = None,
    base_lang: Annotated[str | None, typer.Option("--base-lang", help="pl or en.")] = None,
    rate: Annotated[float | None, typer.Option("--rate", help="Rate, 0.5 to 3.0.")] = None,
    volume: Annotated[
        int | None, typer.Option("--volume", help="Volume, 0 to 100; 50 = even level.")
    ] = None,
    idle: Annotated[
        str | None, typer.Option("--idle", metavar="DUR", help="Idle timeout: 0, 90s, 10m, 1h.")
    ] = None,
    autostart: Annotated[
        bool | None,
        typer.Option(
            "--autostart/--no-autostart", help="Start the service when reading.", show_default=False
        ),
    ] = None,
    accept_model_license: Annotated[
        bool,
        typer.Option("--accept-model-license", help="Accept the model license (download)."),
    ] = False,
    check: Annotated[
        bool,
        typer.Option(
            "--check", help="Change nothing: print one JSON line; exit 0 when ready, 1 if not."
        ),
    ] = False,
    profile: Annotated[
        str | None,
        typer.Option(
            "--profile", metavar="NAME", help="Set up this profile only (created when missing)."
        ),
    ] = None,
    lang: Annotated[
        str | None, typer.Option("--lang", help="auto, pl or en (needs --profile).")
    ] = None,
    steps: Annotated[
        int | None, typer.Option("--steps", help="Quality steps, 4 to 32 (needs --profile).")
    ] = None,
    progress: Annotated[
        str,
        typer.Option(
            "--progress", metavar="text|json", help="json: download progress as JSON lines."
        ),
    ] = "text",
) -> int:
    """Set up voxlektor (engine, model, voice, language, rate, volume, service; profiles)."""
    from voxlektor.bootstrap import environment  # noqa: PLC0415  # setup only

    flags: dict[str, object] = {
        "engine": engine,
        "voice": voice,
        "lang": lang,
        "base_lang": base_lang,
        "rate": rate,
        "volume": volume,
        "steps": steps,
        "idle_timeout": idle,
        "autostart": autostart,
    }
    given = any(value is not None for value in flags.values())
    _check_options(check=check, yes=yes, accept=accept_model_license, given=given)
    _check_profile_options(profile, flags, progress, yes=yes)
    ctx, profiles = _client()
    current = profiles.file_default  # the file's values, not the env's
    if check:
        environment.prepare()
        return _check(ctx, profiles, profile or _DEFAULT_PROFILE)
    if not yes and (refusal := _interactive_refusal(ctx, profile, given=given)) is not None:
        print(refusal, file=sys.stderr)
        return EXIT_USER
    environment.prepare()
    as_json = progress == "json"
    # A profile run: --yes with --profile (any), or interactive with a non-default profile.
    by_profile = profile is not None and (yes or profile != _DEFAULT_PROFILE)
    try:
        if yes:
            result = _flag_mode(ctx, current, flags, accept_model_license, profile, as_json)
        elif by_profile and profile is not None:
            result = _interactive_profile(ctx, profile)
        else:
            result = _interactive(ctx, current)
    except (DomainError, OSError) as err:
        if as_json:
            _emit({"event": "error", "message": _message(err)})
        raise
    if _needs_stop(ctx, result, profile, yes=yes, by_profile=by_profile):
        _stop_daemon(ctx)
    _summary(ctx, result, profile if by_profile else None, as_json=as_json)
    if not yes and not by_profile and result.values["autostart"] == "true":
        _test_sentence(ctx, result)
    if as_json:
        _emit({"event": "done"})
    return EXIT_OK


def _needs_stop(
    ctx: ClientContext,
    result: "SetupResult",
    profile: str | None,
    *,
    yes: bool,
    by_profile: bool,
) -> bool:
    """Whether a running daemon must be stopped after this run.

    A non-default profile never reaches the daemon (its values travel with every request), so
    its runs leave the daemon alone; only an interactive one with ``autostart = false`` stops
    the daemon it started for the samples. The full interactive setup always stops it; flag
    mode for ``default`` (or the globals) only when a value changed.
    """
    if by_profile and profile != _DEFAULT_PROFILE:
        return not yes and not ctx.settings.autostart
    return not yes or result.changed


def _interactive_refusal(ctx: ClientContext, profile: str | None, *, given: bool) -> str | None:
    """Why the interactive mode cannot run (``None``: it can)."""
    if given:
        raise typer.BadParameter("setting options need --yes (the mode without questions)")
    if not _has_terminal():
        return NO_TERMINAL
    if profile not in {None, _DEFAULT_PROFILE} and not ctx.store.setup_completed():
        return NEEDS_SETUP
    return None


def _summary(
    ctx: ClientContext, result: "SetupResult", profile: str | None, *, as_json: bool
) -> None:
    out = sys.stderr if as_json else sys.stdout  # with --progress json stdout is JSON only
    if profile is not None:
        print(f"profile: {profile}", file=out)
    for key, value in result.values.items():
        print(f"{key} = {value}", file=out)
    print(f"config: {ctx.config_path}", file=out)


def _check_options(*, check: bool, yes: bool, accept: bool, given: bool) -> None:
    if check and (yes or accept or given):
        raise typer.BadParameter("--check takes no other options (except --profile)")


def _check_profile_options(
    profile: str | None, flags: dict[str, object], progress: str, *, yes: bool
) -> None:
    if progress not in {"text", "json"}:
        raise typer.BadParameter("--progress must be text or json")
    if progress == "json" and not yes:
        raise typer.BadParameter("--progress json needs --yes")
    if profile is None and (flags["lang"] is not None or flags["steps"] is not None):
        raise typer.BadParameter(
            "--lang and --steps are profile settings; add --profile NAME (e.g. --profile default)"
        )
    if profile is not None and any(flags[key] is not None for key in _GLOBAL_FLAGS):
        raise typer.BadParameter(
            "--engine, --idle and --autostart are global settings; use them without --profile"
        )


def _message(err: DomainError | OSError) -> str:
    return err.message if isinstance(err, DomainError) else str(err)


def _emit(event: dict[str, object]) -> None:
    print(json.dumps(event), flush=True)


def _client() -> tuple[ClientContext, TolerantProfiles]:
    try:
        return build_tolerant_client(_warn)
    except PlatformError as err:
        raise PlatformError(
            f"{err.message}: voxlektor runs on macOS 14 or later on Apple Silicon. "
            f"Ask for your platform: {ISSUES_URL} or contribute it: {PULLS_URL}"
        ) from None


def _check(ctx: ClientContext, profiles: TolerantProfiles, profile: str) -> int:
    """Report readiness for tools; read-only (no daemon, no download, nothing written).

    ``settings`` are the effective ones of ``profile`` (env included, invalid values reported
    and replaced by the next layer's), as a reading would use them; an unknown profile shows
    what it would inherit (default's), with ``profile_exists`` false and ``ready`` false.
    """
    from voxlektor.bootstrap.client import model_store  # noqa: PLC0415  # huggingface_hub
    from voxlektor.bootstrap.registry import engine_models  # noqa: PLC0415  # importlib.metadata

    exists, settings, _ = profiles.resolve(profile)
    info = next(
        (m for m in engine_models(include_hidden=True) if m.engine_id == settings.engine), None
    )
    model: dict[str, object] | None = None
    if info is not None:
        model = {
            "present": model_store().is_present(info),
            "repo": info.repo,
            "revision": info.revision,
            "size_bytes": info.size_bytes,
            "license_name": info.license_name,
            "license_url": info.license_url,
            "commercial_use": info.commercial_use,
        }
    completed = ctx.store.setup_completed()
    ready = exists and completed and model is not None and model["present"] is True
    report = {
        "ready": ready,
        "setup_completed": completed,
        "profile": profile,
        "profile_exists": exists,
        "version": voxlektor.__version__,
        "protocol": PROTOCOL_VERSION,
        "config_path": str(ctx.config_path),
        "engine": settings.engine,
        "model": model,
        "settings": {
            "voice": settings.voice,
            "lang": settings.lang,
            "base_lang": settings.base_lang,
            "rate": settings.rate,
            "volume": settings.volume,
            "steps": settings.steps,
            "idle_timeout_s": settings.idle_timeout_s,
            "autostart": settings.autostart,
        },
        "features": list(FEATURES),
    }
    print(json.dumps(report))
    return EXIT_OK if ready else EXIT_USER


def _warn(message: str) -> None:
    print(message, file=sys.stderr)


def _has_terminal() -> bool:
    stdin, stdout = sys.stdin, sys.stdout
    return stdin is not None and stdin.isatty() and stdout is not None and stdout.isatty()


def _interactive(ctx: ClientContext, current: Settings) -> "SetupResult":
    from voxlektor.app.setup import run_interactive  # noqa: PLC0415  # setup only
    from voxlektor.bootstrap.client import model_store, setup_prompter  # noqa: PLC0415
    from voxlektor.bootstrap.registry import engine_models  # noqa: PLC0415  # importlib.metadata

    return run_interactive(
        ctx.store,
        setup_prompter(),
        DaemonSamplePlayer(ctx),
        model_store(),
        engine_models(),
        ctx.catalog().list(),
        current,
    )


def _interactive_profile(ctx: ClientContext, profile: str) -> "SetupResult":
    from voxlektor.app.setup import run_profile_interactive  # noqa: PLC0415  # setup only
    from voxlektor.bootstrap.client import setup_prompter  # noqa: PLC0415

    return run_profile_interactive(
        ctx.store, setup_prompter(), DaemonSamplePlayer(ctx), ctx.catalog().list(), profile
    )


def _flag_mode(  # noqa: PLR0913, PLR0917  # the parsed options, passed on
    ctx: ClientContext,
    current: Settings,
    flags: dict[str, object],
    accept: bool,
    profile: str | None,
    as_json: bool,
) -> "SetupResult":
    """``apply_flags`` without ``--profile``; with it, ``apply_profile_flags`` (any profile)."""
    from voxlektor.app.setup import apply_flags, apply_profile_flags  # noqa: PLC0415  # setup
    from voxlektor.bootstrap.client import model_store  # noqa: PLC0415
    from voxlektor.bootstrap.registry import engine_models  # noqa: PLC0415  # importlib.metadata

    voice = flags["voice"]
    if isinstance(voice, str):
        ctx.catalog().resolve(voice)  # UnknownVoice lists the available voices
    models = _JsonProgress(model_store()) if as_json else _StderrProgress(model_store())
    engines = engine_models(include_hidden=True)
    if profile is None:
        return apply_flags(ctx.store, models, engines, current, flags, accept)
    return apply_profile_flags(ctx.store, models, engines, current, profile, flags, accept)


class _JsonProgress:
    """A ModelStore wrapper that prints download progress as JSON lines on stdout.

    ``{"event": "download", "done": N, "total": M}`` (bytes) whenever the whole percentage or
    the total changes; reports without a total yet are not printed.
    """

    def __init__(self, inner: ModelStore) -> None:
        self._inner = inner

    def is_present(self, info: ModelInfo) -> bool:
        return self._inner.is_present(info)

    def download(self, info: ModelInfo, progress: Callable[[int, int], None]) -> None:
        last = [(-1, -1)]  # the last printed (percent, total)

        def report(done: int, total: int) -> None:
            progress(done, total)
            if total <= 0:
                return
            now = (done * 100 // total, total)
            if now == last[0]:
                return
            last[0] = now
            _emit({"event": "download", "done": done, "total": total})

        self._inner.download(info, report)


class _StderrProgress:
    """A ModelStore wrapper that prints download progress to stderr (flag mode)."""

    def __init__(self, inner: ModelStore) -> None:
        self._inner = inner

    def is_present(self, info: ModelInfo) -> bool:
        return self._inner.is_present(info)

    def download(self, info: ModelInfo, progress: Callable[[int, int], None]) -> None:
        """One line per step; when the total grows (files add their size as they start), the
        percentage is printed again from its new, lower value (same rule as ``app.setup``)."""
        said = [-1, 0]  # the last printed percentage and the largest total seen

        def report(done: int, total: int) -> None:
            progress(done, total)
            if total <= 0:
                return
            percent = min(100, done * 100 // total) // _PROGRESS_STEP * _PROGRESS_STEP
            grew = total > said[1]
            said[1] = max(total, said[1])
            if percent == said[0] or (percent < said[0] and not grew):
                return
            said[0] = percent
            print(f"downloading the model: {percent}%", file=sys.stderr, flush=True)

        self._inner.download(info, report)


def _stop_daemon(ctx: ClientContext) -> None:
    """Stop a running daemon and wait until it is gone; nothing to do when none runs."""
    try:
        ctx.send("shutdown")
    except NotRunning:
        return
    print("stopped the running service so it picks up the new settings", file=sys.stderr)
    _wait_gone(ctx.platform.socket_path(), ctx.platform.pid_path())


def _wait_gone(*paths: Path) -> None:
    deadline = _monotonic() + _STOP_WAIT_S
    while any(path.exists() for path in paths) and _monotonic() < deadline:
        _sleep(_STOP_POLL_S)


def _test_sentence(ctx: ClientContext, result: "SetupResult") -> None:
    values = result.values
    lang = "en" if values["base_lang"] == "en" else "pl"
    DaemonSamplePlayer(ctx).play(
        _TEST_SENTENCES[lang], values["voice"], lang, float(values["rate"]), int(values["volume"])
    )
