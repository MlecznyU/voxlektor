"""``vxl setup``: the guided, re-runnable first run (menus) or its silent flag mode (``--yes``).

Order: platform check (unsupported: exit 3 with where to ask for support), then
``app.setup``. Interactive mode needs a terminal (without one, and without ``--yes``, it refuses
with exit 1 and a hint); it plays samples through the daemon, which it starts even with
``autostart = false``. Flag mode asks nothing, plays nothing and starts no daemon; a model that
has to be downloaded needs ``--accept-model-license``; download progress goes to stderr.

``--check`` (no other options) changes nothing and starts nothing: it prints one JSON line
(``ready``, ``setup_completed``, ``version``, ``config_path``, ``engine``, ``model`` with
``present`` and its license, the effective ``settings``) and exits 0 when setup has completed
and the model is present, 1 otherwise. It needs no terminal; tools such as dialogue-box use it.

An invalid value in ``config.toml`` or a ``VOXLEKTOR_*`` variable does not stop setup (it is
how a broken config gets repaired): it is reported (``ignoring invalid …``), its default is
used, and setup then writes valid values.

At the end a running daemon is stopped, so its next start picks up the new defaults:
interactive mode always does this (the daemon it started for the samples must not stay with
``autostart = false``), flag mode only when a value changed. Then the summary and the config
path are printed, and interactive mode reads a test sentence when ``autostart = true``
(through a fresh autostart).

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
from voxlektor.bootstrap.client import ClientContext, DaemonSamplePlayer, build_setup_client
from voxlektor.contracts.engine import ModelInfo
from voxlektor.domain.errors import NotRunning, PlatformError
from voxlektor.domain.settings import Settings
from voxlektor.entrypoints.cli.output import EXIT_OK, EXIT_USER

if TYPE_CHECKING:
    from voxlektor.app.setup import SetupResult

ISSUES_URL = "https://github.com/MlecznyU/voxlektor/issues"
PULLS_URL = "https://github.com/MlecznyU/voxlektor/pulls"
NO_TERMINAL = "vxl setup needs a terminal; for tools use: vxl setup --yes [options]"
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
) -> int:
    """Set up voxlektor (engine, model, voice, language, rate, volume, service)."""
    from voxlektor.bootstrap import environment  # noqa: PLC0415  # setup only

    flags: dict[str, object] = {
        "engine": engine,
        "voice": voice,
        "base_lang": base_lang,
        "rate": rate,
        "volume": volume,
        "idle_timeout": idle,
        "autostart": autostart,
    }
    if check and (yes or accept_model_license or any(v is not None for v in flags.values())):
        raise typer.BadParameter("--check takes no other options")
    ctx, current = _client()  # current: the file's values, not the env's
    if check:
        environment.prepare()
        return _check(ctx)
    if not yes:
        if any(value is not None for value in flags.values()):
            raise typer.BadParameter("setting options need --yes (the mode without questions)")
        if not _has_terminal():
            print(NO_TERMINAL, file=sys.stderr)
            return EXIT_USER
    environment.prepare()
    if yes:
        result = _flag_mode(ctx, current, flags, accept_model_license)
    else:
        result = _interactive(ctx, current)
    if result.changed or not yes:
        _stop_daemon(ctx)
    for key, value in result.values.items():
        print(f"{key} = {value}")
    print(f"config: {ctx.config_path}")
    if not yes and result.values["autostart"] == "true":
        _test_sentence(ctx, result)
    return EXIT_OK


def _client() -> tuple[ClientContext, Settings]:
    try:
        return build_setup_client(_warn)
    except PlatformError as err:
        raise PlatformError(
            f"{err.message}: voxlektor runs on macOS 14 or later on Apple Silicon. "
            f"Ask for your platform: {ISSUES_URL} or contribute it: {PULLS_URL}"
        ) from None


def _check(ctx: ClientContext) -> int:
    """Report readiness for tools; read-only (no daemon, no download, nothing written).

    ``settings`` are the effective ones (env included, invalid values reported and replaced by
    their defaults), as a reading would use them.
    """
    from voxlektor.bootstrap.client import model_store  # noqa: PLC0415  # huggingface_hub
    from voxlektor.bootstrap.registry import engine_models  # noqa: PLC0415  # importlib.metadata

    settings = ctx.settings
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
    ready = completed and model is not None and model["present"] is True
    report = {
        "ready": ready,
        "setup_completed": completed,
        "version": voxlektor.__version__,
        "config_path": str(ctx.config_path),
        "engine": settings.engine,
        "model": model,
        "settings": {
            "voice": settings.voice,
            "lang": settings.lang,
            "base_lang": settings.base_lang,
            "rate": settings.rate,
            "volume": settings.volume,
            "idle_timeout_s": settings.idle_timeout_s,
            "autostart": settings.autostart,
        },
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


def _flag_mode(
    ctx: ClientContext, current: Settings, flags: dict[str, object], accept: bool
) -> "SetupResult":
    from voxlektor.app.setup import apply_flags  # noqa: PLC0415  # setup only
    from voxlektor.bootstrap.client import model_store  # noqa: PLC0415
    from voxlektor.bootstrap.registry import engine_models  # noqa: PLC0415  # importlib.metadata

    voice = flags["voice"]
    if isinstance(voice, str):
        ctx.catalog().resolve(voice)  # UnknownVoice lists the available voices
    models = _StderrProgress(model_store())
    return apply_flags(
        ctx.store, models, engine_models(include_hidden=True), current, flags, accept
    )


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
