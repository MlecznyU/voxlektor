"""What the CLI prints and the exit codes it returns.

Exit codes: 0 ok, 1 user error, 2 service unavailable, 3 unsupported platform, 130 interrupted
(Ctrl+C; produced by typer, see ``app``). Error replies map
by their wire code: codes about the request or its arguments (``invalid_argument``,
``unknown_voice``, ``unknown_engine``, ``invalid_duration``, ``invalid_setting``, ``bad_frame``,
``unsupported``) are user errors; codes that mean the daemon cannot serve this client
(``bad_version``, ``unknown_command``, ``internal``, ``engine_failed``, ``audio_failed``) are 2.
"""

import sys
from collections.abc import Mapping

from voxlektor.domain.errors import DomainError, PlatformError, ServiceUnavailable, UnknownVoice

EXIT_OK = 0
EXIT_USER = 1
EXIT_UNAVAILABLE = 2
EXIT_UNSUPPORTED = 3

NOT_RUNNING = "not running"
OFF = "OFF"
NOTHING_TO_READ = "nothing to read"

_SERVICE_CODES = frozenset(
    {"bad_version", "unknown_command", "internal", "engine_failed", "audio_failed"}
)


def exit_code(err: DomainError) -> int:
    if isinstance(err, PlatformError):
        return EXIT_UNSUPPORTED
    if isinstance(err, ServiceUnavailable) or err.code in _SERVICE_CODES:
        return EXIT_UNAVAILABLE
    return EXIT_USER


def message(err: DomainError) -> str:
    if isinstance(err, UnknownVoice):
        return f"{err.message}; available: {', '.join(err.available)}"
    return err.message


def fail(err: DomainError) -> int:
    """Print the error on stderr and return its exit code."""
    print(message(err), file=sys.stderr)
    return exit_code(err)


def duration(seconds: float) -> str:
    """``90`` -> ``1m30s``; whole seconds, largest units first."""
    hours, rest = divmod(round(seconds), 3600)
    minutes, secs = divmod(rest, 60)
    units = ((hours, "h"), (minutes, "m"), (secs, "s"))
    parts = [f"{value}{unit}" for value, unit in units if value]
    return "".join(parts) or "0s"


def status_lines(data: Mapping[str, object]) -> list[str]:
    """One-liners for ``vxl status``; values of an unexpected type are skipped."""
    named = ((key, _text(data, key)) for key in ("state", "engine", "voice"))
    lines = [f"{key}: {value}" for key, value in named if value]
    idle = _idle_line(data)
    if idle is not None:
        lines.append(idle)
    lines.append(_reading_line(data))
    metrics = _metrics_line(data)
    if metrics is not None:
        lines.append(metrics)
    error = _text(data, "last_error")
    if error:
        lines.append(f"error: {error}")
    return lines


def _idle_line(data: Mapping[str, object]) -> str | None:
    remaining = _number(data, "idle_remaining_s")
    if remaining is not None:
        return f"idle: unloads in {duration(remaining)}"
    timeout = _number(data, "idle_timeout_s")
    if timeout is None:
        return None
    if timeout == 0:
        return "idle: never unloads"
    return f"idle: timeout {duration(timeout)}"


def _reading_line(data: Mapping[str, object]) -> str:
    queued = _number(data, "queued") or 0
    if data.get("paused") is True:
        doing = "paused"
    elif data.get("playing") is True:
        doing = "playing"
    elif queued:
        doing = "preparing"
    else:
        return "reading: none"
    return f"reading: {doing}, {queued:.0f} queued"


def _metrics_line(data: Mapping[str, object]) -> str | None:
    chunks = _number(data, "chunks")
    if not chunks:
        return None
    parts = [f"{chunks:.0f} chunks"]
    first = _number(data, "first_audio_s")
    if first is not None:
        parts.append(f"first audio {first:.2f} s")
    parts.append(f"gen {_number(data, 'gen_seconds') or 0.0:.2f} s")
    parts.append(f"audio {_number(data, 'audio_seconds') or 0.0:.2f} s")
    parts.append(f"{_number(data, 'underruns') or 0:.0f} underruns")
    return "last: " + ", ".join(parts)


def _text(data: Mapping[str, object], key: str) -> str | None:
    value = data.get(key)
    return value if isinstance(value, str) else None


def _number(data: Mapping[str, object], key: str) -> float | None:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return value
