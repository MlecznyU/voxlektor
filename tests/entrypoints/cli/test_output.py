"""Exit codes, messages and status formatting."""

import pytest

from voxlektor.domain.errors import (
    DomainError,
    InvalidDuration,
    NotRunning,
    PlatformError,
    ServiceUnavailable,
    UnknownVoice,
)
from voxlektor.entrypoints.cli.output import duration, exit_code, message, status_lines

BASE: dict[str, object] = {
    "state": "COLD",
    "engine": "omnivoice",
    "voice": "v0-default",
    "idle_timeout_s": 600,
    "idle_remaining_s": None,
    "queued": 0,
    "playing": False,
    "paused": False,
    "underruns": 0,
    "first_audio_s": None,
    "chunks": 0,
    "gen_seconds": 0.0,
    "audio_seconds": 0.0,
    "last_error": None,
}


@pytest.mark.parametrize(
    ("err", "code"),
    [
        (ServiceUnavailable(), 2),
        (NotRunning(), 2),
        (PlatformError(), 3),
        (InvalidDuration("bad"), 1),
        (UnknownVoice("x", ("a",)), 1),
        (DomainError("engine_failed", "boom"), 2),
        (DomainError("something_new", "?"), 1),
    ],
    ids=repr,
)
def test_exit_codes(err: DomainError, code: int) -> None:
    assert exit_code(err) == code


def test_unknown_voice_message_lists_the_voices() -> None:
    assert message(UnknownVoice("x", ("a", "b"))) == "unknown voice 'x'; available: a, b"
    assert message(InvalidDuration("bad")) == "bad"


@pytest.mark.parametrize(
    ("seconds", "text"),
    [(0, "0s"), (5, "5s"), (60, "1m"), (90, "1m30s"), (3600, "1h"), (3725.4, "1h2m5s")],
)
def test_duration(seconds: float, text: str) -> None:
    assert duration(seconds) == text


def test_cold_daemon() -> None:
    assert status_lines(BASE) == [
        "state: COLD",
        "engine: omnivoice",
        "voice: v0-default",
        "idle: timeout 10m",
        "reading: none",
    ]


def test_warm_playing_daemon_with_metrics_and_error() -> None:
    data = {
        **BASE,
        "state": "WARM",
        "queued": 2,
        "playing": True,
        "first_audio_s": 0.42,
        "chunks": 3,
        "gen_seconds": 1.5,
        "audio_seconds": 9.0,
        "underruns": 1,
        "last_error": "audio device lost",
    }
    assert status_lines(data) == [
        "state: WARM",
        "engine: omnivoice",
        "voice: v0-default",
        "idle: timeout 10m",
        "reading: playing, 2 queued",
        "last: 3 chunks, first audio 0.42 s, gen 1.50 s, audio 9.00 s, 1 underruns",
        "error: audio device lost",
    ]


def test_paused_reading_idle_countdown_and_never_unload() -> None:
    paused = {**BASE, "paused": True, "queued": 1, "idle_remaining_s": 12.0}
    lines = status_lines(paused)
    assert "reading: paused, 1 queued" in lines
    assert "idle: unloads in 12s" in lines
    never = status_lines({**BASE, "idle_timeout_s": 0})
    assert "idle: never unloads" in never


def test_queued_but_not_yet_playing_is_preparing() -> None:
    assert "reading: preparing, 3 queued" in status_lines({**BASE, "queued": 3})


def test_metrics_without_first_audio() -> None:
    data = {**BASE, "chunks": 1, "gen_seconds": 0.5, "audio_seconds": 2.0}
    assert "last: 1 chunks, gen 0.50 s, audio 2.00 s, 0 underruns" in status_lines(data)


def test_unexpected_values_are_skipped() -> None:
    data: dict[str, object] = {"state": "WARM", "chunks": "many", "idle_timeout_s": "?"}
    assert status_lines(data) == ["state: WARM", "reading: none"]


def test_no_ram_line() -> None:
    assert not any(line.startswith("RAM") for line in status_lines(BASE))
