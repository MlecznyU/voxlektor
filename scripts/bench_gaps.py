"""Gap benchmark: how well does generation keep ahead of playback?

Runs the real engine (OmniVoice, about 3 GB of RAM) inside this process, one daemon for the whole
run, and reads the same three-paragraph text for every voice x rate x language. Playback is
simulated in real time (nothing is played aloud): a chunk completes after its audio duration, so
look-ahead generation faces the same timing as with a sound device. After each reading the
table row comes from ``Status``: first audio, underruns (gaps between chunks) and the ratio of
generation time to audio time (below 1.0 means faster than real time).

Run from the repository root: ``uv run python scripts/bench_gaps.py``. It takes several minutes.
The repository root is put on ``sys.path`` so that the sample texts are imported from
``tests/fixtures/samples.py``, their single source.
"""

import asyncio
import sys
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.fixtures.samples import BENCH_PARAGRAPHS_EN, BENCH_PARAGRAPHS_PL
from voxlektor.adapters.audio.memory_out import MemoryCuePlayer
from voxlektor.app.commands import Shutdown, Speak, Status
from voxlektor.bootstrap.daemon import Daemon, build_daemon
from voxlektor.domain.settings import Settings
from voxlektor.domain.types import ChunkId, Lang

VOICES = ("v0-default", "f2-narrator", "m1-calm")
RATES = (1.0, 1.25, 1.5)
LANGS: tuple[Lang, ...] = ("pl", "en")
TEXTS: dict[Lang, str] = {
    "pl": "\n\n".join(BENCH_PARAGRAPHS_PL),
    "en": "\n\n".join(BENCH_PARAGRAPHS_EN),
}
_POLL_S = 0.1
_READING_TIMEOUT_S = 600.0
_HEADER = ("voice", "rate", "lang", "first_audio_s", "underruns", "gen/audio ratio")


class RealtimeOutput:
    """An ``AudioOutput`` that completes each chunk after its real duration."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._on_done: Callable[[ChunkId], None] | None = None
        self._on_failed: Callable[[ChunkId, str], None] | None = None
        self._chunk: ChunkId | None = None
        self._remaining = 0.0
        self._armed_at = 0.0
        self._timer: threading.Timer | None = None
        self._paused = False
        self._serial = 0

    def set_callbacks(
        self,
        on_done: Callable[[ChunkId], None],
        on_failed: Callable[[ChunkId, str], None],
    ) -> None:
        self._on_done = on_done
        self._on_failed = on_failed

    def start(self, chunk: ChunkId, samples: object, sample_rate: int) -> None:
        if not isinstance(samples, np.ndarray):
            raise TypeError(f"expected a numpy array, got {type(samples).__name__}")
        seconds = float(samples.size) / sample_rate  # mono
        with self._lock:
            self._cancel()
            self._serial += 1
            self._chunk = chunk
            self._remaining = seconds
            self._paused = False
            self._arm()

    def pause(self) -> None:
        with self._lock:
            if self._chunk is None or self._paused:
                return
            self._remaining = max(0.0, self._remaining - (time.monotonic() - self._armed_at))
            self._cancel()
            self._paused = True

    def resume(self) -> None:
        with self._lock:
            if self._chunk is None or not self._paused:
                return
            self._paused = False
            self._arm()

    def abort(self) -> None:
        with self._lock:
            self._cancel()
            self._serial += 1
            self._chunk = None

    def _arm(self) -> None:
        self._armed_at = time.monotonic()
        timer = threading.Timer(self._remaining, self._finish, args=(self._serial,))
        timer.daemon = True
        self._timer = timer
        timer.start()

    def _cancel(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    def _finish(self, serial: int) -> None:
        with self._lock:
            if serial != self._serial or self._chunk is None:
                return
            chunk, self._chunk, self._timer = self._chunk, None, None
        if self._on_done is not None:
            self._on_done(chunk)


def _number(data: Mapping[str, object], key: str) -> float:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return 0.0
    return float(value)


async def _status(daemon: Daemon) -> Mapping[str, object]:
    return (await daemon.loop.handle(Status())).data


async def _read(daemon: Daemon, voice: str, rate: float, lang: Lang) -> Mapping[str, object]:
    """Read the text for ``lang``, wait until everything is generated and played, return Status."""
    reply = await daemon.loop.handle(Speak(TEXTS[lang], voice, lang, rate, now=True))
    if not reply.ok:
        raise RuntimeError(f"speak failed: {reply.error}")
    planned = _number(reply.data, "chunks")
    deadline = time.monotonic() + _READING_TIMEOUT_S
    while True:
        await asyncio.sleep(_POLL_S)
        data = await _status(daemon)
        if data.get("last_error"):
            raise RuntimeError(f"reading failed: {data['last_error']}")
        idle = not data["playing"] and not data["queued"]
        if idle and _number(data, "chunks") >= planned:
            return data
        if time.monotonic() > deadline:
            raise TimeoutError(f"{voice} {rate} {lang}: reading did not finish")


def _row(voice: str, rate: float, lang: Lang, data: Mapping[str, object]) -> tuple[str, ...]:
    audio = _number(data, "audio_seconds")
    ratio = _number(data, "gen_seconds") / audio if audio else float("nan")
    return (
        voice,
        f"{rate:g}",
        lang,
        f"{_number(data, 'first_audio_s'):.2f}",
        f"{_number(data, 'underruns'):.0f}",
        f"{ratio:.2f}",
    )


def _table(rows: list[tuple[str, ...]]) -> str:
    lines = [_HEADER, *rows]
    widths = [max(len(line[i]) for line in lines) for i in range(len(_HEADER))]
    return "\n".join(
        " | ".join(cell.ljust(w) for cell, w in zip(line, widths, strict=True)) for line in lines
    )


async def _bench(daemon: Daemon) -> list[tuple[str, ...]]:
    started = time.monotonic()
    await _read(daemon, VOICES[0], 1.0, "pl")  # cold load, not part of the table
    print(f"warm-up (model load + first reading): {time.monotonic() - started:.1f} s", flush=True)
    rows: list[tuple[str, ...]] = []
    for voice in VOICES:
        await _read(daemon, voice, 1.0, "pl")  # builds or loads this voice's prompt, not measured
        for rate in RATES:
            for lang in LANGS:
                row = _row(voice, rate, lang, await _read(daemon, voice, rate, lang))
                rows.append(row)
                print(" | ".join(row), flush=True)
    return rows


async def main() -> None:
    # Never unload while benchmarking; cues are recorded, not played.
    settings = replace(Settings(), idle_timeout_s=0)
    daemon = build_daemon(
        settings, transport=False, output=RealtimeOutput(), cues=MemoryCuePlayer()
    )
    running = asyncio.create_task(daemon.run())
    started = time.monotonic()
    try:
        rows = await _bench(daemon)
    finally:
        await daemon.loop.handle(Shutdown())
        await running
    print()
    print(_table(rows))
    print(f"\ntotal run time: {time.monotonic() - started:.0f} s")


if __name__ == "__main__":
    asyncio.run(main())
