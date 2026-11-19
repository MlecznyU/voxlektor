"""Latency benchmark: first audio, stalls and silence heard for a five-paragraph reply.

Runs the real daemon pipeline in this process (OmniVoice, about 3 GB of RAM): ``build_daemon``
with the real Session, engine worker (native rate, trim, fade, level), pause policy and caps.
The reply (Polish with English terms, ``tests/fixtures/samples.py``) is sent as one ``speak``
per paragraph into one group, as dialogue-box does, so the first paragraph starts from idle
(first and second chunk capped) and the others are appended with normal chunk sizes.

Playback is simulated in real time (nothing is played aloud) with the output adapter's rules:
a chunk's ``lead_s`` is a minimum gap (silence already elapsed since the previous chunk's last
sample counts toward it), and a chunk completes after its audio duration. Per reading it
reports first audio, total time, per-chunk generation vs audio, stalls (silence beyond the
pause the policy asked for), and silence heard = pauses inserted + stalls + edge silence left
inside the chunks (20 ms frames below -50 dBFS at the start and end of each chunk).

The first reading runs on a cold engine (model load and voice prompt included); then every
``--steps`` x ``--rate`` combination is read warm. Run from the repository root:
``uv run python scripts/bench_latency.py [--steps 16 8] [--rate 1.0 1.1] [--voice NAME]
[--json]``. One reading takes about 40 to 80 s.
"""

import argparse
import asyncio
import json
import sys
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import voxlektor.bootstrap.daemon as daemon_module
from tests.fixtures.samples import LATENCY_REPLY_PL
from voxlektor.adapters.audio.memory_out import MemoryCuePlayer
from voxlektor.app.commands import Shutdown, Speak, Status
from voxlektor.bootstrap.daemon import Daemon, build_daemon
from voxlektor.bootstrap.registry import load_engine
from voxlektor.contracts.engine import AudioChunk, Engine, VoicePrompt, VoiceRef
from voxlektor.domain.settings import Settings
from voxlektor.domain.types import ChunkId

OWNER = "bench"
_POLL_S = 0.05
_READING_TIMEOUT_S = 600.0
_STALL_MIN_S = 0.05  # extra silence below this is loop round-trip noise, not a stall
_EDGE_FRAME_S = 0.02
_EDGE_GATE_DB = -50.0
# Measured 2026-11-15 before S2 (docs/superpowers/specs/2026-11-15-latency-benchmark.md):
# f3-bright, 16 steps, rate 1.0, warm; silence = engine edge silence 9.7 + gaps 1.96.
BASELINE = {"first_audio_s": 3.0, "total_s": 84.4, "silence_heard_s": 11.7, "stall_s": 0.8}


@dataclass
class Synth:
    """One ``synthesize`` call as the engine worker made it."""

    text: str
    lang: str
    rate: float
    steps: int | None
    started: float
    ended: float
    raw_s: float  # audio length before trim, stretch and fade


@dataclass
class Played:
    """One chunk as the simulated output played it (times from ``time.monotonic``)."""

    chunk: int
    called: float  # ``start`` was called (chunk generated and its turn)
    lead_s: float  # the pause the policy asked for
    begin: float  # first sample heard
    end: float  # last sample heard
    audio_s: float
    edge_s: float  # silence left at the chunk's edges


class TimedEngine:
    """Wraps the registry's engine and records each ``synthesize`` (load and prompt too)."""

    def __init__(self, inner: Engine, log: "EngineLog") -> None:
        self._inner = inner
        self._log = log
        self.spec = inner.spec

    def load(self) -> None:
        started = time.monotonic()
        self._inner.load()
        self._log.load_s += time.monotonic() - started

    def unload(self) -> None:
        self._inner.unload()

    def prepare_voice(self, ref: VoiceRef) -> VoicePrompt:
        started = time.monotonic()
        prompt = self._inner.prepare_voice(ref)
        self._log.prompt_s += time.monotonic() - started
        return prompt

    def serialize_prompt(self, p: VoicePrompt) -> bytes:
        return self._inner.serialize_prompt(p)

    def deserialize_prompt(self, voice: str, data: bytes) -> VoicePrompt:
        started = time.monotonic()
        prompt = self._inner.deserialize_prompt(voice, data)
        self._log.prompt_s += time.monotonic() - started
        return prompt

    def synthesize(
        self,
        text: str,
        lang: str,
        prompt: VoicePrompt,
        *,
        rate: float = 1.0,
        steps: int | None = None,
    ) -> Iterator[AudioChunk]:
        started = time.monotonic()
        pieces = list(self._inner.synthesize(text, lang, prompt, rate=rate, steps=steps))
        ended = time.monotonic()
        samples = sum(piece.samples.size for piece in pieces)
        raw_s = samples / self.spec.sample_rate
        self._log.synths.append(Synth(text, lang, rate, steps, started, ended, raw_s))
        return iter(pieces)


class EngineLog:
    """What the timed engine saw; ``provider`` replaces the registry lookup of the daemon."""

    def __init__(self) -> None:
        self.import_s = 0.0
        self.load_s = 0.0
        self.prompt_s = 0.0
        self.synths: list[Synth] = []

    def provider(self, engine_id: str) -> Engine:
        started = time.monotonic()
        engine = load_engine(engine_id)  # through the registry, as the daemon does
        self.import_s += time.monotonic() - started
        return TimedEngine(engine, self)


class SimulatedOutput:
    """An ``AudioOutput`` in real time with the device adapter's minimum-gap ``lead_s``."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._on_done: Callable[[ChunkId], None] | None = None
        self._timer: threading.Timer | None = None
        self._serial = 0
        self._last_end: float | None = None  # the previous chunk's last sample
        self.played: list[Played] = []

    def reset(self) -> None:
        """A new reading from idle: no previous chunk, nothing recorded."""
        with self._lock:
            self._last_end = None
            self.played = []

    def set_callbacks(
        self,
        on_done: Callable[[ChunkId], None],
        on_failed: Callable[[ChunkId, str], None],
    ) -> None:
        self._on_done = on_done

    def start(
        self, chunk: ChunkId, samples: object, sample_rate: int, *, lead_s: float = 0.0
    ) -> None:
        if not isinstance(samples, np.ndarray):
            raise TypeError(f"expected a numpy array, got {type(samples).__name__}")
        audio = cast("npt.NDArray[np.float32]", samples)
        now = time.monotonic()
        with self._lock:
            elapsed = None if self._last_end is None else now - self._last_end
            wait = lead_s if elapsed is None else max(0.0, lead_s - elapsed)
            seconds = audio.size / sample_rate
            begin = now + wait
            played = Played(
                int(chunk),
                now,
                lead_s,
                begin,
                begin + seconds,
                seconds,
                _edge_silence(audio, sample_rate),
            )
            self.played.append(played)
            self._cancel()
            self._serial += 1
            timer = threading.Timer(played.end - now, self._finish, args=(self._serial, played))
            timer.daemon = True
            self._timer = timer
            timer.start()

    def pause(self) -> None:  # the benchmark never pauses
        pass

    def resume(self) -> None:
        pass

    def abort(self) -> None:
        with self._lock:
            self._cancel()
            self._serial += 1

    def _cancel(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    def _finish(self, serial: int, played: Played) -> None:
        with self._lock:
            if serial != self._serial:
                return
            self._timer = None
            self._last_end = played.end
        if self._on_done is not None:
            self._on_done(ChunkId(played.chunk))


def _edge_silence(samples: npt.NDArray[np.float32], sample_rate: int) -> float:
    """Seconds below the gate at the start and the end (20 ms RMS frames, as the baseline)."""
    frame = round(sample_rate * _EDGE_FRAME_S)
    count = samples.size // frame
    if count == 0:
        return samples.size / sample_rate
    frames = samples[: count * frame].reshape(count, frame).astype(np.float64)
    db = 10.0 * np.log10(np.mean(np.square(frames), axis=1) + 1e-20)
    loud = np.flatnonzero(db > _EDGE_GATE_DB)
    if loud.size == 0:
        return samples.size / sample_rate
    tail = (samples.size - count * frame) / sample_rate
    return float(loud[0] + (count - 1 - loud[-1])) * _EDGE_FRAME_S + tail


@dataclass
class ChunkRow:
    i: int
    para: int
    chars: int
    lang: str
    gen_s: float
    raw_s: float
    audio_s: float
    lead_s: float
    gap_s: float  # silence heard before the chunk (0 for the first)
    stall_s: float  # silence beyond the lead (generation was late)
    ready_s: float  # generated, from the request
    start_s: float  # first sample heard, from the request
    text: str


@dataclass
class Reading:
    name: str
    voice: str
    steps: int
    rate: float
    chunks: int
    first_audio_s: float  # request to the first sample heard
    daemon_first_audio_s: float  # the Session's own metric (to the first StartPlayback)
    total_s: float  # request to the last sample heard
    gen_s: float
    audio_s: float
    gen_per_audio: float
    pauses_s: float  # leads the pause policy asked for
    stall_s: float  # silence beyond the leads (stalls >= 50 ms)
    stalls: list[tuple[int, float]]  # (chunk, seconds)
    gaps_s: float  # silence heard between chunks (pauses + stalls + round trips)
    edge_silence_s: float
    silence_heard_s: float  # gaps + edges
    rows: list[ChunkRow]
    import_s: float | None = None  # cold reading only: engine import, load, voice prompt
    load_s: float | None = None
    prompt_s: float | None = None


def _number(data: Mapping[str, object], key: str) -> float:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return 0.0
    return float(value)


async def _read(  # noqa: PLR0913  # one reading's knobs
    daemon: Daemon,
    output: SimulatedOutput,
    log: EngineLog,
    name: str,
    *,
    voice: str,
    steps: int,
    rate: float,
) -> Reading:
    """Read the reply once from idle and return the reading's summary and chunk rows."""
    output.reset()
    first_synth = len(log.synths)
    t0 = time.monotonic()
    paras: list[int] = []
    for p, text in enumerate(LATENCY_REPLY_PL):
        speak = Speak(
            text,
            voice,
            "auto",
            rate,
            False,
            base_lang="pl",
            owner=OWNER,
            group=name,
            steps=steps,
        )
        reply = await daemon.loop.handle(speak)
        if not reply.ok:
            raise RuntimeError(f"speak failed: {reply.error}")
        paras += [p] * int(_number(reply.data, "chunks"))
    deadline = t0 + _READING_TIMEOUT_S
    while True:
        await asyncio.sleep(_POLL_S)
        data = (await daemon.loop.handle(Status())).data
        if data.get("last_error"):
            raise RuntimeError(f"reading failed: {data['last_error']}")
        if len(output.played) == len(paras) and not data["playing"] and not data["queued"]:
            break
        if time.monotonic() > deadline:
            raise TimeoutError(f"{name}: reading did not finish")
    synths = log.synths[first_synth:]
    if len(synths) != len(paras):
        raise RuntimeError(f"{name}: {len(synths)} generations for {len(paras)} chunks")
    rows: list[ChunkRow] = []
    previous_end: float | None = None
    for k, (played, synth) in enumerate(zip(output.played, synths, strict=True)):
        gap = 0.0 if previous_end is None else played.begin - previous_end
        stall = max(0.0, gap - played.lead_s) if previous_end is not None else 0.0
        rows.append(
            ChunkRow(
                k,
                paras[k] + 1,
                len(synth.text),
                synth.lang,
                synth.ended - synth.started,
                synth.raw_s,
                played.audio_s,
                played.lead_s,
                gap,
                stall,
                synth.ended - t0,
                played.begin - t0,
                synth.text,
            )
        )
        previous_end = played.end
    stalls = [(r.i, round(r.stall_s, 2)) for r in rows if r.stall_s >= _STALL_MIN_S]
    pauses = sum(r.lead_s for r in rows[1:])
    gaps = sum(r.gap_s for r in rows)
    edges = sum(p.edge_s for p in output.played)
    gen = sum(r.gen_s for r in rows)
    audio = sum(r.audio_s for r in rows)
    return Reading(
        name=name,
        voice=voice,
        steps=steps,
        rate=rate,
        chunks=len(rows),
        first_audio_s=output.played[0].begin - t0,
        daemon_first_audio_s=_number(data, "first_audio_s"),
        total_s=output.played[-1].end - t0,
        gen_s=gen,
        audio_s=audio,
        gen_per_audio=gen / audio if audio else float("nan"),
        pauses_s=pauses,
        stall_s=sum(r.stall_s for r in rows if r.stall_s >= _STALL_MIN_S),
        stalls=stalls,
        gaps_s=gaps,
        edge_silence_s=edges,
        silence_heard_s=gaps + edges,
        rows=rows,
    )


def _summary_line(r: Reading) -> str:
    shown = ", ".join(f"c{i} {s:.2f}" for i, s in r.stalls)
    return (
        f"{r.name:<10} first {r.first_audio_s:5.2f}  total {r.total_s:6.2f}  "
        f"gen {r.gen_s:5.1f} / audio {r.audio_s:5.1f} ({r.gen_per_audio:.2f})  "
        f"silence {r.silence_heard_s:5.2f} (pauses {r.pauses_s:.2f}, stalls {r.stall_s:.2f}, "
        f"edges {r.edge_silence_s:.2f}, round trips "
        f"{r.gaps_s - r.pauses_s - r.stall_s:.2f})  stalls: {shown or 'none'}"
    )


def _chunk_table(r: Reading) -> str:
    head = " i  p  chars lang  gen_s  raw_s audio_s  lead_s  gap_s stall_s ready_s start_s"
    lines = [head]
    for row in r.rows:
        lines.append(
            f"{row.i:2d} {row.para:2d} {row.chars:6d} {row.lang:>4}"
            f" {row.gen_s:6.2f} {row.raw_s:6.2f} {row.audio_s:7.2f}"
            f" {row.lead_s:7.2f} {row.gap_s:6.2f} {row.stall_s:7.2f}"
            f" {row.ready_s:7.2f} {row.start_s:7.2f}"
        )
    return "\n".join(lines)


def _progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


async def _bench(args: argparse.Namespace) -> list[Reading]:
    log = EngineLog()
    daemon_module.load_engine = log.provider  # the daemon's registry lookup, timed
    output = SimulatedOutput()
    settings = replace(Settings(), idle_timeout_s=0)  # never unload while benchmarking
    daemon = build_daemon(settings, transport=False, output=output, cues=MemoryCuePlayer())
    running = asyncio.create_task(daemon.run())
    voice: str = args.voice
    steps_list: list[int] = args.steps
    rates: list[float] = args.rate
    readings: list[Reading] = []
    try:
        first_steps, first_rate = steps_list[0], rates[0]
        cold = await _read(
            daemon, output, log, "cold", voice=voice, steps=first_steps, rate=first_rate
        )
        cold.import_s, cold.load_s, cold.prompt_s = log.import_s, log.load_s, log.prompt_s
        readings.append(cold)
        _progress(_summary_line(cold))
        for steps in steps_list:
            for rate in rates:
                name = f"s{steps}_r{rate:g}"
                reading = await _read(
                    daemon, output, log, name, voice=voice, steps=steps, rate=rate
                )
                readings.append(reading)
                _progress(_summary_line(reading))
    finally:
        await daemon.loop.handle(Shutdown())
        await running
    return readings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0] if __doc__ else None)
    parser.add_argument("--steps", type=int, nargs="+", default=[16], help="4..32; several ok")
    parser.add_argument("--rate", type=float, nargs="+", default=[1.0], help="several ok")
    parser.add_argument("--voice", default="f3-bright", help="the baseline's voice by default")
    parser.add_argument("--json", action="store_true", help="print one JSON object")
    args = parser.parse_args()
    readings = asyncio.run(_bench(args))
    if args.json:
        result = {"baseline": BASELINE, "readings": [asdict(r) for r in readings]}
        print(json.dumps(result, ensure_ascii=False))
        return
    b = BASELINE
    print(
        f"\nbaseline 2026-11-15 (before S2, 16 steps, rate 1.0, warm): first {b['first_audio_s']}"
        f"  total {b['total_s']}  silence {b['silence_heard_s']}  stall {b['stall_s']}"
    )
    for reading in readings:
        print(f"\n{_summary_line(reading)}")
        if reading.load_s is not None:
            print(
                f"  cold: engine import {reading.import_s or 0.0:.2f} s, load "
                f"{reading.load_s:.2f} s, voice prompt {reading.prompt_s or 0.0:.2f} s"
            )
        print(_chunk_table(reading))


if __name__ == "__main__":
    main()
