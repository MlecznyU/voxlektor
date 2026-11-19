"""Daemon smoke test with the real OmniVoice engine (``-m model``; Apple Silicon, ~3 GB RAM).

The whole graph from ``build_daemon`` (registry, FsVoiceRepository with the bundled voices, the
engine thread, time-stretch) reads one short sentence into a MemoryOutput.
"""

import asyncio
import time
from pathlib import Path

import numpy as np
import pytest

from voxlektor.adapters.audio.memory_out import MemoryCuePlayer, MemoryOutput
from voxlektor.adapters.platform.macos import MacOSPlatform
from voxlektor.app.commands import Shutdown, Speak, Status
from voxlektor.bootstrap.daemon import build_daemon
from voxlektor.domain.settings import Settings
from voxlektor.domain.types import ChunkId

pytestmark = [pytest.mark.model, pytest.mark.e2e]

TIMEOUT = 600.0


class _RecordingOutput(MemoryOutput):
    def __init__(self) -> None:
        super().__init__(auto_complete=True)
        self.samples: list[int] = []
        self.started_event = asyncio.Event()

    def start(
        self, chunk: ChunkId, samples: object, sample_rate: int, *, lead_s: float = 0.0
    ) -> None:
        assert sample_rate == 24000
        self.samples.append(np.asarray(samples, dtype=np.float32).size)
        self.started_event.set()
        super().start(chunk, samples, sample_rate, lead_s=lead_s)


def test_omnivoice_reads_a_sentence_through_the_daemon(short_tmp: Path) -> None:
    async def main() -> None:
        output = _RecordingOutput()
        cues = MemoryCuePlayer()
        built = build_daemon(
            Settings(), transport=False, platform=MacOSPlatform(short_tmp), output=output, cues=cues
        )
        runner = asyncio.create_task(built.run())
        started = time.monotonic()
        reply = await built.loop.handle(
            Speak("Hello, this is a short smoke test.", None, "en", None, False)
        )
        assert reply.ok
        await asyncio.wait_for(output.started_event.wait(), TIMEOUT)
        first_audio = time.monotonic() - started
        status = (await built.loop.handle(Status())).data
        await built.loop.handle(Shutdown())
        await asyncio.wait_for(runner, TIMEOUT)
        print(f"\nfirst audio after {first_audio:.1f} s (incl. model load); status: {status}")
        assert status["state"] == "warm"
        assert status["last_error"] is None
        assert cues.played == []
        assert output.samples[0] > 24000  # more than a second of speech

    asyncio.run(main())
