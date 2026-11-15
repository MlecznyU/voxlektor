"""The Simulator itself: how it carries out the effects that only the runtime knows about."""

from pathlib import Path

from voxlektor.app.commands import Speak
from voxlektor.app.effects import CancelGeneration, Emit, Generate
from voxlektor.app.session import Session
from voxlektor.app.voices import VoiceCatalog
from voxlektor.contracts.engine import VoiceRef
from voxlektor.domain.reading import Chunk
from voxlektor.domain.settings import Settings
from voxlektor.domain.types import ChunkId, EngineId, Epoch
from voxlektor.testing.fakes import FakeClock, FakeEngine, InMemoryVoiceRepository, MemoryOutput
from voxlektor.testing.simulator import Simulator

VOICE = "v0-default"
SENTENCE = "The quick brown fox jumps over the lazy dog near the river bank while the sun rises."


def _sim() -> Simulator:
    clock = FakeClock()
    repo = InMemoryVoiceRepository([VoiceRef(VOICE, Path("/voices/v0/ref.wav"), "reference")])
    session = Session(Settings(engine=EngineId("fake")), VoiceCatalog(repo), clock)
    return Simulator(session, FakeEngine(sample_rate=100), MemoryOutput(), clock)


def test_emit_is_recorded_in_order() -> None:
    sim = _sim()
    first = Emit("state", {"owner": None, "group": None})
    second = Emit("drained", {"owner": "a", "group": "g"})
    sim.apply([first, second])
    assert sim.events == [first, second]
    assert sim.effects == [first, second]


def test_cancel_generation_drops_the_queued_jobs_of_those_epochs_only() -> None:
    sim = _sim()
    sim.auto_run = False
    texts = [f"Chunk {n} {SENTENCE}" for n in range(3)]
    assert sim.handle(Speak("\n\n".join(texts), None, "en", None, False)).ok
    sim.step()  # LoadEngine
    sim.step()  # PrepareVoice; then Generate of chunks 0 and 1 (epoch 0) wait (ahead = 2)
    assert len(sim.generate_states) == 2
    other = Generate(Chunk(ChunkId(100), "another epoch", "en", 1.0, Epoch(5)), VOICE)
    sim.apply([other, CancelGeneration((Epoch(0),))])
    sim.step()
    sim.step()
    # Chunk 0 counts as running and cannot be cancelled; chunk 1 is skipped; epoch 5 is spared.
    assert [text for text, _ in sim.engine.synth_calls] == [texts[0], "another epoch"]


def test_cancel_generation_with_no_jobs_does_nothing() -> None:
    sim = _sim()
    sim.apply([CancelGeneration((Epoch(0),))])
    assert sim.effects == [CancelGeneration((Epoch(0),))]
