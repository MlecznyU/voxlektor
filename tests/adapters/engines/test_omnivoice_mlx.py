"""Real-model tests (``-m model``): Apple Silicon, ~3 GB RAM, the pinned revision is downloaded."""

import time
from pathlib import Path

import pytest

from tests.fixtures.samples import PL_PARAGRAPH_110
from voxlektor.adapters.engines.omnivoice_mlx import OMNIVOICE_REVISION, OmniVoiceEngine
from voxlektor.adapters.storage.voices_fs import FsVoiceRepository
from voxlektor.app.voice_prompts import get_or_build_prompt
from voxlektor.testing.engine_contract import run_engine_contract

pytestmark = pytest.mark.model

VOICE = "v0-default"


def _bundled_repo(tmp_path: Path) -> FsVoiceRepository:
    repo = FsVoiceRepository(tmp_path / "voices")
    repo.install_bundled()
    return repo


def test_spec_pins_the_measured_revision() -> None:
    spec = OmniVoiceEngine().spec
    assert spec.id == "omnivoice"
    assert spec.revision == OMNIVOICE_REVISION
    assert len(OMNIVOICE_REVISION) == 40
    assert spec.sample_rate == 24000


def test_engine_contract_with_the_real_model(tmp_path: Path) -> None:
    engine = OmniVoiceEngine()
    try:
        run_engine_contract(engine, _bundled_repo(tmp_path).ref(VOICE))
    finally:
        engine.unload()


def test_polish_paragraph_latency_with_a_warm_cached_prompt(tmp_path: Path) -> None:
    assert len(PL_PARAGRAPH_110) == 110
    repo = _bundled_repo(tmp_path)
    ref = repo.ref(VOICE)
    engine = OmniVoiceEngine()
    engine.load()
    try:
        get_or_build_prompt(engine, repo, ref)  # builds and writes the cache
        prompt = get_or_build_prompt(engine, repo, ref)  # served from the cache
        list(engine.synthesize("Rozgrzewka.", "pl", prompt))  # warm up kernels
        started = time.perf_counter()
        chunks = list(engine.synthesize(PL_PARAGRAPH_110, "pl", prompt))
        elapsed = time.perf_counter() - started
    finally:
        engine.unload()
    seconds = sum(len(chunk.samples) for chunk in chunks) / engine.spec.sample_rate
    print(f"\nLATENCY 110-char PL paragraph: {elapsed:.2f}s for {seconds:.1f}s of audio")
    assert chunks
