"""The suite every engine passes (the fake always; real engines under ``-m model``)."""

from voxlektor.contracts.engine import AudioChunk, Engine, VoicePrompt, VoiceRef

_TEXT = "This is a short sentence for the engine contract."
_LENGTH_TOLERANCE = 0.10


def run_engine_contract(engine: Engine, ref: VoiceRef) -> None:
    """Raise ``AssertionError`` when ``engine`` breaks the Engine SPI."""
    engine.load()
    engine.load()  # idempotent
    prompt = engine.prepare_voice(ref)
    _require(
        prompt.prompt_format == engine.spec.prompt_format,
        f"prompt format {prompt.prompt_format} != spec {engine.spec.prompt_format}",
    )
    lang = "en" if "en" in engine.spec.langs else sorted(engine.spec.langs)[0]
    direct = _synthesize(engine, lang, prompt)

    restored = engine.deserialize_prompt(ref.name, engine.serialize_prompt(prompt))
    again = _synthesize(engine, lang, restored)
    _require(
        abs(again - direct) <= _LENGTH_TOLERANCE * direct,
        f"prompt round trip changed the audio length: {direct} -> {again} samples",
    )

    engine.unload()
    engine.load()
    _synthesize(engine, lang, prompt)


def _synthesize(engine: Engine, lang: str, prompt: VoicePrompt) -> int:
    """Validate every chunk and return the total number of samples."""
    chunks: list[AudioChunk] = list(engine.synthesize(_TEXT, lang, prompt))
    _require(len(chunks) >= 1, "synthesize must yield at least one chunk")
    for chunk in chunks:
        _require(chunk.samples.dtype.name == "float32", "chunks must be float32")
        _require(chunk.samples.ndim == 1, "chunks must be mono (1-D)")
        _require(
            chunk.sample_rate == engine.spec.sample_rate,
            f"chunk sample rate {chunk.sample_rate} != spec {engine.spec.sample_rate}",
        )
    return sum(len(chunk.samples) for chunk in chunks)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
