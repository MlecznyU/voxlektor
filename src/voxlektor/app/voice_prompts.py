"""Voice prompt cache: build a prompt once per engine revision and reuse it from disk."""

from voxlektor.app.ports import VoiceRepository
from voxlektor.contracts.engine import Engine, VoicePrompt, VoiceRef
from voxlektor.contracts.voicefmt import prompt_cache_name


def get_or_build_prompt(
    engine: Engine, repo: VoiceRepository, ref: VoiceRef, *, fresh: bool = False
) -> VoicePrompt:
    """Return the cached prompt for ``ref``, building and caching it when missing or corrupt.

    The cache key is ``prompt_cache_name(spec.id, spec.revision, spec.prompt_format)``.
    ``fresh`` skips the cache read: the prompt is rebuilt from the reference audio and the cache
    entry is overwritten (the retry after a prompt failure).
    """
    spec = engine.spec
    key = prompt_cache_name(spec.id, spec.revision, spec.prompt_format)
    cached = None if fresh else _read_cached(engine, repo, ref, key)
    if cached is not None:
        return cached
    built = engine.prepare_voice(ref)
    repo.write_cache(ref.name, key, engine.serialize_prompt(built))
    return built


def _read_cached(
    engine: Engine, repo: VoiceRepository, ref: VoiceRef, key: str
) -> VoicePrompt | None:
    data = repo.read_cache(ref.name, key)
    if data is None:
        return None
    try:
        return engine.deserialize_prompt(ref.name, data)
    except ValueError:  # the Engine SPI: bad data raises ValueError, and a bad cache is rebuilt
        return None
