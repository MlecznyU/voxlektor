"""OmniVoice prompt serialization and eager validation: no model is loaded here."""

import io
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from huggingface_hub.errors import LocalEntryNotFoundError

from tests.fixtures.samples import PL_PARAGRAPH_110
from voxlektor.adapters.engines import omnivoice_mlx
from voxlektor.adapters.engines.omnivoice_mlx import OmniVoiceEngine
from voxlektor.contracts.engine import Capability, VoicePrompt, VoiceRef

REF = VoiceRef("v0", Path("/voices/v0/ref.wav"), PL_PARAGRAPH_110)
TOKENS = np.arange(40, dtype=np.int32).reshape(5, 8)  # frames x codebooks


PACE = np.array(12.5, dtype=np.float64)
LANG = np.array("pl")


def _blob(
    tokens: npt.NDArray[Any],
    ref_text: npt.NDArray[Any],
    pace: npt.NDArray[Any] = PACE,
    ref_lang: npt.NDArray[Any] = LANG,
) -> bytes:
    buffer = io.BytesIO()
    np.savez(buffer, tokens=tokens, ref_text=ref_text, pace=pace, ref_lang=ref_lang)
    return buffer.getvalue()


def _valid() -> bytes:
    return _blob(TOKENS, np.array(PL_PARAGRAPH_110))


def test_the_engine_speaks_rates_itself() -> None:
    assert Capability.RATE in OmniVoiceEngine().spec.capabilities


def test_prompt_round_trip_keeps_tokens_polish_text_and_pace() -> None:
    engine = OmniVoiceEngine()
    prompt = engine.deserialize_prompt("v0", _valid())
    assert prompt.voice == "v0"
    assert prompt.prompt_format == engine.spec.prompt_format == 2
    again = engine.serialize_prompt(prompt)
    with np.load(io.BytesIO(again), allow_pickle=False) as archive:
        assert np.array_equal(archive["tokens"], TOKENS)
        assert str(archive["ref_text"]) == PL_PARAGRAPH_110
        assert float(archive["pace"]) == 12.5
        assert str(archive["ref_lang"]) == "pl"
    assert engine.deserialize_prompt("v0", again).data is not None


def test_serialized_prompt_contains_no_pickle() -> None:
    engine = OmniVoiceEngine()
    blob = engine.serialize_prompt(engine.deserialize_prompt("v0", _valid()))
    with np.load(io.BytesIO(blob), allow_pickle=False) as archive:
        assert sorted(archive.files) == ["pace", "ref_lang", "ref_text", "tokens"]


def test_a_format_1_prompt_without_pace_is_rejected() -> None:
    old = io.BytesIO()
    np.savez(old, tokens=TOKENS, ref_text=np.array(PL_PARAGRAPH_110))
    with pytest.raises(ValueError, match="OmniVoice prompt"):
        OmniVoiceEngine().deserialize_prompt("v0", old.getvalue())


@pytest.mark.parametrize(
    ("pace", "ref_lang"),
    [
        (np.array(12.5, dtype=np.float32), LANG),
        (np.array([12.5]), LANG),
        (np.array(0.0), LANG),
        (np.array(-3.0), LANG),
        (np.array(np.nan), LANG),
        (np.array(np.inf), LANG),
        (np.array("12.5"), LANG),
        (PACE, np.array("de")),
        (PACE, np.array(["pl"])),
        (PACE, np.array(1)),
    ],
    ids=[
        "pace-f32",
        "pace-vector",
        "pace-zero",
        "pace-negative",
        "pace-nan",
        "pace-inf",
        "pace-text",
        "lang-unknown",
        "lang-vector",
        "lang-int",
    ],
)
def test_deserialize_rejects_a_bad_pace_or_language(
    pace: npt.NDArray[Any], ref_lang: npt.NDArray[Any]
) -> None:
    with pytest.raises(ValueError, match="OmniVoice prompt"):
        OmniVoiceEngine().deserialize_prompt(
            "v0", _blob(TOKENS, np.array("x"), pace=pace, ref_lang=ref_lang)
        )


@pytest.mark.parametrize(
    ("rate", "steps", "message"),
    [(0.0, None, "rate"), (-1.0, None, "rate"), (1.0, 0, "steps"), (1.0, -4, "steps")],
)
def test_synthesize_rejects_a_bad_rate_or_steps_eagerly(
    rate: float, steps: int | None, message: str
) -> None:
    engine = OmniVoiceEngine()
    prompt = engine.deserialize_prompt("v0", _valid())
    with pytest.raises(ValueError, match=message):
        engine.synthesize("x", "pl", prompt, rate=rate, steps=steps)


@pytest.mark.parametrize(
    "garbage",
    [b"", b"garbage", _valid()[:-20], _valid()[:10], b"PK\x03\x04" + b"\x00" * 30],
    ids=["empty", "garbage", "truncated", "tiny", "bad-zip"],
)
def test_deserialize_rejects_garbage(garbage: bytes) -> None:
    with pytest.raises(ValueError, match="OmniVoice prompt"):
        OmniVoiceEngine().deserialize_prompt("v0", garbage)


@pytest.mark.parametrize(
    ("tokens", "ref_text"),
    [
        (TOKENS.astype(np.int64), np.array("x")),
        (TOKENS.astype(np.float32), np.array("x")),
        (TOKENS.reshape(-1), np.array("x")),
        (TOKENS, np.array(["x", "y"])),
        (TOKENS, np.array(7)),
        (TOKENS, np.array(b"bytes")),
    ],
    ids=["int64", "float32", "1-d", "text-vector", "text-int", "text-bytes"],
)
def test_deserialize_rejects_wrong_layouts(
    tokens: npt.NDArray[Any], ref_text: npt.NDArray[Any]
) -> None:
    with pytest.raises(ValueError, match="OmniVoice prompt"):
        OmniVoiceEngine().deserialize_prompt("v0", _blob(tokens, ref_text))


def test_deserialize_rejects_a_bare_array_and_missing_keys() -> None:
    bare = io.BytesIO()
    np.save(bare, TOKENS)
    only_tokens = io.BytesIO()
    np.savez(only_tokens, tokens=TOKENS)
    for data in (bare.getvalue(), only_tokens.getvalue()):
        with pytest.raises(ValueError, match="OmniVoice prompt"):
            OmniVoiceEngine().deserialize_prompt("v0", data)


def test_pickled_object_arrays_are_refused() -> None:
    buffer = io.BytesIO()
    np.savez(buffer, tokens=TOKENS, ref_text=np.array({"a": 1}, dtype=object), allow_pickle=True)
    with pytest.raises(ValueError, match="OmniVoice prompt"):
        OmniVoiceEngine().deserialize_prompt("v0", buffer.getvalue())


def test_serialize_rejects_foreign_prompt_data() -> None:
    with pytest.raises(TypeError, match="OmniVoice prompt"):
        OmniVoiceEngine().serialize_prompt(VoicePrompt("v0", 1, "not omni"))


def test_synthesize_rejects_an_unsupported_language_eagerly() -> None:
    engine = OmniVoiceEngine()
    prompt = engine.deserialize_prompt("v0", _valid())
    with pytest.raises(ValueError, match="language 'de'"):
        engine.synthesize("x", "de", prompt)


def test_synthesize_rejects_a_foreign_prompt_eagerly() -> None:
    with pytest.raises(TypeError, match="OmniVoice prompt"):
        OmniVoiceEngine().synthesize("x", "en", VoicePrompt("v0", 1, "not omni"))


class _Hub:
    """Fake ``snapshot_download``: records calls; ``cached`` decides the offline answer."""

    def __init__(self, *, cached: bool) -> None:
        self.cached = cached
        self.calls: list[dict[str, object]] = []

    def __call__(self, repo_id: str, *, revision: str | None = None, **kwargs: object) -> str:
        self.calls.append({"repo_id": repo_id, "revision": revision, **kwargs})
        if kwargs.get("local_files_only") and not self.cached:
            raise LocalEntryNotFoundError("not in the cache")
        return "/hub/snapshot"


@pytest.mark.parametrize("cached", [True, False], ids=["cache-hit", "cache-miss"])
def test_load_is_offline_first(monkeypatch: pytest.MonkeyPatch, cached: bool) -> None:
    """S3: no Hub call per load when the pinned snapshot is cached."""
    hub = _Hub(cached=cached)
    loaded: list[Path] = []

    def load_model(path: Path) -> object:
        loaded.append(path)
        return object()

    monkeypatch.setattr(omnivoice_mlx, "snapshot_download", hub)
    monkeypatch.setattr(omnivoice_mlx, "load_model", load_model)
    OmniVoiceEngine().load()
    pinned: dict[str, object] = {
        "repo_id": omnivoice_mlx.REPO,
        "revision": omnivoice_mlx.OMNIVOICE_REVISION,
    }
    expected: list[dict[str, object]] = [{**pinned, "local_files_only": True}]
    if not cached:
        expected.append(pinned)  # the normal download, only on a cache miss
    assert hub.calls == expected
    assert loaded == [Path("/hub/snapshot")]


class _Tokenizer:
    pass


class _LoadedModel:
    audio_tokenizer = _Tokenizer()


@pytest.mark.parametrize(
    "case",
    [
        (8.0, 200, 1.0),  # whole reference: the whole transcript
        (20.0, 250, 0.5),  # cut to 10 s by the library: half of the transcript counts
    ],
)
def test_pace_of_a_reference_cut_by_the_library_counts_only_the_kept_part(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    case: tuple[float, int, float],
) -> None:
    import soundfile as sf  # noqa: PLC0415  # test only

    wav_seconds, frames, kept = case

    from voxlektor.adapters.engines.omnivoice_pace import speech_weight  # noqa: PLC0415

    wav = tmp_path / "ref.wav"
    sf.write(str(wav), np.zeros(round(wav_seconds * 24000), dtype=np.float32), 24000)
    text = "abcd efgh " * 12 * round(1 / kept)  # 12.6 weight/s of what the prompt holds
    pace = speech_weight(text.strip()) * kept / (frames / 25)
    assert 8.0 < pace < 18.0  # inside PACE_RANGE: no clamp hides the result
    warned = kept < 1.0

    def clone(path: str, *, tokenizer: object, max_duration_s: float) -> npt.NDArray[np.int32]:
        return np.zeros((frames, 8), dtype=np.int32)

    monkeypatch.setattr(omnivoice_mlx, "create_voice_clone_prompt", clone)
    engine = OmniVoiceEngine()
    engine._model = _LoadedModel()  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue]
    prompt = engine.prepare_voice(VoiceRef("v0", wav, text))
    blob = engine.serialize_prompt(prompt)
    with np.load(io.BytesIO(blob), allow_pickle=False) as archive:
        assert float(archive["pace"]) == pytest.approx(pace)
    assert ("cut" in caplog.text) is warned


def test_calls_before_load_fail_clearly_and_eagerly() -> None:
    engine = OmniVoiceEngine()
    prompt = engine.deserialize_prompt("v0", _valid())
    with pytest.raises(RuntimeError, match="not loaded"):
        engine.synthesize("x", "pl", prompt)
    with pytest.raises(RuntimeError, match="not loaded"):
        engine.prepare_voice(REF)
