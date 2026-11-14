"""OmniVoice via mlx-audio: pinned model revision, numpy copies out, no file I/O of its own.

(The only file access is the library reading the reference WAV in ``prepare_voice``.)
"""

import gc
import io
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import mlx.core as mx
import numpy as np
import numpy.typing as npt
from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError
from mlx_audio.tts.models.omnivoice.utils import create_voice_clone_prompt
from mlx_audio.tts.utils import load_model

from voxlektor.contracts.engine import AudioChunk, Capability, EngineSpec, VoicePrompt, VoiceRef

REPO = "mlx-community/OmniVoice-bf16"
OMNIVOICE_REVISION = "8fb0b754cad788aaefec690cd55c207e8a628f85"
NUM_STEPS = 16
LANGS = {"pl": "polish", "en": "english"}
PROMPT_FORMAT = 1
_SAMPLE_RATE = 24000
_MAX_REF_SECONDS = 10.0
_TOKEN_DIMS = 2  # frames x codebooks, as returned by mlx-audio: [T_ref, 8]


class _Result(Protocol):
    audio: mx.array


class _Model(Protocol):
    audio_tokenizer: object

    def generate(
        self, *, text: str, language: str, num_steps: int, ref_tokens: mx.array, ref_text: str
    ) -> Iterator[_Result]: ...


@dataclass(frozen=True)
class _OmniPrompt:
    """Reference audio tokens (frames x codebooks, numpy) and the transcript they came from."""

    tokens: npt.NDArray[np.int32]
    ref_text: str


class OmniVoiceEngine:
    PROMPT_FORMAT: int = PROMPT_FORMAT

    def __init__(self) -> None:
        self.spec = EngineSpec(
            id="omnivoice",
            revision=OMNIVOICE_REVISION,
            prompt_format=PROMPT_FORMAT,
            sample_rate=_SAMPLE_RATE,
            langs=frozenset(LANGS),
            capabilities=frozenset({Capability.LIVE, Capability.RENDER, Capability.CLONE}),
        )
        self._model: _Model | None = None

    def load(self) -> None:
        if self._model is not None:
            return
        self._model = load_model(Path(_model_path()))

    def unload(self) -> None:
        self._model = None
        gc.collect()
        mx.clear_cache()

    def prepare_voice(self, ref: VoiceRef) -> VoicePrompt:
        model = self._require_model()
        tokens = create_voice_clone_prompt(
            str(ref.ref_wav), tokenizer=model.audio_tokenizer, max_duration_s=_MAX_REF_SECONDS
        )
        data = _OmniPrompt(np.array(tokens, dtype=np.int32), ref.ref_text)
        return VoicePrompt(ref.name, PROMPT_FORMAT, data)

    def serialize_prompt(self, p: VoicePrompt) -> bytes:
        data = _omni_prompt(p)
        buffer = io.BytesIO()
        np.savez(buffer, tokens=data.tokens, ref_text=np.array(data.ref_text))
        return buffer.getvalue()

    def deserialize_prompt(self, voice: str, data: bytes) -> VoicePrompt:
        try:
            with np.load(io.BytesIO(data), allow_pickle=False) as archive:
                tokens = np.asarray(archive["tokens"])
                ref_text = np.asarray(archive["ref_text"])
        except Exception as exc:
            raise ValueError("corrupt OmniVoice prompt") from exc
        if (
            tokens.ndim != _TOKEN_DIMS
            or tokens.dtype != np.int32
            or ref_text.shape != ()
            or ref_text.dtype.kind != "U"
        ):
            raise ValueError("unexpected OmniVoice prompt layout")
        return VoicePrompt(voice, PROMPT_FORMAT, _OmniPrompt(tokens, str(ref_text)))

    def synthesize(self, text: str, lang: str, prompt: VoicePrompt) -> Iterator[AudioChunk]:
        language = LANGS.get(lang)
        if language is None:
            raise ValueError(f"unsupported language {lang!r}")
        data = _omni_prompt(prompt)
        model = self._require_model()
        return self._chunks(model, text, language, data)

    def _chunks(
        self, model: _Model, text: str, language: str, data: _OmniPrompt
    ) -> Iterator[AudioChunk]:
        results = model.generate(
            text=text,
            language=language,
            num_steps=NUM_STEPS,
            ref_tokens=mx.array(data.tokens),
            ref_text=data.ref_text,
        )
        for result in results:
            yield AudioChunk(np.array(result.audio, np.float32).copy(), _SAMPLE_RATE)

    def _require_model(self) -> _Model:
        if self._model is None:
            raise RuntimeError("engine not loaded")
        return self._model


def _model_path() -> str:
    """The pinned snapshot, offline first: the Hub is contacted only on a cache miss."""
    try:
        return snapshot_download(REPO, revision=OMNIVOICE_REVISION, local_files_only=True)
    except LocalEntryNotFoundError:
        return snapshot_download(REPO, revision=OMNIVOICE_REVISION)


def _omni_prompt(p: VoicePrompt) -> _OmniPrompt:
    if not isinstance(p.data, _OmniPrompt):
        raise TypeError("not an OmniVoice prompt")
    return p.data
