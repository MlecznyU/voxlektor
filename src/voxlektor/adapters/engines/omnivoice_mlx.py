"""OmniVoice via mlx-audio: pinned model revision, numpy copies out, no file I/O of its own.

(The only file access is the library reading the reference WAV in ``prepare_voice``.)

MLX keeps freed buffers in its cache for reuse and never returns them on its own: in a warm daemon
it grew to 22.8 GB over 20 generates on a 32 GB Mac, and generation slowed 2-3x (swap). So the
cache is cleared after every synthesis (exhausted, failed or closed early) and after
``prepare_voice``; reuse within one generate is kept. No ``mx.set_cache_limit`` cap: clearing at
these points already bounds the cache, and a cap below one generate's working set (which reaches
a few GB) would only add allocations in the middle of a generate.

Native rate: the model fills exactly the duration it is asked for, so every synthesis passes
``duration_s`` sized from the voice's own pace (stored in the prompt, ``PROMPT_FORMAT`` 2) and
divided by ``rate``; see ``omnivoice_pace``.
"""

import gc
import io
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import mlx.core as mx
import numpy as np
import numpy.typing as npt
from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError
from mlx_audio.audio_io import read as audio_read
from mlx_audio.tts.models.omnivoice.utils import create_voice_clone_prompt
from mlx_audio.tts.utils import load_model

from voxlektor.adapters.engines.omnivoice_model import OMNIVOICE_REVISION, REPO
from voxlektor.adapters.engines.omnivoice_pace import ref_lang, ref_pace, target_seconds
from voxlektor.contracts.engine import (
    ENGINE_API,
    AudioChunk,
    Capability,
    EngineSpec,
    VoicePrompt,
    VoiceRef,
)

NUM_STEPS = 16
LANGS = {"pl": "polish", "en": "english"}
PROMPT_FORMAT = 2  # 2: + the voice's pace and reference language (native rate)
_SAMPLE_RATE = 24000
_FRAMES_PER_SECOND = _SAMPLE_RATE / 960  # one audio token frame = 960 samples (25 per second)
_MAX_REF_SECONDS = 10.0
_TOKEN_DIMS = 2  # frames x codebooks, as returned by mlx-audio: [T_ref, 8]

_log = logging.getLogger(__name__)


class _Result(Protocol):
    audio: mx.array


class _Model(Protocol):
    audio_tokenizer: object

    def generate(  # noqa: PLR0913  # mirrors the keyword-only library call
        self,
        *,
        text: str,
        language: str,
        num_steps: int,
        ref_tokens: mx.array,
        ref_text: str,
        duration_s: float,
    ) -> Iterator[_Result]: ...


@dataclass(frozen=True)
class _OmniPrompt:
    """Reference tokens (frames x codebooks, numpy), their transcript and the voice's pace.

    ``pace`` is speech weight per second of the reference the tokens hold (see
    ``omnivoice_pace``); ``ref_lang`` is the transcript's language.
    """

    tokens: npt.NDArray[np.int32]
    ref_text: str
    pace: float
    ref_lang: str


class OmniVoiceEngine:
    PROMPT_FORMAT: int = PROMPT_FORMAT

    def __init__(self) -> None:
        self.spec = EngineSpec(
            id="omnivoice",
            revision=OMNIVOICE_REVISION,
            prompt_format=PROMPT_FORMAT,
            sample_rate=_SAMPLE_RATE,
            langs=frozenset(LANGS),
            capabilities=frozenset(
                {Capability.LIVE, Capability.RENDER, Capability.CLONE, Capability.RATE}
            ),
            api=ENGINE_API,
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
        try:
            tokens = create_voice_clone_prompt(
                str(ref.ref_wav), tokenizer=model.audio_tokenizer, max_duration_s=_MAX_REF_SECONDS
            )
            frames = np.array(tokens, dtype=np.int32)
        finally:
            mx.clear_cache()  # encoding the reference leaves ~2.5 GiB of freed buffers cached
        # The pace is measured on what the prompt really holds: the library has already removed
        # long silences (and trimmed an over-long reference) before encoding.
        seconds = frames.shape[0] / _FRAMES_PER_SECOND
        pace = ref_pace(ref.ref_text, seconds, kept=_kept_fraction(ref, seconds))
        data = _OmniPrompt(frames, ref.ref_text, pace, ref_lang(ref.ref_text))
        return VoicePrompt(ref.name, PROMPT_FORMAT, data)

    def serialize_prompt(self, p: VoicePrompt) -> bytes:
        data = _omni_prompt(p)
        buffer = io.BytesIO()
        np.savez(
            buffer,
            tokens=data.tokens,
            ref_text=np.array(data.ref_text),
            pace=np.array(data.pace, dtype=np.float64),
            ref_lang=np.array(data.ref_lang),
        )
        return buffer.getvalue()

    def deserialize_prompt(self, voice: str, data: bytes) -> VoicePrompt:
        try:
            with np.load(io.BytesIO(data), allow_pickle=False) as archive:
                tokens = np.asarray(archive["tokens"])
                ref_text = np.asarray(archive["ref_text"])
                pace = np.asarray(archive["pace"])
                lang = np.asarray(archive["ref_lang"])
        except Exception as exc:
            raise ValueError("corrupt OmniVoice prompt") from exc
        if (
            tokens.ndim != _TOKEN_DIMS
            or tokens.dtype != np.int32
            or ref_text.shape != ()
            or ref_text.dtype.kind != "U"
            or pace.shape != ()
            or pace.dtype != np.float64
            or not (np.isfinite(pace) and float(pace) > 0.0)
            or lang.shape != ()
            or lang.dtype.kind != "U"
            or str(lang) not in LANGS
        ):
            raise ValueError("unexpected OmniVoice prompt layout")
        prompt = _OmniPrompt(tokens, str(ref_text), float(pace), str(lang))
        return VoicePrompt(voice, PROMPT_FORMAT, prompt)

    def synthesize(
        self,
        text: str,
        lang: str,
        prompt: VoicePrompt,
        *,
        rate: float = 1.0,
        steps: int | None = None,
    ) -> Iterator[AudioChunk]:
        """``rate`` sets the requested duration (``omnivoice_pace``); ``steps`` the diffusion."""
        language = LANGS.get(lang)
        if language is None:
            raise ValueError(f"unsupported language {lang!r}")
        if steps is not None and steps < 1:
            raise ValueError(f"steps must be at least 1, got {steps}")
        data = _omni_prompt(prompt)
        duration_s = target_seconds(text, lang, data.pace, data.ref_lang, rate)
        model = self._require_model()
        return self._chunks(
            model, text, language, data, duration_s=duration_s, num_steps=steps or NUM_STEPS
        )

    def _chunks(  # noqa: PLR0913  # private: the validated arguments of one synthesis
        self,
        model: _Model,
        text: str,
        language: str,
        data: _OmniPrompt,
        *,
        duration_s: float,
        num_steps: int,
    ) -> Iterator[AudioChunk]:
        results = model.generate(
            text=text,
            language=language,
            num_steps=num_steps,
            ref_tokens=mx.array(data.tokens),
            ref_text=data.ref_text,
            duration_s=duration_s,
        )
        try:
            for result in results:
                yield AudioChunk(np.array(result.audio, np.float32).copy(), _SAMPLE_RATE)
        finally:  # exhausted, failed or closed early (a cancelled chunk)
            mx.clear_cache()

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


def _kept_fraction(ref: VoiceRef, seconds: float) -> float:
    """The share of the reference the prompt holds: 1.0 unless the library cut it.

    A reference longer than ``_MAX_REF_SECONDS`` is cut (at its largest silence) while the
    transcript stays whole. ``seconds`` (after the cut and silence removal) over the file's
    length underestimates what was kept, so the pace errs on the slow side, never rushed.
    """
    audio, rate = audio_read(str(ref.ref_wav), always_2d=True)  # the library's own reader
    total = audio.shape[0] / rate
    if total <= _MAX_REF_SECONDS:
        return 1.0
    kept = min(1.0, seconds / total)
    _log.warning(
        "voice %s: the reference (%.1f s) was cut to %.0f s; its pace counts %.0f %% of the "
        "transcript",
        ref.name,
        total,
        _MAX_REF_SECONDS,
        kept * 100,
    )
    return kept


def _omni_prompt(p: VoicePrompt) -> _OmniPrompt:
    if not isinstance(p.data, _OmniPrompt):
        raise TypeError("not an OmniVoice prompt")
    return p.data
