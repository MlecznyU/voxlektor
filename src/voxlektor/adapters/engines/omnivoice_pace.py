"""OmniVoice speaking pace: how long to ask the model to speak (no MLX, plain arithmetic).

OmniVoice is non-autoregressive: it fills exactly the duration it is given (measured: trimmed
speech within 0-10 % of the request for every bundled voice), speaking slower or faster to fit.
So the duration *is* the pace. mlx-audio's default sizes it against "Nice to meet you." + 15 %,
not the voice, and the model pads the surplus with silence; we size it from the voice instead.

Length is measured in *speech weight*, the phonetic weights of mlx-audio's
``RuleDurationEstimator`` (Latin letter 1.0, space 0.2, punctuation 0.5, digit 3.5, combining
mark 0; any other letter 1.0 - only Polish and English are spoken here), so a number such as
"2026" gets the time it takes to say it, which a plain character count would not give.

``ref_pace`` is the voice's own pace: weight of the reference transcript per second of the
reference audio the prompt really holds (after the library's silence removal and trimming).
For the other language a fixed correction ratio applies (``LANG_CORRECTION``), measured once on
the bundled voices by ``tests/adapters/engines/test_omnivoice_mlx.py``.
"""

import unicodedata

from voxlektor.domain.text.lang import detect_lang

# The weights of mlx-audio's RuleDurationEstimator for the scripts we speak.
_LETTER = 1.0
_SPACE = 0.2
_PUNCTUATION = 0.5
_DIGIT = 3.5
_MARK = 0.0

# Requested length = max(estimate * MARGIN / rate, weight / MAX_PACE) + TAIL_S.
# MARGIN leaves a little room in proportion; TAIL_S (not divided by the rate) gives the last
# word time to finish: without it the model fills the window to its last sample and syllables
# were lost at rate 1.5 (measured with an envelope syllable count). MAX_PACE caps how fast the
# model is asked to speak (syllables went missing at ~17 weight/s without the tail), so a fast
# voice at a high rate speaks a little slower than asked rather than dropping words. Trimming
# removes whatever silence the room produces.
MARGIN = 1.05
TAIL_S = 0.3
MAX_PACE = 18.0
MIN_SECONDS = 0.5  # the library's own floor is 10 frames (0.4 s)
# Plausible speaking paces (speech weight per second); the bundled voices measure 11.2-14.5.
PACE_RANGE = (8.0, 18.0)

# pace(lang) = ref_pace * LANG_CORRECTION[(reference language, lang)]; 1.0 when they match.
# Measured 2026-11-16 on the 8 bundled (Polish) voices at equal weight pace, two runs: syllable
# rate EN / PL = 0.995 (sd 0.09) and 1.030 (sd 0.064), i.e. within noise of 1: speech weight
# already carries the difference between the languages.
LANG_CORRECTION: dict[tuple[str, str], float] = {("pl", "en"): 1.0, ("en", "pl"): 1.0}


def speech_weight(text: str) -> float:
    """The phonetic weight of ``text`` (see the module doc)."""
    return sum(_char_weight(ch) for ch in text)


def _char_weight(ch: str) -> float:
    category = unicodedata.category(ch)
    if category.startswith("M"):
        return _MARK
    if ch == " " or category.startswith("Z"):
        return _SPACE
    if category.startswith(("P", "S")):
        return _PUNCTUATION
    if category.startswith("N"):
        return _DIGIT
    if category.startswith("C"):  # tabs and newlines count as a word boundary
        return _SPACE
    return _LETTER


def ref_pace(ref_text: str, ref_seconds: float, *, kept: float = 1.0) -> float:
    """Speech weight per second of the reference, kept within ``PACE_RANGE``.

    ``kept`` is the fraction of the reference the prompt still holds: a reference longer than
    the library's ``max_duration_s`` is cut while the transcript stays whole, so only that
    fraction of the transcript's weight counts (otherwise the pace comes out too fast and the
    speech is rushed). The clamp guards against any other mismatch between transcript and
    audio. ``ValueError`` on an empty reference or a ``kept`` outside (0, 1].
    """
    if not 0.0 < kept <= 1.0:
        raise ValueError(f"kept must be in (0, 1], got {kept}")
    weight = speech_weight(ref_text.strip()) * kept
    if ref_seconds <= 0.0 or weight <= 0.0:
        raise ValueError("the voice reference has no speech to measure its pace from")
    low, high = PACE_RANGE
    return min(high, max(low, weight / ref_seconds))


def ref_lang(ref_text: str) -> str:
    """The language of the reference transcript ("pl" or "en"; ambiguous text counts as "en")."""
    return detect_lang(ref_text, "en")


def pace_for(pace: float, reference_lang: str, lang: str) -> float:
    """The voice's pace (speech weight per second) when it speaks ``lang``."""
    if lang == reference_lang:
        return pace
    return pace * LANG_CORRECTION.get((reference_lang, lang), 1.0)


def target_seconds(text: str, lang: str, pace: float, reference_lang: str, rate: float) -> float:
    """How long to ask the model to speak ``text`` in ``lang`` at ``rate`` (1.0 = own pace)."""
    if rate <= 0.0:
        raise ValueError(f"rate must be positive, got {rate}")
    weight = speech_weight(text.strip())
    spoken = weight / pace_for(pace, reference_lang, lang) * MARGIN / rate
    return max(MIN_SECONDS, max(spoken, weight / MAX_PACE) + TAIL_S)
