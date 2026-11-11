"""Generate ``src/voxlektor/bundled_cues/stopped_en.wav`` once, with the real OmniVoice engine.

Run: ``uv run python scripts/make_cues.py`` (downloads/loads the model, ~3 GB RAM). The clip is
package data afterwards; rerun only to change the wording or the voice.
"""

from importlib import resources
from pathlib import Path

import numpy as np
import soundfile as sf

from voxlektor.bootstrap.registry import load_engine
from voxlektor.contracts.engine import VoiceRef

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "src" / "voxlektor" / "bundled_cues" / "stopped_en.wav"
ENGINE = "omnivoice"
VOICE = "v0-default"
LANG = "en"
TEXT = "Reading stopped because of an error. Run vxl status for details."


def main() -> None:
    voice_dir = resources.files("voxlektor.bundled_voices").joinpath(VOICE)
    with resources.as_file(voice_dir) as path:
        ref = VoiceRef(VOICE, path / "ref.wav", (path / "ref.txt").read_text(encoding="utf-8"))
        engine = load_engine(ENGINE)
        engine.load()
        try:
            prompt = engine.prepare_voice(ref)
            pieces = [piece.samples for piece in engine.synthesize(TEXT, LANG, prompt)]
        finally:
            engine.unload()
    samples = np.concatenate(pieces).astype(np.float32)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(OUTPUT), samples, engine.spec.sample_rate, subtype="PCM_16")
    seconds = len(samples) / engine.spec.sample_rate
    print(f"wrote {OUTPUT.relative_to(ROOT)}: {seconds:.2f} s at {engine.spec.sample_rate} Hz")


if __name__ == "__main__":
    main()
