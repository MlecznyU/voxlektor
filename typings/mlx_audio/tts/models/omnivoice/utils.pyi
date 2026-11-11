import mlx.core as mx

def create_voice_clone_prompt(
    ref_audio_path: str,
    tokenizer: object = ...,
    ref_text: str | None = ...,
    preprocess: bool = ...,
    max_duration_s: float = ...,
) -> mx.array: ...
