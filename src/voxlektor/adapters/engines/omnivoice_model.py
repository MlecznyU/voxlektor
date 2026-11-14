"""What OmniVoice needs from Hugging Face, importable without MLX (setup reads it).

``omnivoice_mlx`` takes its repository and revision from here, so the two cannot drift apart.
"""

from voxlektor.contracts.engine import ModelInfo

REPO = "mlx-community/OmniVoice-bf16"
OMNIVOICE_REVISION = "8fb0b754cad788aaefec690cd55c207e8a628f85"

MODEL = ModelInfo(
    engine_id="omnivoice",
    title="OmniVoice (k2-fsa), 0.6B, bf16",
    repo=REPO,
    revision=OMNIVOICE_REVISION,
    # Measured 2026-11-14: the pinned snapshot's blobs sum to 1,639,492,513 bytes; rounded up
    # to 0.1 GB.
    size_bytes=1_700_000_000,
    license_name="CC-BY-NC-4.0",
    license_url="https://huggingface.co/k2-fsa/OmniVoice",
    commercial_use=False,
)
