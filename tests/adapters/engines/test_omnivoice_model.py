"""The light OmniVoice model description: the values setup shows, without importing MLX."""

import subprocess
import sys

from voxlektor.adapters.engines import omnivoice_mlx
from voxlektor.adapters.engines.omnivoice_model import MODEL


def test_model_values() -> None:
    assert MODEL.engine_id == "omnivoice"
    assert MODEL.title == "OmniVoice (k2-fsa), 0.6B, bf16"
    assert MODEL.repo == "mlx-community/OmniVoice-bf16"
    assert MODEL.revision == "8fb0b754cad788aaefec690cd55c207e8a628f85"
    assert MODEL.license_name == "CC-BY-NC-4.0"
    assert MODEL.license_url == "https://huggingface.co/k2-fsa/OmniVoice"
    assert MODEL.commercial_use is False
    assert MODEL.hidden is False
    # Measured: the cached snapshot's blobs sum to 1,639,492,513 bytes, rounded up to 0.1 GB.
    assert MODEL.size_bytes == 1_700_000_000


def test_engine_uses_the_same_repo_and_revision() -> None:
    assert MODEL.repo == omnivoice_mlx.REPO
    assert MODEL.revision == omnivoice_mlx.OMNIVOICE_REVISION


def test_import_does_not_load_mlx() -> None:
    code = (
        "import sys, voxlektor.adapters.engines.omnivoice_model; "
        "bad = {'mlx', 'mlx_audio', 'numpy', 'huggingface_hub'} & set(sys.modules); "
        "assert not bad, bad"
    )
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
